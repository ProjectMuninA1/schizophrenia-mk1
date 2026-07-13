"""End-to-end orchestration: train, predict, and retrain."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import Config
from .data import clean_data, load_market_data
from .dataset import prepare_dataset
from .evaluate import evaluate_predictions
from .feature_selection import select_top_features
from .features import FEATURE_COLUMNS, add_features
from .labels import add_labels
from .models import LSTMModel, SklearnModel, build_models
from .risk import build_trade_plan, to_signal


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def load_and_engineer(cfg: Config) -> pd.DataFrame:
    """Download -> clean -> add technical features. Returns a raw (unscaled) frame."""
    raw = load_market_data(
        cfg.data.ticker, cfg.data.start, cfg.data.end, cfg.data.interval
    )
    clean = clean_data(raw)
    return add_features(clean)


def _artifacts_dir(cfg: Config) -> Path:
    d = Path(cfg.paths.artifacts_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_filename(name: str, kind: str) -> str:
    return f"model_{name}.pt" if kind == "lstm" else f"model_{name}.joblib"


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(cfg: Config, verbose: bool = True) -> dict[str, Any]:
    """Run the full training pipeline and persist artifacts. Returns a summary."""
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"[data] downloading {cfg.data.ticker} ({cfg.data.start} -> {cfg.data.end or 'today'})")
    frame = load_and_engineer(cfg)
    frame = add_labels(frame, cfg.labels.horizon)
    log(f"[data] {len(frame)} rows after feature engineering")

    ds = prepare_dataset(
        frame,
        cfg.dataset.train_frac,
        cfg.dataset.val_frac,
        cfg.dataset.test_frac,
        feature_names=FEATURE_COLUMNS,
    )
    log(f"[split] train={len(ds.X_train)} val={len(ds.X_val)} test={len(ds.X_test)}")

    selected, importances = select_top_features(
        ds.X_train, ds.y_train, cfg.feature_selection.top_n
    )
    log(f"[select] top {len(selected)} features: {', '.join(selected)}")
    ds_sel = ds.select(selected)

    models = build_models(cfg.models)
    metrics: dict[str, dict[str, dict]] = {}
    for name, model in models.items():
        log(f"[train] fitting {name} ...")
        model.fit(ds_sel)
        val_metrics = evaluate_predictions(ds_sel.y_val.values, model.predict_proba(ds_sel.X_val))
        test_metrics = evaluate_predictions(
            ds_sel.y_test.values, model.predict_proba(ds_sel.X_test)
        )
        metrics[name] = {"val": val_metrics, "test": test_metrics}
        log(
            f"[eval]  {name:<20} "
            f"val_{cfg.selection_metric}={val_metrics[cfg.selection_metric]:.4f}  "
            f"test_{cfg.selection_metric}={test_metrics[cfg.selection_metric]:.4f}"
        )

    metric = cfg.selection_metric

    def _score(name: str) -> float:
        v = metrics[name]["val"][metric]
        return v if v == v else -1.0  # treat NaN as worst

    best_name = max(metrics, key=_score)
    best_model = models[best_name]
    log(f"[best]  {best_name} (by val {metric}={metrics[best_name]['val'][metric]:.4f})")

    # Persist artifacts.
    out = _artifacts_dir(cfg)
    joblib.dump(ds.scaler, out / "scaler.joblib")
    (out / "selected_features.json").write_text(json.dumps(selected, indent=2))
    (out / "scaler_features.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (out / "feature_importances.json").write_text(
        json.dumps(importances.round(6).to_dict(), indent=2)
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    for name, model in models.items():
        model.save(out / _model_filename(name, model.kind))

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "ticker": cfg.data.ticker,
        "best_model": best_name,
        "best_model_kind": best_model.kind,
        "best_model_file": _model_filename(best_name, best_model.kind),
        "selection_metric": metric,
        "metrics": metrics,
        "config": cfg.to_dict(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log(f"[save]  artifacts written to {out.resolve()}")

    return metadata


# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
def _load_best_model(out: Path, metadata: dict):
    kind = metadata["best_model_kind"]
    path = out / metadata["best_model_file"]
    if kind == "lstm":
        return LSTMModel.load(path)
    return SklearnModel.load(path)


def predict(cfg: Config, verbose: bool = True) -> dict[str, Any]:
    """Download the latest data, recompute indicators, and emit a trade signal."""
    out = _artifacts_dir(cfg)
    meta_path = out / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No trained model found in {out.resolve()}. Run `train` first."
        )

    metadata = json.loads(meta_path.read_text())
    scaler = joblib.load(out / "scaler.joblib")
    selected = json.loads((out / "selected_features.json").read_text())
    scaler_features = json.loads((out / "scaler_features.json").read_text())
    model = _load_best_model(out, metadata)

    frame = load_and_engineer(cfg)
    frame = frame.dropna(subset=scaler_features)
    if frame.empty:
        raise ValueError("Not enough recent data to compute a full feature vector.")

    scaled_all = pd.DataFrame(
        scaler.transform(frame[scaler_features].values),
        index=frame.index,
        columns=scaler_features,
    )
    x_selected = scaled_all[selected]

    proba = model.predict_proba(x_selected)
    p_up = float(proba[-1])

    last = frame.iloc[-1]
    entry_price = float(last["Close"])
    atr = float(last["ATR14"])
    as_of = frame.index[-1]

    signal = to_signal(p_up, cfg.signal)
    plan = build_trade_plan(signal, p_up, entry_price, atr, cfg.risk)

    result = {
        "ticker": cfg.data.ticker,
        "as_of": str(as_of.date()) if hasattr(as_of, "date") else str(as_of),
        "model": metadata["best_model"],
        "probability_up": round(p_up, 4),
        **plan.as_dict(),
    }

    if verbose:
        print(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------- #
# Retrain
# --------------------------------------------------------------------------- #
def retrain(cfg: Config, verbose: bool = True) -> dict[str, Any]:
    """Retrain on the newest data and archive a timestamped copy of metadata."""
    metadata = train(cfg, verbose=verbose)
    out = _artifacts_dir(cfg)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = out / "history"
    archive.mkdir(exist_ok=True)
    (archive / f"metadata_{stamp}.json").write_text(json.dumps(metadata, indent=2))
    if verbose:
        print(f"[retrain] archived metadata snapshot -> {archive / f'metadata_{stamp}.json'}")
    return metadata
