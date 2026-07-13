"""Model definitions with a common interface.

Every model exposes:
    fit(dataset)                -> None
    predict_proba(X)            -> np.ndarray of P(class == 1 / BUY)
    save(path) / load(path)

Included: Random Forest, XGBoost, Logistic Regression, and an optional LSTM
(requires torch; skipped gracefully if torch is unavailable).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

try:  # torch is optional
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    _TORCH_AVAILABLE = False


def torch_available() -> bool:
    return _TORCH_AVAILABLE


# --------------------------------------------------------------------------- #
# Tabular (scikit-learn / xgboost) models
# --------------------------------------------------------------------------- #
class SklearnModel:
    """Wraps any estimator exposing ``fit`` and ``predict_proba``."""

    kind = "sklearn"

    def __init__(self, name: str, estimator: Any):
        self.name = name
        self.estimator = estimator

    def fit(self, dataset) -> None:
        self.estimator.fit(dataset.X_train.values, dataset.y_train.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X.values)[:, 1]

    def save(self, path: str | Path) -> None:
        joblib.dump({"name": self.name, "estimator": self.estimator}, path)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnModel":
        blob = joblib.load(path)
        return cls(blob["name"], blob["estimator"])


def build_random_forest(cfg: dict) -> SklearnModel:
    return SklearnModel(
        "random_forest",
        RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", None),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=42,
            n_jobs=-1,
        ),
    )


def build_xgboost(cfg: dict) -> SklearnModel:
    return SklearnModel(
        "xgboost",
        XGBClassifier(
            n_estimators=cfg.get("n_estimators", 400),
            max_depth=cfg.get("max_depth", 5),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=cfg.get("subsample", 0.9),
            colsample_bytree=cfg.get("colsample_bytree", 0.9),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    )


def build_logistic_regression(cfg: dict) -> SklearnModel:
    return SklearnModel(
        "logistic_regression",
        LogisticRegression(
            max_iter=cfg.get("max_iter", 1000),
            C=cfg.get("C", 1.0),
            class_weight="balanced",
        ),
    )


# --------------------------------------------------------------------------- #
# LSTM (optional, torch)
# --------------------------------------------------------------------------- #
def _make_sequences(values: np.ndarray, seq_len: int) -> np.ndarray:
    """Turn a (n, f) matrix into (n, seq_len, f) with left-padding for early rows."""
    if len(values) == 0:
        return values.reshape(0, seq_len, values.shape[-1] if values.ndim > 1 else 0)
    pad = np.repeat(values[:1], seq_len - 1, axis=0)
    padded = np.vstack([pad, values])
    return np.stack([padded[i : i + seq_len] for i in range(len(values))])


if _TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):
        def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)


class LSTMModel:
    kind = "lstm"

    def __init__(self, cfg: dict):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is not installed; LSTM is unavailable.")
        self.name = "lstm"
        self.seq_len = int(cfg.get("seq_len", 20))
        self.hidden_size = int(cfg.get("hidden_size", 64))
        self.num_layers = int(cfg.get("num_layers", 1))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.epochs = int(cfg.get("epochs", 30))
        self.lr = float(cfg.get("lr", 1e-3))
        self.batch_size = int(cfg.get("batch_size", 32))
        self.n_features: int | None = None
        self.net: Any = None
        self._device = torch.device("cpu")

    def _build_net(self, n_features: int):
        self.n_features = n_features
        self.net = _LSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout).to(
            self._device
        )

    def fit(self, dataset) -> None:
        X = dataset.X_train.values.astype(np.float32)
        y = dataset.y_train.values.astype(np.float32)
        self._build_net(X.shape[1])

        seqs = torch.tensor(_make_sequences(X, self.seq_len), dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32)

        n_pos = float(max(y.sum(), 1.0))
        n_neg = float(max(len(y) - y.sum(), 1.0))
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        ds = torch.utils.data.TensorDataset(seqs, targets)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                logits = self.net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        seqs = torch.tensor(
            _make_sequences(X.values.astype(np.float32), self.seq_len), dtype=torch.float32
        )
        self.net.eval()
        with torch.no_grad():
            logits = self.net(seqs.to(self._device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def save(self, path: str | Path) -> None:
        path = Path(path)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "hparams": {
                    "seq_len": self.seq_len,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "batch_size": self.batch_size,
                },
                "n_features": self.n_features,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(blob["hparams"])
        model._build_net(blob["n_features"])
        model.net.load_state_dict(blob["state_dict"])
        return model


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_models(models_cfg: dict) -> dict[str, Any]:
    """Instantiate all enabled models from config."""
    models: dict[str, Any] = {
        "xgboost": build_xgboost(models_cfg.get("xgboost", {})),
        "random_forest": build_random_forest(models_cfg.get("random_forest", {})),
        "logistic_regression": build_logistic_regression(
            models_cfg.get("logistic_regression", {})
        ),
    }
    lstm_cfg = models_cfg.get("lstm", {})
    if lstm_cfg.get("enabled", True):
        if _TORCH_AVAILABLE:
            models["lstm"] = LSTMModel(lstm_cfg)
        else:
            print("[warn] torch not installed -> skipping LSTM model.")
    return models
