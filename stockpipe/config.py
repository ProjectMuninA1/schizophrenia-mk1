"""Typed configuration loaded from a YAML file."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: Optional[str] = None
    interval: str = "1d"


@dataclass
class LabelConfig:
    horizon: int = 1


@dataclass
class DatasetConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


@dataclass
class FeatureSelectionConfig:
    top_n: int = 20


@dataclass
class SignalConfig:
    buy_threshold: float = 0.70
    sell_threshold: float = 0.30


@dataclass
class RiskConfig:
    account_equity: float = 10000.0
    risk_per_trade: float = 0.01
    atr_stop_mult: float = 2.0
    atr_take_mult: float = 3.0


@dataclass
class PathsConfig:
    artifacts_dir: str = "artifacts"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    models: dict[str, Any] = field(default_factory=dict)
    selection_metric: str = "roc_auc"
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            labels=LabelConfig(**raw.get("labels", {})),
            dataset=DatasetConfig(**raw.get("dataset", {})),
            feature_selection=FeatureSelectionConfig(**raw.get("feature_selection", {})),
            models=raw.get("models", {}),
            selection_metric=raw.get("selection_metric", "roc_auc"),
            signal=SignalConfig(**raw.get("signal", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            paths=PathsConfig(**raw.get("paths", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
