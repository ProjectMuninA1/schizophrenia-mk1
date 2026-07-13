"""Dataset preparation: drop incomplete rows, split chronologically, normalize."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS
from .labels import LABEL_COLUMN


@dataclass
class Dataset:
    feature_names: list[str]
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    scaler: StandardScaler

    def select(self, feature_names: list[str]) -> "Dataset":
        """Return a copy restricted to ``feature_names`` (used after selection)."""
        return Dataset(
            feature_names=list(feature_names),
            X_train=self.X_train[feature_names],
            y_train=self.y_train,
            X_val=self.X_val[feature_names],
            y_val=self.y_val,
            X_test=self.X_test[feature_names],
            y_test=self.y_test,
            scaler=self.scaler,
        )


def prepare_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    feature_names: Optional[list[str]] = None,
) -> Dataset:
    """Drop incomplete rows, split chronologically, and normalize features.

    The scaler is fit on the training split only to avoid look-ahead leakage.
    """
    feature_names = feature_names or FEATURE_COLUMNS
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {total}")

    cols = feature_names + [LABEL_COLUMN]
    clean = df.dropna(subset=cols).copy()
    if len(clean) < 50:
        raise ValueError(
            f"Only {len(clean)} complete rows after dropping NaNs; need more history."
        )

    n = len(clean)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = clean.iloc[:n_train]
    val = clean.iloc[n_train:n_train + n_val]
    test = clean.iloc[n_train + n_val:]

    scaler = StandardScaler()
    scaler.fit(train[feature_names].values)

    def _scaled(frame: pd.DataFrame) -> pd.DataFrame:
        arr = scaler.transform(frame[feature_names].values)
        return pd.DataFrame(arr, index=frame.index, columns=feature_names)

    return Dataset(
        feature_names=list(feature_names),
        X_train=_scaled(train),
        y_train=train[LABEL_COLUMN].astype(int),
        X_val=_scaled(val),
        y_val=val[LABEL_COLUMN].astype(int),
        X_test=_scaled(test),
        y_test=test[LABEL_COLUMN].astype(int),
        scaler=scaler,
    )
