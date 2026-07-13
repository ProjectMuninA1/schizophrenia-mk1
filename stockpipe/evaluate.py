"""Model evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute standard classification metrics from probabilities."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)

    # roc_auc needs both classes present in y_true.
    try:
        roc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        roc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
    }
