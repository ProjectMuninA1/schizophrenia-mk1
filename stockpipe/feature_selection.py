"""Feature selection via Random Forest importance."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 20,
    random_state: int = 42,
) -> tuple[list[str], pd.Series]:
    """Train a Random Forest and return the top-N features by importance.

    Returns ``(selected_feature_names, importance_series)`` where the importance
    series is sorted descending and covers all input features.
    """
    top_n = min(top_n, X_train.shape[1])
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train.values, y_train.values)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    selected = importances.head(top_n).index.tolist()
    return selected, importances
