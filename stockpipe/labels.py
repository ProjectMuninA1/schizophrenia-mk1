"""Label creation: BUY (1) if the future close is higher, else SELL (0)."""

from __future__ import annotations

import pandas as pd

LABEL_COLUMN = "Label"


def add_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add a binary ``Label`` column.

    Label = 1 (BUY) if Close[t + horizon] > Close[t], else 0 (SELL).
    The last ``horizon`` rows have no future close and are set to NaN
    (they are dropped during dataset preparation).
    """
    out = df.copy()
    future_close = out["Close"].shift(-horizon)
    out[LABEL_COLUMN] = (future_close > out["Close"]).astype("float")
    out.loc[future_close.isna(), LABEL_COLUMN] = float("nan")
    return out
