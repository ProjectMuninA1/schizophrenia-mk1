"""Load and clean historical OHLCV market data from a public source (yfinance)."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def load_market_data(
    ticker: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download historical OHLCV data for ``ticker`` from Yahoo Finance.

    Returns a DataFrame indexed by date with columns Open/High/Low/Close/Volume.
    """
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for ticker {ticker!r}. "
            "Check the symbol, date range, and your network connection."
        )

    # yfinance can return a MultiIndex column frame (even for a single ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        # Prefer the price-field level; drop the ticker level.
        level0 = raw.columns.get_level_values(0)
        if set(OHLCV).issubset(set(level0)):
            raw.columns = level0
        else:
            raw.columns = raw.columns.get_level_values(-1)

    missing = [c for c in OHLCV if c not in raw.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    df = raw[OHLCV].copy()
    df.index.name = "Date"
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV data.

    - sort chronologically
    - drop duplicate rows / duplicate dates
    - drop rows with missing values
    - drop non-positive prices (bad ticks)
    """
    out = df.copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.drop_duplicates()
    out = out.dropna(subset=OHLCV)
    out = out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    out = out[out["Volume"] >= 0]
    return out
