"""Feature engineering: technical indicators computed from OHLCV data.

All indicators are implemented with pandas/numpy (no external TA dependency) so
the exact formulas are transparent and reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA): EMA with alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


# --------------------------------------------------------------------------- #
# Trend indicators
# --------------------------------------------------------------------------- #
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd = _ema(close, fast) - _ema(close, slow)
    macd_signal = _ema(macd, signal)
    return macd, macd_signal


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = _true_range(high, low, close)
    atr = _rma(tr, period)
    plus_di = 100.0 * _rma(plus_dm, period) / atr
    minus_di = 100.0 * _rma(minus_dm, period) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _rma(dx, period)


# --------------------------------------------------------------------------- #
# Momentum indicators
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _stochastic(high, low, close, k_period: int = 14, d_period: int = 3):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    percent_k = 100.0 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    percent_d = percent_k.rolling(d_period).mean()
    return percent_k, percent_d


def _williams_r(high, low, close, period: int = 14) -> pd.Series:
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100.0 * (highest - close) / (highest - lowest).replace(0, np.nan)


def _cci(high, low, close, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period).mean()
    mean_dev = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015 * mean_dev.replace(0, np.nan))


# --------------------------------------------------------------------------- #
# Volatility indicators
# --------------------------------------------------------------------------- #
def _true_range(high, low, close) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(high, low, close, period: int = 14) -> pd.Series:
    return _rma(_true_range(high, low, close), period)


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, lower, width


# --------------------------------------------------------------------------- #
# Volume indicators
# --------------------------------------------------------------------------- #
def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def _chaikin_money_flow(high, low, close, volume, period: int = 20) -> pd.Series:
    denom = (high - low).replace(0, np.nan)
    mf_mult = ((close - low) - (high - close)) / denom
    mf_vol = mf_mult * volume
    return mf_vol.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def _volume_oscillator(volume: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    short_ema = _ema(volume, short)
    long_ema = _ema(volume, long)
    return 100.0 * (short_ema - long_ema) / long_ema.replace(0, np.nan)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    # trend
    "EMA20", "EMA50", "EMA100", "MACD", "MACD_Signal", "ADX",
    # momentum
    "RSI14", "Stoch_K", "Stoch_D", "Williams_R", "CCI",
    # volatility
    "ATR14", "BB_Upper", "BB_Lower", "BB_Width", "Rolling_Std",
    # volume
    "OBV", "CMF", "Volume_Osc",
    # additional
    "Daily_Return", "Return_5D", "Return_20D",
    "Price_Change_Pct", "Volume_Change_Pct", "High_Low_Spread",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all technical-indicator feature columns added.

    ``df`` must contain Open/High/Low/Close/Volume columns.
    """
    out = df.copy()
    high, low, close, volume = out["High"], out["Low"], out["Close"], out["Volume"]

    # Trend
    out["EMA20"] = _ema(close, 20)
    out["EMA50"] = _ema(close, 50)
    out["EMA100"] = _ema(close, 100)
    out["MACD"], out["MACD_Signal"] = _macd(close)
    out["ADX"] = _adx(high, low, close, 14)

    # Momentum
    out["RSI14"] = _rsi(close, 14)
    out["Stoch_K"], out["Stoch_D"] = _stochastic(high, low, close, 14, 3)
    out["Williams_R"] = _williams_r(high, low, close, 14)
    out["CCI"] = _cci(high, low, close, 20)

    # Volatility
    out["ATR14"] = _atr(high, low, close, 14)
    out["BB_Upper"], out["BB_Lower"], out["BB_Width"] = _bollinger(close, 20, 2.0)
    out["Rolling_Std"] = close.rolling(20).std()

    # Volume
    out["OBV"] = _obv(close, volume)
    out["CMF"] = _chaikin_money_flow(high, low, close, volume, 20)
    out["Volume_Osc"] = _volume_oscillator(volume, 5, 20)

    # Additional
    out["Daily_Return"] = close.pct_change()
    out["Return_5D"] = close.pct_change(5)
    out["Return_20D"] = close.pct_change(20)
    out["Price_Change_Pct"] = close.pct_change() * 100.0
    out["Volume_Change_Pct"] = volume.pct_change() * 100.0
    out["High_Low_Spread"] = (high - low) / close.replace(0, np.nan)

    # Replace inf produced by divisions with NaN so they get dropped later.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
