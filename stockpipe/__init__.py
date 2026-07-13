"""Stock signal ML pipeline.

Turns OHLCV market data into technical features, trains several classifiers to
predict next-day direction, evaluates them, and emits a BUY / SELL / HOLD signal
with ATR-based risk management.
"""

__version__ = "0.1.0"
