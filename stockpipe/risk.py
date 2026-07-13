"""Signal generation and ATR-based risk management."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from .config import RiskConfig, SignalConfig


def to_signal(probability: float, cfg: SignalConfig) -> str:
    """Map a BUY probability to BUY / SELL / HOLD."""
    if probability > cfg.buy_threshold:
        return "BUY"
    if probability < cfg.sell_threshold:
        return "SELL"
    return "HOLD"


@dataclass
class TradePlan:
    signal: str
    probability: float
    entry_price: float
    atr: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_amount: Optional[float]
    position_size: Optional[float]  # units / shares

    def as_dict(self) -> dict:
        return asdict(self)


def build_trade_plan(
    signal: str,
    probability: float,
    entry_price: float,
    atr: float,
    risk: RiskConfig,
) -> TradePlan:
    """Compute stop-loss, take-profit and position size for a signal.

    Position size targets risking ``risk_per_trade`` of account equity given a
    stop distance of ``atr_stop_mult * ATR``.
    """
    if signal == "HOLD" or atr <= 0:
        return TradePlan(
            signal=signal,
            probability=probability,
            entry_price=entry_price,
            atr=atr,
            stop_loss=None,
            take_profit=None,
            risk_amount=None,
            position_size=None,
        )

    stop_distance = risk.atr_stop_mult * atr
    take_distance = risk.atr_take_mult * atr
    risk_amount = risk.account_equity * risk.risk_per_trade
    position_size = risk_amount / stop_distance if stop_distance > 0 else 0.0

    if signal == "BUY":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + take_distance
    else:  # SELL / short
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - take_distance

    return TradePlan(
        signal=signal,
        probability=probability,
        entry_price=entry_price,
        atr=atr,
        stop_loss=round(stop_loss, 4),
        take_profit=round(take_profit, 4),
        risk_amount=round(risk_amount, 2),
        position_size=round(position_size, 4),
    )
