"""
Position Sizing Module — calculates entry, stop-loss, take-profit,
and share count based on risk parameters.
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from signal_engine import TradeAlert

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class PositionPlan:
    """Fully computed trade plan for a single alert."""
    ticker: str
    direction: str
    entry_price: float
    stop_loss: float
    risk_per_share: float
    shares: int
    position_value: float
    target_1: float  # 1.5x risk
    target_2: float  # 3x risk
    target_3: float  # Fib extension
    max_loss: float
    risk_reward_t1: float
    risk_reward_t2: float


def load_portfolio_config() -> dict:
    """Load portfolio parameters from portfolio.json, with .env overrides."""
    config = {
        "total_portfolio_value": float(os.getenv("TOTAL_PORTFOLIO_VALUE", "50000")),
        "available_cash": float(os.getenv("AVAILABLE_CASH", "10000")),
        "max_risk_per_trade_pct": float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0")),
        "max_position_size_pct": float(os.getenv("MAX_POSITION_SIZE_PCT", "10.0")),
    }

    # Try to load from portfolio.json for holdings info
    try:
        with open("portfolio.json") as f:
            file_config = json.load(f)
            for key in config:
                if key in file_config:
                    config[key] = float(file_config[key])
    except FileNotFoundError:
        pass

    return config


def compute(alert: TradeAlert) -> Optional[PositionPlan]:
    """Calculate position sizing for a trade alert.

    Returns a PositionPlan with entry, stop, targets, and share count.
    Returns None if the trade cannot be sized (e.g., no valid stop level).
    """
    config = load_portfolio_config()
    tech = alert.technical

    if tech is None:
        logger.error("No technical signals for %s — cannot size position", alert.ticker)
        return None

    portfolio_value = config["total_portfolio_value"]
    available_cash = config["available_cash"]
    max_risk = portfolio_value * (config["max_risk_per_trade_pct"] / 100)
    max_position = portfolio_value * (config["max_position_size_pct"] / 100)

    entry_price = tech.current_price

    # --- Stop-loss placement (ATR-enhanced) ---
    if alert.direction == "BUY":
        stop_candidates = []
        if tech.ema21 > 0 and tech.ema21 < entry_price:
            stop_candidates.append(tech.ema21)
        if tech.support_level > 0 and tech.support_level < entry_price:
            stop_candidates.append(tech.support_level)
        # ATR-based stop: most reliable volatility-adjusted level
        if tech.atr_stop_long > 0 and tech.atr_stop_long < entry_price:
            stop_candidates.append(tech.atr_stop_long)

        if not stop_candidates:
            stop_loss = entry_price * 0.97
        else:
            # Use the highest support level (tightest stop)
            stop_loss = max(stop_candidates)
            stop_loss *= 0.995
    else:
        stop_candidates = []
        if tech.ema21 > 0 and tech.ema21 > entry_price:
            stop_candidates.append(tech.ema21)
        if tech.resistance_level > 0 and tech.resistance_level > entry_price:
            stop_candidates.append(tech.resistance_level)
        if tech.atr_stop_short > 0 and tech.atr_stop_short > entry_price:
            stop_candidates.append(tech.atr_stop_short)

        if not stop_candidates:
            stop_loss = entry_price * 1.03
        else:
            stop_loss = min(stop_candidates)
            stop_loss *= 1.005

    risk_per_share = abs(entry_price - stop_loss)

    if risk_per_share <= 0:
        logger.warning("Zero risk per share for %s — skipping", alert.ticker)
        return None

    # --- Share count ---
    shares_by_risk = int(max_risk / risk_per_share)
    shares_by_position = int(max_position / entry_price)
    shares_by_cash = int(available_cash / entry_price)

    shares = max(1, min(shares_by_risk, shares_by_position, shares_by_cash))
    position_value = shares * entry_price
    max_loss = shares * risk_per_share

    # --- Take-profit targets ---
    if alert.direction == "BUY":
        target_1 = entry_price + 1.5 * risk_per_share
        target_2 = entry_price + 3.0 * risk_per_share
        # Fib extension
        fib_ext = tech.fib_levels.get(161.8) or tech.fib_levels.get(127.2)
        target_3 = fib_ext if fib_ext and fib_ext > entry_price else entry_price + 4.0 * risk_per_share
    else:
        target_1 = entry_price - 1.5 * risk_per_share
        target_2 = entry_price - 3.0 * risk_per_share
        target_3 = entry_price - 4.0 * risk_per_share

    rr_t1 = 1.5
    rr_t2 = 3.0

    plan = PositionPlan(
        ticker=alert.ticker,
        direction=alert.direction,
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        risk_per_share=round(risk_per_share, 2),
        shares=shares,
        position_value=round(position_value, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        target_3=round(target_3, 2),
        max_loss=round(max_loss, 2),
        risk_reward_t1=rr_t1,
        risk_reward_t2=rr_t2,
    )

    logger.info(
        "%s %s plan: entry=%.2f, stop=%.2f, shares=%d, T1=%.2f, T2=%.2f, T3=%.2f",
        plan.ticker, plan.direction, plan.entry_price, plan.stop_loss,
        plan.shares, plan.target_1, plan.target_2, plan.target_3,
    )

    return plan
