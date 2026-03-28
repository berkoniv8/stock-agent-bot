"""
Signal Confluence Engine — combines technical and fundamental signals,
applies scoring weights, and decides whether to fire a trade alert.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List

from dotenv import load_dotenv

from technical_analysis import TechnicalSignals
from fundamental_analysis import FundamentalSignals

load_dotenv()
logger = logging.getLogger(__name__)

SIGNAL_THRESHOLD = int(os.getenv("SIGNAL_THRESHOLD", "5"))

# Scoring weights for each signal
SIGNAL_WEIGHTS = {
    "ema_cross_bullish": 2,
    "price_above_200sma": 1,
    "breakout_with_volume": 3,
    "double_bottom": 2,
    "inverse_head_shoulders": 3,
    "fib_bounce_618": 2,
    "fib_bounce_382": 1,
    "rsi_oversold_bounce": 2,
    "macd_bullish_cross": 2,
    "bb_squeeze_breakout": 2,
    "rsi_bullish_divergence": 3,
    "macd_bullish_divergence": 2,
    "price_above_vwap": 1,
    "fundamental_score_8plus": 2,     # 8+ out of 15 fundamentals (replaces 4plus)
    "fundamental_score_12plus": 2,    # 12+ out of 15 (exceptional, replaces 6)
    "positive_news_sentiment": 1,
    # Money flow signals
    "mfi_oversold_bounce": 2,         # MFI < 20 with confirming reversal
    "obv_bullish_divergence": 2,      # Price down but OBV accumulating
    "accumulation_confirmed": 1,      # A/D line bullish + OBV bullish
    # Ichimoku signals
    "ichimoku_bullish": 2,            # Price above cloud + Tenkan > Kijun
    "ichimoku_bearish_warning": 0,    # Not scored, used for direction
    # Stochastic RSI
    "stoch_rsi_oversold_bounce": 2,   # StochRSI K crosses above D below 20
    # ADX trend confirmation
    "adx_strong_bullish_trend": 2,    # ADX > 25 with +DI > -DI
    # TTM Squeeze
    "ttm_squeeze_fired_bullish": 3,   # TTM Squeeze just fired + bullish momentum (rare, high value)
    # Relative Strength
    "relative_strength_leader": 1,    # Stock outperforming SPY
    # Pivot support bounce
    "pivot_support_bounce": 1,        # Price near pivot support level
    # Gap signals
    "gap_up_momentum": 1,             # Gap up with volume = institutional buying
    # New patterns
    "cup_and_handle": 3,              # Classic bullish continuation (rare, high value)
    "ascending_triangle": 2,          # Bullish breakout pattern
    "bull_flag": 2,                   # Bullish continuation
    # EMA ribbon
    "ema_ribbon_aligned": 1,          # All EMAs in bullish order
    # 52-week context
    "near_52w_high_momentum": 1,      # Near highs = strength
    # Enhanced fundamental signals
    "fcf_yield_strong": 1,            # Free cash flow yield > 4%
    "insider_buying": 2,              # Insiders buying their own stock (strong signal)
    "earnings_beat_streak": 1,        # Consecutive earnings beats
    "short_squeeze_risk": 1,          # High short interest = potential squeeze on good news
}


@dataclass
class TradeAlert:
    """Actionable trade alert produced by the confluence engine."""
    ticker: str
    signal_score: int
    direction: str  # "BUY" or "SELL"
    triggered_signals: list = field(default_factory=list)
    technical: Optional[TechnicalSignals] = None
    fundamental: Optional[FundamentalSignals] = None


def evaluate(
    tech: TechnicalSignals,
    fund: FundamentalSignals,
    threshold: Optional[int] = None,
) -> Optional[TradeAlert]:
    """Score combined signals and return a TradeAlert if threshold is met.

    Returns None if the combined score is below the threshold.
    """
    if threshold is None:
        threshold = SIGNAL_THRESHOLD

    score = 0
    triggered = []

    # --- Technical signals ---
    if tech.ema_cross_bullish:
        score += SIGNAL_WEIGHTS["ema_cross_bullish"]
        triggered.append(("ema_cross_bullish", SIGNAL_WEIGHTS["ema_cross_bullish"]))

    if tech.price_above_200sma:
        score += SIGNAL_WEIGHTS["price_above_200sma"]
        triggered.append(("price_above_200sma", SIGNAL_WEIGHTS["price_above_200sma"]))

    if tech.breakout_with_volume:
        score += SIGNAL_WEIGHTS["breakout_with_volume"]
        triggered.append(("breakout_with_volume", SIGNAL_WEIGHTS["breakout_with_volume"]))

    if tech.double_bottom:
        score += SIGNAL_WEIGHTS["double_bottom"]
        triggered.append(("double_bottom", SIGNAL_WEIGHTS["double_bottom"]))

    if tech.inverse_head_shoulders:
        score += SIGNAL_WEIGHTS["inverse_head_shoulders"]
        triggered.append(("inverse_head_shoulders", SIGNAL_WEIGHTS["inverse_head_shoulders"]))

    if tech.fib_bounce_618:
        score += SIGNAL_WEIGHTS["fib_bounce_618"]
        triggered.append(("fib_bounce_618", SIGNAL_WEIGHTS["fib_bounce_618"]))

    if tech.fib_bounce_382:
        score += SIGNAL_WEIGHTS["fib_bounce_382"]
        triggered.append(("fib_bounce_382", SIGNAL_WEIGHTS["fib_bounce_382"]))

    # RSI oversold bounce: RSI was oversold and MACD or EMA confirms reversal
    if tech.rsi_oversold and (tech.macd_bullish_cross or tech.ema_cross_bullish):
        score += SIGNAL_WEIGHTS["rsi_oversold_bounce"]
        triggered.append(("rsi_oversold_bounce", SIGNAL_WEIGHTS["rsi_oversold_bounce"]))

    # MACD bullish crossover (standalone, not double-counting with RSI combo)
    if tech.macd_bullish_cross and not tech.rsi_oversold:
        score += SIGNAL_WEIGHTS["macd_bullish_cross"]
        triggered.append(("macd_bullish_cross", SIGNAL_WEIGHTS["macd_bullish_cross"]))

    # Bollinger Band squeeze followed by upper breakout (volatility expansion)
    if tech.bb_squeeze and tech.bb_breakout_upper:
        score += SIGNAL_WEIGHTS["bb_squeeze_breakout"]
        triggered.append(("bb_squeeze_breakout", SIGNAL_WEIGHTS["bb_squeeze_breakout"]))

    # RSI bullish divergence (strong reversal signal)
    if tech.rsi_bullish_divergence:
        score += SIGNAL_WEIGHTS["rsi_bullish_divergence"]
        triggered.append(("rsi_bullish_divergence", SIGNAL_WEIGHTS["rsi_bullish_divergence"]))

    # MACD bullish divergence
    if tech.macd_bullish_divergence:
        score += SIGNAL_WEIGHTS["macd_bullish_divergence"]
        triggered.append(("macd_bullish_divergence", SIGNAL_WEIGHTS["macd_bullish_divergence"]))

    # Price above VWAP (institutional bias)
    if tech.price_above_vwap:
        score += SIGNAL_WEIGHTS["price_above_vwap"]
        triggered.append(("price_above_vwap", SIGNAL_WEIGHTS["price_above_vwap"]))

    # --- Money flow signals ---
    # MFI oversold bounce: MFI was oversold and price confirms reversal
    if tech.mfi_oversold and (tech.macd_bullish_cross or tech.ema_cross_bullish):
        score += SIGNAL_WEIGHTS["mfi_oversold_bounce"]
        triggered.append(("mfi_oversold_bounce", SIGNAL_WEIGHTS["mfi_oversold_bounce"]))

    # OBV bullish divergence: price declining but volume accumulating
    if tech.obv_divergence_bullish:
        score += SIGNAL_WEIGHTS["obv_bullish_divergence"]
        triggered.append(("obv_bullish_divergence", SIGNAL_WEIGHTS["obv_bullish_divergence"]))

    # Accumulation confirmed: both A/D line and OBV trending bullish
    if tech.ad_trend_bullish and tech.obv_trend_bullish:
        score += SIGNAL_WEIGHTS["accumulation_confirmed"]
        triggered.append(("accumulation_confirmed", SIGNAL_WEIGHTS["accumulation_confirmed"]))

    # --- Ichimoku ---
    if getattr(tech, "ichimoku_above_cloud", False) and getattr(tech, "ichimoku_bullish_cross", False):
        score += SIGNAL_WEIGHTS["ichimoku_bullish"]
        triggered.append(("ichimoku_bullish", SIGNAL_WEIGHTS["ichimoku_bullish"]))

    # --- Stochastic RSI ---
    if getattr(tech, "stoch_rsi_bullish_cross", False):
        score += SIGNAL_WEIGHTS["stoch_rsi_oversold_bounce"]
        triggered.append(("stoch_rsi_oversold_bounce", SIGNAL_WEIGHTS["stoch_rsi_oversold_bounce"]))

    # --- ADX trend confirmation ---
    if getattr(tech, "adx_bullish", False) and getattr(tech, "adx_strong_trend", False):
        score += SIGNAL_WEIGHTS["adx_strong_bullish_trend"]
        triggered.append(("adx_strong_bullish_trend", SIGNAL_WEIGHTS["adx_strong_bullish_trend"]))

    # --- TTM Squeeze ---
    if getattr(tech, "ttm_squeeze_fired", False) and getattr(tech, "macd_histogram", 0) > 0:
        score += SIGNAL_WEIGHTS["ttm_squeeze_fired_bullish"]
        triggered.append(("ttm_squeeze_fired_bullish", SIGNAL_WEIGHTS["ttm_squeeze_fired_bullish"]))

    # --- Relative Strength ---
    if getattr(tech, "rs_vs_spy", 0) > 1.0 and getattr(tech, "rs_trending_up", False):
        score += SIGNAL_WEIGHTS["relative_strength_leader"]
        triggered.append(("relative_strength_leader", SIGNAL_WEIGHTS["relative_strength_leader"]))

    # --- Pivot support ---
    if getattr(tech, "near_pivot_support", False):
        score += SIGNAL_WEIGHTS["pivot_support_bounce"]
        triggered.append(("pivot_support_bounce", SIGNAL_WEIGHTS["pivot_support_bounce"]))

    # --- Gap up momentum ---
    if getattr(tech, "gap_up", False) and tech.breakout_with_volume:
        score += SIGNAL_WEIGHTS["gap_up_momentum"]
        triggered.append(("gap_up_momentum", SIGNAL_WEIGHTS["gap_up_momentum"]))

    # --- New patterns ---
    if getattr(tech, "cup_and_handle", False):
        score += SIGNAL_WEIGHTS["cup_and_handle"]
        triggered.append(("cup_and_handle", SIGNAL_WEIGHTS["cup_and_handle"]))

    if getattr(tech, "ascending_triangle", False):
        score += SIGNAL_WEIGHTS["ascending_triangle"]
        triggered.append(("ascending_triangle", SIGNAL_WEIGHTS["ascending_triangle"]))

    if getattr(tech, "bull_flag", False):
        score += SIGNAL_WEIGHTS["bull_flag"]
        triggered.append(("bull_flag", SIGNAL_WEIGHTS["bull_flag"]))

    # --- EMA Ribbon ---
    if getattr(tech, "ema_ribbon_bullish", False):
        score += SIGNAL_WEIGHTS["ema_ribbon_aligned"]
        triggered.append(("ema_ribbon_aligned", SIGNAL_WEIGHTS["ema_ribbon_aligned"]))

    # --- 52-week context ---
    if getattr(tech, "near_52w_high", False) and tech.price_above_200sma:
        score += SIGNAL_WEIGHTS["near_52w_high_momentum"]
        triggered.append(("near_52w_high_momentum", SIGNAL_WEIGHTS["near_52w_high_momentum"]))

    # --- Fundamental signals (enhanced 15-point scale) ---
    if fund.fundamental_score >= 8:
        score += SIGNAL_WEIGHTS["fundamental_score_8plus"]
        triggered.append(("fundamental_score_8plus", SIGNAL_WEIGHTS["fundamental_score_8plus"]))

    if fund.fundamental_score >= 12:
        score += SIGNAL_WEIGHTS["fundamental_score_12plus"]
        triggered.append(("fundamental_score_12plus", SIGNAL_WEIGHTS["fundamental_score_12plus"]))

    if getattr(fund, "fcf_yield_strong", False):
        score += SIGNAL_WEIGHTS["fcf_yield_strong"]
        triggered.append(("fcf_yield_strong", SIGNAL_WEIGHTS["fcf_yield_strong"]))

    if getattr(fund, "insider_net_bullish", False):
        score += SIGNAL_WEIGHTS["insider_buying"]
        triggered.append(("insider_buying", SIGNAL_WEIGHTS["insider_buying"]))

    if getattr(fund, "earnings_quality_strong", False):
        score += SIGNAL_WEIGHTS["earnings_beat_streak"]
        triggered.append(("earnings_beat_streak", SIGNAL_WEIGHTS["earnings_beat_streak"]))

    if getattr(fund, "short_squeeze_risk", False):
        score += SIGNAL_WEIGHTS["short_squeeze_risk"]
        triggered.append(("short_squeeze_risk", SIGNAL_WEIGHTS["short_squeeze_risk"]))

    if fund.news_sentiment_positive:
        score += SIGNAL_WEIGHTS["positive_news_sentiment"]
        triggered.append(("positive_news_sentiment", SIGNAL_WEIGHTS["positive_news_sentiment"]))

    logger.info(
        "%s confluence score: %d (threshold: %d) — signals: %s",
        tech.ticker,
        score,
        threshold,
        ", ".join(f"{s[0]}(+{s[1]})" for s in triggered) or "none",
    )

    if score < threshold:
        return None

    # Determine direction
    direction = "BUY"
    bearish_count = sum([
        tech.head_and_shoulders,
        tech.ema_cross_bearish,
        tech.macd_bearish_cross,
        tech.rsi_overbought,
        tech.bb_breakout_lower,
        tech.rsi_bearish_divergence,
        tech.macd_bearish_divergence,
        tech.mfi_overbought,
        tech.ad_trend_bearish,
        tech.obv_divergence_bearish,
        getattr(tech, "ichimoku_below_cloud", False),
        getattr(tech, "ichimoku_bearish_cross", False),
        getattr(tech, "adx_bearish", False),
        getattr(tech, "stoch_rsi_overbought", False),
        getattr(tech, "gap_down", False),
        getattr(tech, "descending_triangle", False),
        getattr(tech, "ema_ribbon_bearish", False),
    ])
    bullish_count = sum([
        tech.ema_cross_bullish,
        tech.inverse_head_shoulders,
        tech.double_bottom,
        tech.macd_bullish_cross,
        tech.rsi_oversold,
        tech.bb_breakout_upper,
        tech.rsi_bullish_divergence,
        tech.macd_bullish_divergence,
        tech.price_above_vwap,
        tech.mfi_oversold,
        tech.ad_trend_bullish,
        tech.obv_divergence_bullish,
        getattr(tech, "ichimoku_above_cloud", False),
        getattr(tech, "ichimoku_bullish_cross", False),
        getattr(tech, "adx_bullish", False),
        getattr(tech, "stoch_rsi_bullish_cross", False),
        getattr(tech, "cup_and_handle", False),
        getattr(tech, "ascending_triangle", False),
        getattr(tech, "bull_flag", False),
        getattr(tech, "ema_ribbon_bullish", False),
        getattr(tech, "near_52w_high", False),
        getattr(tech, "ttm_squeeze_fired", False),
    ])
    if bearish_count > bullish_count:
        direction = "SELL"

    return TradeAlert(
        ticker=tech.ticker,
        signal_score=score,
        direction=direction,
        triggered_signals=triggered,
        technical=tech,
        fundamental=fund,
    )
