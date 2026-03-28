"""
Multi-Timeframe Analysis — confirms daily signals using intraday (1h) data.

Provides a confirmation score that either strengthens or weakens
the primary daily signal before an alert is sent.
"""

import logging
from dataclasses import dataclass

import pandas as pd

import technical_analysis

logger = logging.getLogger(__name__)


@dataclass
class MTFConfirmation:
    """Result of multi-timeframe confirmation analysis."""
    confirmed: bool = False
    score_adjustment: int = 0  # positive = strengthen, negative = weaken
    intraday_trend: str = ""   # "bullish", "bearish", "neutral"
    details: list = None

    def __post_init__(self):
        if self.details is None:
            self.details = []


def confirm_signal(
    daily_signals: technical_analysis.TechnicalSignals,
    intraday_df: pd.DataFrame,
    direction: str,
) -> MTFConfirmation:
    """Evaluate whether intraday price action confirms the daily signal.

    Checks multiple intraday conditions:
    1. EMA alignment (9/21 on 1h chart)
    2. RSI not diverging from daily signal
    3. MACD histogram direction
    4. VWAP alignment
    5. Volume trend

    Returns an MTFConfirmation with score_adjustment:
      +2  Strong confirmation (3+ confirming factors)
      +1  Moderate confirmation (2 confirming factors)
       0  Neutral (mixed or insufficient data)
      -1  Weak contradiction (2 contradicting factors)
      -2  Strong contradiction (3+ contradicting) — may suppress alert
    """
    result = MTFConfirmation()

    if intraday_df.empty or len(intraday_df) < 30:
        result.details.append("Insufficient intraday data — skipping MTF check")
        return result

    # Analyze intraday data
    intra = technical_analysis.analyze("intraday", intraday_df)

    bullish_factors = 0
    bearish_factors = 0
    details = []

    # 1. EMA alignment
    if intra.ema9 > intra.ema21:
        bullish_factors += 1
        details.append("1h EMA9 > EMA21 (bullish alignment)")
    elif intra.ema9 < intra.ema21:
        bearish_factors += 1
        details.append("1h EMA9 < EMA21 (bearish alignment)")

    # 2. EMA crossover on intraday
    if intra.ema_cross_bullish:
        bullish_factors += 1
        details.append("1h bullish EMA cross")
    if intra.ema_cross_bearish:
        bearish_factors += 1
        details.append("1h bearish EMA cross")

    # 3. RSI direction
    if intra.rsi > 50 and intra.rsi < 70:
        bullish_factors += 1
        details.append(f"1h RSI {intra.rsi:.0f} (bullish zone)")
    elif intra.rsi < 50 and intra.rsi > 30:
        bearish_factors += 1
        details.append(f"1h RSI {intra.rsi:.0f} (bearish zone)")
    elif intra.rsi <= 30:
        # Oversold on intraday — potential bounce (bullish for buys)
        bullish_factors += 1
        details.append(f"1h RSI {intra.rsi:.0f} (oversold — bounce candidate)")
    elif intra.rsi >= 70:
        bearish_factors += 1
        details.append(f"1h RSI {intra.rsi:.0f} (overbought)")

    # 4. MACD histogram
    if intra.macd_histogram > 0:
        bullish_factors += 1
        details.append("1h MACD histogram positive")
    elif intra.macd_histogram < 0:
        bearish_factors += 1
        details.append("1h MACD histogram negative")

    # 5. VWAP alignment
    if intra.price_above_vwap:
        bullish_factors += 1
        details.append("1h price above VWAP")
    elif intra.vwap > 0:
        bearish_factors += 1
        details.append("1h price below VWAP")

    # 6. Volume trend — is volume increasing in the signal direction?
    if len(intraday_df) >= 10:
        recent_vol = intraday_df["Volume"].tail(5).mean()
        earlier_vol = intraday_df["Volume"].tail(10).head(5).mean()
        if earlier_vol > 0 and recent_vol > earlier_vol * 1.2:
            if direction == "BUY" and intra.ema9 > intra.ema21:
                bullish_factors += 1
                details.append("1h volume increasing with bullish trend")
            elif direction == "SELL" and intra.ema9 < intra.ema21:
                bearish_factors += 1
                details.append("1h volume increasing with bearish trend")

    # Determine confirmation
    if direction == "BUY":
        confirming = bullish_factors
        contradicting = bearish_factors
        result.intraday_trend = "bullish" if confirming > contradicting else "bearish" if contradicting > confirming else "neutral"
    else:
        confirming = bearish_factors
        contradicting = bullish_factors
        result.intraday_trend = "bearish" if confirming > contradicting else "bullish" if contradicting > confirming else "neutral"

    # Score adjustment
    if confirming >= 4:
        result.score_adjustment = 2
        result.confirmed = True
    elif confirming >= 3:
        result.score_adjustment = 1
        result.confirmed = True
    elif contradicting >= 4:
        result.score_adjustment = -2
        result.confirmed = False
    elif contradicting >= 3:
        result.score_adjustment = -1
        result.confirmed = False
    else:
        result.score_adjustment = 0
        result.confirmed = confirming >= contradicting

    result.details = details

    logger.info(
        "MTF confirmation: %s (adj=%+d, bull=%d, bear=%d)",
        "CONFIRMED" if result.confirmed else "CONTRADICTED",
        result.score_adjustment,
        bullish_factors,
        bearish_factors,
    )

    return result
