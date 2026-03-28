"""
Trade Grader — grades every trade suggestion with a letter grade (A+ through F),
detailed logic explanation, and risk assessment.

Called by other modules; no CLI interface.
"""

import json
import logging
import os
from typing import Optional, List, Dict

from dotenv import load_dotenv

from signal_engine import TradeAlert
from position_sizing import PositionPlan

load_dotenv()
logger = logging.getLogger(__name__)

PORTFOLIO_PATH = os.path.join(os.path.dirname(__file__), "portfolio.json")

# ---------------------------------------------------------------------------
# Signal category mapping — used for diversity scoring
# ---------------------------------------------------------------------------
SIGNAL_CATEGORIES: Dict[str, str] = {
    # Technical
    "ema_cross_bullish": "technical",
    "price_above_200sma": "technical",
    "macd_bullish_cross": "technical",
    "bb_squeeze_breakout": "technical",
    "ichimoku_bullish": "technical",
    "stoch_rsi_oversold_bounce": "technical",
    "adx_strong_bullish_trend": "technical",
    "ttm_squeeze_fired_bullish": "technical",
    "ema_ribbon_aligned": "technical",
    "rsi_oversold_bounce": "technical",
    "pivot_support_bounce": "technical",
    # Fundamental
    "fundamental_score_8plus": "fundamental",
    "fundamental_score_12plus": "fundamental",
    "fcf_yield_strong": "fundamental",
    "insider_buying": "fundamental",
    "earnings_beat_streak": "fundamental",
    "positive_news_sentiment": "fundamental",
    # Volume
    "breakout_with_volume": "volume",
    "obv_bullish_divergence": "volume",
    "accumulation_confirmed": "volume",
    "mfi_oversold_bounce": "volume",
    "gap_up_momentum": "volume",
    # Pattern
    "double_bottom": "pattern",
    "inverse_head_shoulders": "pattern",
    "cup_and_handle": "pattern",
    "ascending_triangle": "pattern",
    "bull_flag": "pattern",
    "near_52w_high_momentum": "pattern",
    # Divergence
    "rsi_bullish_divergence": "divergence",
    "macd_bullish_divergence": "divergence",
    "fib_bounce_618": "divergence",
    "fib_bounce_382": "divergence",
    "relative_strength_leader": "divergence",
    "short_squeeze_risk": "divergence",
}

# Volume-related signal names checked for volume confirmation scoring
VOLUME_SIGNALS = [
    "breakout_with_volume",
    "obv_trend_bullish",
    "ad_trend_bullish",
    "mfi_oversold_bounce",
]

# Letter grade thresholds (lower bound inclusive)
GRADE_MAP = [
    (90, "A+"),
    (85, "A"),
    (80, "A-"),
    (75, "B+"),
    (70, "B"),
    (65, "B-"),
    (60, "C+"),
    (55, "C"),
    (50, "C-"),
    (40, "D"),
    (0, "F"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_to_grade(score: int) -> str:
    """Map a 0-100 numeric score to a letter grade."""
    for threshold, grade in GRADE_MAP:
        if score >= threshold:
            return grade
    return "F"


def _load_portfolio() -> dict:
    """Load portfolio.json, returning empty dict on failure."""
    try:
        with open(PORTFOLIO_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _triggered_signal_names(alert: TradeAlert) -> List[str]:
    """Extract signal name strings from alert.triggered_signals.

    Each entry is typically a tuple (name, weight) or just a string.
    """
    names: List[str] = []
    for item in getattr(alert, "triggered_signals", []):
        if isinstance(item, (list, tuple)):
            names.append(str(item[0]))
        else:
            names.append(str(item))
    return names


def _safe_rr(plan: PositionPlan) -> float:
    """Compute risk/reward ratio from plan fields, returning 0.0 on failure."""
    try:
        entry = plan.entry_price
        stop = plan.stop_loss
        target = plan.target_1
        risk = abs(entry - stop)
        if risk == 0:
            return 0.0
        reward = abs(target - entry)
        return round(reward / risk, 2)
    except (AttributeError, TypeError, ZeroDivisionError):
        return 0.0


# ---------------------------------------------------------------------------
# Scoring sub-functions
# ---------------------------------------------------------------------------

def _score_signal_strength(alert: TradeAlert) -> int:
    """Signal Strength (25 pts max)."""
    score = getattr(alert, "signal_score", 0) or 0
    if score >= 11:
        return 25
    if score >= 9:
        return 20
    if score >= 7:
        return 15
    if score >= 5:
        return 10
    return 0


def _score_signal_diversity(alert: TradeAlert) -> int:
    """Signal Diversity (20 pts max) — count distinct categories triggered."""
    names = _triggered_signal_names(alert)
    categories = set()
    for name in names:
        cat = SIGNAL_CATEGORIES.get(name)
        if cat:
            categories.add(cat)
    count = len(categories)
    if count >= 4:
        return 20
    if count >= 3:
        return 14
    if count >= 2:
        return 8
    if count >= 1:
        return 4
    return 0


def _score_fundamental_backing(alert: TradeAlert) -> int:
    """Fundamental Backing (15 pts max)."""
    fund = getattr(alert, "fundamental", None)
    if fund is None:
        return 0
    fs = getattr(fund, "fundamental_score", 0) or 0
    if fs >= 12:
        return 15
    if fs >= 8:
        return 10
    if fs >= 4:
        return 5
    return 0


def _score_risk_reward(plan: PositionPlan) -> int:
    """Risk/Reward Ratio (15 pts max)."""
    rr = _safe_rr(plan)
    if rr > 3.0:
        return 15
    if rr >= 2.0:
        return 10
    if rr >= 1.5:
        return 5
    return 0


def _score_volume_confirmation(alert: TradeAlert) -> float:
    """Volume Confirmation (10 pts max, 2.5 per confirmed signal)."""
    tech = getattr(alert, "technical", None)
    if tech is None:
        return 0.0
    confirmed = 0.0
    for sig in VOLUME_SIGNALS:
        if getattr(tech, sig, False):
            confirmed += 2.5
    return min(confirmed, 10.0)


def _score_trend_alignment(alert: TradeAlert) -> int:
    """Trend Alignment (10 pts max)."""
    tech = getattr(alert, "technical", None)
    if tech is None:
        return 0
    above_200 = getattr(tech, "price_above_200sma", False)
    ribbon = getattr(tech, "ema_ribbon_bullish", False)
    if above_200 and ribbon:
        return 10
    if above_200:
        return 5
    return 0


def _score_market_regime(alert: TradeAlert, regime_info: Optional[dict]) -> int:
    """Market Regime (5 pts max)."""
    if regime_info is None:
        return 3  # neutral
    regime = str(regime_info.get("regime", "")).upper()
    direction = getattr(alert, "direction", "BUY").upper()

    bull_regimes = {"BULL", "BULL_STRONG", "BULL_WEAK"}
    bear_regimes = {"BEAR", "BEAR_STRONG", "BEAR_WEAK"}

    if direction == "BUY":
        if regime in bull_regimes:
            return 5
        if regime in bear_regimes:
            return 0
    elif direction == "SELL":
        if regime in bear_regimes:
            return 5
        if regime in bull_regimes:
            return 0
    return 3


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def grade_trade(
    alert: TradeAlert,
    plan: PositionPlan,
    regime_info: Optional[dict] = None,
) -> dict:
    """Grade a trade suggestion on a 0-100 scale and return a detailed breakdown.

    Parameters
    ----------
    alert : TradeAlert
        From signal_engine — contains triggered signals, technical & fundamental data.
    plan : PositionPlan
        From position_sizing — entry, stop, targets, shares.
    regime_info : dict, optional
        Market regime dict with at least a ``"regime"`` key.

    Returns
    -------
    dict
        Keys: grade, score, breakdown, logic, risks, confidence.
    """
    breakdown = {
        "signal_strength": _score_signal_strength(alert),
        "signal_diversity": _score_signal_diversity(alert),
        "fundamental_backing": _score_fundamental_backing(alert),
        "risk_reward": _score_risk_reward(plan),
        "volume_confirmation": _score_volume_confirmation(alert),
        "trend_alignment": _score_trend_alignment(alert),
        "market_regime": _score_market_regime(alert, regime_info),
    }

    total = int(sum(breakdown.values()))
    total = max(0, min(total, 100))
    grade = _score_to_grade(total)

    # Confidence bucket
    if total >= 80:
        confidence = "HIGH"
    elif total >= 60:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    logic = generate_logic(alert, plan, {"grade": grade, "score": total, "breakdown": breakdown})
    risks = assess_risks(alert, plan, regime_info)

    result = {
        "grade": grade,
        "score": total,
        "breakdown": breakdown,
        "logic": logic,
        "risks": risks,
        "confidence": confidence,
    }
    logger.info("Graded %s %s — %s (%d/100)", alert.direction, alert.ticker, grade, total)
    return result


# ---------------------------------------------------------------------------
# Logic generation
# ---------------------------------------------------------------------------

_SIGNAL_DESCRIPTIONS: Dict[str, str] = {
    "ema_cross_bullish": "bullish EMA 9/21 crossover",
    "price_above_200sma": "trading above the 200-day SMA",
    "breakout_with_volume": "breaking out on above-average volume",
    "macd_bullish_cross": "MACD bullish cross",
    "bb_squeeze_breakout": "Bollinger Band squeeze breakout",
    "rsi_oversold_bounce": "RSI bounce from oversold territory",
    "rsi_bullish_divergence": "bullish RSI divergence (price lower low, RSI higher low)",
    "macd_bullish_divergence": "bullish MACD divergence",
    "double_bottom": "double-bottom reversal pattern",
    "inverse_head_shoulders": "inverse head-and-shoulders pattern",
    "cup_and_handle": "cup-and-handle continuation pattern",
    "ascending_triangle": "ascending triangle breakout",
    "bull_flag": "bull flag continuation",
    "ttm_squeeze_fired_bullish": "TTM Squeeze fired with bullish momentum",
    "ichimoku_bullish": "Ichimoku cloud bullish confirmation",
    "stoch_rsi_oversold_bounce": "Stochastic RSI bounce from oversold",
    "adx_strong_bullish_trend": "strong bullish trend confirmed by ADX",
    "obv_bullish_divergence": "on-balance volume bullish divergence",
    "accumulation_confirmed": "accumulation confirmed by A/D line and OBV",
    "mfi_oversold_bounce": "Money Flow Index bounce from oversold",
    "gap_up_momentum": "gap-up with institutional volume",
    "ema_ribbon_aligned": "EMA ribbon in bullish alignment",
    "near_52w_high_momentum": "near 52-week high with momentum",
    "fib_bounce_618": "bounce off the 0.618 Fibonacci retracement",
    "fib_bounce_382": "bounce off the 0.382 Fibonacci retracement",
    "pivot_support_bounce": "bounce off pivot support",
    "relative_strength_leader": "outperforming SPY on relative strength",
    "fundamental_score_8plus": "solid fundamental score (8+/15)",
    "fundamental_score_12plus": "exceptional fundamental score (12+/15)",
    "fcf_yield_strong": "strong free cash flow yield",
    "insider_buying": "insider buying activity",
    "earnings_beat_streak": "consecutive earnings beats",
    "positive_news_sentiment": "positive news sentiment",
    "short_squeeze_risk": "elevated short interest with squeeze potential",
}


def generate_logic(alert: TradeAlert, plan: PositionPlan, grade_info: dict) -> str:
    """Build an analyst-style explanation of the trade thesis.

    Parameters
    ----------
    alert : TradeAlert
    plan : PositionPlan
    grade_info : dict
        Must contain ``grade``, ``score``, and ``breakdown`` keys.

    Returns
    -------
    str
        Multi-sentence explanation suitable for display.
    """
    ticker = getattr(alert, "ticker", "???")
    direction = getattr(alert, "direction", "BUY")
    grade = grade_info.get("grade", "?")
    score = grade_info.get("score", 0)

    # Collect human-readable signal descriptions
    names = _triggered_signal_names(alert)
    tech_descs: List[str] = []
    fund_descs: List[str] = []
    for name in names:
        desc = _SIGNAL_DESCRIPTIONS.get(name)
        if desc is None:
            continue
        cat = SIGNAL_CATEGORIES.get(name, "technical")
        if cat == "fundamental":
            fund_descs.append(desc)
        else:
            tech_descs.append(desc)

    # Opening sentence
    parts: List[str] = [
        f"{ticker} presents a {direction} opportunity with a {grade} grade ({score}/100)."
    ]

    # Technical summary
    if tech_descs:
        joined = ", ".join(tech_descs[:4])
        if len(tech_descs) > 4:
            joined += f", and {len(tech_descs) - 4} additional technical signal(s)"
        parts.append(f"The stock is showing {joined}.")

    # TTM Squeeze callout
    tech = getattr(alert, "technical", None)
    if tech and getattr(tech, "ttm_squeeze_fired", False):
        parts.append(
            "The TTM Squeeze has just fired, suggesting an imminent volatility expansion."
        )

    # Fundamental summary
    fund = getattr(alert, "fundamental", None)
    if fund:
        fs = getattr(fund, "fundamental_score", 0) or 0
        if fs > 0:
            detail_bits: List[str] = []
            roe = getattr(fund, "roe", None)
            if roe is not None and roe > 0:
                detail_bits.append(f"ROE {roe:.0f}%")
            fcf = getattr(fund, "fcf_yield", None)
            if fcf is not None and fcf > 0:
                detail_bits.append(f"FCF yield {fcf:.1f}%")
            streak = getattr(fund, "earnings_beat_streak", 0) or 0
            if streak > 0:
                detail_bits.append(f"{streak} consecutive earnings beat{'s' if streak != 1 else ''}")
            detail_str = ""
            if detail_bits:
                detail_str = f" with {', '.join(detail_bits)}"
            parts.append(f"Fundamentally, the company scores {fs}/15{detail_str}.")

    # Risk/reward
    rr = _safe_rr(plan)
    try:
        parts.append(
            f"The risk/reward ratio is {rr:.1f}:1 with entry at ${plan.entry_price:.2f}, "
            f"stop at ${plan.stop_loss:.2f}, and first target at ${plan.target_1:.2f}."
        )
    except (AttributeError, TypeError):
        pass

    # Volume confirmation note
    if tech:
        vol_items: List[str] = []
        if getattr(tech, "breakout_with_volume", False):
            vol_items.append("breakout volume")
        if getattr(tech, "obv_trend_bullish", False):
            vol_items.append("rising OBV")
        if getattr(tech, "ad_trend_bullish", False):
            vol_items.append("bullish accumulation/distribution")
        if vol_items:
            parts.append(f"Volume confirms the move with {', '.join(vol_items)}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

def assess_risks(
    alert: TradeAlert,
    plan: PositionPlan,
    regime_info: Optional[dict] = None,
) -> List[str]:
    """Return a list of specific, actionable risk factors for the trade.

    Dynamically generated from position sizing, technical levels, regime,
    earnings proximity, sector concentration, and indicator readings.
    """
    risks: List[str] = []
    tech = getattr(alert, "technical", None)
    fund = getattr(alert, "fundamental", None)
    direction = getattr(alert, "direction", "BUY").upper()
    ticker = getattr(alert, "ticker", "???")

    # 1. Max loss from plan
    try:
        portfolio = _load_portfolio()
        portfolio_value = portfolio.get("total_portfolio_value", 0)
        max_loss = getattr(plan, "max_loss", 0) or 0
        stop = getattr(plan, "stop_loss", 0) or 0
        if max_loss > 0 and portfolio_value > 0:
            pct = (max_loss / portfolio_value) * 100
            risks.append(
                f"Max loss: ${max_loss:,.0f} ({pct:.1f}% of portfolio) if stop at ${stop:.2f} is hit"
            )
        elif max_loss > 0:
            risks.append(f"Max loss: ${max_loss:,.0f} if stop at ${stop:.2f} is hit")
    except (AttributeError, TypeError):
        pass

    # 2. RSI warning
    if tech:
        rsi = getattr(tech, "rsi", 0) or 0
        if direction == "BUY" and rsi > 65:
            risks.append(f"RSI at {rsi:.0f} — approaching overbought territory, entry timing risk")
        elif direction == "SELL" and rsi < 35:
            risks.append(f"RSI at {rsi:.0f} — approaching oversold territory, short timing risk")

    # 3. Market regime
    if regime_info:
        regime = str(regime_info.get("regime", "")).upper()
        bear_regimes = {"BEAR", "BEAR_STRONG", "BEAR_WEAK"}
        bull_regimes = {"BULL", "BULL_STRONG", "BULL_WEAK"}
        if direction == "BUY" and regime in bear_regimes:
            risks.append(f"Market regime is {regime} — counter-trend trade, lower probability")
        elif direction == "SELL" and regime in bull_regimes:
            risks.append(f"Market regime is {regime} — counter-trend short, lower probability")

    # 4. Earnings proximity
    try:
        from earnings_guard import check_earnings_safe
        earnings_info = check_earnings_safe(ticker)
        if earnings_info and earnings_info.get("days_until") is not None:
            days = earnings_info["days_until"]
            if days <= 14:
                risks.append(
                    f"Earnings in {days} day{'s' if days != 1 else ''} — "
                    "consider reducing position size or closing before report"
                )
    except (ImportError, Exception) as e:
        logger.debug("Earnings guard unavailable: %s", e)

    # 5. 52-week high proximity
    if tech:
        pct_from_high = getattr(tech, "pct_from_52w_high", 0) or 0
        if direction == "BUY" and 0 < pct_from_high <= 5:
            risks.append(
                f"Within {pct_from_high:.1f}% of 52-week high — resistance may cap upside"
            )

    # 6. Sector concentration
    try:
        portfolio = _load_portfolio()
        holdings = portfolio.get("holdings", [])
        if fund:
            # Attempt to determine sector from portfolio or fundamentals
            ticker_sector = None
            for h in holdings:
                if h.get("ticker", "").upper() == ticker.upper():
                    ticker_sector = h.get("sector")
                    break
            if ticker_sector is None:
                ticker_sector = getattr(fund, "sector", None)
            if ticker_sector and ticker_sector not in ("ETF", None, ""):
                same_sector = [
                    h["ticker"] for h in holdings
                    if h.get("sector") == ticker_sector
                    and h.get("ticker", "").upper() != ticker.upper()
                ]
                if same_sector:
                    risks.append(
                        f"High correlation with existing {', '.join(same_sector[:3])} "
                        f"position{'s' if len(same_sector) > 1 else ''} ({ticker_sector} sector)"
                    )
    except Exception:
        pass

    # 7. Bollinger Band position
    if tech:
        bb_upper = getattr(tech, "bb_upper", 0) or 0
        price = getattr(tech, "current_price", 0) or 0
        if direction == "BUY" and bb_upper > 0 and price > 0:
            if price >= bb_upper * 0.98:
                risks.append(
                    "Price at or above upper Bollinger Band — extended move, mean-reversion risk"
                )

    # 8. Gap risk
    if tech:
        if getattr(tech, "gap_up", False) and direction == "BUY":
            gap_pct = getattr(tech, "gap_up_pct", 0) or 0
            if gap_pct > 2:
                risks.append(
                    f"Gapped up {gap_pct:.1f}% — gap-fill risk could pull price back to pre-gap level"
                )

    # 9. Short interest
    if fund:
        si = getattr(fund, "short_interest_pct", None)
        if si is not None and si > 10:
            risks.append(
                f"Short interest at {si:.1f}% — potential squeeze but also elevated volatility risk"
            )

    # 10. MFI overbought warning
    if tech:
        mfi = getattr(tech, "mfi", 0) or 0
        if direction == "BUY" and mfi > 80:
            risks.append(f"MFI at {mfi:.0f} — overbought money flow, potential near-term pullback")

    return risks


# ---------------------------------------------------------------------------
# Formatted alert output
# ---------------------------------------------------------------------------

def format_graded_alert(
    alert: TradeAlert,
    plan: PositionPlan,
    grade_info: dict,
    options_suggestion: Optional[dict] = None,
) -> str:
    """Format a complete graded alert message suitable for Telegram or console.

    Parameters
    ----------
    alert : TradeAlert
    plan : PositionPlan
    grade_info : dict
        Output of ``grade_trade()``.
    options_suggestion : dict, optional
        If provided, should contain keys like ``strategy``, ``expiry``,
        ``buy_leg``, ``sell_leg``, ``cost``, ``max_profit``, ``rr``.

    Returns
    -------
    str
        Multi-line formatted string.
    """
    ticker = getattr(alert, "ticker", "???")
    direction = getattr(alert, "direction", "BUY")
    grade = grade_info.get("grade", "?")
    score = grade_info.get("score", 0)
    confidence = grade_info.get("confidence", "MEDIUM")
    logic = grade_info.get("logic", "")
    risks = grade_info.get("risks", [])

    bar = "\u2550" * 38  # ══════════════════════════════════════

    lines: List[str] = [
        bar,
        f"  {direction} {ticker} \u2014 Grade: {grade} ({score}/100)",
        bar,
        f"  Confidence: {confidence}",
        "",
        "  LOGIC:",
    ]

    # Word-wrap logic text at ~50 chars for readability
    for paragraph in _wrap_text(logic, width=50, indent="  "):
        lines.append(paragraph)

    # Position plan
    lines.append("")
    lines.append("  POSITION PLAN:")
    try:
        rr1 = getattr(plan, "risk_reward_t1", _safe_rr(plan))
        rr2 = getattr(plan, "risk_reward_t2", 0)
        lines.append(f"  Entry:    ${plan.entry_price:.2f}")
        lines.append(f"  Stop:     ${plan.stop_loss:.2f}")
        lines.append(f"  T1:       ${plan.target_1:.2f} ({rr1:.1f}:1 R/R)")
        lines.append(f"  T2:       ${plan.target_2:.2f} ({rr2:.1f}:1 R/R)")
        lines.append(f"  Shares:   {plan.shares}")
        lines.append(f"  Max Loss: ${plan.max_loss:,.0f}")
    except (AttributeError, TypeError):
        lines.append("  (position data unavailable)")

    # Risks
    if risks:
        lines.append("")
        lines.append("  RISKS:")
        for risk in risks:
            lines.append(f"  \u2022 {risk}")

    # Options play (optional)
    if options_suggestion:
        lines.append("")
        lines.append("  OPTIONS PLAY:")
        strategy = options_suggestion.get("strategy", "")
        expiry = options_suggestion.get("expiry", "")
        if strategy or expiry:
            lines.append(f"  {strategy} \u2014 {expiry}")
        buy_leg = options_suggestion.get("buy_leg", "")
        if buy_leg:
            lines.append(f"  BUY {buy_leg}")
        sell_leg = options_suggestion.get("sell_leg", "")
        if sell_leg:
            lines.append(f"  SELL {sell_leg}")
        cost = options_suggestion.get("cost")
        max_profit = options_suggestion.get("max_profit")
        rr = options_suggestion.get("rr")
        detail_parts: List[str] = []
        if cost is not None:
            detail_parts.append(f"Cost: ${cost:,.0f}")
        if max_profit is not None:
            detail_parts.append(f"Max Profit: ${max_profit:,.0f}")
        if rr is not None:
            detail_parts.append(f"R/R: {rr:.1f}:1")
        if detail_parts:
            lines.append(f"  {' | '.join(detail_parts)}")

    lines.append(bar)
    return "\n".join(lines)


def _wrap_text(text: str, width: int = 50, indent: str = "  ") -> List[str]:
    """Simple word-wrap that respects an indent prefix."""
    words = text.split()
    wrapped: List[str] = []
    current_line = indent
    for word in words:
        if len(current_line) + len(word) + 1 > width + len(indent):
            wrapped.append(current_line)
            current_line = indent + word
        else:
            if current_line == indent:
                current_line += word
            else:
                current_line += " " + word
    if current_line.strip():
        wrapped.append(current_line)
    return wrapped
