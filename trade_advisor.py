#!/usr/bin/env python3
"""
Trade Advisor — plain English investment advice engine.

Takes a ticker (or a natural language question) and generates clear,
actionable advice that any investor can understand.  No jargon dumps —
just "buy / don't buy / wait" with the reasoning in simple terms.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path("portfolio.json")


# ---------------------------------------------------------------------------
# Ticker extraction from natural language
# ---------------------------------------------------------------------------

# Common tickers that might appear in questions
KNOWN_TICKERS = set()

def _load_known_tickers():
    global KNOWN_TICKERS
    try:
        if PORTFOLIO_FILE.exists():
            p = json.load(open(PORTFOLIO_FILE))
            KNOWN_TICKERS = set(h["ticker"] for h in p.get("holdings", []))
    except Exception:
        pass
    # Add common tickers people ask about
    KNOWN_TICKERS.update([
        "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "SPY", "QQQ", "NFLX", "AVGO", "AMD", "CRM", "ADBE", "INTC",
    ])

_load_known_tickers()


def extract_ticker(text: str) -> Optional[str]:
    """Extract a stock ticker from natural language text."""
    upper = text.upper()

    # Direct mention of a ticker (1-5 uppercase letters)
    # Check known tickers first
    for ticker in KNOWN_TICKERS:
        # Match as a whole word
        if re.search(r'\b' + re.escape(ticker) + r'\b', upper):
            return ticker

    # Try to find any 1-5 letter word that looks like a ticker
    words = re.findall(r'\b([A-Z]{1,5})\b', upper)
    # Filter out common English words
    stop_words = {
        "I", "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF",
        "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO",
        "UP", "WE", "THE", "AND", "ARE", "BUT", "CAN", "DID", "FOR",
        "GET", "GOT", "HAD", "HAS", "HER", "HIM", "HIS", "HOW", "ITS",
        "LET", "MAY", "NEW", "NOT", "NOW", "OLD", "OUR", "OUT", "OWN",
        "PUT", "SAY", "SHE", "TOO", "USE", "WAS", "WAY", "WHO", "WHY",
        "YES", "YET", "YOU", "ALL", "ANY", "BIG", "BUY", "DAY", "END",
        "FAR", "FEW", "GOD", "GUY", "HIT", "HOT", "JOB", "KID", "LOT",
        "MAN", "MEN", "RAN", "RED", "RUN", "SET", "SIT", "TEN", "TOP",
        "TRY", "TWO", "WAR", "WIN", "WON", "YET", "WHAT", "WHEN",
        "WILL", "WITH", "THAT", "THIS", "THEY", "THAN", "THEM", "THEN",
        "FROM", "HAVE", "BEEN", "WERE", "SAID", "EACH", "MUCH", "GOOD",
        "VERY", "JUST", "OVER", "SUCH", "SELL", "HOLD", "WAIT", "LONG",
        "SHORT", "ABOUT", "THINK", "SHOULD", "COULD", "WOULD", "STOCK",
        "MARKET", "TRADE", "PRICE", "VALUE", "MONEY", "RISK", "SAFE",
        "HIGH", "DOWN", "LIKE", "KEEP", "WANT", "NEED", "KNOW", "TELL",
        "HELP", "MAKE", "TAKE", "GIVE", "COME", "LOOK", "FIND", "CALL",
        "EXPLAIN", "UNDERSTAND", "STRATEGY", "OPTIONS", "GRADE",
    }
    for w in words:
        if w not in stop_words and len(w) >= 2:
            return w

    return None


def detect_intent(text: str) -> str:
    """Detect what the user is asking about."""
    lower = text.lower()

    # Buy/sell decision questions
    if any(p in lower for p in [
        "should i buy", "should i sell", "good buy", "good investment",
        "worth buying", "worth selling", "time to buy", "time to sell",
        "buy or sell", "hold or sell", "keep or sell", "entry",
        "should i enter", "good entry", "good time",
        "would you buy", "would you sell", "is it a buy",
    ]):
        return "decision"

    # Explanation requests
    if any(p in lower for p in [
        "explain", "what does", "what do", "don't understand",
        "dont understand", "what should i do", "what to do",
        "what is", "what are", "mean", "meaning",
        "can you explain", "help me understand", "break it down",
        "in simple terms", "plain english", "eli5",
    ]):
        return "explain"

    # Risk questions
    if any(p in lower for p in [
        "risk", "dangerous", "safe", "how much can i lose",
        "downside", "worst case", "stop loss",
    ]):
        return "risk"

    # Options questions
    if any(p in lower for p in [
        "option", "call", "put", "spread", "strike", "premium",
        "expiry", "expiration", "iron condor", "straddle",
    ]):
        return "options"

    # General questions about a position
    if any(p in lower for p in [
        "how is", "how's", "what about", "update on", "news on",
        "any news", "what happened", "why is it",
    ]):
        return "update"

    # Portfolio-level questions
    if any(p in lower for p in [
        "portfolio", "all positions", "overall", "how am i doing",
        "my stocks", "my holdings", "my money",
    ]):
        return "portfolio"

    # Default — try to give advice
    return "general"


# ---------------------------------------------------------------------------
# Advice generation
# ---------------------------------------------------------------------------

def _get_holding_info(ticker: str) -> Optional[dict]:
    """Get holding info from portfolio.json if ticker is held."""
    try:
        if not PORTFOLIO_FILE.exists():
            return None
        portfolio = json.load(open(PORTFOLIO_FILE))
        for h in portfolio.get("holdings", []):
            if h["ticker"] == ticker:
                return h
    except Exception:
        pass
    return None


def _run_full_analysis(ticker: str) -> dict:
    """Run the full analysis pipeline and return all data."""
    result = {
        "ticker": ticker,
        "tech": None,
        "fund": None,
        "alert": None,
        "plan": None,
        "grade": None,
        "options": None,
        "holding": _get_holding_info(ticker),
        "error": None,
    }

    try:
        import data_layer
        import technical_analysis
        import fundamental_analysis
        import signal_engine
        import position_sizing

        df = data_layer.fetch_daily_ohlcv(ticker)
        if df.empty:
            result["error"] = "No price data available for %s" % ticker
            return result

        tech = technical_analysis.analyze(ticker, df)
        fund = fundamental_analysis.analyze(ticker, "Technology")
        result["tech"] = tech
        result["fund"] = fund

        # Use threshold=1 so we always get a signal for grading
        alert = signal_engine.evaluate(tech, fund, threshold=1)
        result["alert"] = alert

        if alert:
            plan = position_sizing.compute(alert)
            result["plan"] = plan

            try:
                import trade_grader
                result["grade"] = trade_grader.grade_trade(alert, plan)
            except Exception:
                pass

            try:
                import options_analyzer
                result["options"] = options_analyzer.analyze_ticker_options(
                    ticker, alert.direction, alert.signal_score
                )
            except Exception:
                pass

    except Exception as e:
        result["error"] = str(e)

    return result


def advise_decision(ticker: str, analysis: dict) -> str:
    """Generate a clear buy/sell/hold/wait recommendation."""
    tech = analysis.get("tech")
    fund = analysis.get("fund")
    alert = analysis.get("alert")
    plan = analysis.get("plan")
    grade = analysis.get("grade")
    holding = analysis.get("holding")

    if analysis.get("error"):
        return "Sorry, I couldn't analyze %s: %s" % (ticker, analysis["error"])

    lines = []

    # Start with the bottom line
    if not alert or not grade:
        lines.append("%s — NO CLEAR SIGNAL right now." % ticker)
        lines.append("")
        if tech:
            lines.append("The stock is at $%.2f." % tech.current_price)
            if tech.rsi < 30:
                lines.append("It's oversold (RSI %.0f) which could mean a bounce is coming, but there's no confirmed reversal yet." % tech.rsi)
            elif tech.rsi > 70:
                lines.append("It's overbought (RSI %.0f) — not a great time to buy." % tech.rsi)
            else:
                lines.append("RSI is neutral at %.0f — no extreme reading." % tech.rsi)
            if not tech.price_above_200sma:
                lines.append("It's trading BELOW its 200-day moving average — the long-term trend is down. Wait for it to reclaim the 200 SMA before buying.")
            else:
                lines.append("It's above the 200-day moving average which is good for the long-term trend, but no specific entry signal is firing right now.")
        lines.append("")
        lines.append("My advice: WAIT. Don't chase it. Let a clear setup develop.")
        return "\n".join(lines)

    grade_letter = grade.get("grade", "?")
    grade_score = grade.get("score", 0)
    direction = alert.direction
    score = alert.signal_score
    confidence = grade.get("confidence", "LOW")

    # Clear recommendation based on grade
    if direction == "BUY":
        if grade_score >= 80:
            verdict = "STRONG BUY"
            emoji = "🟢"
            advice = "This is a high-conviction setup. The technicals and fundamentals are aligned."
        elif grade_score >= 65:
            verdict = "BUY (moderate confidence)"
            emoji = "🟡"
            advice = "Decent setup but not perfect. Consider a smaller position size than usual."
        elif grade_score >= 50:
            verdict = "WEAK BUY — be cautious"
            emoji = "🟠"
            advice = "The signal is there but it's not strong. Only enter if you have high conviction from your own research."
        else:
            verdict = "DON'T BUY — signal is too weak"
            emoji = "🔴"
            advice = "Even though there's a technical signal, the overall grade is too low. Wait for a better setup."
    else:  # SELL
        if holding:
            pnl_pct = float(holding.get("pnl_pct", 0))
            strategy = holding.get("strategy", "trade")
            if strategy == "long_term":
                if pnl_pct < -30:
                    verdict = "CONSIDER TRIMMING"
                    emoji = "🟠"
                    advice = "This is a long-term hold but it's down significantly. Consider reducing your position by 25-50%% to manage risk."
                else:
                    verdict = "HOLD — it's a long-term position"
                    emoji = "🟡"
                    advice = "Sell signals on long-term holds are normal during pullbacks. Unless something fundamentally changed, keep holding."
            else:
                if grade_score >= 60:
                    verdict = "SELL"
                    emoji = "🔴"
                    advice = "Multiple signals suggest exiting this trade position."
                else:
                    verdict = "WATCH closely"
                    emoji = "🟡"
                    advice = "Bearish signals are appearing but not strong enough for a full exit yet. Set a tight stop."
        else:
            verdict = "BEARISH — avoid buying"
            emoji = "🔴"
            advice = "The analysis is showing bearish signals. Don't enter a new position."

    lines.append("%s %s — %s" % (emoji, ticker, verdict))
    lines.append("Grade: %s (%d/100) | Confidence: %s" % (grade_letter, grade_score, confidence))
    lines.append("")
    lines.append(advice)

    # Add key numbers
    if tech:
        lines.append("")
        lines.append("Price: $%.2f | RSI: %.0f | Fundamental: %d/15" % (
            tech.current_price, tech.rsi,
            fund.fundamental_score if fund else 0))

    # If they hold it, show their P&L
    if holding:
        pnl = float(holding.get("unrealized_pnl", 0))
        pnl_pct = float(holding.get("pnl_pct", 0))
        avg = float(holding.get("avg_price", 0))
        lines.append("")
        lines.append("Your position: %d shares @ $%.2f avg" % (
            holding.get("shares", 0), avg))
        lines.append("P&L: ${:+,.0f} ({:+.1f}%)".format(pnl, pnl_pct))

    # Entry plan if it's a buy
    if direction == "BUY" and plan and grade_score >= 50:
        lines.append("")
        lines.append("If you decide to buy:")
        lines.append("  Entry: $%.2f" % plan.entry_price)
        lines.append("  Stop loss: $%.2f (exit if it drops here)" % plan.stop_loss)
        lines.append("  Target: $%.2f (take profit here)" % plan.target_1)
        lines.append("  Max risk: $%.0f" % plan.max_loss)

    return "\n".join(lines)


def advise_explain(ticker: str, analysis: dict) -> str:
    """Explain the analysis in simple terms."""
    tech = analysis.get("tech")
    fund = analysis.get("fund")
    alert = analysis.get("alert")
    grade = analysis.get("grade")
    holding = analysis.get("holding")

    if analysis.get("error"):
        return "Couldn't analyze %s: %s" % (ticker, analysis["error"])

    lines = ["Here's what's happening with %s in simple terms:" % ticker]
    lines.append("")

    if tech:
        # Price context
        lines.append("PRICE: $%.2f" % tech.current_price)
        if getattr(tech, "pct_from_52w_high", 0):
            lines.append("  %.0f%% below its 52-week high" % abs(tech.pct_from_52w_high))

        # Trend
        if tech.price_above_200sma:
            lines.append("TREND: Uptrend (above 200-day average)")
        else:
            lines.append("TREND: Downtrend (below 200-day average)")

        if getattr(tech, "ema_ribbon_bullish", False):
            lines.append("  All moving averages are lined up bullishly — strong momentum")
        elif getattr(tech, "ema_ribbon_bearish", False):
            lines.append("  All moving averages are lined up bearishly — weak momentum")

        # Momentum
        lines.append("")
        if tech.rsi < 30:
            lines.append("MOMENTUM: Oversold (RSI %.0f) — the stock has been beaten down. Could bounce, but don't catch a falling knife." % tech.rsi)
        elif tech.rsi > 70:
            lines.append("MOMENTUM: Overbought (RSI %.0f) — the stock has run up a lot. Might pull back soon." % tech.rsi)
        elif tech.rsi > 50:
            lines.append("MOMENTUM: Mildly bullish (RSI %.0f) — buyers have slight control." % tech.rsi)
        else:
            lines.append("MOMENTUM: Mildly bearish (RSI %.0f) — sellers have slight control." % tech.rsi)

        # Key patterns
        patterns = []
        if tech.double_bottom:
            patterns.append("Double bottom (bullish reversal pattern)")
        if tech.head_and_shoulders:
            patterns.append("Head & Shoulders (bearish reversal — watch out)")
        if tech.inverse_head_shoulders:
            patterns.append("Inverse Head & Shoulders (bullish reversal)")
        if getattr(tech, "cup_and_handle", False):
            patterns.append("Cup & Handle (strong bullish continuation)")
        if getattr(tech, "bull_flag", False):
            patterns.append("Bull Flag (bullish continuation)")
        if getattr(tech, "ascending_triangle", False):
            patterns.append("Ascending Triangle (bullish breakout pattern)")
        if getattr(tech, "ttm_squeeze_fired", False):
            patterns.append("TTM Squeeze fired (volatility about to explode)")
        if patterns:
            lines.append("")
            lines.append("PATTERNS DETECTED:")
            for p in patterns:
                lines.append("  • %s" % p)

    if fund:
        lines.append("")
        lines.append("FUNDAMENTALS: %d/15" % fund.fundamental_score)
        if fund.fundamental_score >= 12:
            lines.append("  Excellent fundamentals — strong company")
        elif fund.fundamental_score >= 8:
            lines.append("  Good fundamentals — solid company")
        elif fund.fundamental_score >= 5:
            lines.append("  Average fundamentals — some concerns")
        else:
            lines.append("  Weak fundamentals — proceed with caution")

        # Key metrics in plain English
        details = []
        if getattr(fund, "roe", None) and fund.roe:
            roe_pct = fund.roe * 100 if fund.roe < 1 else fund.roe
            details.append("ROE: %.0f%% (%s)" % (roe_pct, "strong" if roe_pct > 15 else "weak"))
        if getattr(fund, "fcf_yield", None) and fund.fcf_yield:
            details.append("Free cash flow yield: %.1f%%" % fund.fcf_yield)
        if getattr(fund, "earnings_beat_streak", 0) and fund.earnings_beat_streak > 0:
            details.append("Beat earnings %d quarters in a row" % fund.earnings_beat_streak)
        if getattr(fund, "insider_net_bullish", False):
            details.append("Insiders are buying their own stock (bullish)")
        if details:
            for d in details:
                lines.append("  • %s" % d)

    # Bottom line
    if alert and grade:
        lines.append("")
        grade_letter = grade.get("grade", "?")
        if alert.direction == "BUY":
            if grade.get("score", 0) >= 70:
                lines.append("BOTTOM LINE: The stock looks good for buying. Grade %s." % grade_letter)
            elif grade.get("score", 0) >= 50:
                lines.append("BOTTOM LINE: There's a weak buy signal but it's not convincing. Grade %s." % grade_letter)
            else:
                lines.append("BOTTOM LINE: Not a good time to buy. Wait. Grade %s." % grade_letter)
        else:
            lines.append("BOTTOM LINE: Bearish signals — avoid or consider selling. Grade %s." % grade_letter)
    else:
        lines.append("")
        lines.append("BOTTOM LINE: No clear signal right now. Best to wait.")

    return "\n".join(lines)


def advise_risk(ticker: str, analysis: dict) -> str:
    """Explain the risks in plain terms."""
    tech = analysis.get("tech")
    plan = analysis.get("plan")
    holding = analysis.get("holding")
    grade = analysis.get("grade")

    lines = ["RISK ASSESSMENT for %s:" % ticker]
    lines.append("")

    if grade and grade.get("risks"):
        for r in grade["risks"]:
            lines.append("⚠️ %s" % r)
        lines.append("")

    if tech:
        if tech.rsi > 70:
            lines.append("🔴 Overbought — high chance of a pullback")
        if tech.rsi < 30:
            lines.append("🟡 Oversold — could bounce but could also keep falling")
        if not tech.price_above_200sma:
            lines.append("🔴 Below 200-day average — long-term trend is against you")
        if getattr(tech, "gap_down", False):
            lines.append("🔴 Recent gap down — might have more downside")
        if tech.bb_breakout_lower:
            lines.append("🟡 Below Bollinger Band — extended to the downside")

    if holding:
        pnl_pct = float(holding.get("pnl_pct", 0))
        if pnl_pct < -20:
            lines.append("🔴 You're down %.0f%% — consider if your thesis still holds" % abs(pnl_pct))
        cost = float(holding.get("cost_basis", 0))
        lines.append("")
        lines.append("Your cost basis: $%.0f" % cost)
        lines.append("If it drops 10%% more: you'd lose another ~$%.0f" % (cost * 0.10))

    if plan:
        lines.append("")
        lines.append("If entering now:")
        lines.append("  Stop loss at: $%.2f" % plan.stop_loss)
        lines.append("  Max you'd lose: $%.0f per position" % plan.max_loss)

    if not lines[2:]:  # No real risks found
        lines.append("No specific risk flags detected — but always use a stop loss!")

    return "\n".join(lines)


def advise_options_simple(ticker: str, analysis: dict) -> str:
    """Explain the options strategy in simple terms."""
    opts = analysis.get("options")
    alert = analysis.get("alert")

    if not opts or opts.get("strategy_name") == "No options data available":
        return ("No options data available for %s. "
                "This stock might not have liquid options, or markets are closed." % ticker)

    lines = ["OPTIONS PLAY for %s:" % ticker]
    lines.append("")

    strategy = opts.get("strategy_name", "Unknown")
    lines.append("Strategy: %s" % strategy)
    lines.append("")

    # Explain the strategy in plain English
    strat_lower = strategy.lower()
    if "bull call" in strat_lower:
        lines.append("What this means: You're betting the stock goes UP.")
        lines.append("You buy a call option and sell a higher one to reduce cost.")
        lines.append("You profit if the stock rises above your breakeven by expiration.")
    elif "bull put" in strat_lower:
        lines.append("What this means: You're collecting premium betting the stock STAYS ABOVE a level.")
        lines.append("You sell a put and buy a lower one for protection.")
        lines.append("You keep the credit if the stock stays above your short put strike.")
    elif "bear call" in strat_lower or "bear put" in strat_lower:
        lines.append("What this means: You're betting the stock goes DOWN or stays flat.")
    elif "long call" in strat_lower:
        lines.append("What this means: You're betting the stock goes UP.")
        lines.append("Simple and direct — you buy a call option.")
    elif "long put" in strat_lower:
        lines.append("What this means: You're buying insurance / betting the stock goes DOWN.")
    elif "iron condor" in strat_lower:
        lines.append("What this means: You're betting the stock STAYS in a range.")
        lines.append("You collect premium from both sides.")

    lines.append("")

    # Key numbers
    max_profit = opts.get("max_profit", 0)
    max_loss = opts.get("max_loss", 0)
    breakeven = opts.get("breakeven", 0)
    rr = opts.get("risk_reward_ratio", 0)
    prob = opts.get("probability_of_profit", 0)

    if max_profit:
        lines.append("Max you can make: $%.0f per contract" % (max_profit * 100 if max_profit < 50 else max_profit))
    if max_loss:
        lines.append("Max you can lose: $%.0f per contract" % (abs(max_loss) * 100 if abs(max_loss) < 50 else abs(max_loss)))
    if breakeven:
        lines.append("Breakeven price: $%.2f" % breakeven)
    if prob:
        lines.append("Estimated win rate: ~%d%%" % prob)

    # Legs
    legs = opts.get("legs", [])
    if legs:
        lines.append("")
        lines.append("The trade:")
        for leg in legs:
            action = leg.get("action", "?")
            opt_type = leg.get("type", "?")
            strike = leg.get("strike", 0)
            premium = leg.get("premium", 0)
            lines.append("  %s %s $%.0f strike @ $%.2f" % (action, opt_type, strike, premium))

    lines.append("")
    lines.append("Expiry: %s" % opts.get("legs", [{}])[0].get("expiry", "N/A"))

    return "\n".join(lines)


def advise_update(ticker: str, analysis: dict) -> str:
    """Quick update on a ticker — price, trend, any notable changes."""
    tech = analysis.get("tech")
    fund = analysis.get("fund")
    holding = analysis.get("holding")

    if analysis.get("error"):
        return "Couldn't get data for %s: %s" % (ticker, analysis["error"])

    lines = ["%s Quick Update:" % ticker]
    lines.append("")

    if tech:
        lines.append("Price: $%.2f" % tech.current_price)
        lines.append("RSI: %.0f | MACD: %.2f" % (tech.rsi, tech.macd_value))

        if tech.price_above_200sma:
            lines.append("Trend: ✅ Above 200 SMA (bullish)")
        else:
            lines.append("Trend: ❌ Below 200 SMA (bearish)")

        if getattr(tech, "gap_up", False):
            lines.append("⬆️ Gapped up today (%.1f%%)" % getattr(tech, "gap_up_pct", 0))
        if getattr(tech, "gap_down", False):
            lines.append("⬇️ Gapped down today (%.1f%%)" % getattr(tech, "gap_down_pct", 0))

    if holding:
        pnl = float(holding.get("unrealized_pnl", 0))
        pnl_pct = float(holding.get("pnl_pct", 0))
        lines.append("")
        lines.append("Your P&L: ${:+,.0f} ({:+.1f}%)".format(pnl, pnl_pct))

    if fund:
        lines.append("Fundamental score: %d/15" % fund.fundamental_score)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_question(text: str) -> str:
    """Take a natural language question and return plain English advice."""
    ticker = extract_ticker(text)
    intent = detect_intent(text)

    # Portfolio-level questions (no ticker needed)
    if intent == "portfolio":
        try:
            from telegram_bot import cmd_status
            return cmd_status()
        except Exception:
            return "Couldn't load portfolio. Try /status"

    if not ticker:
        # Try to be helpful even without a ticker
        lower = text.lower()
        if any(w in lower for w in ["buy", "invest", "opportunity"]):
            try:
                from telegram_bot import cmd_buy
                return "Let me scan for opportunities...\n\n" + cmd_buy()
            except Exception:
                return "Try /buy to see current opportunities, or ask about a specific stock like 'should I buy TSLA?'"

        if any(w in lower for w in ["sell", "exit", "close"]):
            try:
                from telegram_bot import cmd_sell
                return cmd_sell()
            except Exception:
                return "Try /sell to check your positions."

        return ("I'm not sure which stock you're asking about. "
                "Try asking like:\n\n"
                "• 'Should I buy TSLA?'\n"
                "• 'Explain META'\n"
                "• 'Is NVDA a good investment?'\n"
                "• 'What's the risk on AMZN?'\n"
                "• 'Options play for AAPL'\n"
                "\nOr type /help for all commands.")

    # Run analysis
    analysis = _run_full_analysis(ticker)

    if intent == "decision":
        return advise_decision(ticker, analysis)
    elif intent == "explain":
        return advise_explain(ticker, analysis)
    elif intent == "risk":
        return advise_risk(ticker, analysis)
    elif intent == "options":
        return advise_options_simple(ticker, analysis)
    elif intent == "update":
        return advise_update(ticker, analysis)
    else:
        # General — give decision advice (most useful default)
        return advise_decision(ticker, analysis)
