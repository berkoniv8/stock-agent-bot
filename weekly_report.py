#!/usr/bin/env python3
"""
Weekly Portfolio Report — generates a comprehensive Sunday digest covering:
  - Portfolio P&L for the week (Monday open vs Friday close)
  - Top movers (best and worst weekly performers)
  - Position alerts from position_monitor
  - Upcoming earnings (next 7 days)
  - Market regime (from market_regime)
  - Sector rotation signals (from sector_rotation)
  - Tax-loss harvesting opportunities (from tax_harvesting if available)
  - 3-5 actionable recommendations

Usage:
    python3 weekly_report.py          # Generate and print report
    python3 weekly_report.py --send   # Also email the HTML version
    python3 weekly_report.py --json   # Output raw dict as JSON
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"
REPORTS_DIR = Path(__file__).parent / "logs" / "weekly_reports"


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def _load_portfolio() -> dict:
    """Load portfolio.json."""
    if not PORTFOLIO_FILE.exists():
        return {"holdings": [], "total_portfolio_value": 0}
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Weekly P&L calculation
# ---------------------------------------------------------------------------

def _fetch_weekly_returns(tickers: List[str]) -> Dict[str, dict]:
    """Fetch Monday-open vs Friday-close returns for each ticker.

    Falls back to a 5-trading-day window when exact Mon/Fri closes are
    unavailable (e.g. shortened holiday week).

    Returns dict keyed by ticker:
        {"open_price": float, "close_price": float, "week_return_pct": float}
    """
    results: Dict[str, dict] = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="10d")
            if hist.empty or len(hist) < 2:
                continue
            open_price = float(hist["Close"].iloc[-min(5, len(hist))])
            close_price = float(hist["Close"].iloc[-1])
            week_return = ((close_price - open_price) / open_price * 100) if open_price else 0.0
            results[ticker] = {
                "open_price": round(open_price, 2),
                "close_price": round(close_price, 2),
                "week_return_pct": round(week_return, 2),
            }
        except Exception as exc:
            logger.debug("%s: failed to fetch weekly data — %s", ticker, exc)
    return results


# ---------------------------------------------------------------------------
# Tax harvesting stub (module may or may not exist)
# ---------------------------------------------------------------------------

def _safe_tax_summary() -> dict:
    """Call tax_harvesting.analyze_harvesting() if the module exists."""
    try:
        import tax_harvesting  # type: ignore
        return tax_harvesting.analyze_harvesting()
    except ImportError:
        logger.debug("tax_harvesting module not found — skipping")
        return {}
    except Exception as exc:
        logger.warning("tax_harvesting error: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_weekly_report() -> dict:
    """Compile the full weekly report dict.

    Returns
    -------
    dict with keys: week_ending, portfolio_summary, weekly_pnl, top_movers,
    position_alerts, earnings_next_week, market_regime, sector_rotation,
    tax_summary, recommendations.
    """
    today = date.today()
    portfolio = _load_portfolio()
    holdings = portfolio.get("holdings", [])
    tickers = [h["ticker"] for h in holdings]

    # ------------------------------------------------------------------
    # 1. Weekly price returns for all holdings
    # ------------------------------------------------------------------
    weekly_returns = _fetch_weekly_returns(tickers)

    weekly_pnl = sum(
        h.get("shares", 0) * (
            weekly_returns.get(h["ticker"], {}).get("close_price", h.get("current_price", 0))
            - weekly_returns.get(h["ticker"], {}).get("open_price", h.get("current_price", 0))
        )
        for h in holdings
    )

    top_movers: List[dict] = []
    for h in holdings:
        ret = weekly_returns.get(h["ticker"])
        if ret:
            top_movers.append(
                {
                    "ticker": h["ticker"],
                    "week_return": ret["week_return_pct"],
                    "close_price": ret["close_price"],
                    "shares": h.get("shares", 0),
                }
            )
    top_movers.sort(key=lambda x: abs(x["week_return"]), reverse=True)

    # ------------------------------------------------------------------
    # 2. Position alerts
    # ------------------------------------------------------------------
    position_alerts: List[dict] = []
    try:
        import position_monitor
        all_results = position_monitor.check_all_positions()
        position_alerts = position_monitor.get_actionable(all_results)
    except Exception as exc:
        logger.warning("position_monitor error: %s", exc)

    # ------------------------------------------------------------------
    # 3. Upcoming earnings (next 7 days)
    # ------------------------------------------------------------------
    earnings_next_week: List[dict] = []
    try:
        import earnings_calendar
        all_earnings = earnings_calendar.get_portfolio_earnings()
        earnings_next_week = [
            e for e in all_earnings
            if e.get("days_away") is not None and 0 <= e["days_away"] <= 7
        ]
    except Exception as exc:
        logger.warning("earnings_calendar error: %s", exc)

    # ------------------------------------------------------------------
    # 4. Market regime
    # ------------------------------------------------------------------
    market_regime: dict = {}
    try:
        import market_regime as mr
        market_regime = mr.detect_regime()
    except Exception as exc:
        logger.warning("market_regime error: %s", exc)

    # ------------------------------------------------------------------
    # 5. Sector rotation
    # ------------------------------------------------------------------
    sector_rotation: dict = {}
    try:
        import sector_rotation as sr
        rotation = sr.detect_rotation()
        sector_rotation = rotation if isinstance(rotation, dict) else {}
    except Exception as exc:
        logger.warning("sector_rotation error: %s", exc)

    # ------------------------------------------------------------------
    # 6. Tax harvesting
    # ------------------------------------------------------------------
    tax_summary = _safe_tax_summary()

    # ------------------------------------------------------------------
    # 7. Actionable recommendations
    # ------------------------------------------------------------------
    recommendations: List[str] = []

    # a) Hard sell/trim signals
    urgent = [a for a in position_alerts if a.get("action") in ("SELL", "TRIM")]
    for a in urgent[:2]:
        recommendations.append(
            "{} {}: {} (P&L {:.1f}%)".format(
                a["action"], a["ticker"],
                "; ".join(a.get("reasons", []))[:80],
                a.get("pnl_pct", 0),
            )
        )

    # b) Earnings caution
    if earnings_next_week:
        tickers_earning = ", ".join(e["ticker"] for e in earnings_next_week[:3])
        recommendations.append(
            "Earnings this week for %s — review positions before reports." % tickers_earning
        )

    # c) Market regime guidance
    regime_name = market_regime.get("regime", "")
    if regime_name in ("BEAR_STRONG", "BEAR_WEAK"):
        recommendations.append(
            "Market regime is %s — reduce new position sizes and keep stop-losses tight." % regime_name
        )
    elif regime_name == "BULL_STRONG":
        recommendations.append("Market regime is BULL_STRONG — lean into momentum, consider adding to leaders.")

    # d) Top weekly loser — DCA candidate if long_term
    long_term_losers = [
        m for m in top_movers
        if m["week_return"] < -3
        and any(
            h["ticker"] == m["ticker"] and h.get("strategy") == "long_term"
            for h in holdings
        )
    ]
    if long_term_losers:
        worst = long_term_losers[0]
        recommendations.append(
            "%s down %.1f%% this week — run dca_advisor.py for averaging levels."
            % (worst["ticker"], worst["week_return"])
        )

    # Keep top 5 recommendations
    recommendations = recommendations[:5]
    if not recommendations:
        recommendations.append("No urgent actions identified — monitor positions as usual.")

    # ------------------------------------------------------------------
    # Portfolio summary block
    # ------------------------------------------------------------------
    total_val = float(portfolio.get("total_portfolio_value", 0))
    total_unrealized = sum(float(h.get("unrealized_pnl", 0) or 0) for h in holdings)
    total_invested = sum(float(h.get("current_value", 0) or 0) for h in holdings)
    realized_ytd = float(portfolio.get("realized_pnl_ytd", 0))

    portfolio_summary = {
        "total_portfolio_value": total_val,
        "available_cash": float(portfolio.get("available_cash", 0)),
        "positions": len(holdings),
        "total_invested": round(total_invested, 2),
        "unrealized_pnl": round(total_unrealized, 2),
        "realized_pnl_ytd": realized_ytd,
        "total_pnl_ytd": round(total_unrealized + realized_ytd, 2),
    }

    return {
        "week_ending": today.isoformat(),
        "generated_at": datetime.now().isoformat(),
        "portfolio_summary": portfolio_summary,
        "weekly_pnl": round(weekly_pnl, 2),
        "top_movers": top_movers,
        "position_alerts": position_alerts,
        "earnings_next_week": earnings_next_week,
        "market_regime": market_regime,
        "sector_rotation": sector_rotation,
        "tax_summary": tax_summary,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Plain-text formatter
# ---------------------------------------------------------------------------

def format_report(report: dict) -> str:
    """Render the weekly report dict as a readable plain-text string."""
    lines: List[str] = []
    week = report.get("week_ending", "N/A")
    generated = report.get("generated_at", "")

    lines.append("=" * 65)
    lines.append("  WEEKLY PORTFOLIO REPORT — Week ending %s" % week)
    lines.append("=" * 65)

    # --- Portfolio summary ---
    ps = report.get("portfolio_summary", {})
    lines.append("")
    lines.append("  PORTFOLIO SUMMARY")
    lines.append("  " + "─" * 61)
    lines.append("  Total value     : ${:>12,.2f}".format(ps.get("total_portfolio_value", 0)))
    lines.append("  Cash            : ${:>12,.2f}".format(ps.get("available_cash", 0)))
    lines.append("  Invested        : ${:>12,.2f}".format(ps.get("total_invested", 0)))
    lines.append("  Positions       :  {:>12d}".format(ps.get("positions", 0)))
    lines.append("  Unrealized P&L  : ${:>+12,.2f}".format(ps.get("unrealized_pnl", 0)))
    lines.append("  Realized YTD    : ${:>+12,.2f}".format(ps.get("realized_pnl_ytd", 0)))
    lines.append("  Total P&L YTD   : ${:>+12,.2f}".format(ps.get("total_pnl_ytd", 0)))

    # --- Weekly P&L ---
    wpnl = report.get("weekly_pnl", 0)
    lines.append("")
    lines.append("  WEEKLY P&L      : ${:>+12,.2f}".format(wpnl))

    # --- Top movers ---
    movers = report.get("top_movers", [])
    if movers:
        lines.append("")
        lines.append("  TOP MOVERS THIS WEEK")
        lines.append("  " + "─" * 61)
        lines.append("  %-6s  %s" % ("Ticker", "Week Return"))
        for m in movers[:8]:
            sign = "+" if m["week_return"] >= 0 else ""
            lines.append("  %-6s  %s%.1f%%" % (m["ticker"], sign, m["week_return"]))

    # --- Position alerts ---
    alerts = report.get("position_alerts", [])
    if alerts:
        lines.append("")
        lines.append("  POSITION ALERTS  (%d)" % len(alerts))
        lines.append("  " + "─" * 61)
        for a in alerts:
            reasons = "; ".join(a.get("reasons", [])[:2])
            lines.append(
                "  [%s] %-6s  P&L %.1f%%  — %s"
                % (a.get("action", "?"), a.get("ticker", ""), a.get("pnl_pct", 0), reasons[:55])
            )
    else:
        lines.append("")
        lines.append("  POSITION ALERTS  — None requiring attention.")

    # --- Earnings next week ---
    earnings = report.get("earnings_next_week", [])
    if earnings:
        lines.append("")
        lines.append("  EARNINGS NEXT 7 DAYS")
        lines.append("  " + "─" * 61)
        for e in earnings:
            days = e.get("days_away", "?")
            eps = e.get("estimate_eps")
            eps_str = "  EPS est: $%.2f" % eps if eps is not None else ""
            lines.append(
                "  %-6s  %s  (%s days away)%s"
                % (e.get("ticker", ""), e.get("date", ""), days, eps_str)
            )
    else:
        lines.append("")
        lines.append("  EARNINGS NEXT 7 DAYS  — No portfolio holdings reporting.")

    # --- Market regime ---
    regime = report.get("market_regime", {})
    if regime:
        lines.append("")
        lines.append("  MARKET REGIME")
        lines.append("  " + "─" * 61)
        lines.append("  Regime          : %s" % regime.get("regime", "N/A"))
        lines.append("  Description     : %s" % regime.get("description", "N/A"))
        lines.append("  SPY vs 200 SMA  : %s" % regime.get("spy_vs_200sma", "N/A"))
        lines.append("  VIX level       : %s" % regime.get("vix_level", "N/A"))

    # --- Sector rotation ---
    rotation = report.get("sector_rotation", {})
    if rotation:
        lines.append("")
        lines.append("  SECTOR ROTATION")
        lines.append("  " + "─" * 61)
        leaders = rotation.get("leaders", [])
        laggards = rotation.get("laggards", [])
        if leaders:
            lines.append("  Leading    : %s" % ", ".join(s.get("sector", str(s)) if isinstance(s, dict) else str(s) for s in leaders[:3]))
        if laggards:
            lines.append("  Lagging    : %s" % ", ".join(s.get("sector", str(s)) if isinstance(s, dict) else str(s) for s in laggards[:3]))

    # --- Tax summary ---
    tax = report.get("tax_summary", {})
    if tax:
        lines.append("")
        lines.append("  TAX HARVESTING")
        lines.append("  " + "─" * 61)
        harvest_candidates = tax.get("harvest_candidates", [])
        if harvest_candidates:
            for c in harvest_candidates[:3]:
                ticker = c.get("ticker", "?")
                loss = c.get("unrealized_pnl", 0)
                lines.append("  %-6s  Harvestable loss: ${:+,.0f}".format(loss) % ticker)
        else:
            lines.append("  No harvest candidates identified.")

    # --- Recommendations ---
    lines.append("")
    lines.append("  RECOMMENDATIONS")
    lines.append("  " + "─" * 61)
    for i, rec in enumerate(report.get("recommendations", []), 1):
        lines.append("  %d. %s" % (i, rec))

    # --- Footer ---
    lines.append("")
    lines.append("  Generated: %s" % generated[:19] if generated else "")
    lines.append("=" * 65)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML formatter
# ---------------------------------------------------------------------------

def format_html_report(report: dict) -> str:
    """Render the weekly report as a styled HTML email."""
    week = report.get("week_ending", "N/A")
    ps = report.get("portfolio_summary", {})
    wpnl = report.get("weekly_pnl", 0)
    wpnl_color = "#2e7d32" if wpnl >= 0 else "#c62828"
    upnl = ps.get("unrealized_pnl", 0)
    upnl_color = "#2e7d32" if upnl >= 0 else "#c62828"

    def _pnl_span(val: float) -> str:
        color = "#2e7d32" if val >= 0 else "#c62828"
        sign = "+" if val >= 0 else ""
        return '<span style="color:{};font-weight:bold;">{}{:,.2f}</span>'.format(color, sign, val)

    # Movers table rows
    mover_rows = ""
    for m in report.get("top_movers", [])[:8]:
        ret = m["week_return"]
        color = "#2e7d32" if ret >= 0 else "#c62828"
        sign = "+" if ret >= 0 else ""
        mover_rows += (
            "<tr>"
            "<td style='padding:4px 8px;'><strong>%s</strong></td>"
            "<td style='padding:4px 8px;color:%s;'>%s%.1f%%</td>"
            "</tr>"
        ) % (m["ticker"], color, sign, ret)

    # Alerts
    alert_rows = ""
    action_colors = {"SELL": "#c62828", "TRIM": "#e65100", "WATCH": "#f57f17", "HOLD": "#388e3c"}
    for a in report.get("position_alerts", []):
        action = a.get("action", "?")
        col = action_colors.get(action, "#555")
        reasons = "; ".join(a.get("reasons", [])[:2])[:80]
        alert_rows += (
            "<tr>"
            "<td style='padding:4px 8px;'><strong style='color:{col};'>{action}</strong></td>"
            "<td style='padding:4px 8px;'>{ticker}</td>"
            "<td style='padding:4px 8px;'>{pnl:+.1f}%</td>"
            "<td style='padding:4px 8px;font-size:12px;color:#555;'>{reasons}</td>"
            "</tr>"
        ).format(
            col=col, action=action,
            ticker=a.get("ticker", ""),
            pnl=a.get("pnl_pct", 0),
            reasons=reasons,
        )

    # Earnings
    earnings_rows = ""
    for e in report.get("earnings_next_week", []):
        days = e.get("days_away", "?")
        eps = e.get("estimate_eps")
        eps_str = "$%.2f" % eps if eps is not None else "N/A"
        earnings_rows += (
            "<tr>"
            "<td style='padding:4px 8px;'><strong>%s</strong></td>"
            "<td style='padding:4px 8px;'>%s</td>"
            "<td style='padding:4px 8px;'>%s days</td>"
            "<td style='padding:4px 8px;'>%s</td>"
            "</tr>"
        ) % (e.get("ticker", ""), e.get("date", "N/A"), days, eps_str)

    # Regime
    regime = report.get("market_regime", {})
    regime_name = regime.get("regime", "N/A")
    regime_desc = regime.get("description", "")
    regime_color = (
        "#2e7d32" if "BULL" in regime_name
        else "#c62828" if "BEAR" in regime_name
        else "#555"
    )

    # Recommendations
    rec_items = "".join(
        "<li style='margin-bottom:6px;'>%s</li>" % r
        for r in report.get("recommendations", [])
    )

    html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"/></head>
<body style="font-family:Arial,sans-serif;max-width:640px;margin:0 auto;padding:16px;color:#222;">

  <div style="background:#1565c0;color:white;padding:20px;border-radius:8px 8px 0 0;">
    <h1 style="margin:0;font-size:20px;">Weekly Portfolio Report</h1>
    <p style="margin:4px 0 0;opacity:.85;">Week ending {week}</p>
  </div>

  <div style="border:1px solid #ddd;padding:16px;border-radius:0 0 8px 8px;margin-bottom:16px;">

    <!-- Summary -->
    <h2 style="font-size:16px;border-bottom:1px solid #eee;padding-bottom:6px;">Portfolio Summary</h2>
    <table style="width:100%;border-collapse:collapse;">
      <tr><td style="padding:4px 0;">Total Value</td><td style="text-align:right;font-weight:bold;">${total_val:,.2f}</td></tr>
      <tr><td style="padding:4px 0;">Cash</td><td style="text-align:right;">${cash:,.2f}</td></tr>
      <tr><td style="padding:4px 0;">Open Positions</td><td style="text-align:right;">{positions}</td></tr>
      <tr><td style="padding:4px 0;">Unrealized P&amp;L</td><td style="text-align:right;">{upnl_span}</td></tr>
      <tr><td style="padding:4px 0;">Realized YTD</td><td style="text-align:right;">{ryz_span}</td></tr>
      <tr style="border-top:1px solid #eee;"><td style="padding:4px 0;"><strong>Weekly P&amp;L</strong></td>
        <td style="text-align:right;font-size:18px;">{wpnl_span}</td></tr>
    </table>

    <!-- Top movers -->
    <h2 style="font-size:16px;border-bottom:1px solid #eee;padding-bottom:6px;margin-top:20px;">Top Movers</h2>
    {mover_section}

    <!-- Alerts -->
    <h2 style="font-size:16px;border-bottom:1px solid #eee;padding-bottom:6px;margin-top:20px;">Position Alerts</h2>
    {alert_section}

    <!-- Earnings -->
    <h2 style="font-size:16px;border-bottom:1px solid #eee;padding-bottom:6px;margin-top:20px;">Earnings Next 7 Days</h2>
    {earnings_section}

    <!-- Market regime -->
    <h2 style="font-size:16px;border-bottom:1px solid #eee;padding-bottom:6px;margin-top:20px;">Market Regime</h2>
    <p style="margin:4px 0;">
      <strong style="color:{regime_color};">{regime_name}</strong>
      {regime_desc_html}
    </p>

    <!-- Recommendations -->
    <h2 style="font-size:16px;border-bottom:1px solid #eee;padding-bottom:6px;margin-top:20px;">Recommendations</h2>
    <ol style="padding-left:20px;">{rec_items}</ol>

  </div>
  <p style="font-size:11px;color:#aaa;text-align:center;">
    Generated by Stock Agent &mdash; {generated}
  </p>
</body>
</html>""".format(
        week=week,
        total_val=ps.get("total_portfolio_value", 0),
        cash=ps.get("available_cash", 0),
        positions=ps.get("positions", 0),
        upnl_span=_pnl_span(upnl),
        ryz_span=_pnl_span(ps.get("realized_pnl_ytd", 0)),
        wpnl_span=_pnl_span(wpnl),
        mover_section=(
            "<table style='width:100%;border-collapse:collapse;'>"
            + mover_rows
            + "</table>"
        ) if mover_rows else "<p style='color:#888;'>No data available.</p>",
        alert_section=(
            "<table style='width:100%;border-collapse:collapse;'>"
            "<tr style='background:#f5f5f5;'>"
            "<th style='padding:4px 8px;text-align:left;'>Action</th>"
            "<th style='padding:4px 8px;text-align:left;'>Ticker</th>"
            "<th style='padding:4px 8px;text-align:left;'>P&L</th>"
            "<th style='padding:4px 8px;text-align:left;'>Reason</th>"
            "</tr>"
            + alert_rows
            + "</table>"
        ) if alert_rows else "<p style='color:#2e7d32;'>No positions requiring action.</p>",
        earnings_section=(
            "<table style='width:100%;border-collapse:collapse;'>"
            "<tr style='background:#f5f5f5;'>"
            "<th style='padding:4px 8px;text-align:left;'>Ticker</th>"
            "<th style='padding:4px 8px;text-align:left;'>Date</th>"
            "<th style='padding:4px 8px;text-align:left;'>Days Away</th>"
            "<th style='padding:4px 8px;text-align:left;'>EPS Est</th>"
            "</tr>"
            + earnings_rows
            + "</table>"
        ) if earnings_rows else "<p style='color:#888;'>No holdings report earnings in the next 7 days.</p>",
        regime_color=regime_color,
        regime_name=regime_name,
        regime_desc_html="&mdash; %s" % regime_desc if regime_desc else "",
        rec_items=rec_items if rec_items else "<li>No urgent actions.</li>",
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    return html


# ---------------------------------------------------------------------------
# Email sender
# ---------------------------------------------------------------------------

def send_weekly_report(report: dict) -> None:
    """Send the HTML weekly report via notifications.send_email_text."""
    try:
        import notifications
    except ImportError:
        logger.error("notifications module not found — cannot send email")
        return

    text = format_report(report)
    html = format_html_report(report)
    week = report.get("week_ending", datetime.now().strftime("%Y-%m-%d"))
    subject = "Weekly Portfolio Report — %s" % week
    notifications.send_email_text(text, subject=subject, html=html)

    # Telegram
    try:
        import telegram_bot
        telegram_bot.send_message(text)
        logger.info("Weekly report sent to Telegram")
    except Exception as e:
        logger.debug("Telegram weekly report failed: %s", e)

    logger.info("Weekly report sent for week ending %s", week)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly Portfolio Report")
    parser.add_argument("--send", action="store_true", help="Email the HTML report")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output raw dict as JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for lib in ("urllib3", "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    report = generate_weekly_report()

    if args.as_json:
        print(json.dumps(report, indent=2, default=str))
        return

    text_report = format_report(report)
    print(text_report)

    # Save plain-text copy
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / ("weekly_report_%s.txt" % report["week_ending"])
    with open(report_path, "w") as f:
        f.write(text_report)
    print("\n  Saved to: %s" % report_path)

    if args.send:
        send_weekly_report(report)
        print("  Report emailed.")


if __name__ == "__main__":
    main()
