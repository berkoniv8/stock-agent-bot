#!/usr/bin/env python3
"""
Tax Loss Harvesting Analyzer — scans portfolio.json for unrealized losses that
can be harvested to offset capital gains, and provides actionable recommendations.

Only "trade" strategy positions are recommended for harvesting (not long_term).
Tax rates are configurable via environment variables:
    TAX_RATE_SHORT=0.30   (default 30%)
    TAX_RATE_LONG=0.15    (default 15%)

Usage:
    python3 tax_harvesting.py         # Print report
    python3 tax_harvesting.py --json  # Output as JSON
    python3 tax_harvesting.py --send  # Send via email
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_harvesting(portfolio: dict) -> dict:
    """Analyze a portfolio dict for tax loss harvesting opportunities.

    Only positions with strategy == "trade" are recommended for harvesting.
    Long-term positions are reported in the gains/losses lists but flagged
    as ineligible for harvesting.

    Returns:
        {
            "losses": [
                {
                    "ticker": "BMNR",
                    "unrealized_pnl": -2305.84,
                    "pnl_pct": -29.8,
                    "cost_basis": 7749.28,
                    "current_value": 5443.44,
                    "shares": 296,
                    "strategy": "trade",
                    "harvestable": True,
                }
            ],
            "gains": [
                {
                    "ticker": "QQQ",
                    "unrealized_pnl": 15955.09,
                    "pnl_pct": 51.9,
                    "cost_basis": 30739.05,
                    "current_value": 46694.14,
                    "shares": 83,
                    "strategy": "long_term",
                    "harvestable": False,
                }
            ],
            "total_harvestable_loss": -4800.0,
            "total_gains": 21125.0,
            "net_tax_exposure": 16325.0,
            "potential_tax_savings": 1440.0,
            "tax_rate_short": 0.30,
            "tax_rate_long": 0.15,
            "recommendations": [
                "Sell BMNR to harvest $2,306 loss, offsetting gains ..."
            ],
        }
    """
    tax_rate_short = float(os.getenv("TAX_RATE_SHORT", "0.30"))
    tax_rate_long = float(os.getenv("TAX_RATE_LONG", "0.15"))

    holdings = portfolio.get("holdings", [])

    losses: List[dict] = []
    gains: List[dict] = []

    for h in holdings:
        pnl = h.get("unrealized_pnl", 0.0) or 0.0
        strategy = h.get("strategy", "")
        harvestable = strategy == "trade" and pnl < 0

        entry = {
            "ticker": h["ticker"],
            "unrealized_pnl": round(pnl, 2),
            "pnl_pct": round(h.get("pnl_pct", 0.0) or 0.0, 2),
            "cost_basis": round(h.get("cost_basis", 0.0) or 0.0, 2),
            "current_value": round(h.get("current_value", 0.0) or 0.0, 2),
            "shares": h.get("shares", 0),
            "strategy": strategy,
            "sector": h.get("sector", ""),
            "harvestable": harvestable,
        }

        if pnl < 0:
            losses.append(entry)
        elif pnl > 0:
            gains.append(entry)

    # Sort losses: most negative first (biggest harvesting opportunity)
    losses.sort(key=lambda x: x["unrealized_pnl"])
    # Sort gains: largest first
    gains.sort(key=lambda x: x["unrealized_pnl"], reverse=True)

    total_harvestable_loss = sum(
        item["unrealized_pnl"] for item in losses if item["harvestable"]
    )
    total_gains = sum(item["unrealized_pnl"] for item in gains)

    net_tax_exposure = total_gains + total_harvestable_loss  # loss is negative
    # Savings = tax on the harvested losses (short-term rate applied to losses)
    potential_tax_savings = abs(total_harvestable_loss) * tax_rate_short

    # Build recommendations
    recommendations: List[str] = []

    harvestable_items = [item for item in losses if item["harvestable"]]
    if not harvestable_items:
        recommendations.append(
            "No trade-strategy positions with unrealized losses to harvest."
        )
    else:
        gain_tickers = [g["ticker"] for g in gains[:3]]
        gain_desc = (
            "against gains in %s" % ", ".join(gain_tickers)
            if gain_tickers
            else "to reduce taxable income"
        )
        for item in harvestable_items:
            loss_amt = abs(item["unrealized_pnl"])
            tax_saved = loss_amt * tax_rate_short
            recommendations.append(
                "Sell %s (%d shares) to harvest $%s loss %s "
                "(saves ~$%s in taxes at %.0f%% rate)"
                % (
                    item["ticker"],
                    item["shares"],
                    _fmt_dollar(loss_amt),
                    gain_desc,
                    _fmt_dollar(tax_saved),
                    tax_rate_short * 100,
                )
            )

    if total_gains > 0 and harvestable_items:
        recommendations.append(
            "Total harvestable losses ($%s) can offset $%s of gains, "
            "reducing net taxable gain to $%s"
            % (
                _fmt_dollar(abs(total_harvestable_loss)),
                _fmt_dollar(total_gains),
                _fmt_dollar(max(0.0, net_tax_exposure)),
            )
        )

    # Note long-term losses that are NOT harvested
    long_term_losses = [item for item in losses if not item["harvestable"]]
    if long_term_losses:
        tickers_str = ", ".join(item["ticker"] for item in long_term_losses)
        recommendations.append(
            "Note: %s have unrealized losses but are flagged long_term — "
            "review before harvesting to avoid disrupting long-term positions."
            % tickers_str
        )

    return {
        "losses": losses,
        "gains": gains,
        "total_harvestable_loss": round(total_harvestable_loss, 2),
        "total_gains": round(total_gains, 2),
        "net_tax_exposure": round(net_tax_exposure, 2),
        "potential_tax_savings": round(potential_tax_savings, 2),
        "tax_rate_short": tax_rate_short,
        "tax_rate_long": tax_rate_long,
        "recommendations": recommendations,
    }


def _fmt_dollar(amount: float) -> str:
    """Format a dollar amount with comma separators, no decimal for whole numbers."""
    if amount == int(amount):
        return "{:,.0f}".format(amount)
    return "{:,.2f}".format(amount)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(analysis: dict) -> str:
    """Return a human-readable tax harvesting report."""
    lines = [
        "=" * 62,
        "  TAX LOSS HARVESTING ANALYSIS",
        "=" * 62,
    ]

    tax_short = analysis.get("tax_rate_short", 0.30)
    tax_long = analysis.get("tax_rate_long", 0.15)
    lines.append(
        "  Tax rates:  Short-term %.0f%%  |  Long-term %.0f%%"
        % (tax_short * 100, tax_long * 100)
    )
    lines.append("")

    # ---- Losses section ----
    losses = analysis.get("losses", [])
    if losses:
        lines.append("  UNREALIZED LOSSES")
        lines.append("  " + "─" * 58)
        lines.append(
            "  %-6s  %-10s  %9s  %8s  %9s  %s"
            % ("Ticker", "Strategy", "P&L", "P&L %", "Cost", "Harvest?")
        )
        for item in losses:
            harvest_flag = "YES" if item["harvestable"] else "no (long)"
            lines.append(
                "  %-6s  %-10s  %9s  %7.1f%%  %9s  %s"
                % (
                    item["ticker"],
                    item["strategy"],
                    "-$" + _fmt_dollar(abs(item["unrealized_pnl"])),
                    item["pnl_pct"],
                    "$" + _fmt_dollar(item["cost_basis"]),
                    harvest_flag,
                )
            )
        lines.append("")

    # ---- Gains section ----
    gains = analysis.get("gains", [])
    if gains:
        lines.append("  UNREALIZED GAINS")
        lines.append("  " + "─" * 58)
        lines.append(
            "  %-6s  %-10s  %9s  %8s  %9s"
            % ("Ticker", "Strategy", "P&L", "P&L %", "Cost")
        )
        for item in gains:
            lines.append(
                "  %-6s  %-10s  %9s  %7.1f%%  %9s"
                % (
                    item["ticker"],
                    item["strategy"],
                    "+$" + _fmt_dollar(item["unrealized_pnl"]),
                    item["pnl_pct"],
                    "$" + _fmt_dollar(item["cost_basis"]),
                )
            )
        lines.append("")

    # ---- Summary ----
    lines.append("  SUMMARY")
    lines.append("  " + "─" * 58)
    lines.append(
        "  Total harvestable losses:  -$%s"
        % _fmt_dollar(abs(analysis.get("total_harvestable_loss", 0.0)))
    )
    lines.append(
        "  Total unrealized gains:    +$%s"
        % _fmt_dollar(analysis.get("total_gains", 0.0))
    )
    net = analysis.get("net_tax_exposure", 0.0)
    net_sign = "+" if net >= 0 else "-"
    lines.append(
        "  Net tax exposure:           %s$%s"
        % (net_sign, _fmt_dollar(abs(net)))
    )
    lines.append(
        "  Potential tax savings:      $%s  (at %.0f%% short-term rate)"
        % (_fmt_dollar(analysis.get("potential_tax_savings", 0.0)), tax_short * 100)
    )
    lines.append("")

    # ---- Recommendations ----
    recs = analysis.get("recommendations", [])
    if recs:
        lines.append("  RECOMMENDATIONS")
        lines.append("  " + "─" * 58)
        for i, rec in enumerate(recs, 1):
            # Word-wrap at 56 chars
            words = rec.split()
            current_line = "  %d. " % i
            for word in words:
                if len(current_line) + len(word) + 1 > 60:
                    lines.append(current_line)
                    current_line = "     " + word
                else:
                    current_line += (" " if current_line.strip() else "") + word
            if current_line.strip():
                lines.append(current_line)
        lines.append("")

    lines.append("=" * 62)
    lines.append(
        "  Generated: %s" % datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    lines.append("=" * 62)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------

def send_report(analysis: dict) -> None:
    """Email the tax harvesting report via notifications.send_email_text."""
    import notifications

    report = format_report(analysis)
    savings = analysis.get("potential_tax_savings", 0.0)
    subject = "Tax Harvesting Analysis — potential savings $%s" % _fmt_dollar(savings)
    notifications.send_email_text(report, subject=subject)
    logger.info("Tax harvesting report emailed")


# ---------------------------------------------------------------------------
# Portfolio loader
# ---------------------------------------------------------------------------

def load_portfolio() -> dict:
    """Load and return the portfolio.json dict."""
    if not PORTFOLIO_FILE.exists():
        logger.error("portfolio.json not found at %s", PORTFOLIO_FILE)
        return {}
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tax loss harvesting analyzer for portfolio.json"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output analysis as JSON",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send report via email (requires SMTP config in .env)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    portfolio = load_portfolio()
    if not portfolio:
        print("Error: could not load portfolio.json")
        return

    analysis = analyze_harvesting(portfolio)

    if args.as_json:
        print(json.dumps(analysis, indent=2, default=str))
        return

    print(format_report(analysis))

    if args.send:
        send_report(analysis)
        print("Report sent via email.")


if __name__ == "__main__":
    main()
