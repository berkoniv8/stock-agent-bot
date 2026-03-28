#!/usr/bin/env python3
"""
Portfolio Rebalancer — analyzes current position weights, sector exposure,
and risk concentration, then suggests adjustments to align with targets.

Features:
- Weight analysis: compare actual vs target position sizing
- Sector balance: flag over/under-exposure by sector
- Risk budget: ensure no single position exceeds risk limits
- Trim/add suggestions: concrete share counts to rebalance

Usage:
    python3 rebalancer.py                    # Full rebalance analysis
    python3 rebalancer.py --target equal     # Equal-weight target
    python3 rebalancer.py --target sector    # Sector-weighted target
    python3 rebalancer.py --max-weight 10    # Max position weight %
"""

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "10.0"))
DEFAULT_MAX_SECTOR_PCT = float(os.getenv("MAX_SECTOR_PCT", "30.0"))
REBALANCE_THRESHOLD_PCT = float(os.getenv("REBALANCE_THRESHOLD_PCT", "2.0"))


def get_portfolio_snapshot():
    # type: () -> Tuple[List[Dict], float, float]
    """Get current open positions with market values.

    Returns:
        (positions, total_equity, cash)
        Each position dict has: ticker, sector, direction, shares, entry_price,
        current_price, market_value, weight_pct, unrealized_pnl.
    """
    try:
        import paper_trader
        state = paper_trader.load_state()
    except Exception:
        return [], 0.0, 0.0

    positions = state.get("open_positions", [])
    cash = state.get("cash", 0)

    if not positions:
        starting = state.get("starting_capital", 100000)
        return [], starting, cash

    enriched = []
    for pos in positions:
        # Use highest_price as proxy for current price if available
        current = pos.get("highest_price", pos["entry_price"])
        shares = pos["shares"]
        market_value = shares * current
        entry_value = shares * pos["entry_price"]

        if pos["direction"] == "BUY":
            unrealized = market_value - entry_value
        else:
            unrealized = entry_value - market_value

        enriched.append({
            "ticker": pos["ticker"],
            "sector": pos.get("sector", "Unknown"),
            "direction": pos["direction"],
            "shares": shares,
            "entry_price": pos["entry_price"],
            "current_price": current,
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(unrealized, 2),
            "stop_loss": pos.get("current_stop", pos.get("stop_loss", 0)),
            "risk_per_share": pos.get("risk_per_share", 0),
        })

    total_invested = sum(p["market_value"] for p in enriched)
    total_equity = total_invested + cash

    # Compute weights
    for p in enriched:
        p["weight_pct"] = round(p["market_value"] / total_equity * 100, 2) if total_equity > 0 else 0

    return enriched, total_equity, cash


def analyze_weights(
    positions,          # type: List[Dict]
    total_equity,       # type: float
    max_position_pct=None,  # type: Optional[float]
):
    # type: (...) -> Dict
    """Analyze position weights and flag overweight positions.

    Returns dict with weight analysis details.
    """
    max_pct = max_position_pct or DEFAULT_MAX_POSITION_PCT

    overweight = []
    underweight = []
    n = len(positions)

    if n == 0:
        return {"positions": [], "overweight": [], "equal_target": 0, "max_allowed": max_pct}

    equal_target = min(100.0 / n, max_pct)

    for pos in positions:
        weight = pos["weight_pct"]
        deviation = weight - equal_target

        entry = {
            "ticker": pos["ticker"],
            "weight_pct": weight,
            "target_pct": round(equal_target, 2),
            "deviation_pct": round(deviation, 2),
        }

        if weight > max_pct:
            overweight.append(entry)
        elif weight < equal_target - REBALANCE_THRESHOLD_PCT:
            underweight.append(entry)

    return {
        "positions": [{
            "ticker": p["ticker"],
            "weight_pct": p["weight_pct"],
            "market_value": p["market_value"],
        } for p in positions],
        "overweight": overweight,
        "underweight": underweight,
        "equal_target": round(equal_target, 2),
        "max_allowed": max_pct,
    }


def analyze_sectors(
    positions,          # type: List[Dict]
    total_equity,       # type: float
    max_sector_pct=None,  # type: Optional[float]
):
    # type: (...) -> Dict
    """Analyze sector exposure and flag concentration.

    Returns dict with sector weights and warnings.
    """
    max_pct = max_sector_pct or DEFAULT_MAX_SECTOR_PCT

    sector_totals = {}  # type: Dict[str, float]
    sector_tickers = {}  # type: Dict[str, List[str]]

    for pos in positions:
        sector = pos.get("sector", "Unknown")
        value = pos["market_value"]
        sector_totals[sector] = sector_totals.get(sector, 0) + value
        if sector not in sector_tickers:
            sector_tickers[sector] = []
        sector_tickers[sector].append(pos["ticker"])

    sectors = []
    warnings = []

    for sector, value in sorted(sector_totals.items(), key=lambda x: -x[1]):
        weight = value / total_equity * 100 if total_equity > 0 else 0
        entry = {
            "sector": sector,
            "market_value": round(value, 2),
            "weight_pct": round(weight, 2),
            "positions": len(sector_tickers[sector]),
            "tickers": sector_tickers[sector],
        }
        sectors.append(entry)

        if weight > max_pct:
            warnings.append(
                "%s is %.1f%% of portfolio (limit: %.1f%%) — consider trimming %s"
                % (sector, weight, max_pct, ", ".join(sector_tickers[sector]))
            )

    return {
        "sectors": sectors,
        "warnings": warnings,
        "max_sector_pct": max_pct,
    }


def analyze_risk(positions, total_equity):
    # type: (List[Dict], float) -> Dict
    """Analyze risk exposure — max loss if all stops hit.

    Returns dict with risk metrics.
    """
    total_risk = 0.0
    position_risks = []

    for pos in positions:
        stop = pos.get("stop_loss", 0)
        current = pos["current_price"]
        shares = pos["shares"]

        if pos["direction"] == "BUY":
            risk_per_share = max(0, current - stop)
        else:
            risk_per_share = max(0, stop - current)

        position_risk = risk_per_share * shares
        risk_pct = position_risk / total_equity * 100 if total_equity > 0 else 0

        position_risks.append({
            "ticker": pos["ticker"],
            "risk_amount": round(position_risk, 2),
            "risk_pct": round(risk_pct, 2),
            "stop_loss": stop,
            "current_price": current,
        })
        total_risk += position_risk

    total_risk_pct = total_risk / total_equity * 100 if total_equity > 0 else 0

    return {
        "total_risk": round(total_risk, 2),
        "total_risk_pct": round(total_risk_pct, 2),
        "position_risks": sorted(position_risks, key=lambda x: -x["risk_pct"]),
    }


def generate_suggestions(
    positions,      # type: List[Dict]
    total_equity,   # type: float
    cash,           # type: float
    max_position_pct=None,  # type: Optional[float]
):
    # type: (...) -> List[Dict]
    """Generate concrete rebalancing suggestions.

    Returns list of suggestion dicts with action, ticker, shares, reason.
    """
    max_pct = max_position_pct or DEFAULT_MAX_POSITION_PCT
    suggestions = []

    if not positions:
        return suggestions

    n = len(positions)
    target_value = total_equity * min(100.0 / n, max_pct) / 100

    for pos in positions:
        weight = pos["weight_pct"]
        current_value = pos["market_value"]
        price = pos["current_price"]

        if weight > max_pct + REBALANCE_THRESHOLD_PCT:
            # Overweight — suggest trim
            excess_value = current_value - (total_equity * max_pct / 100)
            trim_shares = int(excess_value / price) if price > 0 else 0
            if trim_shares > 0:
                suggestions.append({
                    "action": "TRIM",
                    "ticker": pos["ticker"],
                    "shares": trim_shares,
                    "current_weight": weight,
                    "target_weight": max_pct,
                    "reason": "Overweight: %.1f%% > %.1f%% limit" % (weight, max_pct),
                })

        elif weight < (target_value / total_equity * 100) - REBALANCE_THRESHOLD_PCT:
            # Underweight — suggest adding if cash available
            deficit_value = target_value - current_value
            add_shares = int(deficit_value / price) if price > 0 else 0
            cost = add_shares * price
            if add_shares > 0 and cost <= cash:
                target_w = target_value / total_equity * 100
                suggestions.append({
                    "action": "ADD",
                    "ticker": pos["ticker"],
                    "shares": add_shares,
                    "current_weight": weight,
                    "target_weight": round(target_w, 2),
                    "estimated_cost": round(cost, 2),
                    "reason": "Underweight: %.1f%% vs %.1f%% target" % (weight, target_w),
                })

    return suggestions


def full_analysis(max_position_pct=None, max_sector_pct=None):
    # type: (Optional[float], Optional[float]) -> Dict
    """Run complete rebalancing analysis.

    Returns dict with all analysis results.
    """
    positions, total_equity, cash = get_portfolio_snapshot()

    return {
        "total_equity": round(total_equity, 2),
        "cash": round(cash, 2),
        "cash_pct": round(cash / total_equity * 100, 2) if total_equity > 0 else 0,
        "num_positions": len(positions),
        "weights": analyze_weights(positions, total_equity, max_position_pct),
        "sectors": analyze_sectors(positions, total_equity, max_sector_pct),
        "risk": analyze_risk(positions, total_equity),
        "suggestions": generate_suggestions(positions, total_equity, cash, max_position_pct),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_report(analysis):
    # type: (Dict) -> None
    """Print formatted rebalancing report."""
    print("\n" + "=" * 60)
    print("  PORTFOLIO REBALANCE ANALYSIS")
    print("=" * 60)

    print("\n  OVERVIEW")
    print("  " + "-" * 50)
    print("  Total Equity:     $%s" % "{:,.2f}".format(analysis["total_equity"]))
    print("  Cash:             $%s (%.1f%%)" % (
        "{:,.2f}".format(analysis["cash"]), analysis["cash_pct"]))
    print("  Positions:        %d" % analysis["num_positions"])

    # Weights
    weights = analysis["weights"]
    if weights["positions"]:
        print("\n  POSITION WEIGHTS")
        print("  " + "-" * 50)
        for p in sorted(weights["positions"], key=lambda x: -x["weight_pct"]):
            bar = "#" * int(p["weight_pct"] / 2)
            print("  %-6s %5.1f%%  $%10s  %s" % (
                p["ticker"], p["weight_pct"],
                "{:,.2f}".format(p["market_value"]), bar))

        if weights["overweight"]:
            print("\n  OVERWEIGHT:")
            for o in weights["overweight"]:
                print("    %s: %.1f%% (target: %.1f%%, +%.1f%%)" % (
                    o["ticker"], o["weight_pct"], o["target_pct"], o["deviation_pct"]))

    # Sectors
    sectors = analysis["sectors"]
    if sectors["sectors"]:
        print("\n  SECTOR EXPOSURE")
        print("  " + "-" * 50)
        for s in sectors["sectors"]:
            bar = "#" * int(s["weight_pct"] / 2)
            print("  %-25s %5.1f%%  (%d pos)  %s" % (
                s["sector"], s["weight_pct"], s["positions"], bar))

        for w in sectors["warnings"]:
            print("\n  WARNING: %s" % w)

    # Risk
    risk = analysis["risk"]
    if risk["position_risks"]:
        print("\n  RISK EXPOSURE (if all stops hit)")
        print("  " + "-" * 50)
        print("  Total at risk:    $%s (%.1f%%)" % (
            "{:,.2f}".format(risk["total_risk"]), risk["total_risk_pct"]))
        for r in risk["position_risks"]:
            print("    %-6s  $%8s  (%4.1f%%)  stop=$%.2f" % (
                r["ticker"],
                "{:,.2f}".format(r["risk_amount"]),
                r["risk_pct"], r["stop_loss"]))

    # Suggestions
    suggestions = analysis["suggestions"]
    if suggestions:
        print("\n  REBALANCING SUGGESTIONS")
        print("  " + "-" * 50)
        for s in suggestions:
            if s["action"] == "TRIM":
                print("  TRIM %-6s  sell %d shares  (%.1f%% -> %.1f%%)" % (
                    s["ticker"], s["shares"], s["current_weight"], s["target_weight"]))
            elif s["action"] == "ADD":
                print("  ADD  %-6s  buy  %d shares  (%.1f%% -> %.1f%%)  cost: $%s" % (
                    s["ticker"], s["shares"], s["current_weight"], s["target_weight"],
                    "{:,.2f}".format(s.get("estimated_cost", 0))))
        print("  %s" % s["reason"])
    else:
        if analysis["num_positions"] > 0:
            print("\n  Portfolio is balanced within thresholds.")

    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Portfolio Rebalancer")
    parser.add_argument("--max-weight", type=float, default=None,
                        help="Max position weight %% (default from env)")
    parser.add_argument("--max-sector", type=float, default=None,
                        help="Max sector weight %% (default from env)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of formatted report")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    analysis = full_analysis(
        max_position_pct=args.max_weight,
        max_sector_pct=args.max_sector,
    )

    if args.json:
        import json
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print_report(analysis)


if __name__ == "__main__":
    main()
