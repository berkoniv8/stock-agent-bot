#!/usr/bin/env python3
"""
IB Sync — READ-ONLY connection to Interactive Brokers TWS / IB Gateway.

IMPORTANT: This module never places orders or modifies any brokerage state.
It only reads positions and account summary data and optionally writes the
results to the local portfolio.json (preserving user-defined strategy tags).

Environment variables (configured in .env):
    IB_HOST        TWS/Gateway host (default: 127.0.0.1)
    IB_PORT        TWS/Gateway port (default: 7496 live / 7497 paper)
    IB_CLIENT_ID   Client ID for this connection (default: 10)
    IB_ACCOUNT     Account ID (optional, for multi-account setup)
    IB_PAPER       Set to 1 to default to paper-trading port 7497

Requires: ib_insync  (pip install ib_insync)
Falls back gracefully if the library is not installed or TWS is unreachable.

Usage:
    python3 ib_sync.py             # Full sync — updates portfolio.json
    python3 ib_sync.py --dry-run   # Show what would change, no writes
    python3 ib_sync.py --status    # Check connection only
    python3 ib_sync.py --json      # Output sync result as JSON
"""

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Environment / config helpers
# ---------------------------------------------------------------------------

def _resolve_port() -> int:
    """Return the TWS port: explicit IB_PORT > paper default > live default."""
    explicit = os.environ.get("IB_PORT", "")
    if explicit:
        return int(explicit)
    return 7497 if os.environ.get("IB_PAPER", "0") == "1" else 7496


IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORT = _resolve_port()
IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "10"))
IB_ACCOUNT = os.environ.get("IB_ACCOUNT", "")

PORTFOLIO_JSON = str(Path(__file__).parent / "portfolio.json")

# Fields from portfolio.json that must never be overwritten by IB data.
_PRESERVE_FIELDS = ("strategy", "sector")


# ---------------------------------------------------------------------------
# 1. is_available
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """
    Return True only if ib_insync is importable AND IB (TWS/Gateway) is
    reachable on IB_HOST:IB_PORT.  Never raises.
    """
    try:
        import ib_insync  # noqa: F401
    except ImportError:
        return False

    port = _resolve_port()
    try:
        sock = socket.create_connection((IB_HOST, port), timeout=3)
        sock.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. connect
# ---------------------------------------------------------------------------

def connect() -> Optional[object]:
    """
    Create and return a read-only IB connection, or None on failure.
    Times out after 5 seconds.
    """
    try:
        import ib_insync
    except ImportError:
        print("[ib_sync] ib_insync not installed — cannot connect.", file=sys.stderr)
        return None

    port = _resolve_port()
    account = IB_ACCOUNT
    try:
        ib = ib_insync.IB()
        ib.connect(
            host=IB_HOST,
            port=port,
            clientId=IB_CLIENT_ID,
            readonly=True,
            timeout=10,
        )
        if account:
            try:
                ib.reqAccountUpdates(subscribe=True, acctCode=account)
            except Exception:
                pass  # non-fatal for single-account setups
        return ib
    except Exception as exc:
        print(f"[ib_sync] Connection failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 3. fetch_positions
# ---------------------------------------------------------------------------

def fetch_positions(ib) -> list:
    """
    Retrieve all USD positions from IB and return a list of dicts.

    Each dict contains:
        ticker, shares, avg_cost, market_price, market_value,
        unrealized_pnl, currency
    """
    try:
        import ib_insync
    except ImportError:
        return []

    results = []

    try:
        positions = ib.positions()
    except Exception as exc:
        print(f"[ib_sync] ib.positions() failed: {exc}", file=sys.stderr)
        return []

    for pos in positions:
        try:
            contract = pos.contract
            currency = getattr(contract, "currency", "USD")
            if currency != "USD":
                continue

            ticker_sym = getattr(contract, "symbol", None)
            if not ticker_sym:
                continue

            shares = float(pos.position)
            avg_cost = float(pos.avgCost)
            cost_basis = shares * avg_cost

            # --- attempt live market price via reqMktData ---
            market_price = None
            try:
                ticker_data = ib.reqMktData(contract, "", False, False)
                ib.sleep(1.5)  # allow snapshot to populate
                price = ticker_data.last
                if price is None or price != price:  # NaN check
                    price = ticker_data.close
                if price is not None and price == price and price > 0:
                    market_price = float(price)
            except Exception:
                market_price = None

            # --- fallback: yfinance ---
            if market_price is None:
                try:
                    import yfinance as yf
                    yf_ticker = yf.Ticker(ticker_sym)
                    hist = yf_ticker.history(period="1d")
                    if not hist.empty:
                        market_price = float(hist["Close"].iloc[-1])
                except Exception:
                    market_price = avg_cost  # last resort: use avg cost

            market_value = shares * market_price if market_price else cost_basis
            unrealized_pnl = market_value - cost_basis

            results.append({
                "ticker": ticker_sym,
                "shares": shares,
                "avg_cost": avg_cost,
                "market_price": market_price,
                # market_value is exposed directly as a convenience alias
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "currency": currency,
            })
        except Exception as exc:
            sym = getattr(getattr(pos, "contract", None), "symbol", "?")
            print(f"[ib_sync] Skipping position {sym}: {exc}", file=sys.stderr)
            continue

    return results


# ---------------------------------------------------------------------------
# 4. fetch_account_summary
# ---------------------------------------------------------------------------

def fetch_account_summary(ib) -> dict:
    """
    Return selected account summary values as a dict of floats.

    Keys: NetLiquidation, TotalCashValue, UnrealizedPnL, RealizedPnL,
          GrossPositionValue
    """
    _WANTED = {
        "NetLiquidation",
        "TotalCashValue",
        "UnrealizedPnL",
        "RealizedPnL",
        "GrossPositionValue",
    }

    summary = {}
    acct_filter = IB_ACCOUNT or "All"

    try:
        rows = ib.accountSummary(account=acct_filter)
    except TypeError:
        # Older ib_insync versions don't accept the account keyword
        try:
            rows = ib.accountSummary()
        except Exception as exc:
            print(f"[ib_sync] ib.accountSummary() failed: {exc}", file=sys.stderr)
            return summary
    except Exception as exc:
        print(f"[ib_sync] ib.accountSummary() failed: {exc}", file=sys.stderr)
        return summary

    for row in rows:
        try:
            tag = getattr(row, "tag", None)
            currency = getattr(row, "currency", "USD")
            if tag in _WANTED and currency == "USD":
                value = getattr(row, "value", None)
                if value is not None:
                    summary[tag] = float(value)
        except Exception:
            continue

    return summary


# ---------------------------------------------------------------------------
# 5. sync_portfolio
# ---------------------------------------------------------------------------

def sync_portfolio(dry_run: bool = False) -> dict:
    """
    Full sync: connect -> fetch positions + account summary -> disconnect.

    If not dry_run, updates portfolio.json while preserving "strategy" and
    "sector" tags for every holding.

    Returns a result dict:
        synced (bool), positions (int), net_liq (float),
        cash (float), timestamp (str), error (str or None)
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "synced": False,
        "positions": 0,
        "net_liq": 0.0,
        "cash": 0.0,
        "gross_position_value": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "timestamp": timestamp,
        "error": None,
    }

    # --- connect ---
    ib = connect()
    if ib is None:
        result["error"] = "Could not connect to IB (TWS/Gateway not running or ib_insync not installed)."
        return result

    try:
        # --- fetch data ---
        positions = fetch_positions(ib)
        account = fetch_account_summary(ib)

        result["positions"] = len(positions)
        result["net_liq"] = account.get("NetLiquidation", 0.0)
        result["cash"] = account.get("TotalCashValue", 0.0)
        result["gross_position_value"] = account.get("GrossPositionValue", 0.0)
        result["unrealized_pnl"] = account.get("UnrealizedPnL", 0.0)
        result["realized_pnl"] = account.get("RealizedPnL", 0.0)

        if dry_run:
            print("[ib_sync] DRY RUN — no files will be written.")
            _print_positions(positions, account)
            result["synced"] = True
            return result

        # --- load existing portfolio.json to preserve metadata ---
        existing_holdings = {}
        try:
            with open(PORTFOLIO_JSON, "r") as fh:
                existing_data = json.load(fh)
            for h in existing_data.get("holdings", []):
                ticker = h.get("ticker")
                if ticker:
                    existing_holdings[ticker] = h
        except Exception as exc:
            print(f"[ib_sync] Could not read existing portfolio.json: {exc}", file=sys.stderr)
            existing_data = {}

        # --- build updated holdings list ---
        updated_holdings = []
        for pos in positions:
            ticker = pos["ticker"]
            shares = pos["shares"]
            avg_cost = pos["avg_cost"]
            market_price = pos["market_price"] if pos["market_price"] else avg_cost
            market_value = pos["market_value"]
            cost_basis = shares * avg_cost
            unrealized_pnl = market_value - cost_basis
            pnl_pct = (unrealized_pnl / cost_basis * 100.0) if cost_basis else 0.0

            existing = existing_holdings.get(ticker, {})

            holding = {
                "ticker": ticker,
                "shares": shares,
                "avg_cost": avg_cost,
                "current_price": market_price,
                "current_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pnl": unrealized_pnl,
                "pnl_pct": pnl_pct,
                # Preserved fields — fall back to empty string if not present
                "sector": existing.get("sector", ""),
                "strategy": existing.get("strategy", ""),
            }
            updated_holdings.append(holding)

        # --- compute total portfolio value ---
        total_value = sum(h["current_value"] for h in updated_holdings) + result["cash"]

        # --- write portfolio.json ---
        portfolio_out = dict(existing_data)
        portfolio_out["total_portfolio_value"] = total_value
        portfolio_out["available_cash"] = result["cash"]
        portfolio_out["holdings"] = updated_holdings
        portfolio_out["last_ib_sync"] = timestamp

        try:
            with open(PORTFOLIO_JSON, "w") as fh:
                json.dump(portfolio_out, fh, indent=4)
            print(f"[ib_sync] portfolio.json updated — {len(updated_holdings)} positions.")
        except Exception as exc:
            result["error"] = f"Failed to write portfolio.json: {exc}"
            return result

        result["synced"] = True

    except Exception as exc:
        result["error"] = str(exc)
    finally:
        disconnect(ib)

    return result


# ---------------------------------------------------------------------------
# 6. disconnect
# ---------------------------------------------------------------------------

def disconnect(ib) -> None:
    """Safely disconnect from IB."""
    if ib is None:
        return
    try:
        ib.disconnect()
    except Exception as exc:
        print(f"[ib_sync] disconnect() warning: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_positions(positions: list, account: dict) -> None:
    """Pretty-print positions and account summary to stdout."""
    print(f"\n{'Ticker':<8} {'Shares':>8} {'Avg Cost':>10} {'Mkt Price':>10} {'Mkt Value':>12} {'Unreal PnL':>12}")
    print("-" * 66)
    for p in positions:
        print(
            f"{p['ticker']:<8} "
            f"{p['shares']:>8.2f} "
            f"{p['avg_cost']:>10.2f} "
            f"{(p['market_price'] or 0):>10.2f} "
            f"{p['market_value']:>12.2f} "
            f"{p['unrealized_pnl']:>12.2f}"
        )
    print("-" * 66)

    if account:
        print("\nAccount Summary:")
        for key, val in account.items():
            print(f"  {key}: {val:,.2f}")
    print()


# ---------------------------------------------------------------------------
# 7. CLI entry-point
# ---------------------------------------------------------------------------

def _cmd_status() -> None:
    """Test connectivity and print status."""
    port = _resolve_port()
    paper = os.environ.get("IB_PAPER", "0") == "1"
    mode = "paper" if paper else "live"
    print(f"[ib_sync] Checking IB connection at {IB_HOST}:{port} ({mode} mode) ...")
    if is_available():
        print("[ib_sync] Status: CONNECTED (ib_insync available + TWS/Gateway reachable)")
    else:
        # Distinguish between missing library and unreachable host
        try:
            import ib_insync  # noqa: F401
            lib_ok = True
        except ImportError:
            lib_ok = False

        if not lib_ok:
            print("[ib_sync] Status: UNAVAILABLE — ib_insync is not installed.")
            print("          Install with: pip install ib_insync")
        else:
            print(f"[ib_sync] Status: UNAVAILABLE — cannot reach TWS/IB Gateway at {IB_HOST}:{port}.")
            print("          Make sure Trader Workstation or IB Gateway is running and API is enabled.")


def _cmd_dry_run() -> None:
    """Run sync in dry-run mode (no file writes) and print what would change."""
    print("[ib_sync] Starting dry-run sync ...")
    result = sync_portfolio(dry_run=True)
    print(f"[ib_sync] Result: {json.dumps(result, indent=2, default=str)}")


def _cmd_sync_result(result: dict) -> None:
    """Print the result of a completed full sync."""
    if result.get("synced"):
        print(
            f"[ib_sync] Sync complete — {result['positions']} positions, "
            f"net liq ${result['net_liq']:,.2f}, "
            f"cash ${result['cash']:,.2f}"
        )
        print(
            f"[ib_sync] Gross pos value: ${result.get('gross_position_value', 0):,.2f}  "
            f"Unrealized P&L: ${result.get('unrealized_pnl', 0):+,.2f}  "
            f"Realized P&L: ${result.get('realized_pnl', 0):+,.2f}"
        )
    else:
        print(f"[ib_sync] Sync failed: {result.get('error', 'unknown error')}")
    print(f"[ib_sync] Timestamp: {result['timestamp']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only IB portfolio sync. NEVER places orders."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing portfolio.json",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Test IB connection and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output sync result as JSON",
    )
    args = parser.parse_args()

    if args.status:
        _cmd_status()
    elif args.dry_run:
        _cmd_dry_run()
    else:
        result = sync_portfolio(dry_run=False)
        if args.as_json:
            print(json.dumps(result, indent=2, default=str))
        else:
            _cmd_sync_result(result)


if __name__ == "__main__":
    main()
