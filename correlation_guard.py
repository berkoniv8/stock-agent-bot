#!/usr/bin/env python3
"""
Correlation Guard — prevents over-concentration in correlated positions.

Checks two dimensions before allowing a new entry:
1. Sector concentration: limits how many open positions share the same sector.
2. Price correlation: computes rolling correlation of returns between the
   candidate ticker and each existing position, blocking entry when the
   average pairwise correlation is too high.

Environment variables:
    MAX_SECTOR_POSITIONS   Max positions per sector (default 2)
    MAX_CORRELATION        Block if avg pairwise correlation > this (default 0.75)
    CORRELATION_LOOKBACK   Trading days to compute correlation (default 60)

Usage:
    from correlation_guard import check_correlation_safe
    safe, info = check_correlation_safe("MSFT", "Technology", open_positions, data_cache)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MAX_SECTOR_POSITIONS = int(os.getenv("MAX_SECTOR_POSITIONS", "2"))
MAX_CORRELATION = float(os.getenv("MAX_CORRELATION", "0.75"))
CORRELATION_LOOKBACK = int(os.getenv("CORRELATION_LOOKBACK", "60"))

# Known sector groupings — tickers in the same group are treated as same-sector
# even if the watchlist sector labels differ slightly.
SECTOR_ALIASES = {
    "Financial Services": "Financials",
    "Communication Services": "Communication",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
}


def normalize_sector(sector):
    # type: (str) -> str
    """Normalize sector name to canonical form."""
    if not sector:
        return "Unknown"
    return SECTOR_ALIASES.get(sector, sector)


def check_sector_concentration(
    candidate_sector,   # type: str
    open_positions,     # type: List[Dict]
    max_per_sector=None,  # type: Optional[int]
):
    # type: (...) -> Tuple[bool, Dict]
    """Check if adding a position in this sector would exceed the limit.

    Args:
        candidate_sector: Sector of the candidate ticker.
        open_positions: List of position dicts with at least 'ticker' and 'sector' keys.
        max_per_sector: Override for MAX_SECTOR_POSITIONS.

    Returns:
        (is_safe, info_dict)
    """
    limit = max_per_sector if max_per_sector is not None else MAX_SECTOR_POSITIONS
    norm_sector = normalize_sector(candidate_sector)

    same_sector = []
    for pos in open_positions:
        pos_sector = normalize_sector(pos.get("sector", "Unknown"))
        if pos_sector == norm_sector:
            same_sector.append(pos["ticker"])

    count = len(same_sector)
    is_safe = count < limit

    info = {
        "sector": norm_sector,
        "current_count": count,
        "limit": limit,
        "same_sector_tickers": same_sector,
        "reason": "",
    }

    if not is_safe:
        info["reason"] = (
            "Sector concentration limit reached: %d/%d positions in %s (%s)"
            % (count, limit, norm_sector, ", ".join(same_sector))
        )
        logger.warning("CORRELATION GUARD — %s", info["reason"])

    return is_safe, info


def compute_pairwise_correlation(
    returns_a,  # type: pd.Series
    returns_b,  # type: pd.Series
):
    # type: (...) -> float
    """Compute Pearson correlation between two return series.

    Returns 0.0 if insufficient overlapping data.
    """
    if returns_a is None or returns_b is None:
        return 0.0
    if len(returns_a) < 10 or len(returns_b) < 10:
        return 0.0

    # Align on common dates
    combined = pd.concat([returns_a, returns_b], axis=1, join="inner")
    if len(combined) < 10:
        return 0.0

    corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
    if np.isnan(corr):
        return 0.0
    return float(corr)


def get_returns_from_data(df, lookback=None):
    # type: (pd.DataFrame, Optional[int]) -> Optional[pd.Series]
    """Extract daily returns from OHLCV DataFrame.

    Args:
        df: DataFrame with 'Close' column.
        lookback: Number of recent trading days to use.

    Returns:
        Series of daily percentage returns, or None.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None

    lookback = lookback or CORRELATION_LOOKBACK
    close = df["Close"].tail(lookback + 1)
    if len(close) < 10:
        return None

    returns = close.pct_change().dropna()
    return returns


def check_price_correlation(
    candidate_ticker,   # type: str
    candidate_data,     # type: Optional[pd.DataFrame]
    open_positions,     # type: List[Dict]
    data_cache,         # type: Dict[str, pd.DataFrame]
    max_correlation=None,  # type: Optional[float]
    lookback=None,         # type: Optional[int]
):
    # type: (...) -> Tuple[bool, Dict]
    """Check if the candidate is too correlated with existing positions.

    Args:
        candidate_ticker: Ticker symbol to evaluate.
        candidate_data: OHLCV DataFrame for the candidate.
        open_positions: List of position dicts with 'ticker' key.
        data_cache: Dict mapping ticker -> OHLCV DataFrame for existing positions.
        max_correlation: Override for MAX_CORRELATION threshold.
        lookback: Override for CORRELATION_LOOKBACK.

    Returns:
        (is_safe, info_dict)
    """
    threshold = max_correlation if max_correlation is not None else MAX_CORRELATION
    lb = lookback or CORRELATION_LOOKBACK

    if not open_positions:
        return True, {
            "avg_correlation": 0.0,
            "max_pairwise": 0.0,
            "threshold": threshold,
            "correlations": {},
            "reason": "",
        }

    candidate_returns = get_returns_from_data(candidate_data, lb)
    if candidate_returns is None:
        # Can't compute correlation — allow entry but warn
        return True, {
            "avg_correlation": 0.0,
            "max_pairwise": 0.0,
            "threshold": threshold,
            "correlations": {},
            "reason": "Insufficient data for correlation check",
        }

    correlations = {}
    for pos in open_positions:
        pos_ticker = pos["ticker"]
        if pos_ticker == candidate_ticker:
            continue
        pos_data = data_cache.get(pos_ticker)
        pos_returns = get_returns_from_data(pos_data, lb)
        corr = compute_pairwise_correlation(candidate_returns, pos_returns)
        correlations[pos_ticker] = round(corr, 3)

    if not correlations:
        return True, {
            "avg_correlation": 0.0,
            "max_pairwise": 0.0,
            "threshold": threshold,
            "correlations": correlations,
            "reason": "",
        }

    avg_corr = sum(correlations.values()) / len(correlations)
    max_corr = max(correlations.values())
    is_safe = avg_corr <= threshold

    info = {
        "avg_correlation": round(avg_corr, 3),
        "max_pairwise": round(max_corr, 3),
        "threshold": threshold,
        "correlations": correlations,
        "reason": "",
    }

    if not is_safe:
        most_corr = max(correlations, key=correlations.get)
        info["reason"] = (
            "High correlation: avg=%.2f (threshold=%.2f), "
            "most correlated with %s (%.2f)"
            % (avg_corr, threshold, most_corr, correlations[most_corr])
        )
        logger.warning("CORRELATION GUARD — %s blocked: %s", candidate_ticker, info["reason"])

    return is_safe, info


def check_correlation_safe(
    candidate_ticker,   # type: str
    candidate_sector,   # type: str
    open_positions,     # type: List[Dict]
    data_cache=None,    # type: Optional[Dict[str, pd.DataFrame]]
    candidate_data=None,  # type: Optional[pd.DataFrame]
):
    # type: (...) -> Tuple[bool, Dict]
    """Combined correlation + sector concentration check.

    This is the main entry point. Checks sector concentration first
    (cheap), then price correlation (requires data).

    Args:
        candidate_ticker: Ticker to evaluate.
        candidate_sector: Sector of the candidate.
        open_positions: List of position dicts with 'ticker' and 'sector'.
        data_cache: Dict mapping ticker -> OHLCV DataFrame. If None, skips
                    price correlation check.
        candidate_data: OHLCV DataFrame for the candidate.

    Returns:
        (is_safe, info_dict) where info_dict contains details from both checks.
    """
    # 1. Sector concentration (fast, no data needed)
    sector_safe, sector_info = check_sector_concentration(
        candidate_sector, open_positions
    )
    if not sector_safe:
        return False, {
            "check": "sector_concentration",
            "sector": sector_info,
            "correlation": None,
        }

    # 2. Price correlation (requires price data)
    if data_cache is not None:
        corr_safe, corr_info = check_price_correlation(
            candidate_ticker, candidate_data, open_positions, data_cache
        )
        if not corr_safe:
            return False, {
                "check": "price_correlation",
                "sector": sector_info,
                "correlation": corr_info,
            }
    else:
        corr_info = {"avg_correlation": 0.0, "reason": "Skipped (no data cache)"}

    return True, {
        "check": "passed",
        "sector": sector_info,
        "correlation": corr_info,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Show correlation matrix for current paper positions."""
    import argparse

    parser = argparse.ArgumentParser(description="Correlation Guard")
    parser.add_argument("--ticker", type=str, help="Check a specific ticker against open positions")
    parser.add_argument("--sector", type=str, default="Technology", help="Sector of the ticker")
    parser.add_argument("--matrix", action="store_true", help="Show full correlation matrix")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    # Load paper positions
    try:
        import paper_trader
        state = paper_trader.load_state()
        positions = state.get("open_positions", [])
    except Exception:
        positions = []

    if not positions:
        print("\n  No open positions to check correlations against.\n")
        return

    # Fetch data for all tickers
    import data_layer
    tickers = [p["ticker"] for p in positions]
    data_cache = {}
    for t in tickers:
        try:
            data_cache[t] = data_layer.fetch_daily_ohlcv(t, period="6mo")
        except Exception as e:
            logger.error("Failed to fetch %s: %s", t, e)

    if args.ticker:
        # Check a specific candidate
        try:
            cand_data = data_layer.fetch_daily_ohlcv(args.ticker, period="6mo")
        except Exception:
            cand_data = None

        safe, info = check_correlation_safe(
            args.ticker, args.sector, positions, data_cache, cand_data
        )
        status = "SAFE" if safe else "BLOCKED"
        print("\n  %s: %s" % (args.ticker, status))
        if info.get("correlation") and isinstance(info["correlation"], dict):
            corr_data = info["correlation"]
            if corr_data.get("correlations"):
                print("  Correlations:")
                for t, c in sorted(corr_data["correlations"].items(), key=lambda x: -x[1]):
                    bar = "#" * int(abs(c) * 20)
                    print("    %s: %+.3f  %s" % (t, c, bar))
                print("  Avg: %.3f  Max: %.3f  Threshold: %.2f"
                      % (corr_data["avg_correlation"], corr_data["max_pairwise"],
                         corr_data["threshold"]))
        if info.get("sector") and isinstance(info["sector"], dict):
            si = info["sector"]
            print("  Sector: %s (%d/%d)" % (si["sector"], si["current_count"], si["limit"]))
        print()
        return

    if args.matrix:
        # Full correlation matrix
        if len(tickers) < 2:
            print("\n  Need at least 2 positions for correlation matrix.\n")
            return

        returns_map = {}
        for t in tickers:
            r = get_returns_from_data(data_cache.get(t))
            if r is not None:
                returns_map[t] = r

        valid_tickers = list(returns_map.keys())
        n = len(valid_tickers)
        if n < 2:
            print("\n  Insufficient data for correlation matrix.\n")
            return

        # Build matrix
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = compute_pairwise_correlation(
                        returns_map[valid_tickers[i]], returns_map[valid_tickers[j]]
                    )

        # Print
        max_label = max(len(t) for t in valid_tickers)
        header = " " * (max_label + 2) + "  ".join("%6s" % t[:6] for t in valid_tickers)
        print("\n  CORRELATION MATRIX")
        print("  " + "=" * len(header))
        print("  " + header)
        for i, t in enumerate(valid_tickers):
            row = t.ljust(max_label) + "  "
            for j in range(n):
                val = matrix[i][j]
                row += " %+.2f " % val
            print("  " + row)
        print("  " + "=" * len(header))

        # Warn about high correlations
        for i in range(n):
            for j in range(i + 1, n):
                if abs(matrix[i][j]) > MAX_CORRELATION:
                    print("  WARNING: %s <-> %s correlation %.2f exceeds threshold %.2f"
                          % (valid_tickers[i], valid_tickers[j], matrix[i][j], MAX_CORRELATION))
        print()


if __name__ == "__main__":
    main()
