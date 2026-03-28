"""
Data Layer — fetches price/OHLCV, fundamental, and news data for watchlist tickers.
"""

import os
import csv
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Watchlist & Portfolio helpers
# ---------------------------------------------------------------------------

def load_watchlist(path: str = "watchlist.csv") -> List[Dict]:
    """Return list of dicts with keys: ticker, sector, notes."""
    if os.getenv("USE_SQLITE", "0") == "1":
        try:
            import database as db
            wl = db.get_watchlist()
            if wl:
                return wl
        except Exception:
            pass
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_portfolio(path: str = "portfolio.json") -> dict:
    """Return portfolio config dict.

    Prefers the JSON file if it has IB-synced holdings (real portfolio).
    Falls back to SQLite only if the JSON file has no holdings.
    """
    # Always try the JSON file first — it has real IB-synced data
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("holdings"):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback to SQLite if JSON has no holdings
    if os.getenv("USE_SQLITE", "0") == "1":
        try:
            import database as db
            config = db.get_portfolio_config()
            holdings = db.get_holdings()
            return {
                "total_portfolio_value": config.get("total_portfolio_value", 0),
                "available_cash": config.get("available_cash", 0),
                "max_risk_per_trade_pct": config.get("max_risk_per_trade_pct", 1.0),
                "max_position_size_pct": config.get("max_position_size_pct", 10.0),
                "holdings": holdings,
            }
        except Exception:
            pass

    # Last resort: return empty portfolio
    return {
        "total_portfolio_value": 0,
        "available_cash": 0,
        "max_risk_per_trade_pct": 1.0,
        "max_position_size_pct": 10.0,
        "holdings": [],
    }


# ---------------------------------------------------------------------------
# Price / OHLCV data  (primary: yfinance, fallback: Alpha Vantage)
# ---------------------------------------------------------------------------

def fetch_daily_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV candles via yfinance.

    Returns DataFrame with columns: Open, High, Low, Close, Volume
    indexed by Date.
    """
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval="1d")
        if df.empty:
            logger.warning("yfinance returned empty data for %s, trying Alpha Vantage", ticker)
            return _fetch_daily_alpha_vantage(ticker)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logger.error("yfinance error for %s: %s", ticker, e)
        return _fetch_daily_alpha_vantage(ticker)


def _fetch_daily_alpha_vantage(ticker: str) -> pd.DataFrame:
    """Fallback: fetch daily OHLCV from Alpha Vantage."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        logger.error("Alpha Vantage API key not configured")
        return pd.DataFrame()

    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={ticker}"
        f"&outputsize=full&apikey={api_key}"
    )
    resp = requests.get(url, timeout=30)
    data = resp.json()
    ts = data.get("Time Series (Daily)", {})
    if not ts:
        logger.error("Alpha Vantage returned no data for %s", ticker)
        return pd.DataFrame()

    rows = []
    for date_str, vals in ts.items():
        rows.append({
            "Date": pd.Timestamp(date_str),
            "Open": float(vals["1. open"]),
            "High": float(vals["2. high"]),
            "Low": float(vals["3. low"]),
            "Close": float(vals["4. close"]),
            "Volume": int(vals["5. volume"]),
        })
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df


def fetch_intraday_ohlcv(ticker: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    """Fetch intraday (1h) candles for confirmation signals."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval)
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logger.error("Intraday fetch error for %s: %s", ticker, e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Fundamental data  (Financial Modeling Prep)
# ---------------------------------------------------------------------------

def fetch_fundamentals_fmp(ticker: str) -> dict:
    """Fetch key fundamental metrics from Financial Modeling Prep.

    Returns dict with keys: pe_ratio, eps, eps_growth_yoy,
    revenue_growth_qoq, debt_to_equity, analyst_consensus.
    """
    api_key = os.getenv("FMP_API_KEY", "")
    base = "https://financialmodelingprep.com/api/v3"
    result = {
        "pe_ratio": None,
        "eps": None,
        "eps_growth_yoy": None,
        "revenue_growth_qoq": None,
        "debt_to_equity": None,
        "analyst_consensus": None,
    }

    if not api_key or api_key.startswith("your_"):
        logger.warning("FMP API key not configured — falling back to yfinance fundamentals")
        return _fetch_fundamentals_yfinance(ticker)

    try:
        # Key metrics
        url = f"{base}/key-metrics-ttm/{ticker}?apikey={api_key}"
        resp = requests.get(url, timeout=15)
        metrics = resp.json()
        if metrics and isinstance(metrics, list):
            m = metrics[0]
            result["pe_ratio"] = m.get("peRatioTTM")
            result["debt_to_equity"] = m.get("debtToEquityTTM")

        # Income statement for EPS / revenue growth
        url = f"{base}/income-statement/{ticker}?period=quarter&limit=5&apikey={api_key}"
        resp = requests.get(url, timeout=15)
        stmts = resp.json()
        if stmts and len(stmts) >= 2:
            result["eps"] = stmts[0].get("eps")
            eps_now = stmts[0].get("eps", 0) or 0
            eps_prev = stmts[4].get("eps", 0) if len(stmts) >= 5 else None
            if eps_prev and eps_prev != 0:
                result["eps_growth_yoy"] = (eps_now - eps_prev) / abs(eps_prev) * 100

            rev_now = stmts[0].get("revenue", 0) or 0
            rev_prev = stmts[1].get("revenue", 0) or 0
            if rev_prev and rev_prev != 0:
                result["revenue_growth_qoq"] = (rev_now - rev_prev) / abs(rev_prev) * 100

        # Analyst consensus
        url = f"{base}/analyst-estimates/{ticker}?limit=1&apikey={api_key}"
        resp = requests.get(url, timeout=15)
        est = resp.json()
        if est and isinstance(est, list) and est:
            result["analyst_consensus"] = est[0].get("estimatedEpsAvg")

    except Exception as e:
        logger.error("FMP fundamentals error for %s: %s", ticker, e)
        return _fetch_fundamentals_yfinance(ticker)

    return result


def _fetch_fundamentals_yfinance(ticker: str) -> dict:
    """Fallback: pull fundamentals from yfinance."""
    result = {
        "pe_ratio": None,
        "eps": None,
        "eps_growth_yoy": None,
        "revenue_growth_qoq": None,
        "debt_to_equity": None,
        "analyst_consensus": None,
    }
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        result["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
        result["eps"] = info.get("trailingEps")
        result["debt_to_equity"] = info.get("debtToEquity")
        if result["debt_to_equity"] is not None:
            # yfinance reports D/E as percentage (e.g. 102.63 = 1.0263)
            result["debt_to_equity"] = result["debt_to_equity"] / 100

        # EPS growth — try earnings_history, fall back to earnings_dates
        try:
            earnings = getattr(tk, "earnings_history", None)
            if earnings is not None and hasattr(earnings, "empty") and not earnings.empty and len(earnings) >= 5:
                eps_now = earnings.iloc[-1].get("epsActual", 0) or 0
                eps_prev = earnings.iloc[-5].get("epsActual", 0) or 0
                if eps_prev != 0:
                    result["eps_growth_yoy"] = (eps_now - eps_prev) / abs(eps_prev) * 100
        except Exception:
            pass

        # If EPS growth still not set, try computing from trailing vs forward EPS
        if result["eps_growth_yoy"] is None:
            trailing = info.get("trailingEps")
            forward = info.get("forwardEps")
            if trailing and forward and trailing != 0:
                result["eps_growth_yoy"] = (forward - trailing) / abs(trailing) * 100

        # Revenue growth from quarterly financials
        try:
            qf = tk.quarterly_income_stmt
            if qf is not None and not qf.empty:
                # yfinance may use "Total Revenue" or "TotalRevenue"
                rev_row = None
                for label in ("Total Revenue", "TotalRevenue", "Revenue"):
                    if label in qf.index:
                        rev_row = qf.loc[label]
                        break
                if rev_row is not None and len(rev_row) >= 2:
                    rev_now = float(rev_row.iloc[0]) if pd.notna(rev_row.iloc[0]) else 0
                    rev_prev = float(rev_row.iloc[1]) if pd.notna(rev_row.iloc[1]) else 0
                    if rev_prev and rev_prev != 0:
                        result["revenue_growth_qoq"] = (rev_now - rev_prev) / abs(rev_prev) * 100
        except Exception:
            pass

        # Analyst recommendation
        rec = info.get("recommendationKey", "") or ""
        result["analyst_consensus"] = rec  # e.g. "buy", "strong_buy", "hold"

    except Exception as e:
        logger.error("yfinance fundamentals error for %s: %s", ticker, e)

    return result


def fetch_fundamentals(ticker: str) -> dict:
    """Unified fundamental data fetcher — tries FMP first, falls back to yfinance."""
    return fetch_fundamentals_fmp(ticker)


# ---------------------------------------------------------------------------
# News sentiment  (NewsAPI primary, Finnhub fallback)
# ---------------------------------------------------------------------------

def fetch_news_newsapi(ticker: str, days: int = 7) -> List[Dict]:
    """Fetch recent news headlines from NewsAPI.

    Returns list of dicts with keys: title, description, publishedAt, source.
    """
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key or api_key.startswith("your_"):
        logger.warning("NewsAPI key not configured — trying Finnhub")
        return fetch_news_finnhub(ticker, days)

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={ticker}&from={from_date}&sortBy=publishedAt"
        f"&language=en&pageSize=20&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "publishedAt": a.get("publishedAt", ""),
                "source": a.get("source", {}).get("name", ""),
            }
            for a in articles
        ]
    except Exception as e:
        logger.error("NewsAPI error for %s: %s", ticker, e)
        return fetch_news_finnhub(ticker, days)


def fetch_news_finnhub(ticker: str, days: int = 7) -> List[Dict]:
    """Fallback: fetch news from Finnhub."""
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        logger.warning("Finnhub API key not configured")
        return []

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")
    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={from_date}&to={to_date}"
        f"&token={api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        articles = resp.json()
        if not isinstance(articles, list):
            return []
        return [
            {
                "title": a.get("headline", ""),
                "description": a.get("summary", ""),
                "publishedAt": datetime.fromtimestamp(a.get("datetime", 0)).isoformat(),
                "source": a.get("source", ""),
            }
            for a in articles[:20]
        ]
    except Exception as e:
        logger.error("Finnhub error for %s: %s", ticker, e)
        return []


def fetch_news(ticker: str, days: int = 7) -> List[Dict]:
    """Unified news fetcher."""
    return fetch_news_newsapi(ticker, days)
