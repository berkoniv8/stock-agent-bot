#!/usr/bin/env python3
"""
IB Flex Reports — web-based trade sync that works WITHOUT TWS/Gateway.

Interactive Brokers' Flex Web Service lets you pull executed trades via HTTP
from anywhere (including GitHub Actions). No local connection required.

Setup (one-time, ~2 minutes in IB Account Management):
    1. Log in to https://www.interactivebrokers.com/portal
    2. Go to: Reports → Flex Queries → Create New Flex Query
       - Query type: Activity
       - Name: "StockAgentBot"
       - Sections to include: Trades (check all sub-fields)
       - Date range: Last 1 business day (or "Today")
       - Format: XML
       - Save & note the Query ID shown on the list
    3. Go to: Settings → Account Settings → Flex Web Service
       - Enable Flex Web Service and generate a Token
       - Copy the token

    4. Add to GitHub Secrets:
       IB_FLEX_TOKEN   = <your flex token>
       IB_FLEX_QUERY_ID = <your query ID>

How it works:
    - Bot calls the Flex API every 15 min during market hours
    - Fetches all trades executed today
    - Compares against portfolio.json (tracks known trades in flex_seen.json)
    - For each NEW trade detected: updates portfolio.json, commits to GitHub,
      sends a Telegram message with investor-quality feedback
"""

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, date
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FLEX_REQUEST_URL  = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
FLEX_DOWNLOAD_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.GetStatement"

SEEN_TRADES_FILE  = Path("logs/flex_seen_trades.json")
PORTFOLIO_FILE    = Path("portfolio.json")


# ---------------------------------------------------------------------------
# Flex API helpers
# ---------------------------------------------------------------------------

def is_configured() -> bool:
    token    = os.getenv("IB_FLEX_TOKEN", "")
    query_id = os.getenv("IB_FLEX_QUERY_ID", "")
    return bool(token and query_id and not token.startswith("your_"))


def _request_flex_report() -> str | None:
    """Step 1 — ask IB to prepare the report; returns a reference code."""
    token    = os.getenv("IB_FLEX_TOKEN", "")
    query_id = os.getenv("IB_FLEX_QUERY_ID", "")
    try:
        resp = requests.get(
            FLEX_REQUEST_URL,
            params={"t": token, "q": query_id, "v": "3"},
            timeout=30,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        status = root.findtext("Status")
        if status == "Success":
            ref = root.findtext("ReferenceCode")
            logger.debug("Flex report requested — ref: %s", ref)
            return ref
        err = root.findtext("ErrorMessage") or resp.text[:200]
        logger.warning("Flex request failed: %s", err)
        return None
    except Exception as e:
        logger.error("Flex SendRequest error: %s", e)
        return None


def _download_flex_report(ref_code: str, retries: int = 8, delay: int = 5) -> str | None:
    """Step 2 — download the prepared report (IB may need a few seconds)."""
    token = os.getenv("IB_FLEX_TOKEN", "")
    for attempt in range(retries):
        try:
            resp = requests.get(
                FLEX_DOWNLOAD_URL,
                params={"t": token, "q": ref_code, "v": "3"},
                timeout=30,
            )
            resp.raise_for_status()
            if "<FlexQueryResponse" in resp.text or "<FlexStatementResponse" in resp.text:
                return resp.text
            # IB returns a status XML if still preparing
            root = ET.fromstring(resp.text)
            err  = root.findtext("ErrorMessage") or ""
            if "1019" in err or "Statement generation" in err:
                logger.debug("Flex report not ready yet (attempt %d/%d)", attempt + 1, retries)
                time.sleep(delay)
                continue
            logger.warning("Flex download unexpected response: %s", err or resp.text[:200])
            return None
        except Exception as e:
            logger.error("Flex GetStatement error (attempt %d): %s", attempt + 1, e)
            time.sleep(delay)
    return None


def fetch_todays_trades() -> list[dict]:
    """
    Fetch all executed trades from IB for today via the Flex Web Service.
    Returns a list of trade dicts with keys:
        ticker, action (BUY/SELL), quantity, price, proceeds, pnl,
        currency, date_time, order_ref, exec_id
    """
    if not is_configured():
        logger.debug("IB Flex not configured — set IB_FLEX_TOKEN and IB_FLEX_QUERY_ID")
        return []

    ref = _request_flex_report()
    if not ref:
        return []

    time.sleep(3)  # Give IB a moment to prepare the report
    xml_data = _download_flex_report(ref)
    if not xml_data:
        return []

    return _parse_flex_trades(xml_data)


def _parse_flex_trades(xml_data: str) -> list[dict]:
    """Parse IB Flex XML and extract trade rows."""
    trades = []
    try:
        root = ET.fromstring(xml_data)
        today = date.today().isoformat()

        for trade in root.iter("Trade"):
            attrib = trade.attrib

            # Only equity / stock trades (skip options, bonds, etc.)
            asset_class = attrib.get("assetCategory", "").upper()
            if asset_class not in ("STK", ""):
                continue

            symbol     = attrib.get("symbol", "").strip()
            action     = attrib.get("buySell", "").strip().upper()   # BUY / SELL
            quantity   = float(attrib.get("quantity", 0) or 0)
            price      = float(attrib.get("tradePrice", 0) or 0)
            proceeds   = float(attrib.get("proceeds", 0) or 0)
            realized   = float(attrib.get("fifoPnlRealized", 0) or 0)
            currency   = attrib.get("currency", "USD")
            trade_date = attrib.get("tradeDate", "")          # YYYYMMDD
            date_time  = attrib.get("dateTime", "")           # YYYYMMDD;HHMMSS
            exec_id    = attrib.get("execID", "") or attrib.get("tradeID", "")
            order_ref  = attrib.get("orderReference", "")

            if not symbol or not action or quantity == 0:
                continue

            # Normalise to YYYY-MM-DD
            if trade_date and len(trade_date) == 8:
                trade_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"

            # Only process today's trades — guard against Flex query returning extra days
            if trade_date and trade_date != today:
                continue

            # Normalise action: handle "BUY", "SELL", "BOT", "SLD"
            if action in ("BOT", "B"):
                action = "BUY"
            elif action in ("SLD", "S"):
                action = "SELL"

            trades.append({
                "ticker":     symbol,
                "action":     action,
                "quantity":   abs(quantity),
                "price":      price,
                "proceeds":   proceeds,
                "realized_pnl": realized,
                "currency":   currency,
                "trade_date": trade_date,
                "date_time":  date_time,
                "exec_id":    exec_id,
                "order_ref":  order_ref,
            })

    except Exception as e:
        logger.error("Flex XML parse error: %s", e)

    logger.info("Flex: parsed %d trades", len(trades))
    return trades


# ---------------------------------------------------------------------------
# Seen-trades tracking (so we don't process the same execution twice)
# ---------------------------------------------------------------------------

def _load_seen() -> set:
    try:
        if SEEN_TRADES_FILE.exists():
            data = json.loads(SEEN_TRADES_FILE.read_text())
            return set(data.get("seen", []))
    except Exception:
        pass
    return set()


def _save_seen(seen: set) -> None:
    try:
        SEEN_TRADES_FILE.parent.mkdir(exist_ok=True)
        # Only keep last 500 IDs to prevent unbounded growth
        trimmed = list(seen)[-500:]
        SEEN_TRADES_FILE.write_text(json.dumps({"seen": trimmed}, indent=2))
    except Exception as e:
        logger.debug("Could not save seen trades: %s", e)


def filter_new_trades(trades: list[dict]) -> list[dict]:
    """Return only trades not yet processed, and update the seen-set."""
    seen = _load_seen()
    new  = []
    for t in trades:
        # Build a unique ID for each execution
        uid = t.get("exec_id") or f"{t['ticker']}-{t['action']}-{t['quantity']}-{t['price']}-{t['date_time']}"
        if uid not in seen:
            t["_uid"] = uid
            new.append(t)
    if new:
        seen.update(t["_uid"] for t in new)
        _save_seen(seen)
    return new


# ---------------------------------------------------------------------------
# Portfolio update
# ---------------------------------------------------------------------------

def _load_portfolio() -> dict:
    try:
        return json.loads(PORTFOLIO_FILE.read_text())
    except Exception as e:
        logger.error("Failed to load portfolio.json: %s", e)
        return {}


def _save_portfolio(portfolio: dict) -> None:
    PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2))


def _commit_portfolio(portfolio: dict) -> bool:
    """Commit updated portfolio.json to the GitHub repo so changes survive restarts."""
    import base64
    token = os.getenv("GITHUB_TOKEN", "")
    repo  = os.getenv("GITHUB_REPOSITORY", "")
    if not token or not repo:
        return False
    try:
        content = json.dumps(portfolio, indent=2)
        encoded = base64.b64encode(content.encode()).decode()
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        url     = f"https://api.github.com/repos/{repo}/contents/portfolio.json"
        resp    = requests.get(url, headers=headers, timeout=15)
        sha     = resp.json().get("sha", "")
        payload = {"message": "bot: auto-sync trade from IB Flex", "content": encoded, "sha": sha, "branch": "main"}
        r       = requests.put(url, json=payload, headers=headers, timeout=15)
        if r.status_code in (200, 201):
            logger.info("Portfolio committed to GitHub after Flex sync")
            return True
        logger.warning("GitHub commit failed (%d)", r.status_code)
        return False
    except Exception as e:
        logger.error("GitHub commit error: %s", e)
        return False


def apply_trade_to_portfolio(trade: dict, portfolio: dict) -> dict:
    """Apply a single IB trade to portfolio.json and return updated portfolio."""
    ticker   = trade["ticker"]
    action   = trade["action"]
    qty      = trade["quantity"]
    price    = trade["price"]
    realized = trade.get("realized_pnl", 0)
    holdings = portfolio.get("holdings", [])

    if action == "BUY":
        existing = next((h for h in holdings if h.get("ticker") == ticker), None)
        if existing:
            old_shares = float(existing.get("shares", 0))
            old_cost   = float(existing.get("avg_cost", 0))
            new_shares = old_shares + qty
            new_avg    = (old_shares * old_cost + qty * price) / new_shares if new_shares else price
            existing["shares"]         = round(new_shares, 4)
            existing["avg_cost"]       = round(new_avg, 4)
            existing["current_price"]  = price
            existing["cost_basis"]     = round(new_shares * new_avg, 2)
            existing["current_value"]  = round(new_shares * price, 2)
            existing["unrealized_pnl"] = round((price - new_avg) * new_shares, 2)
            existing["pnl_pct"]        = round((price - new_avg) / new_avg * 100, 2) if new_avg else 0
        else:
            holdings.append({
                "ticker":        ticker,
                "shares":        qty,
                "avg_cost":      round(price, 4),
                "current_price": price,
                "cost_basis":    round(qty * price, 2),
                "current_value": round(qty * price, 2),
                "unrealized_pnl": 0.0,
                "pnl_pct":       0.0,
                "sector":        "Unknown",
                "strategy":      "trade",
            })
        cash = float(portfolio.get("available_cash", 0) or 0) - qty * price
        portfolio["available_cash"] = round(max(cash, 0), 2)

    elif action == "SELL":
        existing = next((h for h in holdings if h.get("ticker") == ticker), None)
        if existing:
            old_shares = float(existing.get("shares", 0))
            avg_cost   = float(existing.get("avg_cost", price))
            sold       = min(qty, old_shares)
            remaining  = old_shares - sold

            if remaining <= 0:
                holdings.remove(existing)
            else:
                existing["shares"]         = round(remaining, 4)
                existing["current_value"]  = round(remaining * price, 2)
                existing["unrealized_pnl"] = round((price - avg_cost) * remaining, 2)
        else:
            avg_cost = price  # Unknown cost basis
            sold     = qty

        realized_pnl = realized if realized != 0 else round((price - avg_cost) * sold, 2)

        realized_trades = portfolio.get("realized_trades_ytd", [])
        realized_trades.append({
            "ticker":       ticker,
            "shares":       round(sold, 4),
            "avg_cost":     avg_cost,
            "exit_price":   price,
            "realized_pnl": realized_pnl,
            "pnl_pct":      round((price - avg_cost) / avg_cost * 100, 2) if avg_cost else 0,
            "date":         trade.get("trade_date", date.today().isoformat()),
            "source":       "ib_flex",
        })
        portfolio["realized_trades_ytd"] = realized_trades
        prev = float(portfolio.get("realized_pnl_ytd", 0) or 0)
        portfolio["realized_pnl_ytd"]    = round(prev + realized_pnl, 2)

        cash = float(portfolio.get("available_cash", 0) or 0) + sold * price
        portfolio["available_cash"] = round(cash, 2)

    portfolio["holdings"]              = holdings
    total_val = sum(float(h.get("current_value", 0) or 0) for h in holdings)
    portfolio["total_portfolio_value"] = round(total_val + float(portfolio.get("available_cash", 0)), 2)
    portfolio["_last_updated"]         = datetime.now().isoformat()
    portfolio["last_flex_sync"]        = datetime.now().isoformat()
    return portfolio


# ---------------------------------------------------------------------------
# Telegram notification for auto-detected trades
# ---------------------------------------------------------------------------

def _send_trade_notification(trade: dict, portfolio: dict) -> None:
    """Send a Telegram message when a new trade is auto-detected."""
    try:
        import telegram_bot
        ticker  = trade["ticker"]
        action  = trade["action"]
        qty     = int(trade["quantity"])
        price   = trade["price"]
        realized = trade.get("realized_pnl", 0)

        if action == "SELL" and realized != 0:
            sign     = "+" if realized >= 0 else ""
            avg_cost = price - realized / qty if qty else price
            pnl_pct  = round((price - avg_cost) / avg_cost * 100, 2) if avg_cost else 0
            header = (
                f"🤖 Auto-detected SELL: {ticker}\n"
                f"{qty} shares @ ${price:.2f}\n"
                f"Realized P&L: {sign}${realized:,.2f} ({sign}{pnl_pct:.1f}%)\n"
            )
        else:
            header = (
                f"🤖 Auto-detected {'BUY' if action == 'BUY' else 'SELL'}: {ticker}\n"
                f"{qty} shares @ ${price:.2f}\n"
            )

        # Get investor feedback
        feedback = _get_feedback(ticker, action, qty, price, portfolio)
        telegram_bot.send_message(header + "\n" + feedback)
    except Exception as e:
        logger.error("Failed to send trade notification: %s", e)


def _get_feedback(ticker, direction, shares, price, portfolio) -> str:
    """Reuse the feedback engine from telegram_bot."""
    try:
        import telegram_bot
        return telegram_bot._get_trade_feedback(ticker, direction, shares, price, portfolio)
    except Exception as e:
        return f"(Feedback unavailable: {e})"


# ---------------------------------------------------------------------------
# Main sync entry point — called by agent.py scheduler
# ---------------------------------------------------------------------------

def sync_new_trades(notify: bool = True) -> list[dict]:
    """
    Full sync cycle:
      1. Fetch today's trades from IB Flex
      2. Filter out already-seen trades
      3. Apply each new trade to portfolio.json
      4. Commit portfolio to GitHub
      5. Optionally send Telegram notifications
    Returns the list of new trades processed.
    """
    if not is_configured():
        logger.debug("IB Flex not configured — skipping auto sync")
        return []

    logger.info("IB Flex: checking for new trades...")
    all_trades = fetch_todays_trades()
    new_trades  = filter_new_trades(all_trades)

    if not new_trades:
        logger.info("IB Flex: no new trades detected")
        return []

    logger.info("IB Flex: %d new trade(s) detected — processing", len(new_trades))
    portfolio = _load_portfolio()

    for trade in new_trades:
        logger.info("  Processing %s %s × %s @ $%s",
                    trade["action"], trade["ticker"],
                    trade["quantity"], trade["price"])
        portfolio = apply_trade_to_portfolio(trade, portfolio)
        _save_portfolio(portfolio)

        if notify:
            _send_trade_notification(trade, portfolio)

    # Commit once after all trades are applied
    _commit_portfolio(portfolio)
    logger.info("IB Flex: portfolio updated and committed (%d trades)", len(new_trades))
    return new_trades


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    Path("logs").mkdir(exist_ok=True)

    if "--test" in sys.argv:
        # Test mode: print raw Flex XML without updating portfolio
        print("Testing IB Flex connection...")
        if not is_configured():
            print("ERROR: Set IB_FLEX_TOKEN and IB_FLEX_QUERY_ID in .env")
            sys.exit(1)
        trades = fetch_todays_trades()
        print(f"Found {len(trades)} trades today:")
        for t in trades:
            print(f"  {t['action']} {t['ticker']} {t['quantity']} @ ${t['price']:.2f}  P&L: ${t['realized_pnl']:+.2f}")
    else:
        processed = sync_new_trades(notify=False)
        print(f"Processed {len(processed)} new trades.")
