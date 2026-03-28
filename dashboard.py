#!/usr/bin/env python3
"""
Dashboard — local web server with live portfolio, paper trading,
sector rotation, performance metrics, and trade history.

Endpoints:
    /                   Full dashboard
    /api/alerts         Alert history JSON
    /api/portfolio      Portfolio JSON
    /api/paper          Paper trading state JSON
    /api/positions      Open positions JSON
    /api/performance    Risk-adjusted metrics JSON
    /api/sectors        Sector rotation JSON

Usage:
    python3 dashboard.py              # Start on port 8050
    python3 dashboard.py --port 9000  # Custom port
"""

import argparse
import csv
import json
import os
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import List, Dict

LOGS_DIR = Path("logs")
DASHBOARD_CSV = LOGS_DIR / "dashboard.csv"
PORTFOLIO_FILE = Path("portfolio.json")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_alerts() -> List[Dict]:
    if os.getenv("USE_SQLITE", "0") == "1":
        try:
            import database as db
            return db.get_alerts(limit=200)
        except Exception:
            pass
    if not DASHBOARD_CSV.exists():
        return []
    with open(DASHBOARD_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_portfolio() -> dict:
    if os.getenv("USE_SQLITE", "0") == "1":
        try:
            import database as db
            config = db.get_portfolio_config()
            holdings = db.get_holdings()
            return {
                "total_portfolio_value": config.get("total_portfolio_value", 0),
                "available_cash": config.get("available_cash", 0),
                "holdings": holdings,
            }
        except Exception:
            pass
    if not PORTFOLIO_FILE.exists():
        return {}
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)


def load_paper_state() -> dict:
    try:
        import paper_trader
        return paper_trader.load_state()
    except Exception:
        return {}


def load_paper_performance() -> dict:
    try:
        import paper_trader
        state = paper_trader.load_state()
        return paper_trader.compute_performance(state)
    except Exception:
        return {}


def load_risk_metrics() -> dict:
    try:
        import risk_metrics
        pnls, capital = risk_metrics.load_paper_pnls()
        if not pnls:
            pnls, capital = risk_metrics.load_backtest_pnls()
        if pnls:
            return risk_metrics.compute_metrics(pnls, starting_capital=capital)
    except Exception:
        pass
    return {}


def load_positions() -> List[Dict]:
    try:
        paper = load_paper_state()
        return paper.get("open_positions", [])
    except Exception:
        return []


def load_attribution() -> dict:
    try:
        import performance_attribution
        return performance_attribution.full_attribution()
    except Exception:
        return {}


def load_monte_carlo() -> dict:
    try:
        import monte_carlo
        pnls, capital = monte_carlo.load_historical_pnls()
        if pnls:
            return monte_carlo.run_simulation(pnls, capital, n_simulations=500, seed=42)
        return {}
    except Exception:
        return {}


def load_briefing() -> dict:
    try:
        import daily_briefing
        return daily_briefing.generate_briefing()
    except Exception:
        return {}


def load_regime() -> dict:
    try:
        import market_regime
        history = market_regime.get_history(1)
        if history:
            return history[0]
        return {}
    except Exception:
        return {}


def load_rebalance() -> dict:
    try:
        import rebalancer
        return rebalancer.full_analysis()
    except Exception:
        return {}


def load_journal_entries() -> List[Dict]:
    try:
        import trade_journal
        return trade_journal.get_entries(limit=100)
    except Exception:
        return []


def load_journal_stats() -> dict:
    try:
        import trade_journal
        return trade_journal.get_stats()
    except Exception:
        return {}


def load_price_alerts() -> dict:
    try:
        import price_alerts
        return {
            "active": price_alerts.get_active_alerts(),
            "all": price_alerts._load_alerts(),
            "position_alerts": price_alerts.generate_position_alerts(),
        }
    except Exception:
        return {}


def load_health() -> dict:
    try:
        report_file = LOGS_DIR / "health_report.json"
        if report_file.exists():
            with open(report_file) as f:
                return json.load(f)
        import health_monitor
        results = health_monitor.run_all_checks()
        return {
            "overall": health_monitor.get_overall_status(results),
            "checks": results,
        }
    except Exception:
        return {}


def load_eod_report(report_date: str = None) -> dict:
    try:
        import eod_report
        if report_date:
            saved = eod_report.load_report(report_date)
            if saved:
                return saved
        return eod_report.generate_report(report_date)
    except Exception:
        return {}


def load_eod_report_list() -> List[str]:
    try:
        import eod_report
        return eod_report.list_reports()
    except Exception:
        return []


def load_optimizer() -> dict:
    try:
        import strategy_optimizer
        weights_file = Path("logs/optimized_weights.json")
        if weights_file.exists():
            with open(weights_file) as f:
                return json.load(f)
        return {}
    except Exception:
        return {}


def load_watchlist_suggestions() -> dict:
    try:
        import watchlist_curator
        wl = watchlist_curator.load_watchlist()
        tickers = [w["ticker"] for w in wl]
        additions = watchlist_curator.suggest_additions(tickers, {}, {})
        removals = watchlist_curator.suggest_removals(wl, {}, {}, {})
        return {"additions": additions, "removals": removals}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html() -> str:
    alerts = load_alerts()
    portfolio = load_portfolio()
    paper = load_paper_state()
    metrics = load_risk_metrics()
    journal_entries = load_journal_entries()
    journal_stats = load_journal_stats()
    regime = load_regime()
    attribution = load_attribution()
    health = load_health()
    price_alert_data = load_price_alerts()

    total_val = portfolio.get("total_portfolio_value", 0)
    cash = portfolio.get("available_cash", 0)
    holdings = portfolio.get("holdings", [])
    invested = sum(float(h.get("current_value", 0)) for h in holdings)
    total_unrealized = sum(float(h.get("unrealized_pnl", 0) or 0) for h in holdings)
    total_cost_basis = sum(float(h.get("cost_basis", 0) or 0) for h in holdings)
    realized_ytd = float(portfolio.get("realized_pnl_ytd", 0) or 0)
    realized_trades = portfolio.get("realized_trades_ytd", [])
    broker = portfolio.get("_broker", "")
    num_holdings = len(holdings)

    # Sector breakdown
    sector_map = {}
    for h in holdings:
        sec = h.get("sector", "Other")
        sector_map[sec] = sector_map.get(sec, 0) + float(h.get("current_value", 0))

    # Winners/losers
    winners = [h for h in holdings if float(h.get("unrealized_pnl", 0) or 0) > 0]
    losers = [h for h in holdings if float(h.get("unrealized_pnl", 0) or 0) < 0]

    total_alerts = len(alerts)
    buy_alerts = sum(1 for a in alerts if a.get("direction") == "BUY")
    sell_alerts = sum(1 for a in alerts if a.get("direction") == "SELL")

    # Paper trading stats
    paper_cash = paper.get("cash", 0)
    paper_starting = paper.get("starting_capital", 100000)
    paper_positions = paper.get("open_positions", [])
    paper_trades = paper.get("closed_trades", [])
    paper_pnl = sum(t.get("pnl", 0) for t in paper_trades)
    paper_wins = sum(1 for t in paper_trades if t.get("pnl", 0) > 0)
    paper_total = len(paper_trades)
    paper_wr = (paper_wins / paper_total * 100) if paper_total > 0 else 0

    # Risk metrics
    sharpe = metrics.get("sharpe_ratio", 0)
    sortino = metrics.get("sortino_ratio", 0)
    max_dd = metrics.get("max_drawdown_pct", 0)
    pf = metrics.get("profit_factor", 0)
    expectancy = metrics.get("expectancy", 0)
    equity_curve = metrics.get("equity_curve", [])

    # Alert table rows
    alert_rows = ""
    for a in list(reversed(alerts))[:50]:
        direction = a.get("direction", "")
        dir_class = "buy" if direction == "BUY" else "sell"
        signals = a.get("signals", "").replace("|", ", ")
        alert_rows += f"""<tr class="{dir_class}">
            <td>{str(a.get('timestamp', ''))[:19]}</td>
            <td><strong>{a.get('ticker', '')}</strong></td>
            <td><span class="badge {dir_class}">{direction}</span></td>
            <td>{a.get('signal_score', '')}</td>
            <td>${float(a.get('entry_price', 0)):,.2f}</td>
            <td>${float(a.get('stop_loss', 0)):,.2f}</td>
            <td>{a.get('shares', '')}</td>
            <td>${float(a.get('target_1', 0)):,.2f}</td>
            <td>${float(a.get('target_2', 0)):,.2f}</td>
            <td class="signals-cell">{signals}</td></tr>"""

    # Holdings rows — mirrors IB portfolio view
    holdings_rows = ""
    sorted_holdings = sorted(holdings, key=lambda h: h.get("ticker", ""))
    for h in sorted_holdings:
        shares = int(h.get("shares", 1))
        avg_cost = float(h.get("avg_cost", 1))
        cur_price = float(h.get("current_price", 0))
        cur_val = float(h.get("current_value", 0))
        cost_basis = float(h.get("cost_basis", 0))
        unrealized = float(h.get("unrealized_pnl", 0) or 0)
        pnl_pct = float(h.get("pnl_pct", 0) or 0)
        gain_class = "positive" if unrealized >= 0 else "negative"
        pct_of_port = (cur_val / invested * 100) if invested > 0 else 0
        holdings_rows += f"""<tr>
            <td><strong>{h.get('ticker', '')}</strong></td>
            <td style="text-align:right">{shares}</td>
            <td style="text-align:right">${cur_price:,.2f}</td>
            <td style="text-align:right">${cost_basis:,.2f}</td>
            <td style="text-align:right">${cur_val:,.2f}</td>
            <td style="text-align:right">${avg_cost:,.2f}</td>
            <td style="text-align:right" class="{gain_class}">${unrealized:+,.2f}</td>
            <td style="text-align:right" class="{gain_class}">{pnl_pct:+.1f}%</td>
            <td style="text-align:right;color:var(--text-muted)">{pct_of_port:.1f}%</td>
            <td>{h.get('sector', '')}</td></tr>"""

    # Realized trades rows
    realized_rows = ""
    for rt in sorted(realized_trades, key=lambda r: abs(r.get("realized_pnl", 0)), reverse=True):
        rpnl = float(rt.get("realized_pnl", 0))
        rpnl_class = "positive" if rpnl >= 0 else "negative"
        realized_rows += f"""<tr>
            <td><strong>{rt.get('ticker', '')}</strong></td>
            <td class="{rpnl_class}" style="text-align:right">${rpnl:+,.2f}</td>
            <td>{rt.get('type', 'short_term').replace('_', ' ').title()}</td></tr>"""

    # Sector breakdown rows
    sector_rows = ""
    for sec, val in sorted(sector_map.items(), key=lambda x: x[1], reverse=True):
        sec_pct = (val / invested * 100) if invested > 0 else 0
        sector_rows += f"""<tr>
            <td>{sec}</td>
            <td style="text-align:right">${val:,.2f}</td>
            <td style="text-align:right">{sec_pct:.1f}%</td>
            <td><div style="background:var(--blue);height:8px;border-radius:4px;width:{sec_pct}%"></div></td></tr>"""

    # Paper positions rows
    paper_pos_rows = ""
    for p in paper_positions:
        t1 = "Y" if p.get("t1_hit") else "-"
        t2 = "Y" if p.get("t2_hit") else "-"
        sigs = ", ".join(p.get("triggered_signals", [])[:3]) if isinstance(p.get("triggered_signals"), list) else str(p.get("triggered_signals", ""))
        paper_pos_rows += f"""<tr>
            <td><strong>{p.get('ticker', '')}</strong></td>
            <td><span class="badge {'buy' if p.get('direction')=='BUY' else 'sell'}">{p.get('direction', '')}</span></td>
            <td>{p.get('shares', 0)}</td><td>${float(p.get('entry_price', 0)):,.2f}</td>
            <td>${float(p.get('current_stop', 0)):,.2f}</td>
            <td>{t1}</td><td>{t2}</td>
            <td class="signals-cell">{sigs}</td></tr>"""

    # Paper trade history rows (last 15)
    paper_trade_rows = ""
    for t in paper_trades[-15:]:
        pnl = t.get("pnl", 0)
        pnl_class = "positive" if pnl >= 0 else "negative"
        paper_trade_rows += f"""<tr>
            <td>{str(t.get('exit_date', ''))[:10]}</td>
            <td><strong>{t.get('ticker', '')}</strong></td>
            <td><span class="badge {'buy' if t.get('direction')=='BUY' else 'sell'}">{t.get('direction', '')}</span></td>
            <td>${float(t.get('entry_price', 0)):,.2f}</td>
            <td>${float(t.get('exit_price', 0)):,.2f}</td>
            <td class="{pnl_class}">${pnl:,.2f}</td>
            <td>{t.get('exit_reason', '')}</td></tr>"""

    # Equity curve JS data
    eq_js = json.dumps(equity_curve) if equity_curve else "[]"

    # --- Journal data ---
    journal_rows = ""
    for j in journal_entries[:30]:
        status = "CLOSED" if j.get("closed") else "OPEN"
        pnl_val = float(j.get("pnl", 0) or 0)
        status_class = "positive" if status == "CLOSED" and pnl_val > 0 else ("negative" if pnl_val < 0 else "")
        tags = ", ".join(j.get("tags", [])[:3])
        journal_rows += f"""<tr>
            <td>{str(j.get('created_at', ''))[:10]}</td>
            <td><strong>{j.get('ticker', '')}</strong></td>
            <td><span class="badge {'buy' if j.get('direction')=='BUY' else 'sell'}">{j.get('direction', '')}</span></td>
            <td>{status}</td>
            <td class="{status_class}">${float(pnl_val):+,.2f}</td>
            <td class="signals-cell">{tags}</td></tr>"""

    js_total = journal_stats.get("total_entries", 0)
    js_reviewed = journal_stats.get("reviewed", 0)
    js_review_pct = journal_stats.get("review_coverage_pct", 0)

    # --- Regime data ---
    regime_name = regime.get("regime", "N/A")
    regime_confidence = regime.get("confidence", 0)
    regime_desc = regime.get("params", {}).get("description", "") if isinstance(regime.get("params"), dict) else ""

    # --- Attribution data ---
    attr_by_signal = attribution.get("by_signal", {})
    attr_by_dir = attribution.get("by_direction", {})
    attr_by_exit = attribution.get("by_exit_reason", {})
    attr_total = attribution.get("total_trades", 0)

    signal_rows = ""
    for sig, s in list(attr_by_signal.items())[:10]:
        signal_rows += f"""<tr>
            <td>{sig}</td><td>{s.get('trades', 0)}</td>
            <td>{s.get('win_rate', 0):.1f}%</td>
            <td class="{'positive' if s.get('total_pnl', 0) >= 0 else 'negative'}">${s.get('total_pnl', 0):,.2f}</td>
            <td>${s.get('avg_pnl', 0):,.2f}</td></tr>"""

    exit_rows = ""
    for reason, e in attr_by_exit.items():
        exit_rows += f"""<tr>
            <td>{reason}</td><td>{e.get('trades', 0)}</td>
            <td>{e.get('win_rate', 0):.1f}%</td>
            <td>${e.get('avg_pnl', 0):,.2f}</td></tr>"""

    # --- Health data ---
    health_checks = health.get("checks", [])
    health_overall = health.get("overall", "N/A")
    health_rows = ""
    status_icons = {"OK": "+", "WARN": "!", "FAIL": "X"}
    status_colors = {"OK": "green", "WARN": "orange", "FAIL": "red"}
    for c in health_checks:
        icon = status_icons.get(c.get("status", ""), "?")
        color = status_colors.get(c.get("status", ""), "text-muted")
        health_rows += f"""<tr>
            <td style="color:var(--{color})"><strong>[{icon}]</strong></td>
            <td>{c.get('name', '')}</td>
            <td style="color:var(--{color})">{c.get('status', '')}</td>
            <td>{c.get('message', '')}</td></tr>"""

    # --- Price alerts data ---
    active_alerts = price_alert_data.get("active", [])
    position_alerts = price_alert_data.get("position_alerts", [])
    price_alert_rows = ""
    for a in active_alerts:
        price_alert_rows += f"""<tr>
            <td><strong>{a.get('ticker', '')}</strong></td>
            <td>{a.get('condition', '')}</td>
            <td>${float(a.get('price', 0)):,.2f}</td>
            <td>{a.get('type', 'CUSTOM')}</td>
            <td>{a.get('note', '')}</td></tr>"""
    for a in position_alerts:
        price_alert_rows += f"""<tr>
            <td><strong>{a.get('ticker', '')}</strong></td>
            <td>{a.get('condition', '')}</td>
            <td>${float(a.get('price', 0)):,.2f}</td>
            <td>{a.get('type', '')}</td>
            <td>{a.get('note', '')}</td></tr>"""

    # Sharpe quality label
    if sharpe >= 2:
        sharpe_label = "Excellent"
        sharpe_color = "var(--green)"
    elif sharpe >= 1:
        sharpe_label = "Good"
        sharpe_color = "var(--green)"
    elif sharpe >= 0.5:
        sharpe_label = "Moderate"
        sharpe_color = "var(--orange)"
    elif sharpe >= 0:
        sharpe_label = "Below avg"
        sharpe_color = "var(--orange)"
    else:
        sharpe_label = "Poor"
        sharpe_color = "var(--red)"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Agent Dashboard</title>
<style>
:root {{
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a;
    --text: #e1e4ed; --text-muted: #8b8fa3;
    --green: #00c853; --red: #ff1744; --blue: #2979ff;
    --purple: #7c4dff; --orange: #ff9100; --cyan: #00e5ff;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5;
}}
.header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border);
}}
.header h1 {{ font-size: 22px; font-weight: 600; }}
.header .sub {{ color: var(--text-muted); font-size: 13px; }}
.tabs {{
    display: flex; gap: 4px; margin-bottom: 24px; border-bottom: 1px solid var(--border);
    padding-bottom: 0;
}}
.tab {{
    padding: 10px 20px; cursor: pointer; border: none; background: none;
    color: var(--text-muted); font-size: 14px; font-weight: 500;
    border-bottom: 2px solid transparent; transition: all 0.2s;
}}
.tab:hover {{ color: var(--text); }}
.tab.active {{ color: var(--blue); border-bottom-color: var(--blue); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.cards {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 24px;
}}
.card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
}}
.card .label {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--text-muted); margin-bottom: 4px;
}}
.card .value {{ font-size: 24px; font-weight: 700; }}
.card .value.green {{ color: var(--green); }}
.card .value.red {{ color: var(--red); }}
.card .value.blue {{ color: var(--blue); }}
.card .value.purple {{ color: var(--purple); }}
.card .value.orange {{ color: var(--orange); }}
.card .value.cyan {{ color: var(--cyan); }}
.card .sub-value {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}
.section {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; margin-bottom: 20px;
}}
.section h2 {{ font-size: 16px; font-weight: 600; margin-bottom: 14px; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{
    text-align: left; padding: 8px 10px; background: var(--bg);
    color: var(--text-muted); font-weight: 500; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.5px; position: sticky; top: 0;
}}
td {{ padding: 8px 10px; border-top: 1px solid var(--border); }}
tr:hover td {{ background: rgba(255,255,255,0.02); }}
.badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
}}
.badge.buy {{ background: rgba(0,200,83,0.15); color: var(--green); }}
.badge.sell {{ background: rgba(255,23,68,0.15); color: var(--red); }}
.positive {{ color: var(--green); }}
.negative {{ color: var(--red); }}
.signals-cell {{ max-width: 200px; font-size: 11px; color: var(--text-muted); }}
.table-wrapper {{ overflow-x: auto; max-height: 500px; overflow-y: auto; }}
.empty-state {{ text-align: center; padding: 40px; color: var(--text-muted); }}
.chart-container {{
    width: 100%; height: 200px; position: relative; margin-top: 8px;
}}
canvas {{ width: 100%; height: 100%; }}
.metric-row {{
    display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid var(--border); font-size: 13px;
}}
.metric-row:last-child {{ border-bottom: none; }}
.metric-label {{ color: var(--text-muted); }}
.metric-value {{ font-weight: 600; }}
.refresh-btn {{
    background: var(--blue); color: white; border: none;
    padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px;
}}
.refresh-btn:hover {{ opacity: 0.85; }}
</style>
</head>
<body>

<div class="header">
    <div><h1>Stock Agent Dashboard</h1>
    <div class="sub">Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div></div>
    <button class="refresh-btn" onclick="location.reload()">Refresh</button>
</div>

<div class="tabs">
    <button class="tab active" onclick="switchTab(event,'overview')">Overview</button>
    <button class="tab" onclick="switchTab(event,'paper')">Paper Trading</button>
    <button class="tab" onclick="switchTab(event,'performance')">Performance</button>
    <button class="tab" onclick="switchTab(event,'analytics')">Analytics</button>
    <button class="tab" onclick="switchTab(event,'journal')">Journal</button>
    <button class="tab" onclick="switchTab(event,'alerts')">Alerts</button>
    <button class="tab" onclick="switchTab(event,'system')">System</button>
</div>

<!-- OVERVIEW TAB -->
<div id="overview" class="tab-content active">
<div class="cards">
    <div class="card"><div class="label">Portfolio Value</div><div class="value blue">${total_val:,.0f}</div>
        <div class="sub-value">{num_holdings} positions{(' | ' + broker) if broker else ''}</div></div>
    <div class="card"><div class="label">Cash Available</div><div class="value green">${cash:,.0f}</div></div>
    <div class="card"><div class="label">Invested</div><div class="value purple">${invested:,.0f}</div>
        <div class="sub-value">Cost basis: ${total_cost_basis:,.0f}</div></div>
    <div class="card"><div class="label">Unrealized P&L</div>
        <div class="value {'green' if total_unrealized >= 0 else 'red'}">${total_unrealized:+,.0f}</div>
        <div class="sub-value">{len(winners)} winners / {len(losers)} losers</div></div>
    <div class="card"><div class="label">Realized P&L YTD</div>
        <div class="value {'green' if realized_ytd >= 0 else 'red'}">${realized_ytd:+,.0f}</div>
        <div class="sub-value">{len(realized_trades)} closed trades</div></div>
    <div class="card"><div class="label">Total P&L</div>
        <div class="value {'green' if (total_unrealized + realized_ytd) >= 0 else 'red'}">${total_unrealized + realized_ytd:+,.0f}</div>
        <div class="sub-value">unrealized + realized</div></div>
</div>

<div class="section">
    <h2>Open Positions ({num_holdings})</h2>
    <div class="table-wrapper">
    {"<table><thead><tr><th>Ticker</th><th style='text-align:right'>Shares</th><th style='text-align:right'>Last</th><th style='text-align:right'>Cost Basis</th><th style='text-align:right'>Mkt Value</th><th style='text-align:right'>Avg Price</th><th style='text-align:right'>Unrealized</th><th style='text-align:right'>P&L %</th><th style='text-align:right'>Weight</th><th>Sector</th></tr></thead><tbody>" + holdings_rows + "</tbody></table>" if holdings_rows else '<div class="empty-state">No holdings</div>'}
    </div>
</div>

<div class="grid-2">
<div class="section">
    <h2>Sector Allocation</h2>
    <div class="table-wrapper">
    {"<table><thead><tr><th>Sector</th><th style='text-align:right'>Value</th><th style='text-align:right'>Weight</th><th style='width:40%'>Allocation</th></tr></thead><tbody>" + sector_rows + "</tbody></table>" if sector_rows else '<div class="empty-state">No sector data</div>'}
    </div>
</div>
<div class="section">
    <h2>Realized Trades YTD ({len(realized_trades)})</h2>
    <div class="table-wrapper" style="max-height:350px">
    {"<table><thead><tr><th>Ticker</th><th style='text-align:right'>Realized P&L</th><th>Type</th></tr></thead><tbody>" + realized_rows + "</tbody></table>" if realized_rows else '<div class="empty-state">No realized trades</div>'}
    </div>
</div>
</div>
</div>

<!-- PAPER TRADING TAB -->
<div id="paper" class="tab-content">
<div class="cards">
    <div class="card"><div class="label">Starting Capital</div><div class="value blue">${paper_starting:,.0f}</div></div>
    <div class="card"><div class="label">Cash</div><div class="value green">${paper_cash:,.0f}</div></div>
    <div class="card"><div class="label">Open Positions</div><div class="value purple">{len(paper_positions)}</div></div>
    <div class="card"><div class="label">Closed Trades</div><div class="value orange">{paper_total}</div></div>
    <div class="card"><div class="label">Total P&L</div>
        <div class="value {'green' if paper_pnl >= 0 else 'red'}">${paper_pnl:+,.2f}</div></div>
    <div class="card"><div class="label">Win Rate</div><div class="value cyan">{paper_wr:.0f}%</div></div>
</div>

<div class="section">
    <h2>Open Positions</h2>
    <div class="table-wrapper">
    {"<table><thead><tr><th>Ticker</th><th>Dir</th><th>Shares</th><th>Entry</th><th>Stop</th><th>T1</th><th>T2</th><th>Signals</th></tr></thead><tbody>" + paper_pos_rows + "</tbody></table>" if paper_pos_rows else '<div class="empty-state">No open positions</div>'}
    </div>
</div>

<div class="section">
    <h2>Trade History (Last 15)</h2>
    <div class="table-wrapper">
    {"<table><thead><tr><th>Date</th><th>Ticker</th><th>Dir</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr></thead><tbody>" + paper_trade_rows + "</tbody></table>" if paper_trade_rows else '<div class="empty-state">No trades yet</div>'}
    </div>
</div>
</div>

<!-- PERFORMANCE TAB -->
<div id="performance" class="tab-content">
<div class="cards">
    <div class="card"><div class="label">Sharpe Ratio</div>
        <div class="value" style="color:{sharpe_color}">{sharpe:.2f}</div>
        <div class="sub-value">{sharpe_label}</div></div>
    <div class="card"><div class="label">Sortino Ratio</div><div class="value cyan">{sortino:.2f}</div></div>
    <div class="card"><div class="label">Max Drawdown</div><div class="value red">{max_dd:.1f}%</div></div>
    <div class="card"><div class="label">Profit Factor</div><div class="value green">{pf}</div></div>
    <div class="card"><div class="label">Expectancy</div>
        <div class="value {'green' if expectancy >= 0 else 'red'}">${expectancy:,.2f}</div>
        <div class="sub-value">per trade</div></div>
    <div class="card"><div class="label">Kelly Criterion</div><div class="value purple">{metrics.get('kelly_criterion', 0):.1f}%</div></div>
</div>

<div class="section">
    <h2>Equity Curve</h2>
    <div class="chart-container"><canvas id="equityChart"></canvas></div>
</div>

<div class="grid-2">
<div class="section">
    <h2>Returns</h2>
    <div class="metric-row"><span class="metric-label">Total P&L</span><span class="metric-value {'positive' if metrics.get('total_pnl',0)>=0 else 'negative'}">${metrics.get('total_pnl', 0):,.2f}</span></div>
    <div class="metric-row"><span class="metric-label">Total Return</span><span class="metric-value">{metrics.get('total_return_pct', 0):+.2f}%</span></div>
    <div class="metric-row"><span class="metric-label">Annualized Return</span><span class="metric-value">{metrics.get('annualized_return_pct', 0):+.2f}%</span></div>
    <div class="metric-row"><span class="metric-label">Annualized Volatility</span><span class="metric-value">{metrics.get('annualized_volatility_pct', 0):.2f}%</span></div>
    <div class="metric-row"><span class="metric-label">Recovery Factor</span><span class="metric-value">{metrics.get('recovery_factor', 0)}</span></div>
</div>
<div class="section">
    <h2>Trade Stats</h2>
    <div class="metric-row"><span class="metric-label">Total Trades</span><span class="metric-value">{metrics.get('total_trades', 0)}</span></div>
    <div class="metric-row"><span class="metric-label">Win Rate</span><span class="metric-value">{metrics.get('win_rate', 0):.1f}%</span></div>
    <div class="metric-row"><span class="metric-label">Avg Win</span><span class="metric-value positive">${metrics.get('avg_win', 0):,.2f}</span></div>
    <div class="metric-row"><span class="metric-label">Avg Loss</span><span class="metric-value negative">${metrics.get('avg_loss', 0):,.2f}</span></div>
    <div class="metric-row"><span class="metric-label">Max Win Streak</span><span class="metric-value">{metrics.get('max_win_streak', 0)}</span></div>
    <div class="metric-row"><span class="metric-label">Max Loss Streak</span><span class="metric-value">{metrics.get('max_loss_streak', 0)}</span></div>
</div>
</div>
</div>

<!-- ANALYTICS TAB -->
<div id="analytics" class="tab-content">
<div class="cards">
    <div class="card"><div class="label">Market Regime</div>
        <div class="value {'green' if 'BULL' in regime_name else ('red' if 'BEAR' in regime_name else 'orange')}">{regime_name}</div>
        <div class="sub-value">{regime_confidence}% confidence</div></div>
    <div class="card"><div class="label">Attribution Trades</div><div class="value blue">{attr_total}</div></div>
    <div class="card"><div class="label">Long Trades</div>
        <div class="value green">{attr_by_dir.get('BUY', dict()).get('trades', 0)}</div>
        <div class="sub-value">WR: {attr_by_dir.get('BUY', dict()).get('win_rate', 0):.0f}%</div></div>
    <div class="card"><div class="label">Short Trades</div>
        <div class="value red">{attr_by_dir.get('SELL', dict()).get('trades', 0)}</div>
        <div class="sub-value">WR: {attr_by_dir.get('SELL', dict()).get('win_rate', 0):.0f}%</div></div>
</div>

{f'<div class="section"><p style="color:var(--text-muted)">{regime_desc}</p></div>' if regime_desc else ''}

<div class="grid-2">
<div class="section">
    <h2>Performance by Signal (Top 10)</h2>
    <div class="table-wrapper">
    {"<table><thead><tr><th>Signal</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th><th>Avg P&L</th></tr></thead><tbody>" + signal_rows + "</tbody></table>" if signal_rows else '<div class="empty-state">No signal data</div>'}
    </div>
</div>
<div class="section">
    <h2>Performance by Exit Reason</h2>
    <div class="table-wrapper">
    {"<table><thead><tr><th>Reason</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th></tr></thead><tbody>" + exit_rows + "</tbody></table>" if exit_rows else '<div class="empty-state">No exit data</div>'}
    </div>
</div>
</div>
</div>

<!-- JOURNAL TAB -->
<div id="journal" class="tab-content">
<div class="cards">
    <div class="card"><div class="label">Journal Entries</div><div class="value blue">{js_total}</div></div>
    <div class="card"><div class="label">Reviewed</div><div class="value green">{js_reviewed}</div></div>
    <div class="card"><div class="label">Review Coverage</div><div class="value {'green' if js_review_pct >= 80 else 'orange'}">{js_review_pct:.0f}%</div></div>
</div>

<div class="section">
    <h2>Recent Journal Entries</h2>
    <div class="table-wrapper">
    {"<table><thead><tr><th>Date</th><th>Ticker</th><th>Dir</th><th>Status</th><th>P&L</th><th>Tags</th></tr></thead><tbody>" + journal_rows + "</tbody></table>" if journal_rows else '<div class="empty-state">No journal entries</div>'}
    </div>
</div>
</div>

<!-- ALERTS TAB -->
<div id="alerts" class="tab-content">
<div class="section">
    <h2>Trade Alerts ({total_alerts} total)</h2>
    <div class="table-wrapper">
    {"<table><thead><tr><th>Time</th><th>Ticker</th><th>Dir</th><th>Score</th><th>Entry</th><th>Stop</th><th>Shares</th><th>T1</th><th>T2</th><th>Signals</th></tr></thead><tbody>" + alert_rows + "</tbody></table>" if alert_rows else '<div class="empty-state">No alerts yet</div>'}
    </div>
</div>

<div class="section">
    <h2>Price Alerts ({len(active_alerts)} active + {len(position_alerts)} position)</h2>
    <div class="table-wrapper">
    {"<table><thead><tr><th>Ticker</th><th>Condition</th><th>Price</th><th>Type</th><th>Note</th></tr></thead><tbody>" + price_alert_rows + "</tbody></table>" if price_alert_rows else '<div class="empty-state">No price alerts</div>'}
    </div>
</div>
</div>

<!-- SYSTEM TAB -->
<div id="system" class="tab-content">
<div class="cards">
    <div class="card"><div class="label">System Health</div>
        <div class="value {'green' if health_overall == 'OK' else ('orange' if health_overall == 'WARN' else 'red')}">{health_overall}</div></div>
    <div class="card"><div class="label">Health Checks</div><div class="value blue">{len(health_checks)}</div></div>
</div>

<div class="section">
    <h2>Health Checks</h2>
    <div class="table-wrapper">
    {"<table><thead><tr><th></th><th>Check</th><th>Status</th><th>Details</th></tr></thead><tbody>" + health_rows + "</tbody></table>" if health_rows else '<div class="empty-state">No health data</div>'}
    </div>
</div>
</div>

<script>
function switchTab(e, tabId) {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    e.target.classList.add('active');
    document.getElementById(tabId).classList.add('active');
}}

// Equity curve chart (pure canvas, no dependencies)
(function() {{
    const data = {eq_js};
    if (!data.length) return;
    const canvas = document.getElementById('equityChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.scale(dpr, dpr);

    const w = rect.width, h = rect.height;
    const pad = {{t: 20, r: 20, b: 30, l: 70}};
    const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

    const min = Math.min(...data), max = Math.max(...data);
    const range = max - min || 1;

    function x(i) {{ return pad.l + (i / (data.length - 1)) * cw; }}
    function y(v) {{ return pad.t + ch - ((v - min) / range) * ch; }}

    // Grid
    ctx.strokeStyle = '#2a2d3a'; ctx.lineWidth = 0.5;
    for (let i = 0; i < 5; i++) {{
        const yy = pad.t + (ch / 4) * i;
        ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
        ctx.fillStyle = '#8b8fa3'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
        const val = max - (range / 4) * i;
        ctx.fillText('$' + val.toLocaleString(undefined, {{maximumFractionDigits: 0}}), pad.l - 8, yy + 4);
    }}

    // Fill area
    ctx.beginPath();
    ctx.moveTo(x(0), y(data[0]));
    for (let i = 1; i < data.length; i++) ctx.lineTo(x(i), y(data[i]));
    ctx.lineTo(x(data.length - 1), pad.t + ch);
    ctx.lineTo(x(0), pad.t + ch);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ch);
    const endColor = data[data.length-1] >= data[0] ? '0,200,83' : '255,23,68';
    grad.addColorStop(0, 'rgba(' + endColor + ',0.3)');
    grad.addColorStop(1, 'rgba(' + endColor + ',0.02)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(x(0), y(data[0]));
    for (let i = 1; i < data.length; i++) ctx.lineTo(x(i), y(data[i]));
    ctx.strokeStyle = data[data.length-1] >= data[0] ? '#00c853' : '#ff1744';
    ctx.lineWidth = 2;
    ctx.stroke();
}})();

setTimeout(() => location.reload(), 60000);
</script>
</body></html>"""
    return html


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = build_html()
            self._send(200, "text/html", html.encode())
        elif self.path == "/api/alerts":
            self._json(load_alerts())
        elif self.path == "/api/portfolio":
            self._json(load_portfolio())
        elif self.path == "/api/paper":
            self._json(load_paper_state())
        elif self.path == "/api/positions":
            self._json(load_positions())
        elif self.path == "/api/performance":
            self._json(load_risk_metrics())
        elif self.path == "/api/journal":
            self._json(load_journal_entries())
        elif self.path == "/api/journal/stats":
            self._json(load_journal_stats())
        elif self.path == "/api/rebalance":
            self._json(load_rebalance())
        elif self.path == "/api/regime":
            self._json(load_regime())
        elif self.path == "/api/briefing":
            self._json(load_briefing())
        elif self.path == "/api/attribution":
            self._json(load_attribution())
        elif self.path.startswith("/api/monte-carlo"):
            self._json(load_monte_carlo())
        elif self.path == "/api/price-alerts":
            self._json(load_price_alerts())
        elif self.path == "/api/health":
            self._json(load_health())
        elif self.path.startswith("/api/eod-report"):
            # /api/eod-report?date=2026-03-25 or /api/eod-report
            report_date = None
            if "?" in self.path:
                params = dict(p.split("=") for p in self.path.split("?")[1].split("&") if "=" in p)
                report_date = params.get("date")
            self._json(load_eod_report(report_date))
        elif self.path == "/api/eod-reports":
            self._json(load_eod_report_list())
        elif self.path == "/api/optimizer":
            self._json(load_optimizer())
        elif self.path == "/api/watchlist-suggestions":
            self._json(load_watchlist_suggestions())
        else:
            self.send_error(404)

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data):
        body = json.dumps(data, indent=2, default=str).encode()
        self._send(200, "application/json", body)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Stock Agent Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to serve on")
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
