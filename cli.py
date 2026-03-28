#!/usr/bin/env python3
"""
Stock Agent CLI — unified command-line interface for all tools.

Usage:
    python3 cli.py scan                    # Run single scan
    python3 cli.py scan --paper            # Paper trading scan
    python3 cli.py schedule --paper        # Run on schedule
    python3 cli.py dashboard               # Start web dashboard
    python3 cli.py backtest                # Backtest full watchlist
    python3 cli.py backtest --ticker AAPL  # Backtest single ticker
    python3 cli.py journal list            # View trade journal
    python3 cli.py journal stats           # Journal statistics
    python3 cli.py alerts list             # List price alerts
    python3 cli.py alerts add AAPL above 180
    python3 cli.py alerts check            # Check all alerts
    python3 cli.py watchlist list          # List watchlist
    python3 cli.py watchlist add TSLA Tech
    python3 cli.py regime                  # Current market regime
    python3 cli.py health                  # System health check
    python3 cli.py briefing                # Daily briefing
    python3 cli.py eod                     # End-of-day report
    python3 cli.py attribution             # Performance attribution
    python3 cli.py optimize                # Strategy optimizer
    python3 cli.py monte-carlo             # Monte Carlo simulation
    python3 cli.py risk                    # Risk metrics
    python3 cli.py rebalance               # Portfolio rebalancer
    python3 cli.py performance             # Performance tracker
    python3 cli.py validate                # Config validator
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def cmd_scan(args):
    import agent
    if args.ticker:
        agent.scan_ticker(args.ticker, paper_mode=args.paper)
    else:
        agent.run_scan(paper_mode=args.paper)
    if args.paper:
        import paper_trader
        paper_trader.print_status(paper_trader.load_state())


def cmd_schedule(args):
    import agent
    agent.main()


def cmd_dashboard(args):
    import dashboard
    from http.server import HTTPServer
    port = args.port
    server = HTTPServer(("0.0.0.0", port), dashboard.DashboardHandler)
    print("Dashboard running at http://localhost:%d" % port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


def cmd_backtest(args):
    import backtester
    import data_layer
    if args.ticker:
        tickers = [{"ticker": args.ticker, "sector": "Technology"}]
    else:
        tickers = data_layer.load_watchlist()
    results = []
    for entry in tickers:
        try:
            r = backtester.backtest_ticker(
                entry["ticker"], period=args.period,
                threshold=args.threshold,
                sector=entry.get("sector", "Technology"),
            )
            results.append(r)
            backtester.print_report(r)
        except Exception as e:
            print("  Error backtesting %s: %s" % (entry["ticker"], e))
    if results:
        backtester.save_results_csv(results)


def cmd_journal(args):
    import trade_journal
    if args.subcmd == "list":
        entries = trade_journal.get_entries(
            ticker=args.ticker, limit=args.limit or 20)
        if not entries:
            print("\n  No journal entries.\n")
            return
        print("\n  TRADE JOURNAL")
        print("  " + "-" * 60)
        for e in entries:
            status = "CLOSED" if e.get("closed") else "OPEN"
            pnl = float(e.get("pnl", 0) or 0)
            tags = ", ".join(e.get("tags", []))
            print("  %-6s %4s  %-7s  P&L: $%+8.2f  %s  %s" % (
                e.get("ticker", ""), e.get("direction", ""), status,
                pnl, tags, str(e.get("created_at", ""))[:10]))
        print()
    elif args.subcmd == "stats":
        stats = trade_journal.get_stats()
        print("\n  JOURNAL STATS")
        print("  " + "-" * 40)
        for k, v in stats.items():
            if isinstance(v, dict):
                print("  %s:" % k)
                for kk, vv in v.items():
                    print("    %-20s %s" % (kk, vv))
            else:
                print("  %-20s %s" % (k, v))
        print()
    elif args.subcmd == "export":
        path = trade_journal.export_csv(args.output or "journal_export.csv")
        print("  Exported to %s" % path)


def cmd_alerts(args):
    import price_alerts
    if args.subcmd == "list":
        price_alerts.print_alerts()
    elif args.subcmd == "add":
        price_alerts.add_alert(args.ticker, args.condition, float(args.price), args.note or "")
        print("  Alert added: %s %s $%s" % (args.ticker.upper(), args.condition, args.price))
    elif args.subcmd == "remove":
        if price_alerts.remove_alert(args.index):
            print("  Alert removed.")
        else:
            print("  Invalid alert index.")
    elif args.subcmd == "check":
        print("\n  Checking price alerts...")
        triggered = price_alerts.check_all_alerts()
        if triggered:
            print("  %d alerts triggered:" % len(triggered))
            price_alerts.send_triggered_alerts(triggered)
        else:
            print("  No alerts triggered.")
        print()


def cmd_watchlist(args):
    import watchlist_cli
    if args.subcmd == "list":
        watchlist_cli.cmd_list(args)
    elif args.subcmd == "add":
        watchlist_cli.cmd_add(args)
    elif args.subcmd == "remove":
        watchlist_cli.cmd_remove(args)
    elif args.subcmd == "validate":
        watchlist_cli.cmd_validate(args)
    elif args.subcmd == "curate":
        import watchlist_curator
        wl = watchlist_curator.load_watchlist()
        tickers = [w["ticker"] for w in wl]
        additions = watchlist_curator.suggest_additions(tickers, {}, {})
        removals = watchlist_curator.suggest_removals(wl, {}, {}, {})
        if additions:
            print("\n  Suggested additions:")
            for a in additions:
                print("    + %s (%s) score: %.1f" % (a["ticker"], a.get("sector", ""), a.get("score", 0)))
        if removals:
            print("\n  Suggested removals:")
            for r in removals:
                print("    - %s: %s" % (r["ticker"], r.get("reason", "")))
        if not additions and not removals:
            print("\n  No watchlist changes suggested.")
        print()


def cmd_regime(args):
    import market_regime
    result = market_regime.detect_regime()
    market_regime.print_regime(result)


def cmd_health(args):
    import health_monitor
    results = health_monitor.run_all_checks()
    health_monitor.print_health_report(results)
    if args.save:
        health_monitor.save_health_report(results)


def cmd_briefing(args):
    import daily_briefing
    briefing = daily_briefing.generate_briefing()
    text = daily_briefing.format_briefing(briefing)
    print(text)
    if args.send:
        daily_briefing.send_briefing(text)


def cmd_eod(args):
    import eod_report
    if args.list:
        reports = eod_report.list_reports()
        if reports:
            print("\n  Saved EOD reports:")
            for r in reports[:20]:
                print("    %s" % r)
        else:
            print("\n  No saved reports.")
        print()
        return
    report = eod_report.generate_report(args.date)
    if args.save:
        eod_report.save_report(report)
    if args.send:
        eod_report.send_report(report)
    else:
        print(eod_report.format_report(report))


def cmd_attribution(args):
    import performance_attribution
    result = performance_attribution.full_attribution()
    if args.json:
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        performance_attribution.print_attribution(result)


def cmd_optimize(args):
    import strategy_optimizer
    trades = strategy_optimizer.load_all_trades()
    if len(trades) < 5:
        print("\n  Not enough trades to optimize (need 5+, have %d).\n" % len(trades))
        return
    perf = strategy_optimizer.compute_signal_performance(trades)
    suggestions = strategy_optimizer.suggest_weight_adjustments(perf)
    if suggestions:
        optimized = strategy_optimizer.compute_optimized_weights(suggestions)
        strategy_optimizer.save_weights(optimized)
        print("\n  STRATEGY OPTIMIZATION")
        print("  " + "-" * 50)
        for sig, data in suggestions.items():
            print("  %-28s weight: %d → %d  (WR: %.0f%%, PF: %.1f)" % (
                sig, data.get("current_weight", 0), data.get("suggested_weight", 0),
                data.get("win_rate", 0), data.get("profit_factor", 0)))
        print()
    else:
        print("\n  No weight adjustments suggested.\n")


def cmd_monte_carlo(args):
    import monte_carlo
    pnls, capital = monte_carlo.load_historical_pnls()
    if not pnls:
        print("\n  No trade data for Monte Carlo simulation.\n")
        return
    result = monte_carlo.run_simulation(
        pnls, capital, n_simulations=args.sims, seed=args.seed)
    monte_carlo.print_simulation(result)


def cmd_risk(args):
    import risk_metrics
    pnls, capital = risk_metrics.load_paper_pnls()
    if not pnls:
        pnls, capital = risk_metrics.load_backtest_pnls()
    if not pnls:
        print("\n  No trade data for risk metrics.\n")
        return
    metrics = risk_metrics.compute_metrics(pnls, starting_capital=capital)
    risk_metrics.print_metrics(metrics)


def cmd_rebalance(args):
    import rebalancer
    result = rebalancer.full_analysis()
    rebalancer.print_analysis(result)


def cmd_performance(args):
    import performance_tracker
    alerts = performance_tracker.load_alerts()
    if args.ticker:
        alerts = [a for a in alerts if a.get("ticker", "").upper() == args.ticker.upper()]
    if not alerts:
        print("  No alerts to evaluate.")
        return
    print("  Evaluating %d alerts..." % len(alerts))
    results = [performance_tracker.evaluate_alert(a) for a in alerts]
    performance_tracker.print_performance(results)


def cmd_validate(args):
    import config_validator
    config_validator.run_all()


def main():
    parser = argparse.ArgumentParser(
        description="Stock Agent — unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # scan
    p = sub.add_parser("scan", help="Run a single scan cycle")
    p.add_argument("--paper", action="store_true", help="Paper trading mode")
    p.add_argument("--ticker", type=str, help="Single ticker to scan")

    # schedule
    p = sub.add_parser("schedule", help="Run on recurring schedule")
    p.add_argument("--paper", action="store_true", help="Paper trading mode")

    # dashboard
    p = sub.add_parser("dashboard", help="Start web dashboard")
    p.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")

    # backtest
    p = sub.add_parser("backtest", help="Run backtester")
    p.add_argument("--ticker", type=str, help="Single ticker")
    p.add_argument("--period", type=str, default="2y", help="Lookback period")
    p.add_argument("--threshold", type=int, default=5, help="Signal threshold")

    # journal
    p = sub.add_parser("journal", help="Trade journal")
    jp = p.add_subparsers(dest="subcmd")
    jp.add_parser("list", help="List entries")
    jp.add_parser("stats", help="Show statistics")
    ep = jp.add_parser("export", help="Export to CSV")
    ep.add_argument("--output", type=str, help="Output file path")
    p.add_argument("--ticker", type=str, help="Filter by ticker")
    p.add_argument("--limit", type=int, help="Max entries")

    # alerts
    p = sub.add_parser("alerts", help="Price alerts")
    ap = p.add_subparsers(dest="subcmd")
    ap.add_parser("list", help="List alerts")
    ap.add_parser("check", help="Check all alerts")
    add_p = ap.add_parser("add", help="Add alert")
    add_p.add_argument("ticker", type=str)
    add_p.add_argument("condition", choices=["above", "below"])
    add_p.add_argument("price", type=str)
    add_p.add_argument("--note", type=str, default="")
    rm_p = ap.add_parser("remove", help="Remove alert by index")
    rm_p.add_argument("index", type=int)

    # watchlist
    p = sub.add_parser("watchlist", help="Watchlist management")
    wp = p.add_subparsers(dest="subcmd")
    wp.add_parser("list", help="List tickers")
    wa = wp.add_parser("add", help="Add ticker")
    wa.add_argument("ticker", type=str)
    wa.add_argument("sector", type=str, nargs="?", default="")
    wa.add_argument("notes", type=str, nargs="?", default="")
    wr = wp.add_parser("remove", help="Remove ticker")
    wr.add_argument("ticker", type=str)
    wp.add_parser("validate", help="Validate all tickers")
    wp.add_parser("curate", help="Get curation suggestions")

    # regime
    sub.add_parser("regime", help="Detect market regime")

    # health
    p = sub.add_parser("health", help="System health check")
    p.add_argument("--save", action="store_true", help="Save report to disk")

    # briefing
    p = sub.add_parser("briefing", help="Generate daily briefing")
    p.add_argument("--send", action="store_true", help="Send via notifications")

    # eod
    p = sub.add_parser("eod", help="End-of-day report")
    p.add_argument("--date", type=str, help="Report date (YYYY-MM-DD)")
    p.add_argument("--save", action="store_true", help="Save to disk")
    p.add_argument("--send", action="store_true", help="Send via notifications")
    p.add_argument("--list", action="store_true", help="List saved reports")

    # attribution
    p = sub.add_parser("attribution", help="Performance attribution")
    p.add_argument("--json", action="store_true", help="Output as JSON")

    # optimize
    sub.add_parser("optimize", help="Run strategy optimizer")

    # monte-carlo
    p = sub.add_parser("monte-carlo", help="Monte Carlo simulation")
    p.add_argument("--sims", type=int, default=1000, help="Number of simulations")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # risk
    sub.add_parser("risk", help="Risk-adjusted metrics")

    # rebalance
    sub.add_parser("rebalance", help="Portfolio rebalancer")

    # performance
    p = sub.add_parser("performance", help="Track past alert performance")
    p.add_argument("--ticker", type=str, help="Filter by ticker")

    # validate
    sub.add_parser("validate", help="Run config validator")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    for lib in ("urllib3", "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    os.makedirs("logs", exist_ok=True)

    commands = {
        "scan": cmd_scan,
        "schedule": cmd_schedule,
        "dashboard": cmd_dashboard,
        "backtest": cmd_backtest,
        "journal": cmd_journal,
        "alerts": cmd_alerts,
        "watchlist": cmd_watchlist,
        "regime": cmd_regime,
        "health": cmd_health,
        "briefing": cmd_briefing,
        "eod": cmd_eod,
        "attribution": cmd_attribution,
        "optimize": cmd_optimize,
        "monte-carlo": cmd_monte_carlo,
        "risk": cmd_risk,
        "rebalance": cmd_rebalance,
        "performance": cmd_performance,
        "validate": cmd_validate,
    }

    if args.command is None:
        parser.print_help()
        return

    fn = commands.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
