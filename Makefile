.PHONY: help setup test run scan paper dashboard backtest briefing eod health regime alerts journal optimize clean volume upgrades performance

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and set up environment
	bash setup.sh

test: ## Run all tests
	python3 -m pytest tests/ -v --tb=short

test-quick: ## Run tests without verbose output
	python3 -m pytest tests/ --tb=short -q

scan: ## Run a single scan cycle
	python3 cli.py scan

paper: ## Run a paper trading scan
	python3 cli.py scan --paper

schedule: ## Start scheduled scanning (paper mode)
	python3 cli.py schedule --paper

dashboard: ## Start web dashboard on port 8050
	python3 cli.py dashboard

backtest: ## Backtest full watchlist
	python3 cli.py backtest

briefing: ## Generate daily briefing
	python3 cli.py briefing

eod: ## Generate end-of-day report
	python3 cli.py eod

health: ## Run system health check
	python3 cli.py health

regime: ## Detect current market regime
	python3 cli.py regime

alerts: ## List price alerts
	python3 cli.py alerts list

journal: ## Show trade journal
	python3 cli.py journal list

optimize: ## Run strategy optimizer
	python3 cli.py optimize

attribution: ## Performance attribution report
	python3 cli.py attribution

risk: ## Risk-adjusted metrics
	python3 cli.py risk

rebalance: ## Portfolio rebalancer
	python3 cli.py rebalance

monte-carlo: ## Monte Carlo simulation
	python3 cli.py monte-carlo

validate: ## Run config validator
	python3 cli.py validate

watchlist: ## List watchlist
	python3 cli.py watchlist list

sell-check: ## Check which positions to sell
	python3 position_monitor.py

sell-notify: ## Check positions and send sell alerts
	python3 position_monitor.py --notify

buy-scan: ## Scan full watchlist for BUY opportunities
	python3 cli.py scan --buy-only

earnings: ## Upcoming earnings for your holdings
	python3 earnings_calendar.py

news: ## Latest significant news on your holdings
	python3 news_monitor.py

tax: ## Tax loss harvesting opportunities
	python3 tax_harvesting.py

rotation: ## Sector rotation analysis
	python3 sector_rotation.py

dca: ## DCA suggestions for long-term positions
	python3 dca_advisor.py

weekly: ## Generate weekly portfolio report
	python3 weekly_report.py

ib-sync: ## Sync portfolio from Interactive Brokers (read-only)
	python3 ib_sync.py

ib-status: ## Check IB connection status
	python3 ib_sync.py --status

performance: ## Portfolio performance history
	python3 performance_tracker.py

telegram: ## Start Telegram bot (two-way chat)
	python3 telegram_bot.py

telegram-setup: ## Get your Telegram chat ID
	python3 telegram_bot.py --get-chat-id

options: ## Options analysis for a ticker (usage: make options TICKER=TSLA)
	python3 options_analyzer.py $(TICKER)

volume: ## Scan for unusual volume spikes
	python3 volume_scanner.py

upgrades: ## Check analyst upgrades/downgrades
	python3 upgrade_tracker.py

clean: ## Remove log files and caches
	rm -rf __pycache__ tests/__pycache__ logs/*.log
	find . -name '*.pyc' -delete
