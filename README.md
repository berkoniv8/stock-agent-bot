# Stock Investment Agent

A Python-based stock analysis and paper trading system. Scans a watchlist on a schedule, runs technical + fundamental analysis through a signal confluence engine, sizes positions with risk management, and sends alerts via multiple channels.

Includes a 7-tab web dashboard, backtesting engine, Monte Carlo simulation, and full paper trading with a virtual portfolio.

## Quick Start

```bash
# 1. Setup
bash setup.sh
source .venv/bin/activate

# 2. Configure
# Edit .env with your API keys (optional — works without them via yfinance)
# Edit watchlist.csv with your tickers
# Edit portfolio.json with your holdings

# 3. Run
python3 cli.py scan              # Single scan
python3 cli.py scan --paper      # Paper trading scan
python3 cli.py dashboard         # Web dashboard at localhost:8050
```

Or use the Makefile:

```bash
make help        # Show all commands
make scan        # Single scan
make paper       # Paper trading scan
make dashboard   # Start web dashboard
make test        # Run tests
```

## Architecture

```
watchlist.csv ─→ data_layer ─→ technical_analysis ─┐
                              fundamental_analysis ─┤
                              multi_timeframe ──────┤
                              market_regime ────────┤
                              sector_rotation ──────┤
                              screener ─────────────┤
                                                    ▼
                                             signal_engine
                                                    │
                              earnings_guard ───────┤
                              correlation_guard ────┤
                                                    ▼
                                            position_sizing
                                                    │
                              paper_trader ─────────┤
                              notifications ────────┤
                              trade_journal ────────┤
                              alert_tracker ────────┘
```

## Modules (37)

### Core

| Module | Description |
|--------|-------------|
| `agent.py` | Main runner with scheduling and market-hours gating |
| `cli.py` | Unified CLI with 18 subcommands |
| `data_layer.py` | Price, OHLCV, and news data fetching (yfinance + optional APIs) |
| `database.py` | SQLite persistence layer |
| `config_validator.py` | Startup configuration health checks |
| `dashboard.py` | 7-tab web dashboard served at localhost:8050 |

### Signal Generation

| Module | Description |
|--------|-------------|
| `signal_engine.py` | Confluence engine combining all signal sources |
| `technical_analysis.py` | EMAs, SMA, RSI, MACD, Bollinger, patterns, Fibonacci |
| `fundamental_analysis.py` | 6-criteria fundamental scoring |
| `multi_timeframe.py` | Intraday (1h) confirmation of daily signals |
| `screener.py` | Broader market scanning beyond the watchlist |
| `sector_rotation.py` | Leading and lagging sector identification |
| `market_regime.py` | Bull / bear / sideways classification |

### Risk & Portfolio

| Module | Description |
|--------|-------------|
| `position_sizing.py` | Entry, stop-loss, take-profit, and share count |
| `correlation_guard.py` | Prevents over-concentration in correlated positions |
| `earnings_guard.py` | Blocks trades near earnings dates |
| `risk_monitor.py` | Exposure, sector concentration, correlation matrix |
| `risk_metrics.py` | Sharpe, Sortino, max drawdown, Kelly criterion |
| `trailing_stop.py` | Dynamic stop-loss management for open positions |
| `rebalancer.py` | Portfolio weight rebalancing suggestions |

### Trading & Tracking

| Module | Description |
|--------|-------------|
| `paper_trader.py` | Virtual portfolio with simulated executions |
| `trade_journal.py` | Structured trade logging with tags and notes |
| `alert_tracker.py` | Alert deduplication and cooldown logic |
| `price_alerts.py` | Price level monitoring and breach alerts |
| `watchlist_cli.py` | Watchlist CRUD operations |
| `watchlist_curator.py` | Auto-suggest watchlist additions and removals |

### Analytics & Reporting

| Module | Description |
|--------|-------------|
| `backtester.py` | Historical replay backtesting engine |
| `performance_attribution.py` | P&L breakdown by sector, signal, and direction |
| `performance_tracker.py` | Past alert accuracy tracking |
| `signal_analytics.py` | Win rate and profit factor by signal type |
| `strategy_optimizer.py` | Auto-tune signal weights from trade history |
| `monte_carlo.py` | Equity curve projections via simulation |
| `daily_briefing.py` | Morning portfolio summary |
| `eod_report.py` | End-of-day P&L report |
| `weekly_report.py` | Weekly digest |
| `health_monitor.py` | System health checks (APIs, disk, data freshness) |
| `notifications.py` | Email, Slack, Telegram, Discord, and webhook delivery |

## CLI Commands

```
python3 cli.py scan [--paper] [--ticker AAPL]   Scan watchlist or single ticker
python3 cli.py schedule --paper                  Run on recurring schedule
python3 cli.py dashboard [--port 8050]           Start web dashboard
python3 cli.py backtest [--ticker AAPL]          Run backtester
python3 cli.py briefing [--send]                 Daily briefing
python3 cli.py eod [--save] [--send]             End-of-day report
python3 cli.py journal list|stats|export         Trade journal
python3 cli.py alerts list|add|remove|check      Price alerts
python3 cli.py watchlist list|add|remove|curate  Watchlist management
python3 cli.py regime                            Market regime detection
python3 cli.py health [--save]                   System health check
python3 cli.py attribution [--json]              Performance attribution
python3 cli.py optimize                          Strategy optimizer
python3 cli.py risk                              Risk-adjusted metrics
python3 cli.py rebalance                         Portfolio rebalancer
python3 cli.py monte-carlo [--sims 1000]         Monte Carlo simulation
python3 cli.py performance [--ticker AAPL]       Alert performance tracker
python3 cli.py validate                          Config validator
```

## Web Dashboard

Start with `python3 cli.py dashboard` and open `http://localhost:8050`.

**7 tabs:**
- **Overview** — Portfolio summary, holdings, recent signals
- **Paper Trading** — Virtual portfolio, open positions, trade history
- **Performance** — Equity curve, P&L attribution by sector/direction
- **Analytics** — Market regime, signal performance, exit reasons
- **Journal** — Trade journal entries and review coverage
- **Alerts** — Price alerts (active + position-based)
- **System** — Health checks, API status, disk/config validation

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

**Required:** None. The agent works out of the box using yfinance for free market data.

**Optional API keys** (for enhanced data):
- `ALPHA_VANTAGE_API_KEY` — Additional fundamental data
- `FMP_API_KEY` — Financial Modeling Prep
- `NEWSAPI_KEY` — News sentiment analysis
- `FINNHUB_API_KEY` — Real-time data

**Notifications** (all optional):
- Email (SMTP), Slack, Telegram, Discord, custom webhook

**Key settings:**
```
SIGNAL_THRESHOLD=5              # Min score to trigger alert (1-10)
TOTAL_PORTFOLIO_VALUE=50000     # Portfolio value for position sizing
MAX_RISK_PER_TRADE_PCT=1.0      # Max risk per trade (%)
MAX_POSITION_SIZE_PCT=10.0      # Max single position (%)
RUN_INTERVAL_MINUTES=15         # Scan frequency in scheduled mode
PAPER_STARTING_CAPITAL=100000   # Paper trading starting balance
```

## Data Files

- `watchlist.csv` — Tickers to scan (ticker, sector, notes)
- `portfolio.json` — Real portfolio holdings for tracking
- `logs/stock_agent.db` — SQLite database (auto-created)
- `logs/eod_reports/` — Saved end-of-day reports

## Docker

```bash
# Web dashboard
docker-compose up dashboard

# Continuous scheduled scanning
docker-compose up scheduler

# One-off scan
docker-compose run scan

# Paper trading scan
docker-compose run paper

# Run tests
docker-compose run test
```

## Testing

```bash
make test                    # Full suite (432 tests)
make test-quick              # Quiet output
python3 -m pytest tests/ -v  # Verbose
```

34 test files covering all 37 modules. Tests use mocking — no network calls or API keys needed.

## Project Stats

- **37** Python modules (~14,500 lines)
- **34** test files (~5,400 lines, 432 tests)
- **18** CLI subcommands
- **7** dashboard tabs
- **5** notification channels
- **0** external JS/CSS dependencies (pure Python + stdlib HTML)
