#!/usr/bin/env bash
# Stock Investment Agent — Setup Script
set -e

echo "=== Stock Agent Setup ==="

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading NLTK data for TextBlob..."
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

# Create logs directory
mkdir -p logs

# Create .env from example if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

# Install test dependencies
pip install pytest

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys (optional — works without them via yfinance)"
echo "  2. Edit watchlist.csv with your tickers"
echo "  3. Edit portfolio.json with your portfolio details"
echo ""
echo "Run commands:"
echo "  source .venv/bin/activate"
echo "  python3 cli.py scan                 # Single scan"
echo "  python3 cli.py scan --ticker AAPL   # Single ticker"
echo "  python3 cli.py scan --paper         # Paper trading scan"
echo "  python3 cli.py schedule --paper     # Scheduled mode"
echo "  python3 cli.py dashboard            # Web dashboard on :8050"
echo "  python3 cli.py backtest             # Run backtester"
echo "  make test                           # Run tests"
echo "  make help                           # Show all commands"
