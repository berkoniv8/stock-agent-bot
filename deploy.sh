#!/bin/bash
# deploy.sh — Deploy stock agent to Fly.io (free tier)
# Usage: bash deploy.sh

set -e

echo "=========================================="
echo "  Stock Agent — Cloud Deployment (Fly.io)"
echo "=========================================="
echo ""

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "Installing Fly.io CLI..."
    curl -L https://fly.io/install.sh | sh
    export FLYCTL_INSTALL="$HOME/.fly"
    export PATH="$FLYCTL_INSTALL/bin:$PATH"
    echo ""
    echo "✅ Fly CLI installed"
fi

# Check if logged in
if ! flyctl auth whoami &> /dev/null 2>&1; then
    echo "📝 You need a Fly.io account (free). Opening signup..."
    echo ""
    flyctl auth signup
    echo ""
fi

# Load env vars from .env
if [ -f .env ]; then
    echo "📋 Loading secrets from .env..."

    # Extract key secrets
    TELEGRAM_BOT_TOKEN=$(grep "^TELEGRAM_BOT_TOKEN=" .env | cut -d'=' -f2-)
    TELEGRAM_CHAT_ID=$(grep "^TELEGRAM_CHAT_ID=" .env | cut -d'=' -f2-)
    SMTP_HOST=$(grep "^SMTP_HOST=" .env | cut -d'=' -f2-)
    SMTP_PORT=$(grep "^SMTP_PORT=" .env | cut -d'=' -f2-)
    SMTP_USER=$(grep "^SMTP_USER=" .env | cut -d'=' -f2-)
    SMTP_PASSWORD=$(grep "^SMTP_PASSWORD=" .env | cut -d'=' -f2-)
    ALERT_EMAIL_TO=$(grep "^ALERT_EMAIL_TO=" .env | cut -d'=' -f2-)
    SMS_GATEWAY=$(grep "^SMS_GATEWAY=" .env | cut -d'=' -f2-)
    NEWSAPI_KEY=$(grep "^NEWSAPI_KEY=" .env | cut -d'=' -f2-)
    FINNHUB_API_KEY=$(grep "^FINNHUB_API_KEY=" .env | cut -d'=' -f2-)

    echo "✅ Secrets loaded"
else
    echo "❌ No .env file found! Create one first."
    exit 1
fi

# Check if app exists
APP_NAME="stock-agent-bot"
if ! flyctl apps list 2>/dev/null | grep -q "$APP_NAME"; then
    echo ""
    echo "🚀 Creating Fly.io app: $APP_NAME"
    flyctl apps create "$APP_NAME" --machines
    echo "✅ App created"
fi

# Set secrets (only non-empty ones)
echo ""
echo "🔐 Setting secrets..."
SECRETS_CMD="flyctl secrets set -a $APP_NAME"

[ -n "$TELEGRAM_BOT_TOKEN" ] && SECRETS_CMD="$SECRETS_CMD TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN"
[ -n "$TELEGRAM_CHAT_ID" ] && SECRETS_CMD="$SECRETS_CMD TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID"
[ -n "$SMTP_HOST" ] && SECRETS_CMD="$SECRETS_CMD SMTP_HOST=$SMTP_HOST"
[ -n "$SMTP_PORT" ] && SECRETS_CMD="$SECRETS_CMD SMTP_PORT=$SMTP_PORT"
[ -n "$SMTP_USER" ] && SECRETS_CMD="$SECRETS_CMD SMTP_USER=$SMTP_USER"
[ -n "$SMTP_PASSWORD" ] && SECRETS_CMD="$SECRETS_CMD SMTP_PASSWORD=$SMTP_PASSWORD"
[ -n "$ALERT_EMAIL_TO" ] && SECRETS_CMD="$SECRETS_CMD ALERT_EMAIL_TO=$ALERT_EMAIL_TO"
[ -n "$SMS_GATEWAY" ] && SECRETS_CMD="$SECRETS_CMD SMS_GATEWAY=$SMS_GATEWAY"
[ -n "$NEWSAPI_KEY" ] && SECRETS_CMD="$SECRETS_CMD NEWSAPI_KEY=$NEWSAPI_KEY"
[ -n "$FINNHUB_API_KEY" ] && SECRETS_CMD="$SECRETS_CMD FINNHUB_API_KEY=$FINNHUB_API_KEY"

eval $SECRETS_CMD
echo "✅ Secrets set"

# Deploy
echo ""
echo "🚀 Deploying to Fly.io..."
echo "   (This may take 2-3 minutes on first deploy)"
echo ""
flyctl deploy -a "$APP_NAME" --wait-timeout 300

echo ""
echo "=========================================="
echo "  ✅ DEPLOYED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Your stock agent is now running 24/7 in the cloud!"
echo ""
echo "📊 Check status:  flyctl status -a $APP_NAME"
echo "📋 View logs:     flyctl logs -a $APP_NAME"
echo "🔄 Redeploy:      bash deploy.sh"
echo "⏹  Stop:          flyctl scale count 0 -a $APP_NAME"
echo "▶️  Start:         flyctl scale count 1 -a $APP_NAME"
echo ""
echo "The bot will answer your Telegram messages 24/7,"
echo "even when your Mac is off! 🎉"
