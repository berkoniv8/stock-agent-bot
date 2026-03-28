#!/bin/bash
# stop_local.sh — Stop the local bot + scheduler (before cloud deploy)
echo "Stopping local stock agent services..."

launchctl unload ~/Library/LaunchAgents/com.stockagent.telegram.plist 2>/dev/null
launchctl unload ~/Library/LaunchAgents/com.stockagent.scheduler.plist 2>/dev/null
launchctl unload ~/Library/LaunchAgents/com.stockagent.nosleep.plist 2>/dev/null

sleep 2
pkill -f "telegram_bot.py" 2>/dev/null
pkill -f "agent.py --schedule" 2>/dev/null

echo "✅ Local services stopped."
echo ""
echo "To restart local services later:"
echo "  launchctl load ~/Library/LaunchAgents/com.stockagent.telegram.plist"
echo "  launchctl load ~/Library/LaunchAgents/com.stockagent.scheduler.plist"
