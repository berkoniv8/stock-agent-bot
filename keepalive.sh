#!/bin/bash
# keepalive.sh — Wrapper that prevents macOS from sleeping while running a command.
# Usage: keepalive.sh <command> [args...]
#
# Uses caffeinate -i -s to prevent both idle sleep and system sleep,
# while still allowing the display to sleep (saves energy).
# The -w $$ flag ties caffeinate to this script's PID so it auto-stops on exit.

exec caffeinate -i -s -w $$ &
exec "$@"
