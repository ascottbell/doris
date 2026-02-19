#!/bin/bash
set -euo pipefail

# Start the Doris daemon
#
# Creates a PID file at ~/.doris/daemon.pid and checks for already-running
# instances before starting. Use stop_daemon.sh or kill $(cat ~/.doris/daemon.pid)
# to stop.

# Change to the Doris project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DORIS_STATE_DIR="${HOME}/.doris"
PID_FILE="${DORIS_STATE_DIR}/daemon.pid"

mkdir -p "$DORIS_STATE_DIR"
mkdir -p logs

# Check if daemon is already running
if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Daemon already running (PID $old_pid). Stop it first or remove $PID_FILE."
        exit 1
    else
        echo "Stale PID file found (PID $old_pid no longer running). Cleaning up."
        rm -f "$PID_FILE"
    fi
fi

# Start daemon in background, redirecting output to logs
nohup .venv/bin/python daemon.py > logs/daemon.log 2> logs/daemon.error.log &
DAEMON_PID=$!

# Write PID file
echo "$DAEMON_PID" > "$PID_FILE"

echo "Daemon started with PID $DAEMON_PID (PID file: $PID_FILE)"
