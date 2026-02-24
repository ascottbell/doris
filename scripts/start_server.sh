#!/bin/bash
#
# Doris Server Startup Wrapper
#
# Called by launchd before uvicorn starts.
# Kills any orphan process on port 8000 and waits for the port to be free.
#
# Update DORIS_ROOT to match your installation path.
#

DORIS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8000
MAX_WAIT=15
LOG="$DORIS_ROOT/logs/server.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [start_server] $1" >> "$LOG"
}

# Ensure log directory exists
mkdir -p "$(dirname "$LOG")"

# Kill anything already on the port
pids=$(lsof -ti :$PORT 2>/dev/null)
if [ -n "$pids" ]; then
    log "Found orphan process(es) on port $PORT: $pids — sending SIGTERM"
    echo "$pids" | xargs kill 2>/dev/null

    # Wait for port to clear
    waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        if ! lsof -ti :$PORT >/dev/null 2>&1; then
            log "Port $PORT free after ${waited}s"
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    # Force kill if still holding
    pids=$(lsof -ti :$PORT 2>/dev/null)
    if [ -n "$pids" ]; then
        log "Port $PORT still held after ${MAX_WAIT}s — sending SIGKILL: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null
        sleep 1
    fi
fi

# Final check
if lsof -ti :$PORT >/dev/null 2>&1; then
    log "FATAL: Port $PORT still in use after cleanup. Aborting."
    exit 1
fi

log "Port $PORT is free — starting uvicorn"

# Exec replaces this shell with uvicorn (launchd tracks the uvicorn PID directly)
exec "$DORIS_ROOT/.venv/bin/python" -m uvicorn main:app --host 0.0.0.0 --port $PORT
