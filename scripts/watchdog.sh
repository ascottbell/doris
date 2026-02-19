#!/bin/bash
#
# Doris Watchdog
#
# External health check that runs via launchd every 5 minutes.
# Authenticates with DORIS_API_TOKEN, checks /status endpoint, parses with jq,
# alerts via ntfy.sh. Completely independent of Doris — if Doris is down, this still runs.
#

DORIS_URL="http://localhost:8000/status"
DORIS_TOKEN="${DORIS_API_TOKEN:-}"
NTFY_TOPIC="${DORIS_NTFY_TOPIC:-}"
NTFY_TOKEN="${DORIS_NTFY_TOKEN:-}"  # ntfy access token for authenticated publishing
DORIS_STATE_DIR="${HOME}/.doris"
mkdir -p "$DORIS_STATE_DIR"
ALERT_COOLDOWN_FILE="${DORIS_STATE_DIR}/watchdog-last-alert"
COOLDOWN_SECONDS=1800  # 30 minutes between alerts

# Require API token to authenticate against /status endpoint
if [ -z "$DORIS_TOKEN" ]; then
    echo "$(date): ERROR: DORIS_API_TOKEN is not set. Watchdog cannot authenticate with /status."
    echo "  Set DORIS_API_TOKEN to the same token used by the Doris server."
    exit 1
fi

# Require explicit topic — refuse to run with no topic set
if [ -z "$NTFY_TOPIC" ]; then
    echo "$(date): ERROR: DORIS_NTFY_TOPIC is not set. Refusing to run."
    echo "  Set DORIS_NTFY_TOPIC to a unique, unguessable topic name."
    echo "  Example: export DORIS_NTFY_TOPIC=doris-$(openssl rand -hex 12)"
    exit 1
fi

# Check if we're within cooldown period
should_alert() {
    if [ ! -f "$ALERT_COOLDOWN_FILE" ]; then
        return 0  # No previous alert, go ahead
    fi
    local last_alert
    last_alert=$(cat "$ALERT_COOLDOWN_FILE" 2>/dev/null || echo 0)
    local now
    now=$(date +%s)
    local elapsed=$((now - last_alert))
    [ "$elapsed" -ge "$COOLDOWN_SECONDS" ]
}

send_alert() {
    local title="$1"
    local message="$2"
    local priority="${3:-high}"

    if ! should_alert; then
        echo "$(date): Alert suppressed (cooldown): $title"
        return
    fi

    echo "$(date): ALERTING: $title — $message"

    local auth_args=()
    if [ -n "$NTFY_TOKEN" ]; then
        auth_args=(-H "Authorization: Bearer $NTFY_TOKEN")
    fi

    curl -s \
        "${auth_args[@]}" \
        -H "Title: $title" \
        -H "Priority: $priority" \
        -H "Tags: warning" \
        -d "$message" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1

    date +%s > "$ALERT_COOLDOWN_FILE"
}

# Fetch status with timeout (authenticated — /status requires Bearer token)
response=$(curl -s --max-time 10 -H "Authorization: Bearer $DORIS_TOKEN" "$DORIS_URL" 2>&1)
curl_exit=$?

# Server unreachable
if [ $curl_exit -ne 0 ]; then
    send_alert "Doris Server Down" "Cannot reach $DORIS_URL (curl exit: $curl_exit)" "urgent"
    exit 1
fi

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "$(date): jq not installed, cannot parse status"
    exit 1
fi

# Parse response
overall=$(echo "$response" | jq -r '.overall // "unknown"' 2>/dev/null)
daemon_status=$(echo "$response" | jq -r '.subsystems.daemon.status // "unknown"' 2>/dev/null)
scouts_failing=$(echo "$response" | jq -r '.scouts.failing // 0' 2>/dev/null)

# Check overall status
if [ "$overall" = "critical" ]; then
    send_alert "Doris Critical" "Overall status is CRITICAL. Daemon: $daemon_status" "urgent"
    exit 1
fi

# Check daemon
if [ "$daemon_status" = "down" ] || [ "$daemon_status" = "unknown" ]; then
    send_alert "Doris Daemon Down" "Daemon status: $daemon_status" "high"
    exit 1
fi

# Check scouts
if [ "$scouts_failing" -gt 0 ]; then
    scout_details=$(echo "$response" | jq -r '.scouts.details[] | "\(.name): \(.last_error // "unknown error") (\(.consecutive_failures) failures)"' 2>/dev/null)
    send_alert "Doris Scouts Failing" "$scouts_failing scout(s) failing:\n$scout_details" "high"
    exit 1
fi

# Check push token health
push_health=$(echo "$response" | jq -r '.subsystems.push.health // "unknown"' 2>/dev/null)
if [ "$push_health" = "critical" ]; then
    send_alert "Doris Push Broken" "No active device tokens — notifications won't be delivered. Open the iOS app to re-register." "high"
    exit 1
fi

# Check degraded
if [ "$overall" = "degraded" ]; then
    send_alert "Doris Degraded" "Overall status is degraded. Check /status for details." "default"
    exit 0
fi

echo "$(date): OK — overall=$overall, daemon=$daemon_status, failing_scouts=$scouts_failing"
