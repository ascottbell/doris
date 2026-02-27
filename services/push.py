"""
Apple Push Notification Service (APNS) integration.

Sends push notifications to the Doris iOS app.

Notification Priority Tiers:
- normal: Regular notifications (respects Do Not Disturb)
- urgent: Time Sensitive notifications (breaks through DND, requires entitlement)
- emergency: Critical Alerts (alarm sounds, requires Apple approval)

Until Critical Alerts are approved, emergency uses time-sensitive.
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime
from enum import Enum

from aioapns import APNs, NotificationRequest, PushType
from aioapns.common import APNS_RESPONSE_CODE


class NotificationPriority(str, Enum):
    """Notification priority levels mapping to iOS interruption levels."""
    NORMAL = "normal"      # Default - respects Focus/DND
    URGENT = "urgent"      # Time Sensitive - breaks through Focus (requires entitlement)
    EMERGENCY = "emergency"  # Critical Alert - alarm sound, requires Apple approval


# iOS interruption levels (maps to UNNotificationInterruptionLevel)
INTERRUPTION_LEVELS = {
    NotificationPriority.NORMAL: "active",         # Standard notification (banner + sound, respects Focus/DND)
    NotificationPriority.URGENT: "time-sensitive",  # Breaks Focus
    NotificationPriority.EMERGENCY: "time-sensitive",  # TODO: Use "critical" once Apple approves
}

# Configuration — all APNS credentials must come from environment variables
import os
from dotenv import load_dotenv
load_dotenv()

APNS_KEY_ID = os.getenv("APNS_KEY_ID", "")
APNS_TEAM_ID = os.getenv("APNS_TEAM_ID", "")
APNS_BUNDLE_ID = os.getenv("APNS_BUNDLE_ID", "com.doris.client")

# APNS key path — set APNS_KEY_PATH env var or place key in credentials/ directory
_key_filename = f"AuthKey_{APNS_KEY_ID}.p8" if APNS_KEY_ID else "AuthKey.p8"
from config import settings
APNS_KEY_PATH = Path(os.getenv("APNS_KEY_PATH", str(Path(__file__).parent.parent / "credentials" / _key_filename)))
DB_PATH = settings.data_dir / "doris.db"

# Bundle IDs per device type (APNS topic must match the receiving app's bundle ID)
BUNDLE_IDS = {
    "ios": "com.doris.client",
    "macos": "com.doris.DorisClientMac",
}

# Token environment determined by how app was built/signed
USE_SANDBOX = True  # Development builds use sandbox APNS


def _get_db():
    return sqlite3.connect(DB_PATH)


def init_push_db():
    """Create device tokens table if it doesn't exist."""
    conn = _get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS device_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            device_name TEXT,
            device_type TEXT DEFAULT 'ios',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_used TEXT DEFAULT CURRENT_TIMESTAMP,
            active INTEGER DEFAULT 1
        )
    """)

    # Migration: add device_type column if missing
    cursor.execute("PRAGMA table_info(device_tokens)")
    columns = [row[1] for row in cursor.fetchall()]
    if "device_type" not in columns:
        cursor.execute("ALTER TABLE device_tokens ADD COLUMN device_type TEXT DEFAULT 'ios'")

    conn.commit()
    conn.close()


def register_device(token: str, device_name: str = None, device_type: str = "ios") -> bool:
    """
    Register a device token for push notifications.

    Called when iOS/macOS app starts and registers with APNS.
    """
    conn = _get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO device_tokens (token, device_name, device_type, last_used)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(token) DO UPDATE SET
                device_name = excluded.device_name,
                device_type = excluded.device_type,
                last_used = excluded.last_used,
                active = 1
        """, (token, device_name, device_type, datetime.now().isoformat()))

        conn.commit()
        print(f"[push] Registered {device_type} device: {device_name or 'Unknown'}")
        return True

    except Exception as e:
        print(f"[push] Failed to register device: {e}")
        return False

    finally:
        conn.close()


def get_active_tokens() -> list[tuple[str, str]]:
    """Get all active device tokens with device type.

    Returns list of (token, device_type) tuples.
    """
    # token -> device_type mapping (deduplicates across sources)
    tokens: dict[str, str] = {}

    # Check SQLite
    try:
        conn = _get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT token, device_type FROM device_tokens WHERE active = 1")
        for row in cursor.fetchall():
            tokens[row[0]] = row[1] or "ios"
        conn.close()
    except Exception as e:
        print(f"[push] SQLite token lookup failed: {e}")

    # Also check JSON file (main.py stores tokens there)
    json_tokens_path = settings.data_dir / "device_tokens.json"
    if json_tokens_path.exists():
        try:
            with open(json_tokens_path) as f:
                json_tokens = json.load(f)
                for token, info in json_tokens.items():
                    if token not in tokens:
                        device_type = info.get("device_type", "ios") if isinstance(info, dict) else "ios"
                        tokens[token] = device_type
        except Exception as e:
            print(f"[push] JSON token lookup failed: {e}")

    return list(tokens.items())


def deactivate_token(token: str):
    """Mark a token as inactive (e.g., after APNS rejection)."""
    conn = _get_db()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE device_tokens SET active = 0 WHERE token = ?
    """, (token,))

    conn.commit()
    conn.close()


def create_apns_client(topic: str = None) -> APNs:
    """Create a fresh APNS client for the given topic (bundle ID).

    A new client is created per call to avoid stale connection/event-loop issues
    when send_push_sync runs in a separate thread with its own event loop.
    """
    topic = topic or APNS_BUNDLE_ID

    if not APNS_KEY_PATH.exists():
        raise FileNotFoundError(f"APNS key not found at {APNS_KEY_PATH}")

    with open(APNS_KEY_PATH, 'r') as f:
        key_content = f.read()

    return APNs(
        key=key_content,
        key_id=APNS_KEY_ID,
        team_id=APNS_TEAM_ID,
        topic=topic,
        use_sandbox=USE_SANDBOX,
    )


async def send_push(
    title: str,
    body: str,
    action_id: str = None,
    action_type: str = None,
    sound: bool = True,
    category: str = "PROACTIVE",
    silent: bool = False,
    data: dict = None,
    priority: NotificationPriority = NotificationPriority.NORMAL,
) -> dict:
    """
    Send a push notification to all registered devices.

    Args:
        title: Notification title
        body: Notification body text
        action_id: ID of the proactive action (for undo/dismiss)
        action_type: Type of action (create_event, notify, etc.)
        sound: Whether to play a sound
        category: Notification category for action buttons
        silent: If True, send background push (no alert, triggers app sync)
        data: Additional data to include in payload
        priority: Notification priority (normal, urgent, emergency)

    Returns:
        dict with success count and any errors
    """
    token_pairs = get_active_tokens()

    if not token_pairs:
        print("[push] No registered devices")
        return {"success": 0, "errors": ["No registered devices"]}

    # Build notification payload
    if silent:
        # Silent/background push - triggers app to sync
        payload = {
            "aps": {
                "content-available": 1,
            },
        }
        if data:
            payload.update(data)
    else:
        # Regular alert push
        interruption_level = INTERRUPTION_LEVELS.get(priority, "passive")

        # Build sound configuration
        if priority == NotificationPriority.EMERGENCY:
            # Critical alerts use a specific sound that can't be silenced
            sound_config = {
                "critical": 1,
                "name": "default",
                "volume": 1.0
            }
        elif sound:
            sound_config = "default"
        else:
            sound_config = None

        payload = {
            "aps": {
                "alert": {
                    "title": title,
                    "body": body,
                },
                "sound": sound_config,
                "category": category,
                "mutable-content": 1,
                "interruption-level": interruption_level,
            },
            "action_id": action_id,
            "action_type": action_type,
            "priority": priority.value,
        }

    success_count = 0
    errors = []

    # Group tokens by topic (bundle ID) since APNS clients are per-topic
    from collections import defaultdict
    tokens_by_topic: dict[str, list[str]] = defaultdict(list)
    for token, device_type in token_pairs:
        topic = BUNDLE_IDS.get(device_type, APNS_BUNDLE_ID)
        tokens_by_topic[topic].append(token)

    for topic, tokens in tokens_by_topic.items():
        try:
            client = create_apns_client(topic)
        except Exception as e:
            print(f"[push] Failed to initialize APNS client for {topic}: {e}")
            errors.append(str(e))
            continue

        for token in tokens:
            request = NotificationRequest(
                device_token=token,
                message=payload,
                push_type=PushType.BACKGROUND if silent else PushType.ALERT,
            )

            try:
                response = await client.send_notification(request)

                if response.is_successful:
                    success_count += 1
                    print(f"[push] Sent to device ({topic}): {token[:20]}...")
                else:
                    error_msg = f"APNS error: {response.description}"
                    errors.append(error_msg)
                    print(f"[push] Failed: {error_msg}")

                    # Deactivate invalid tokens
                    if response.status in ("BadDeviceToken", "Unregistered"):
                        deactivate_token(token)
                        print(f"[push] Deactivated invalid token: {token[:20]}...")

            except Exception as e:
                error_msg = f"Send error ({type(e).__name__}): {e}"
                errors.append(error_msg)
                print(f"[push] {error_msg}")

    return {
        "success": success_count,
        "total": sum(len(t) for t in tokens_by_topic.values()),
        "errors": errors if errors else None
    }


def send_push_sync(
    title: str = "",
    body: str = "",
    action_id: str = None,
    action_type: str = None,
    sound: bool = True,
    silent: bool = False,
    data: dict = None,
    priority: NotificationPriority = NotificationPriority.NORMAL,
    category: str = "PROACTIVE",
) -> dict:
    """
    Synchronous wrapper for send_push.

    Works correctly in both sync and async contexts. When called from inside
    a running event loop (e.g., from a sync function called by FastAPI),
    runs the push in a separate thread with its own event loop to avoid
    fire-and-forget behavior.

    Each thread creates fresh APNs clients (no shared mutable state),
    so concurrent calls cannot poison each other's connections.
    """
    import threading
    import logging
    push_logger = logging.getLogger("doris.push")

    coro_kwargs = dict(
        title=title, body=body, action_id=action_id, action_type=action_type,
        sound=sound, category=category, silent=silent, data=data, priority=priority,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context (e.g., FastAPI handler calling sync tool code).
        # Can't use asyncio.run() on this thread. Run push in a separate thread
        # with its own event loop so we get the actual result back.
        result_container = {}

        def _run_push():
            try:
                result_container["result"] = asyncio.run(send_push(**coro_kwargs))
            except Exception as e:
                push_logger.error(f"Push thread failed ({type(e).__name__}): {e}")
                result_container["result"] = {"success": 0, "errors": [f"{type(e).__name__}: {e}"]}

        t = threading.Thread(target=_run_push, daemon=True)
        t.start()
        t.join(timeout=15)

        if "result" not in result_container:
            push_logger.error("Push timed out after 15 seconds")
            return {"success": 0, "errors": ["Push timed out"]}

        result = result_container["result"]
        if result.get("errors"):
            push_logger.error(f"Push delivery errors: {result['errors']}")
        return result
    else:
        return asyncio.run(send_push(**coro_kwargs))


# Emergency event sources - these get EMERGENCY priority
# When Critical Alerts are approved, these will bypass all Focus modes
EMERGENCY_SOURCES = {
    # Home Assistant entity patterns
    "alarm": ["alarm_control_panel.*", "binary_sensor.*_alarm"],
    "smoke": ["binary_sensor.*smoke*", "binary_sensor.*fire*"],
    "water_leak": ["binary_sensor.*leak*", "binary_sensor.*water*", "binary_sensor.*flood*"],
    "carbon_monoxide": ["binary_sensor.*co_*", "binary_sensor.*carbon*"],
    "security": ["binary_sensor.*door*", "binary_sensor.*window*", "binary_sensor.*motion*"],
}


def is_emergency_event(entity_id: str = None, event_type: str = None) -> bool:
    """
    Check if an event qualifies as an emergency.

    Args:
        entity_id: Home Assistant entity ID
        event_type: Type of event (alarm, smoke, water_leak, etc.)

    Returns:
        True if this should be treated as an emergency
    """
    import re

    if event_type and event_type in EMERGENCY_SOURCES:
        return True

    if entity_id:
        for category, patterns in EMERGENCY_SOURCES.items():
            for pattern in patterns:
                if re.match(pattern.replace("*", ".*"), entity_id):
                    return True

    return False


def send_emergency_alert(
    title: str,
    body: str,
    source: str = None,
    entity_id: str = None,
) -> dict:
    """
    Send an emergency alert (Critical Alert when approved, Time Sensitive for now).

    Use this for alarms, water leaks, smoke detectors, etc.

    Args:
        title: Alert title
        body: Alert body
        source: Source of the emergency (alarm, smoke, water_leak, etc.)
        entity_id: Home Assistant entity ID if applicable

    Returns:
        Push result dict
    """
    return send_push_sync(
        title=f"🚨 {title}",
        body=body,
        priority=NotificationPriority.EMERGENCY,
        category="EMERGENCY",
        data={
            "emergency": True,
            "source": source,
            "entity_id": entity_id,
        }
    )


def send_urgent_notification(title: str, body: str, **kwargs) -> dict:
    """Send a Time Sensitive notification that breaks through Focus."""
    return send_push_sync(title=title, body=body, priority=NotificationPriority.URGENT, **kwargs)


def send_normal_notification(title: str, body: str, **kwargs) -> dict:
    """Send a normal notification that respects Focus/DND."""
    return send_push_sync(title=title, body=body, priority=NotificationPriority.NORMAL, **kwargs)


def get_token_status() -> dict:
    """
    Get device token health status.

    Returns dict with token count, details, and health assessment.
    Used by /status endpoint and daemon startup check.
    """
    token_pairs = get_active_tokens()
    token_details = []

    # Get details from SQLite
    try:
        conn = _get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT token, device_name, last_used, active FROM device_tokens ORDER BY last_used DESC"
        )
        for row in cursor.fetchall():
            token_details.append({
                "token_prefix": row[0][:20] + "..." if row[0] else "unknown",
                "device_name": row[1] or "Unknown",
                "last_used": row[2],
                "active": bool(row[3]),
            })
        conn.close()
    except Exception as e:
        import logging
        logging.getLogger("doris.push").error(f"Failed to get token details: {e}")

    active_count = len(token_pairs)
    if active_count == 0:
        health = "critical"
    else:
        health = "healthy"

    return {
        "health": health,
        "active_tokens": active_count,
        "tokens": token_details,
    }


# Initialize DB on module load
init_push_db()
