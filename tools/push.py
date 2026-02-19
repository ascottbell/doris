"""
Push notification service for Doris iOS app.

Uses APNs (Apple Push Notification service) to send notifications
to registered iOS devices.

Required credentials (from Apple Developer):
- APNs Auth Key (.p8 file)
- Key ID (10 character string)
- Team ID (10 character string)
- Bundle ID (e.g., com.yourorg.DorisClient)

Set via environment variables: APNS_KEY_ID, APNS_TEAM_ID, APNS_BUNDLE_ID
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Optional

# APNs configuration — all credentials from environment variables
APNS_KEY_ID = os.getenv("APNS_KEY_ID", "")
APNS_TEAM_ID = os.getenv("APNS_TEAM_ID", "")
APNS_BUNDLE_ID = os.getenv("APNS_BUNDLE_ID", "com.doris.client")
APNS_USE_SANDBOX = os.getenv("APNS_USE_SANDBOX", "true").lower() == "true"

# Key path — set APNS_KEY_PATH env var or place key in credentials/ directory
_key_filename = f"AuthKey_{APNS_KEY_ID}.p8" if APNS_KEY_ID else "AuthKey.p8"
APNS_KEY_PATH = Path(os.getenv("APNS_KEY_PATH", str(Path(__file__).parent.parent / "credentials" / _key_filename)))

# Device tokens file
DEVICE_TOKENS_FILE = Path(__file__).parent.parent / "data" / "device_tokens.json"


def load_config():
    """Load APNs config from config file or environment (optional override)."""
    global APNS_KEY_ID, APNS_TEAM_ID, APNS_USE_SANDBOX

    # Optional override from config file
    config_file = Path(__file__).parent.parent / "credentials" / "apns_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            APNS_KEY_ID = config.get("key_id", APNS_KEY_ID)
            APNS_TEAM_ID = config.get("team_id", APNS_TEAM_ID)
            APNS_USE_SANDBOX = config.get("use_sandbox", APNS_USE_SANDBOX)


def get_device_tokens() -> dict:
    """Get all registered device tokens (decrypts if encrypted)."""
    if not DEVICE_TOKENS_FILE.exists():
        return {}

    try:
        raw = DEVICE_TOKENS_FILE.read_bytes()

        # Try decrypting first (normal case when DORIS_API_TOKEN is set)
        try:
            from security.crypto import get_fernet
            fernet = get_fernet(os.getenv("DORIS_API_TOKEN", ""), b"doris-device-tokens-v1")
            if fernet:
                plaintext = fernet.decrypt(raw)
                return json.loads(plaintext)
        except Exception:
            pass

        # Fallback: plaintext (legacy or dev mode)
        return json.loads(raw)
    except Exception:
        return {}


async def send_push(
    title: str,
    body: str,
    device_token: Optional[str] = None,
    badge: Optional[int] = None,
    sound: str = "default",
    data: Optional[dict] = None,
) -> dict:
    """
    Send a push notification to one or all registered devices.

    Args:
        title: Notification title
        body: Notification body text
        device_token: Specific device token, or None to send to all
        badge: Badge number to show on app icon
        sound: Sound name ("default" or custom sound file)
        data: Additional data payload

    Returns:
        dict with success/failure info
    """
    load_config()

    # Check credentials
    if not APNS_KEY_PATH.exists():
        print(f"[Push] APNs key not found at {APNS_KEY_PATH}")
        return {
            "success": False,
            "error": "APNs key not configured. Check server credentials."
        }

    if not APNS_KEY_ID or not APNS_TEAM_ID:
        return {
            "success": False,
            "error": "APNs key_id or team_id not configured. Create secrets/apns_config.json"
        }

    try:
        from aioapns import APNs, NotificationRequest, PushType
    except ImportError:
        return {
            "success": False,
            "error": "aioapns not installed. Run: pip install aioapns"
        }

    # Get target devices
    tokens = get_device_tokens()
    if device_token:
        if device_token not in tokens:
            return {"success": False, "error": "Device token not registered"}
        target_tokens = [device_token]
    else:
        target_tokens = list(tokens.keys())

    if not target_tokens:
        return {"success": False, "error": "No devices registered"}

    # Read the key content
    with open(APNS_KEY_PATH) as f:
        key_content = f.read()

    # Create APNs client
    apns = APNs(
        key=key_content,
        key_id=APNS_KEY_ID,
        team_id=APNS_TEAM_ID,
        topic=APNS_BUNDLE_ID,
        use_sandbox=APNS_USE_SANDBOX,
    )

    # Build notification payload
    payload = {
        "aps": {
            "alert": {
                "title": title,
                "body": body,
            },
            "sound": sound,
        }
    }
    if badge is not None:
        payload["aps"]["badge"] = badge
    if data:
        payload.update(data)

    # Send to each device
    results = []
    for token in target_tokens:
        request = NotificationRequest(
            device_token=token,
            message=payload,
            push_type=PushType.ALERT,
        )

        try:
            response = await apns.send_notification(request)
            if response.is_successful:
                results.append({"token": token[:16] + "...", "success": True})
            else:
                results.append({
                    "token": token[:16] + "...",
                    "success": False,
                    "error": response.description
                })
        except Exception as e:
            print(f"[Push] APNs send error for token {token[:16]}...: {e}")
            results.append({
                "token": token[:16] + "...",
                "success": False,
                "error": "Push delivery failed"
            })

    success_count = sum(1 for r in results if r["success"])
    return {
        "success": success_count > 0,
        "sent": success_count,
        "failed": len(results) - success_count,
        "results": results
    }


async def send_doris_notification(message: str) -> dict:
    """
    Send a notification from Doris to all devices.

    This is the main function used by Doris to proactively
    reach out to the user.
    """
    return await send_push(
        title="Doris",
        body=message,
    )


# Sync wrapper for non-async contexts
def send_push_sync(title: str, body: str, **kwargs) -> dict:
    """Synchronous wrapper for send_push."""
    return asyncio.run(send_push(title, body, **kwargs))
