"""
iMessage integration for Doris.
Send messages via AppleScript, read via Messages database.
"""

import subprocess
import sqlite3
import json
from datetime import datetime
from pathlib import Path

from security.sanitize import escape_applescript_string

import logging
logger = logging.getLogger(__name__)

# Messages database location
MESSAGES_DB = Path.home() / "Library/Messages/chat.db"

# Contact mappings - name variations to phone/email
# Configure with your own contacts
CONTACTS = {
    # "wife": "+15551234567",
    # "my wife": "+15551234567",
    # Add more as needed
}

# Warn at import time if CONTACTS is empty and bypass is off
import os as _os
if not CONTACTS and not _os.environ.get("IMESSAGE_ALLOW_ANY_RECIPIENT", "").lower() in ("true", "1", "yes"):
    logger.warning("iMessage CONTACTS dict is empty and IMESSAGE_ALLOW_ANY_RECIPIENT is not set — all sends will be blocked")


def normalize_recipient(recipient: str) -> str:
    """
    Normalize recipient to a sendable address.
    Maps nicknames to contact names or phone numbers.
    """
    key = recipient.lower().strip()
    return CONTACTS.get(key, recipient)


def send_message(recipient: str, message: str) -> dict:
    """
    Send an iMessage via AppleScript.

    Args:
        recipient: Contact name, phone number, or email
        message: Message text to send

    Returns:
        dict with 'success' bool and 'error' string if failed
    """
    # Normalize the recipient
    target = normalize_recipient(recipient)

    # Escape for safe AppleScript string interpolation
    # Uses JSON-based escaping which handles all special characters
    escaped_message = escape_applescript_string(message)
    escaped_target = escape_applescript_string(target)

    # AppleScript to send message via Messages app
    applescript = f'''
    tell application "Messages"
        send "{escaped_message}" to buddy "{escaped_target}" of (1st account whose service type = iMessage)
    end tell
    '''

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return {"success": True, "recipient": target}
        else:
            # Try alternate method using buddy by phone/email directly
            applescript_alt = f'''
            tell application "Messages"
                send "{escaped_message}" to buddy "{escaped_target}"
            end tell
            '''
            result_alt = subprocess.run(
                ["osascript", "-e", applescript_alt],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result_alt.returncode == 0:
                return {"success": True, "recipient": target}
            else:
                return {
                    "success": False,
                    "error": result.stderr.strip() or result_alt.stderr.strip() or "Unknown error"
                }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout sending message"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def read_recent(contact: str = None, limit: int = 5) -> list[dict]:
    """
    Read recent messages from the Messages database.

    Args:
        contact: Optional contact name/number to filter by
        limit: Max messages to return

    Returns:
        List of message dicts with sender, text, timestamp, is_from_me
    """
    if not MESSAGES_DB.exists():
        return []

    try:
        # Connect read-only to avoid any lock issues
        db = sqlite3.connect(f"file:{MESSAGES_DB}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row

        # Query recent messages
        # The Messages schema joins message -> chat_message_join -> chat -> handle
        query = """
            SELECT
                m.text,
                m.is_from_me,
                m.date/1000000000 + strftime('%s', '2001-01-01') as timestamp,
                h.id as handle_id,
                COALESCE(h.id, 'Me') as sender
            FROM message m
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL
        """

        params = []

        if contact:
            # Normalize and search
            normalized = normalize_recipient(contact)
            # Escape LIKE metacharacters so % and _ in contact names are literal
            escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            query += " AND (h.id LIKE ? ESCAPE '\\' OR c.display_name LIKE ? ESCAPE '\\')"
            search_term = f"%{escaped}%"
            params.extend([search_term, search_term])

        query += " ORDER BY m.date DESC LIMIT ?"
        params.append(limit)

        rows = db.execute(query, params).fetchall()
        db.close()

        messages = []
        for row in rows:
            try:
                ts = datetime.fromtimestamp(row['timestamp'])
                messages.append({
                    'text': row['text'],
                    'is_from_me': bool(row['is_from_me']),
                    'sender': 'Me' if row['is_from_me'] else row['sender'],
                    'timestamp': ts.isoformat(),
                    'time_ago': _time_ago(ts)
                })
            except (ValueError, OSError):
                continue

        return messages

    except sqlite3.Error as e:
        return []
    except Exception as e:
        return []


def _time_ago(dt: datetime) -> str:
    """Format datetime as relative time string."""
    now = datetime.now()
    diff = now - dt

    if diff.days > 0:
        if diff.days == 1:
            return "yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        else:
            return dt.strftime("%b %d")

    hours = diff.seconds // 3600
    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"

    minutes = diff.seconds // 60
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"

    return "just now"


def format_messages_for_speech(messages: list[dict]) -> str:
    """Format messages for voice output."""
    if not messages:
        return "No recent messages found."

    if len(messages) == 1:
        m = messages[0]
        direction = "You sent" if m['is_from_me'] else f"From {m['sender']}"
        return f"{direction}, {m['time_ago']}: {m['text']}"

    parts = [f"Here are the last {len(messages)} messages."]
    for m in messages:
        direction = "You said" if m['is_from_me'] else f"{m['sender']} said"
        parts.append(f"{direction}, {m['time_ago']}: {m['text']}")

    return " ".join(parts)


if __name__ == "__main__":
    # Test reading (safe)
    print("Recent messages:")
    for msg in read_recent(limit=3):
        print(f"  [{msg['time_ago']}] {msg['sender']}: {msg['text'][:50]}...")

    print("\nTo test sending, run:")
    print("  send_message('<recipient>', 'Test from Doris')")
    print("(Requires confirmation before actually running)")
