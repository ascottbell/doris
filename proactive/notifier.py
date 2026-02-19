"""
Notifier - Tells the user about proactive actions.

Handles push notifications, notification queuing, and quiet hours.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo
from typing import Optional
from .models import ProactiveAction
from .db import update_action_status, get_recent_actions

EASTERN = ZoneInfo("America/New_York")

# Quiet hours - don't send notifications during these times
QUIET_START = time(22, 0)  # 10 PM
QUIET_END = time(7, 0)     # 7 AM

# Notification queue for messages during quiet hours
_notification_queue: list[dict] = []


def is_quiet_hours() -> bool:
    """Check if we're in quiet hours (no notifications)."""
    now = datetime.now(EASTERN).time()
    if QUIET_START <= QUIET_END:
        return QUIET_START <= now <= QUIET_END
    else:
        # Quiet hours span midnight
        return now >= QUIET_START or now <= QUIET_END


def notify_action(action: ProactiveAction) -> bool:
    """
    Notify the user about an action that was taken.

    Delivery is via push notification (always sent, respects Do Not Disturb on device).

    Args:
        action: The action to notify about

    Returns:
        True if notification was delivered
    """
    if action is None:
        return False

    # Skip if action was skipped/failed
    if action.status in ("skipped", "failed"):
        return False

    # Check if escalation wisdom suggests this deserves higher priority
    if action.event_id:
        try:
            from proactive.db import get_event
            event = get_event(action.event_id)
            if event and event.raw_data:
                from memory.wisdom import should_escalate_based_on_wisdom
                should_upgrade, reason = should_escalate_based_on_wisdom(
                    source=event.source_type,
                    sender=event.raw_data.get("sender", ""),
                    subject=event.raw_data.get("subject", ""),
                    snippet=event.raw_data.get("snippet", ""),
                )
                if should_upgrade:
                    message = _build_notification_message(action)
                    print(f"[notifier] Wisdom priority upgrade: {reason}")
                    return notify_urgent(
                        title="Doris (escalated)",
                        message=message.replace(" ... ", " ").replace("...", "").strip(),
                    )
        except Exception as e:
            print(f"[notifier] Wisdom priority check failed (continuing normally): {e}")

    message = _build_notification_message(action)

    # Send push notification (device handles Do Not Disturb)
    return _announce(message, action)


def _build_notification_message(action: ProactiveAction) -> str:
    """Build a natural language notification message."""
    from proactive.speech import format_time_for_speech, format_date_for_speech

    data = action.action_data

    if action.action_type == "create_event":
        title = data.get("title", "an event")
        date_str = data.get("date", "")
        time_str = data.get("time", "")

        # Format date naturally for speech
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                date_phrase = format_date_for_speech(dt)
            except:
                date_phrase = date_str
        else:
            date_phrase = "your calendar"

        # Format time naturally for speech
        if time_str:
            try:
                hour, minute = map(int, time_str.split(":"))
                t = time(hour, minute)
                time_phrase = format_time_for_speech(t)
            except:
                time_phrase = time_str
        else:
            time_phrase = ""

        # Build message with pauses for natural speech
        if time_phrase:
            return f"Hey, ... I saw an email about {title}. ... I added it to your calendar, ... for {date_phrase}, ... at {time_phrase}."
        else:
            return f"Hey, ... I saw an email about {title}. ... I added it to your calendar for {date_phrase}."

    elif action.action_type == "send_reminder":
        title = data.get("title", "a reminder")
        return f"Hey, ... I set a reminder for you. ... {title}."

    elif action.action_type == "notify":
        message = data.get("message", "You have a notification.")
        # Add a lead-in if the message doesn't have one
        if not message.lower().startswith(("hey", "heads up", "just", "quick")):
            message = f"Hey, ... {message}"
        return message

    else:
        return f"I took action, ... {action.action_type}"


def _announce(message: str, action: ProactiveAction) -> bool:
    """
    Notify the user about an action.

    Primary: Save to conversation history (syncs to iOS/macOS apps)
    Secondary: Send alert push notification (shows banner on device)
    """
    from .db import save_action

    # Save to conversation history (apps will sync this)
    message_saved = _save_to_conversation(message, action)

    # Send alert push notification (shows banner + sound on device)
    push_sent = _send_push_notification(message, action)

    # Mark notification as sent
    action.notification_sent = True
    save_action(action)

    print(f"[notifier] {'Message saved' if message_saved else 'Save failed'}, "
          f"{'push sent' if push_sent else 'push failed'}: {message[:50]}...")
    return message_saved


def _save_to_conversation(message: str, action: ProactiveAction) -> bool:
    """Save proactive message to conversation history for app sync."""
    try:
        import uuid
        from api.conversations import save_message, MessageCreate

        # Strip speech formatting (remove ellipses pauses)
        clean_message = message.replace(" ... ", " ").replace("...", "").strip()
        # Remove "Hey, " prefix since it's redundant in text
        if clean_message.lower().startswith("hey, "):
            clean_message = clean_message[5:].strip()

        # Create message with proactive metadata
        msg = MessageCreate(
            id=str(uuid.uuid4()),
            content=clean_message,
            role="assistant",
            device_id="doris-proactive",
            metadata={
                "proactive": True,
                "action_type": action.action_type,
                "action_id": action.id,
            }
        )

        save_message(msg)
        print(f"[notifier] Saved to conversation: {clean_message[:50]}...")
        return True

    except Exception as e:
        print(f"[notifier] Failed to save to conversation: {e}")
        return False


def _send_silent_push() -> None:
    """Send a silent push to trigger app sync."""
    try:
        from services.push import send_push_sync

        # Silent push with content-available flag
        send_push_sync(
            title="",
            body="",
            silent=True,
            data={"sync": True}
        )
    except Exception as e:
        # Silent push failure is not critical
        print(f"[notifier] Silent push failed (non-critical): {e}")


def _send_push_notification(message: str, action: ProactiveAction) -> bool:
    """Send a push notification to registered iOS devices."""
    try:
        from services.push import send_push_sync

        # Build title based on action type
        title_map = {
            "create_event": "📅 Calendar Updated",
            "send_reminder": "⏰ Reminder Set",
            "notify": "💬 Doris",
            "store_memory": "🧠 Noted",
            "queue_briefing": "📋 For Your Briefing",
        }
        title = title_map.get(action.action_type, "Doris")

        # Strip speech formatting for push (remove ellipses pauses)
        clean_message = message.replace(" ... ", " ").replace("...", "").strip()
        # Remove "Hey, " prefix for push since it's redundant
        if clean_message.lower().startswith("hey, "):
            clean_message = clean_message[5:].strip()

        result = send_push_sync(
            title=title,
            body=clean_message,
            action_id=action.id,
            action_type=action.action_type
        )

        return result.get("success", 0) > 0 or result.get("queued", False)

    except Exception as e:
        print(f"[notifier] Push failed: {e}")
        return False


def notify_emergency(
    title: str,
    message: str,
    source: str = None,
    entity_id: str = None,
) -> bool:
    """
    Send an emergency notification - these ALWAYS get through.

    Emergency notifications:
    1. Save to conversation history (for app sync)
    2. Send Time Sensitive push (Critical Alert when approved)

    Use for: alarms, water leaks, smoke detectors, security alerts.

    Args:
        title: Alert title
        message: Alert message
        source: Source of emergency (alarm, smoke, water_leak, etc.)
        entity_id: Home Assistant entity ID if applicable

    Returns:
        True if notification was delivered
    """
    try:
        import uuid
        from api.conversations import save_message, MessageCreate
        from services.push import send_emergency_alert

        # Save to conversation history
        msg = MessageCreate(
            id=str(uuid.uuid4()),
            content=f"🚨 {title}: {message}",
            role="assistant",
            device_id="doris-proactive",
            metadata={
                "proactive": True,
                "emergency": True,
                "action_type": "emergency",
                "source": source,
                "entity_id": entity_id,
            }
        )
        save_message(msg)
        print(f"[notifier] Emergency saved to conversation: {title}")

        # Send emergency push (Time Sensitive, Critical when approved)
        push_result = send_emergency_alert(
            title=title,
            body=message,
            source=source,
            entity_id=entity_id,
        )
        print(f"[notifier] Emergency push sent: {push_result}")

        return True

    except Exception as e:
        print(f"[notifier] Emergency notification failed: {e}")
        return False


def notify_urgent(title: str, message: str) -> bool:
    """
    Send an urgent (Time Sensitive) notification.

    These break through Focus mode but aren't emergencies.
    Use for: important emails, upcoming events, weather alerts.

    Args:
        title: Notification title
        message: Notification message

    Returns:
        True if notification was delivered
    """
    try:
        import uuid
        from api.conversations import save_message, MessageCreate
        from services.push import send_urgent_notification

        # Save to conversation history
        msg = MessageCreate(
            id=str(uuid.uuid4()),
            content=message,
            role="assistant",
            device_id="doris-proactive",
            metadata={
                "proactive": True,
                "urgent": True,
                "action_type": "notify",
            }
        )
        save_message(msg)

        # Send Time Sensitive push
        send_urgent_notification(title=title, body=message)

        # Trigger app sync
        _send_silent_push()

        print(f"[notifier] Urgent notification sent: {title}")
        return True

    except Exception as e:
        print(f"[notifier] Urgent notification failed: {e}")
        return False


def get_queued_notifications() -> list[dict]:
    """Get notifications that were queued during quiet hours."""
    return list(_notification_queue)


def deliver_queued_notifications() -> int:
    """
    Deliver all queued notifications.

    Call this when quiet hours end or when user wakes up.
    Returns the number of notifications delivered.
    """
    global _notification_queue

    if not _notification_queue:
        return 0

    if is_quiet_hours():
        print("[notifier] Still in quiet hours, not delivering queue")
        return 0

    delivered = 0
    remaining = []

    for item in _notification_queue:
        try:
            # Deliver via push notification
            from services.push import send_push_sync
            send_push_sync(item["message"])
            delivered += 1
        except Exception as e:
            print(f"[notifier] Failed to deliver queued notification: {e}")
            remaining.append(item)

    _notification_queue = remaining
    print(f"[notifier] Delivered {delivered} queued notifications")
    return delivered


def get_recent_action_for_correction() -> Optional[ProactiveAction]:
    """
    Get the most recent action for correction handling.

    When the user says "delete that" or "wrong time", we need to know
    what action he's referring to.
    """
    actions = get_recent_actions(limit=1)
    return actions[0] if actions else None


def handle_correction(correction_type: str, new_value: str = None) -> str:
    """
    Handle a correction from the user about a recent action.

    Args:
        correction_type: 'delete', 'wrong_time', 'wrong_date', 'rename'
        new_value: The corrected value (for wrong_time, wrong_date, rename)

    Returns:
        A confirmation message
    """
    from .executor import undo_action
    from memory.store import store_memory

    action = get_recent_action_for_correction()
    if not action:
        return "I don't have any recent actions to correct."

    if correction_type == "delete":
        if undo_action(action):
            # Learn from this correction
            store_memory(
                content=f"Event '{action.action_data.get('title')}' was incorrect - the user deleted it",
                category="correction",
                subject="proactive",
                source="correction:delete",
                confidence=0.9
            )
            return f"Got it, I've removed {action.action_data.get('title', 'that event')}."
        else:
            return "I couldn't undo that action."

    elif correction_type == "wrong_time" and new_value:
        # TODO: Implement time correction
        return f"Time correction not yet implemented. Please update the event manually to {new_value}."

    elif correction_type == "wrong_date" and new_value:
        # TODO: Implement date correction
        return f"Date correction not yet implemented. Please update the event manually to {new_value}."

    else:
        return "I'm not sure how to handle that correction."
