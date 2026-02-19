"""
Executor - Takes actions based on evaluation results.

Executes calendar events, reminders, notifications, etc.
Logs all actions for potential undo/correction.

Now includes Wisdom integration - records outcomes for learning.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from .models import ProactiveEvent, ProactiveAction, EvaluationResult
from .db import save_action

EASTERN = ZoneInfo("America/New_York")


def _record_wisdom_outcome(wisdom_id: str, outcome: str, details: str = None):
    """Record outcome in wisdom table."""
    if not wisdom_id:
        return
    try:
        from memory.wisdom import record_outcome
        record_outcome(wisdom_id, outcome, details)
        print(f"[executor] Recorded wisdom outcome: {outcome}")
    except Exception as e:
        print(f"[executor] Failed to record wisdom outcome: {e}")


def execute_actions(event: ProactiveEvent, evaluation: EvaluationResult) -> list[ProactiveAction]:
    """
    Execute ALL actions recommended by the evaluator.

    Returns a list of ProactiveAction records.
    """
    if not evaluation.should_act or not evaluation.actions:
        return []

    results = []
    for action_item in evaluation.actions:
        action = _execute_single_action(
            event,
            action_item.action_type,
            action_item.action_data,
            action_item.wisdom_id
        )
        if action:
            results.append(action)

    return results


def _execute_single_action(
    event: ProactiveEvent,
    action_type: str,
    action_data: dict,
    wisdom_id: str = None
) -> ProactiveAction:
    """Execute a single action and return the result."""
    if action_type == "create_event":
        return _create_calendar_event(event, action_data, wisdom_id)
    elif action_type == "send_reminder":
        return _create_reminder(event, action_data, wisdom_id)
    elif action_type == "notify":
        return _create_notification(event, action_data, wisdom_id)
    elif action_type == "store_memory":
        return _store_memory(event, action_data, wisdom_id)
    elif action_type == "queue_briefing":
        return _queue_for_briefing(event, action_data, wisdom_id)
    else:
        print(f"[executor] Unknown action type: {action_type}")
        return None


def execute_action(event: ProactiveEvent, evaluation: EvaluationResult) -> ProactiveAction:
    """
    Execute the action recommended by the evaluator.

    Returns a ProactiveAction record of what was done.
    """
    if not evaluation.should_act:
        return None

    action_type = evaluation.action_type
    action_data = evaluation.action_data or {}

    if action_type == "create_event":
        return _create_calendar_event(event, action_data)
    elif action_type == "send_reminder":
        return _create_reminder(event, action_data)
    elif action_type == "notify":
        return _create_notification(event, action_data)
    elif action_type == "store_memory":
        return _store_memory(event, action_data)
    elif action_type == "queue_briefing":
        return _queue_for_briefing(event, action_data)
    else:
        print(f"[executor] Unknown action type: {action_type}")
        return None


def _create_calendar_event(event: ProactiveEvent, data: dict, wisdom_id: str = None) -> ProactiveAction:
    """Create a calendar event and return the action record."""
    from tools.cal import create_event, list_events

    title = data.get("title", "Untitled Event")
    date_str = data.get("date")
    time_str = data.get("time", "09:00")
    location = data.get("location")
    notes = data.get("notes")

    # Parse date
    try:
        if date_str:
            event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            # Default to tomorrow if no date specified
            event_date = (datetime.now(EASTERN) + timedelta(days=1)).date()
    except ValueError:
        event_date = (datetime.now(EASTERN) + timedelta(days=1)).date()

    # Parse time
    try:
        if time_str:
            hour, minute = map(int, time_str.split(":"))
        else:
            hour, minute = 9, 0
    except ValueError:
        hour, minute = 9, 0

    # Build datetime
    start = datetime(event_date.year, event_date.month, event_date.day,
                     hour, minute, tzinfo=EASTERN)
    end = start + timedelta(hours=1)  # Default 1 hour duration

    # Check for duplicates (same title on same day)
    existing = list_events(
        start.replace(hour=0, minute=0),
        start.replace(hour=23, minute=59)
    )
    for e in existing:
        if e.get("title", "").lower() == title.lower():
            print(f"[executor] Duplicate event detected: {title}")
            _record_wisdom_outcome(wisdom_id, "skipped", "Duplicate event already exists")
            # Return action but mark as skipped
            action = ProactiveAction(
                event_id=event.id,
                action_type="create_event",
                action_data={
                    "title": title,
                    "date": date_str,
                    "time": time_str,
                    "skipped": True,
                    "reason": "duplicate"
                },
                status="skipped",
                wisdom_id=wisdom_id
            )
            save_action(action)
            return action

    # Create the event
    try:
        result = create_event(
            title=title,
            start=start,
            end=end,
            location=location,
            notes=notes,
            source="scout"  # Attribution: created by proactive scout system
        )

        result_id = result.get("id") if isinstance(result, dict) else None

        _record_wisdom_outcome(wisdom_id, "success", f"Created: {title}")
        action = ProactiveAction(
            event_id=event.id,
            action_type="create_event",
            action_data={
                "title": title,
                "date": date_str,
                "time": time_str,
                "location": location,
                "start_iso": start.isoformat(),
                "end_iso": end.isoformat()
            },
            result_id=result_id,
            status="completed",
            wisdom_id=wisdom_id
        )
        save_action(action)
        print(f"[executor] Created event: {title} on {start.strftime('%B %d at %I:%M %p')}")
        return action

    except Exception as e:
        print(f"[executor] Failed to create event: {e}")
        _record_wisdom_outcome(wisdom_id, "failed", str(e))
        action = ProactiveAction(
            event_id=event.id,
            action_type="create_event",
            action_data={"title": title, "error": str(e)},
            status="failed"
        )
        save_action(action)
        return action


def _create_reminder(event: ProactiveEvent, data: dict, wisdom_id: str = None) -> ProactiveAction:
    """Create a reminder and return the action record."""
    from tools.reminders import create_reminder

    title = data.get("title", "Reminder")
    date_str = data.get("date")
    time_str = data.get("time")

    # Build due datetime if provided
    due = None
    if date_str:
        try:
            due_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            hour, minute = 9, 0
            if time_str:
                try:
                    hour, minute = map(int, time_str.split(":"))
                except:
                    pass
            due = datetime(due_date.year, due_date.month, due_date.day,
                          hour, minute, tzinfo=EASTERN)
        except ValueError:
            pass

    try:
        result = create_reminder(title=title, due=due)

        _record_wisdom_outcome(wisdom_id, "success", f"Created: {title}")
        action = ProactiveAction(
            event_id=event.id,
            action_type="send_reminder",
            action_data={
                "title": title,
                "due": due.isoformat() if due else None
            },
            status="completed",
            wisdom_id=wisdom_id
        )
        save_action(action)
        print(f"[executor] Created reminder: {title}")
        return action

    except Exception as e:
        print(f"[executor] Failed to create reminder: {e}")
        _record_wisdom_outcome(wisdom_id, "failed", str(e))
        action = ProactiveAction(
            event_id=event.id,
            action_type="send_reminder",
            action_data={"title": title, "error": str(e)},
            status="failed",
            wisdom_id=wisdom_id
        )
        save_action(action)
        return action


def _create_notification(event: ProactiveEvent, data: dict, wisdom_id: str = None) -> ProactiveAction:
    """
    Create a notification to tell the user something.

    This doesn't execute immediately - it queues for the notifier.
    """
    message = data.get("message", data.get("title", "Notification"))

    _record_wisdom_outcome(wisdom_id, "success", f"Notification queued")
    action = ProactiveAction(
        event_id=event.id,
        action_type="notify",
        action_data={
            "message": message,
            "priority": data.get("priority", "normal")
        },
        status="pending",  # Notifier will mark as completed
        notification_sent=False,
        wisdom_id=wisdom_id
    )
    save_action(action)
    print(f"[executor] Queued notification: {message[:50]}...")
    return action


def _store_memory(event: ProactiveEvent, data: dict, wisdom_id: str = None) -> ProactiveAction:
    """Store a fact in Doris's memory for future reference."""
    from memory.store import store_memory
    from security.injection_scanner import scan_for_injection, strip_invisible_chars

    content = data.get("message", data.get("title", ""))
    subject = data.get("subject")  # Who/what this is about

    if not content:
        print("[executor] No content to store in memory")
        return None

    # Security: scan and sanitize before storing
    content = strip_invisible_chars(content)
    scan_result = scan_for_injection(content, source=f"executor:store_memory:{event.source_type}")
    if scan_result.is_suspicious:
        print(
            f"[executor] WARNING: Suspicious memory content "
            f"(risk={scan_result.risk_level}, source={event.source_type}): "
            f"{content[:100]!r}"
        )

    try:
        # Determine category based on content
        category = "fact"
        content_lower = content.lower()
        if any(word in content_lower for word in ["teacher", "coach", "doctor", "contact"]):
            category = "person"
        elif any(word in content_lower for word in ["prefer", "like", "always", "never"]):
            category = "preference"

        mem_id = store_memory(
            content=content,
            category=category,
            subject=subject,
            source=f"proactive:{event.source_type}",
            confidence=0.85
        )

        _record_wisdom_outcome(wisdom_id, "success", f"Stored memory: {content[:30]}")
        action = ProactiveAction(
            event_id=event.id,
            action_type="store_memory",
            action_data={
                "content": content,
                "category": category,
                "subject": subject,
                "memory_id": mem_id
            },
            result_id=mem_id,
            status="completed",
            wisdom_id=wisdom_id
        )
        save_action(action)
        print(f"[executor] Stored memory: {content[:50]}...")
        return action

    except Exception as e:
        print(f"[executor] Failed to store memory: {e}")
        _record_wisdom_outcome(wisdom_id, "failed", str(e))
        return None


def _queue_for_briefing(event: ProactiveEvent, data: dict, wisdom_id: str = None) -> ProactiveAction:
    """Queue information for the next morning briefing."""
    message = data.get("message", data.get("title", ""))
    priority = data.get("priority", "normal")

    if not message:
        print("[executor] No message to queue for briefing")
        return None

    _record_wisdom_outcome(wisdom_id, "success", "Queued for briefing")
    # Store in a briefing queue (we'll use proactive_actions with special status)
    action = ProactiveAction(
        event_id=event.id,
        action_type="queue_briefing",
        action_data={
            "message": message,
            "priority": priority,
            "source_type": event.source_type
        },
        status="queued",  # Special status for briefing items
        wisdom_id=wisdom_id,
        notification_sent=False
    )
    save_action(action)
    print(f"[executor] Queued for briefing: {message[:50]}...")
    return action


def get_briefing_queue() -> list[ProactiveAction]:
    """Get all items queued for briefing."""
    from .db import get_connection
    import json

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, event_id, action_type, action_data, result_id, executed_at, status, notification_sent
        FROM proactive_actions
        WHERE action_type = 'queue_briefing' AND status = 'queued'
        ORDER BY executed_at ASC
    """)

    actions = [ProactiveAction.from_db_row(row) for row in cursor.fetchall()]
    conn.close()
    return actions


def clear_briefing_queue():
    """Mark all briefing items as delivered."""
    from .db import get_connection

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE proactive_actions
        SET status = 'delivered'
        WHERE action_type = 'queue_briefing' AND status = 'queued'
    """)

    conn.commit()
    conn.close()


def undo_action(action: ProactiveAction) -> bool:
    """
    Undo a previous action (e.g., delete a calendar event).

    Returns True if successfully undone.
    """
    from .db import update_action_status

    if action.action_type == "create_event" and action.result_id:
        from tools.cal import delete_event

        try:
            result = delete_event(action.result_id)
            if result.get("success"):
                update_action_status(action.id, "undone")
                print(f"[executor] Deleted event and marked action as undone: {action.id}")
                return True
            else:
                print(f"[executor] Failed to delete event: {result.get('error')}")
                return False
        except Exception as e:
            print(f"[executor] Error deleting event: {e}")
            return False

    elif action.action_type == "send_reminder":
        # TODO: Implement reminder deletion
        update_action_status(action.id, "undone")
        return True

    return False
