"""
Calendar lookahead source - monitors upcoming events for prep alerts.

Checks for:
- Events happening soon (travel time alerts)
- Events tomorrow that might need preparation
- Important events this week
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tools.cal import list_events, get_todays_events
from proactive.models import ProactiveEvent
from proactive.db import (
    save_event,
    is_event_processed,
    update_event_status,
    update_checkpoint,
)
from proactive.evaluator import evaluate_event
from proactive.executor import execute_action
from proactive.notifier import notify_action

EASTERN = ZoneInfo("America/New_York")

# Events with these keywords are important family events
IMPORTANT_EVENT_KEYWORDS = [
    "recital", "performance", "show", "concert",
    "game", "practice", "match",
    "birthday", "party",
    "doctor", "dentist", "appointment",
    "conference", "meeting",
    "flight", "travel",
]

# Default travel times for common locations (in minutes)
# Customize with your own frequently visited places
TRAVEL_TIMES = {
    "school": 10,
    "gym": 10,
    "office": 20,
    "default": 20,
}


def monitor():
    """
    Main calendar monitor function.

    Called by scheduler hourly. Checks for:
    1. Events starting in the next 2 hours (travel alerts)
    2. Events tomorrow (prep reminders)
    """
    print("[calendar-monitor] Checking upcoming events...")

    now = datetime.now(EASTERN)

    try:
        # Check 1: Events in next 2 hours (travel time alerts)
        _check_upcoming_events(now)

        # Check 2: Tomorrow's events (evening prep reminder)
        # Only run this check between 6-9 PM
        if 18 <= now.hour <= 21:
            _check_tomorrow_events(now)

        update_checkpoint("calendar")
        print("[calendar-monitor] Done")

    except Exception as e:
        print(f"[calendar-monitor] Error: {e}")
        import traceback
        traceback.print_exc()


def _check_upcoming_events(now: datetime):
    """Check for events in the next 2 hours that need travel alerts."""
    window_start = now
    window_end = now + timedelta(hours=2)

    events = list_events(window_start, window_end)

    for event in events:
        event_id = event.get("id", event.get("title", ""))
        alert_id = f"travel-alert-{event_id}-{now.strftime('%Y%m%d')}"

        # Skip if already alerted today
        if is_event_processed("calendar", alert_id):
            continue

        title = event.get("title", "")
        start_str = event.get("start", "")
        location = event.get("location", "")

        # Parse start time
        try:
            if "T" in start_str:
                event_start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                if event_start.tzinfo is None:
                    event_start = event_start.replace(tzinfo=EASTERN)
            else:
                continue  # All-day event, skip travel alert
        except:
            continue

        # Calculate time until event
        minutes_until = (event_start - now).total_seconds() / 60

        # Get travel time for this location
        travel_time = _get_travel_time(title, location)

        # Alert if we need to leave within 30 minutes
        leave_in = minutes_until - travel_time
        if 0 < leave_in <= 30:
            # Create a notification event
            proactive_event = ProactiveEvent(
                source_type="calendar",
                source_id=alert_id,
                raw_data={
                    "type": "travel_alert",
                    "title": title,
                    "start": start_str,
                    "location": location,
                    "minutes_until": int(minutes_until),
                    "leave_in": int(leave_in),
                    "travel_time": travel_time,
                }
            )
            save_event(proactive_event)

            # Build notification message - formatted for speech with pauses
            from proactive.speech import format_time_for_speech, format_duration_for_speech

            if leave_in <= 5:
                urgency = "Hey, ... you need to leave now for"
            elif leave_in <= 15:
                urgency = "Hey, ... time to head out soon for"
            else:
                duration = format_duration_for_speech(int(leave_in))
                urgency = f"Hey, ... heads up. ... Leave in about {duration} for"

            # Convert to Eastern time and format for speech
            event_start_local = event_start.astimezone(EASTERN)
            time_str = format_time_for_speech(event_start_local)
            message = f"{urgency} {title}, ... at {time_str}."
            if location:
                message += f" ... It's at {location}."

            # Direct notify (skip Claude evaluation for travel alerts)
            from proactive.executor import _create_notification
            action = _create_notification(proactive_event, {
                "message": message,
                "priority": "high"
            })

            if action:
                update_event_status(proactive_event.id, "actioned")
                notify_action(action)  # Push notification, voice optional


def _check_tomorrow_events(now: datetime):
    """Check tomorrow's events and alert about important ones."""
    tomorrow = now + timedelta(days=1)
    tomorrow_start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_end = tomorrow_start + timedelta(days=1)

    events = list_events(tomorrow_start, tomorrow_end)

    if not events:
        return

    # Find important events
    important = []
    for event in events:
        title = event.get("title", "").lower()
        if any(keyword in title for keyword in IMPORTANT_EVENT_KEYWORDS):
            important.append(event)

    if not important:
        return

    # Create a single prep alert for tomorrow
    alert_id = f"tomorrow-prep-{tomorrow.strftime('%Y%m%d')}"
    if is_event_processed("calendar", alert_id):
        return

    proactive_event = ProactiveEvent(
        source_type="calendar",
        source_id=alert_id,
        raw_data={
            "type": "tomorrow_prep",
            "date": tomorrow.strftime("%Y-%m-%d"),
            "events": important,
        }
    )
    save_event(proactive_event)

    # Build message - formatted for speech with natural pauses
    from proactive.speech import format_time_for_speech

    if len(important) == 1:
        event = important[0]
        title = event.get("title", "event")
        start_str = event.get("start", "")
        try:
            start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            start_local = start.astimezone(EASTERN)
            time_str = format_time_for_speech(start_local)
            message = f"Hey, ... just a heads up. ... {title} is tomorrow, ... at {time_str}."
        except:
            message = f"Hey, ... just a heads up. ... {title} is tomorrow."
    else:
        message = f"Hey, ... just a heads up. ... You have {len(important)} important things tomorrow. ... "
        titles = [e.get("title", "event") for e in important[:3]]
        message += ", ... ".join(titles)
        if len(important) > 3:
            message += f", ... and {len(important) - 3} more"
        message += "."

    # Notify
    from proactive.executor import _create_notification
    action = _create_notification(proactive_event, {
        "message": message,
        "priority": "normal"
    })

    if action:
        update_event_status(proactive_event.id, "actioned")
        notify_action(action)


def _get_travel_time(title: str, location: str) -> int:
    """Estimate travel time in minutes based on event/location."""
    search_text = f"{title} {location}".lower()

    for keyword, minutes in TRAVEL_TIMES.items():
        if keyword in search_text:
            return minutes

    return TRAVEL_TIMES["default"]


def get_upcoming_events_summary(hours: int = 24) -> str:
    """Get a summary of upcoming events (for briefing)."""
    now = datetime.now(EASTERN)
    end = now + timedelta(hours=hours)

    events = list_events(now, end)

    if not events:
        return "No upcoming events in the next 24 hours."

    lines = []
    for event in events[:5]:
        title = event.get("title", "Untitled")
        start_str = event.get("start", "")
        try:
            start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            if start.date() == now.date():
                time_str = f"today at {start.strftime('%I:%M %p').lstrip('0')}"
            elif start.date() == (now + timedelta(days=1)).date():
                time_str = f"tomorrow at {start.strftime('%I:%M %p').lstrip('0')}"
            else:
                time_str = start.strftime("%A at %I:%M %p").lstrip("0")
        except:
            time_str = start_str

        lines.append(f"- {title}: {time_str}")

    if len(events) > 5:
        lines.append(f"- ...and {len(events) - 5} more")

    return "\n".join(lines)
