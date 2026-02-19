import subprocess
import json
import re
from datetime import datetime as dt, timedelta, timezone
from pathlib import Path

from security.sanitize import sanitize_subprocess_arg, validate_id

CALENDAR_CLI = Path.home() / "Projects/doris-calendar/.build/release/doris-calendar"

# Hidden marker for Doris-created events (appended to notes field)
# Format: [doris:source:timestamp] where source is voice|email|scout|api
DORIS_MARKER_PREFIX = "\n\n[doris:"
# Match marker at end of string, with optional \n\n prefix (for marker-only case)
DORIS_MARKER_PATTERN = re.compile(r'(?:\n\n)?\[doris:[^\]]+\]$')


def _add_doris_marker(notes: str | None, source: str = "api") -> str:
    """Append Doris attribution marker to notes."""
    timestamp = dt.now().strftime('%Y-%m-%dT%H:%M:%S')
    marker = f"{DORIS_MARKER_PREFIX}{source}:{timestamp}]"
    if notes:
        return notes + marker
    return marker.lstrip('\n')


def _strip_doris_marker(notes: str | None) -> tuple[str | None, dict | None]:
    """Remove Doris marker from notes, return (clean_notes, marker_info).

    marker_info is a dict with 'source' and 'timestamp' if marker was found.
    """
    if not notes:
        return None, None

    match = DORIS_MARKER_PATTERN.search(notes)
    if not match:
        return notes, None

    # Parse the marker
    marker = match.group(0)
    clean_notes = notes[:match.start()].rstrip('\n') or None  # Remove trailing newlines, None if empty

    # Extract source and timestamp from [doris:source:timestamp]
    try:
        # Strip whitespace and brackets, then parse
        marker_content = marker.strip().lstrip('\n').strip('[]')
        parts = marker_content.split(':')
        if len(parts) >= 3:
            source = parts[1]
            timestamp = ':'.join(parts[2:])  # Rejoin timestamp (has colons)
            return clean_notes, {'source': source, 'timestamp': timestamp, 'created_by': 'doris'}
    except Exception:
        pass

    return clean_notes, {'created_by': 'doris'}


def list_events(start: dt = None, end: dt = None, include_creator: bool = True) -> list[dict]:
    """List calendar events from Indestructible calendar.

    If include_creator is True (default), adds 'created_by' field to events
    that were created by Doris, and strips the internal marker from notes.
    """
    from zoneinfo import ZoneInfo
    local_tz = ZoneInfo("America/New_York")

    args = [str(CALENDAR_CLI), "list"]

    if start:
        # Ensure timezone info is included so Swift CLI interprets dates correctly
        if start.tzinfo is None:
            start = start.replace(tzinfo=local_tz)
        args.append(start.isoformat())
    if end:
        if end.tzinfo is None:
            end = end.replace(tzinfo=local_tz)
        args.append(end.isoformat())

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    data = json.loads(result.stdout)
    events = data.get("events", [])

    if include_creator:
        for event in events:
            notes = event.get('notes')
            clean_notes, marker_info = _strip_doris_marker(notes)
            event['notes'] = clean_notes or ''
            if marker_info:
                event['created_by'] = 'doris'
                event['doris_source'] = marker_info.get('source')
                event['doris_created_at'] = marker_info.get('timestamp')

    return events


def create_event(title: str, start: dt, end: dt, notes: str = None, location: str = None,
                 source: str = "api", alert_minutes: int = 60,
                 recurrence: str = None, recurrence_end: dt = None) -> dict:
    """Create a calendar event with Doris attribution and default alert.

    Args:
        title: Event title
        start: Start datetime
        end: End datetime
        notes: Optional notes/description
        location: Optional location
        source: Who/what created this event (voice, email, scout, api)
        alert_minutes: Minutes before event to show alert (default 60, None to disable)
        recurrence: Recurrence pattern (daily, weekly, weekdays, biweekly, monthly, yearly)
        recurrence_end: Optional end date for recurring events

    Returns:
        Dict with success status and event id
    """
    # CLI requires ISO8601 with timezone - use Eastern time
    start_iso = start.strftime('%Y-%m-%dT%H:%M:%S-05:00')
    end_iso = end.strftime('%Y-%m-%dT%H:%M:%S-05:00')

    # Add Doris attribution marker to notes
    notes_with_marker = _add_doris_marker(notes, source)

    # Sanitize user-controlled strings before passing to subprocess
    safe_title = sanitize_subprocess_arg(title, "title")
    safe_notes = sanitize_subprocess_arg(notes_with_marker or "", "notes")
    safe_location = sanitize_subprocess_arg(location or "", "location")

    args = [str(CALENDAR_CLI), "create", safe_title, start_iso, end_iso]

    # Args are positional: title, start, end, notes, location, alertMinutes, recurrence, recurrenceEnd
    # Use empty string for optional args we want to skip
    args.append(safe_notes)
    args.append(safe_location)
    args.append(str(alert_minutes) if alert_minutes is not None else "60")
    args.append(recurrence or "")
    if recurrence_end:
        args.append(recurrence_end.strftime('%Y-%m-%dT%H:%M:%S-05:00'))

    result = subprocess.run(args, capture_output=True, text=True)
    return json.loads(result.stdout)


def parse_natural_date(date_str: str, allow_past: bool = False) -> dt:
    """Parse natural language date like 'tomorrow', 'monday', 'january 15'.

    Also validates day-of-week if specified (e.g., 'Tuesday, January 27').
    If the day doesn't match the date, logs a warning but trusts the explicit date.

    Args:
        date_str: Natural language date string
        allow_past: If False (default), dates in the past are bumped to next year.
                   If True, allows past dates (useful for querying historical events).
    """
    date_str_original = date_str
    date_str = date_str.lower().strip()
    today = dt.now()

    # Day-of-week names for validation
    day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    # Check if string contains a day-of-week name (for validation later)
    mentioned_day = None
    for i, day in enumerate(day_names):
        if day in date_str:
            mentioned_day = i
            break

    if date_str == "today":
        return today
    elif date_str == "tomorrow":
        return today + timedelta(days=1)
    elif date_str in day_names:
        # Just a day name like "monday" - find next occurrence
        target_day = day_names.index(date_str)
        current_day = today.weekday()
        days_ahead = target_day - current_day
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    else:
        # Try parsing as a date like "january 15" or "jan 15" or "Tuesday, January 27"
        import dateutil.parser
        try:
            parsed = dateutil.parser.parse(date_str, fuzzy=True)
            # If no year specified and date is in past, assume next year (unless allow_past=True)
            # Compare dates only (not datetime) to avoid issues with time-of-day
            if not allow_past and parsed.year == today.year and parsed.date() < today.date():
                parsed = parsed.replace(year=today.year + 1)

            # Validate day-of-week if one was mentioned
            if mentioned_day is not None:
                actual_day = parsed.weekday()
                if actual_day != mentioned_day:
                    # Day-of-week doesn't match! Log warning but trust the explicit date
                    expected_day_name = day_names[mentioned_day].capitalize()
                    actual_day_name = day_names[actual_day].capitalize()
                    print(f"[cal] WARNING: Date mismatch in '{date_str_original}' - "
                          f"said {expected_day_name} but {parsed.strftime('%B %d, %Y')} is actually {actual_day_name}. "
                          f"Using the explicit date.")

            return parsed
        except Exception:
            raise ValueError(f"Could not parse date: {date_str}")


def parse_time(time_str: str) -> tuple[int, int]:
    """Parse time string like '2pm', '14:00', '2:30 PM' into (hour, minute)."""
    time_str = time_str.lower().strip()

    # Handle formats like "2pm", "2:30pm", "14:00"
    import re

    # Try "2:30pm" or "2:30 pm"
    match = re.match(r'(\d{1,2}):(\d{2})\s*(am|pm)?', time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        ampm = match.group(3)
        if ampm == 'pm' and hour < 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        return (hour, minute)

    # Try "2pm" or "2 pm"
    match = re.match(r'(\d{1,2})\s*(am|pm)', time_str)
    if match:
        hour = int(match.group(1))
        ampm = match.group(2)
        if ampm == 'pm' and hour < 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        return (hour, 0)

    # Try "14:00" (24-hour)
    match = re.match(r'(\d{1,2}):(\d{2})$', time_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    raise ValueError(f"Could not parse time: {time_str}")


def create_event_natural(title: str, date_str: str, time_str: str = None,
                         duration_minutes: int = 60, location: str = None,
                         source: str = "voice", recurrence: str = None,
                         recurrence_end_str: str = None) -> str:
    """Create a calendar event using natural language date/time.

    Args:
        title: Event title
        date_str: Natural language date (e.g., 'tomorrow', 'monday', 'january 15')
        time_str: Optional time (e.g., '2pm', '14:00')
        duration_minutes: Event duration (default 60)
        location: Optional location
        source: What created this (voice, email, scout, api) - for attribution
        recurrence: Recurrence pattern (daily, weekly, weekdays, biweekly, monthly, yearly)
        recurrence_end_str: Optional end date for recurring events (natural language)

    Returns:
        Confirmation message or error string
    """
    try:
        event_date = parse_natural_date(date_str)

        if time_str:
            hour, minute = parse_time(time_str)
            start = event_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        else:
            # Default to 9am if no time specified
            start = event_date.replace(hour=9, minute=0, second=0, microsecond=0)

        end = start + timedelta(minutes=duration_minutes)

        # Parse recurrence end date if provided
        recurrence_end = None
        if recurrence_end_str:
            recurrence_end = parse_natural_date(recurrence_end_str)

        result = create_event(title, start, end, location=location, source=source,
                             recurrence=recurrence, recurrence_end=recurrence_end)
        if result.get("success"):
            msg = f"Created '{title}' on {start.strftime('%A, %B %d')} at {start.strftime('%I:%M %p').lstrip('0')}"
            if recurrence:
                msg += f" (repeats {recurrence})"
            return msg
        else:
            return f"Failed to create event: {result.get('error', 'Unknown error')}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error creating event: {str(e)}"


def get_day_bounds(target_date: dt) -> tuple[dt, dt]:
    """Get start and end of a specific day."""
    start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


def get_events_for_date(target_date: dt) -> list[dict]:
    """Get events for a specific date."""
    start, end = get_day_bounds(target_date)
    events = list_events(start, end)
    date_str = target_date.strftime("%Y-%m-%d")

    # Filter to events that actually start on this date
    # For all-day events, also check if the event spans this date
    result = []
    for e in events:
        event_start = e['start'][:10]  # YYYY-MM-DD
        if event_start == date_str:
            result.append(e)
        elif e['isAllDay']:
            # Check if this all-day event spans our target date
            # Use < for end date (all-day events end at midnight, exclusive)
            event_end = e.get('end', '')[:10]
            if event_start <= date_str < event_end:
                result.append(e)
    return result


def get_todays_events() -> list[dict]:
    """Get today's events only."""
    return get_events_for_date(dt.now())


def get_tomorrows_events() -> list[dict]:
    """Get tomorrow's events."""
    return get_events_for_date(dt.now() + timedelta(days=1))


def get_weeks_events() -> list[dict]:
    """Get this week's events (next 7 days)."""
    now = dt.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=7)
    return list_events(start, end)


def get_weekend_events() -> list[dict]:
    """Get this weekend's events (Saturday and Sunday)."""
    now = dt.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Find next Saturday (weekday 5)
    days_until_saturday = (5 - today.weekday()) % 7
    if days_until_saturday == 0 and today.weekday() == 5:
        # It's Saturday
        saturday = today
    elif today.weekday() == 6:
        # It's Sunday - get today and we're done with "this weekend"
        saturday = today - timedelta(days=1)
    else:
        saturday = today + timedelta(days=days_until_saturday)
    
    sunday_end = saturday + timedelta(days=2)
    return list_events(saturday, sunday_end)


def get_events_for_weekday(weekday_name: str) -> list[dict]:
    """
    Get events for a specific day of the week.
    weekday_name: 'monday', 'tuesday', etc.
    Returns events for the next occurrence of that day.
    """
    weekdays = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    target_weekday = weekdays.get(weekday_name.lower())
    if target_weekday is None:
        return []
    
    now = dt.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    current_weekday = today.weekday()
    
    # Calculate days until target weekday
    days_ahead = target_weekday - current_weekday
    if days_ahead < 0:  # Target day already happened this week
        days_ahead += 7
    elif days_ahead == 0:  # It's today
        pass  # days_ahead stays 0
    
    target_date = today + timedelta(days=days_ahead)
    return get_events_for_date(target_date)


def delete_event(event_id: str) -> dict:
    """Delete a calendar event by ID."""
    safe_id = validate_id(event_id, "event_id")
    args = [str(CALENDAR_CLI), "delete", safe_id]
    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode != 0:
        return {"success": False, "error": result.stderr or "Failed to delete event"}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        # CLI might just return success message
        return {"success": True}


def update_event(event_id: str, title: str = None, start: dt = None,
                 end: dt = None, notes: str = None, location: str = None) -> dict:
    """
    Update a calendar event using create-first/delete-old pattern.

    Creates the replacement event first, then deletes the original. If the
    create step fails the original event is untouched. If the delete step
    fails we return the new event ID and flag the orphaned original.
    """
    # First, get the existing event to preserve any fields not being updated
    all_events = list_events(
        dt.now() - timedelta(days=30),
        dt.now() + timedelta(days=365)
    )

    existing = None
    for e in all_events:
        if e.get('id') == event_id:
            existing = e
            break

    if not existing:
        return {"success": False, "error": f"Event with ID {event_id} not found"}

    # Use existing values for any field not provided
    final_title = title if title is not None else existing.get('title', 'Untitled')

    if start is not None:
        final_start = start
    else:
        start_str = existing.get('start', '')
        final_start = dt.fromisoformat(start_str.replace('Z', '+00:00'))

    if end is not None:
        final_end = end
    else:
        end_str = existing.get('end', '')
        final_end = dt.fromisoformat(end_str.replace('Z', '+00:00'))

    final_location = location if location is not None else existing.get('location')
    final_notes = notes if notes is not None else existing.get('notes')

    # Create-first: create the replacement event before touching the original
    create_result = create_event(final_title, final_start, final_end,
                                  notes=final_notes, location=final_location)

    if not create_result.get('success', False):
        # Create failed — original event is untouched, no data loss
        return {
            "success": False,
            "error": f"Failed to create replacement event: {create_result.get('error')}",
        }

    new_id = create_result.get('id')

    # Delete-old: remove the original now that the replacement exists
    delete_result = delete_event(event_id)
    if not delete_result.get('success', False):
        # Delete failed — new event exists but old one is orphaned
        # This is safe: user has duplicate events, not zero events
        return {
            "success": True,
            "message": f"Updated event '{final_title}' (note: old event could not be removed)",
            "new_id": new_id,
            "warning": f"Original event {event_id} may still exist: {delete_result.get('error')}",
        }

    return {
        "success": True,
        "message": f"Updated event '{final_title}'",
        "new_id": new_id,
    }


def move_event(event_id: str, new_date_str: str, new_time_str: str = None,
               duration_minutes: int = None) -> str:
    """
    Move a calendar event to a new date/time using natural language.

    Args:
        event_id: The ID of the event to move
        new_date_str: Natural language date like "tomorrow", "monday", "january 28"
        new_time_str: Optional new time like "2pm", "14:00" (keeps original if not specified)
        duration_minutes: Optional new duration (keeps original if not specified)
    """
    try:
        # Get existing event
        all_events = list_events(
            dt.now() - timedelta(days=30),
            dt.now() + timedelta(days=365)
        )

        existing = None
        for e in all_events:
            if e.get('id') == event_id:
                existing = e
                break

        if not existing:
            return f"Event with ID {event_id} not found"

        # Parse new date
        new_date = parse_natural_date(new_date_str)

        # Get original start time if no new time specified
        if new_time_str:
            hour, minute = parse_time(new_time_str)
        else:
            # Extract time from original event
            orig_start = dt.fromisoformat(existing['start'].replace('Z', '+00:00'))
            hour, minute = orig_start.hour, orig_start.minute

        new_start = new_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # Calculate duration from original if not specified
        if duration_minutes:
            new_end = new_start + timedelta(minutes=duration_minutes)
        else:
            orig_start = dt.fromisoformat(existing['start'].replace('Z', '+00:00'))
            orig_end = dt.fromisoformat(existing['end'].replace('Z', '+00:00'))
            original_duration = orig_end - orig_start
            new_end = new_start + original_duration

        # Update the event
        result = update_event(
            event_id,
            start=new_start,
            end=new_end
        )

        if result.get('success'):
            return f"Moved '{existing['title']}' to {new_start.strftime('%A, %B %d')} at {new_start.strftime('%I:%M %p').lstrip('0')}"
        else:
            return f"Failed to move event: {result.get('error')}"

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error moving event: {str(e)}"


def parse_calendar_query(message: str) -> tuple[list[dict], str] | None:
    """
    Parse a natural language calendar query and return events + period description.
    Returns (events, period_string) or None if not a calendar query.
    """
    msg_lower = message.lower()
    
    # Check if this is a calendar query
    calendar_triggers = [
        'calendar', 'schedule', 'what do i have', "what's on", 'whats on',
        'any events', 'any meetings', 'am i free', 'do i have anything',
        'what am i doing', "what's happening", 'whats happening', 'plans for'
    ]
    
    if not any(trigger in msg_lower for trigger in calendar_triggers):
        return None
    
    # Check for specific time references
    weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    # Today
    if 'today' in msg_lower:
        return get_todays_events(), 'today'
    
    # Tomorrow
    if 'tomorrow' in msg_lower:
        return get_tomorrows_events(), 'tomorrow'
    
    # This weekend
    if 'weekend' in msg_lower:
        return get_weekend_events(), 'this weekend'
    
    # This week / next few days
    if 'week' in msg_lower or 'next few days' in msg_lower:
        return get_weeks_events(), 'this week'
    
    # Specific weekday
    for day in weekdays:
        if day in msg_lower:
            events = get_events_for_weekday(day)
            # Figure out if it's "this X" or "next X"
            now = dt.now()
            target_weekday = weekdays.index(day)
            days_ahead = target_weekday - now.weekday()
            if days_ahead < 0:
                days_ahead += 7
            
            if days_ahead == 0:
                period = 'today'
            elif days_ahead == 1:
                period = 'tomorrow'
            else:
                period = f'on {day.capitalize()}'
            
            return events, period
    
    # Default to today if no specific time mentioned
    return get_todays_events(), 'today'