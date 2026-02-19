"""Google Calendar API client for Doris.

Manages a dedicated secondary calendar (e.g., school events, activities).
Events can be synced from ICS feeds, email extraction, or manual creation.

This module handles CRUD operations on the calendar via the Google
Calendar API.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger("doris.google_cal")

# Config persistence
DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = DATA_DIR / "school_calendar.json"

# Default calendar name
# Configure with your school calendar name
DEFAULT_CALENDAR_NAME = "School Calendar"


def _load_config() -> dict:
    """Load school calendar config from disk."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load school calendar config: {e}")
    return {}


def _save_config(config: dict) -> None:
    """Save school calendar config to disk."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(config, indent=2))
    except OSError as e:
        logger.error(f"Failed to save school calendar config: {e}")
        raise


def get_calendar_service():
    """Build Google Calendar API service using shared Gmail credentials."""
    from tools.gmail import _get_credentials
    creds = _get_credentials()
    return build('calendar', 'v3', credentials=creds)


def get_calendar_id() -> Optional[str]:
    """Get the stored school calendar ID, or None if not yet created."""
    config = _load_config()
    return config.get("calendar_id")


def make_source_uid(source: str, original_id: str) -> str:
    """Create a deterministic source UID for dedup.

    Args:
        source: Source identifier (e.g., 'parentsquare', 'email')
        original_id: Source-specific unique ID

    Returns:
        Hash string like 'parentsquare:abc123...'
    """
    raw = f"{source}:{original_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def create_school_calendar(name: str = DEFAULT_CALENDAR_NAME) -> str:
    """Create a secondary Google Calendar for school events.

    If a calendar already exists (stored in config), returns that ID.
    Otherwise creates a new one.

    Args:
        name: Calendar display name

    Returns:
        Calendar ID string

    Raises:
        HttpError: If Google API call fails
    """
    config = _load_config()

    # Return existing if we have one
    existing_id = config.get("calendar_id")
    if existing_id:
        # Verify it still exists
        try:
            service = get_calendar_service()
            service.calendars().get(calendarId=existing_id).execute()
            logger.info(f"School calendar already exists: {existing_id}")
            return existing_id
        except HttpError as e:
            if e.resp.status == 404:
                logger.warning("Stored calendar ID no longer exists, creating new one")
            else:
                raise

    service = get_calendar_service()
    calendar_body = {
        "summary": name,
        "description": "School events. Auto-synced from ParentSquare, teacher emails, and class doc.",
        "timeZone": "America/New_York",
    }

    created = service.calendars().insert(body=calendar_body).execute()
    calendar_id = created["id"]

    config["calendar_id"] = calendar_id
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    config["school"] = ""  # Configure with your school name
    config["class"] = ""   # Configure with your class identifier
    config["calendar_name"] = name
    _save_config(config)

    logger.info(f"Created school calendar: {calendar_id}")
    return calendar_id


def upsert_event(calendar_id: str, event: dict, source_uid: str) -> dict:
    """Insert or update a calendar event, deduped by source_uid.

    Args:
        calendar_id: Target Google Calendar ID
        event: Google Calendar event body (summary, start, end, etc.)
        source_uid: Unique identifier from make_source_uid()

    Returns:
        The created/updated event resource

    Raises:
        HttpError: If Google API call fails
    """
    service = get_calendar_service()

    # Tag the event with source_uid for dedup
    event.setdefault("extendedProperties", {})
    event["extendedProperties"].setdefault("private", {})
    event["extendedProperties"]["private"]["source_uid"] = source_uid

    # Check if event with this source_uid already exists
    existing = _find_event_by_source_uid(service, calendar_id, source_uid)

    if existing:
        # Update existing event
        event_id = existing["id"]
        updated = service.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event
        ).execute()
        logger.debug(f"Updated event '{event.get('summary', '?')}' ({event_id})")
        return updated
    else:
        # Insert new event
        created = service.events().insert(
            calendarId=calendar_id,
            body=event
        ).execute()
        logger.info(f"Created event '{event.get('summary', '?')}' ({created['id']})")
        return created


def _find_event_by_source_uid(service, calendar_id: str, source_uid: str) -> Optional[dict]:
    """Find an existing event by its source_uid in extendedProperties.

    Uses sharedExtendedProperty query if possible, falls back to listing.
    """
    try:
        # Google Calendar API supports querying by privateExtendedProperty
        result = service.events().list(
            calendarId=calendar_id,
            privateExtendedProperty=f"source_uid={source_uid}",
            maxResults=1,
            singleEvents=False,
        ).execute()
        items = result.get("items", [])
        return items[0] if items else None
    except HttpError as e:
        logger.error(f"Error searching for event by source_uid: {e}")
        return None


def delete_event(calendar_id: str, event_id: str) -> bool:
    """Delete a calendar event.

    Args:
        calendar_id: Google Calendar ID
        event_id: Event ID to delete

    Returns:
        True if deleted, False if not found or error
    """
    try:
        service = get_calendar_service()
        service.events().delete(
            calendarId=calendar_id,
            eventId=event_id
        ).execute()
        logger.info(f"Deleted event {event_id}")
        return True
    except HttpError as e:
        if e.resp.status == 404:
            logger.debug(f"Event {event_id} already deleted")
            return True  # Already gone, that's fine
        logger.error(f"Failed to delete event {event_id}: {e}")
        return False


def delete_event_by_source_uid(calendar_id: str, source_uid: str) -> bool:
    """Delete a calendar event by its source_uid.

    Returns:
        True if deleted or not found, False on error
    """
    service = get_calendar_service()
    existing = _find_event_by_source_uid(service, calendar_id, source_uid)
    if existing:
        return delete_event(calendar_id, existing["id"])
    return True  # Not found = already gone


def share_calendar(calendar_id: str, public: bool = True) -> Optional[str]:
    """Make the calendar publicly readable and return the subscription URL.

    Args:
        calendar_id: Google Calendar ID
        public: If True, make publicly readable. If False, remove public access.

    Returns:
        Public iCal subscription URL, or None on error
    """
    service = get_calendar_service()

    if public:
        acl_body = {
            "role": "reader",
            "scope": {
                "type": "default",  # Anyone
            }
        }
        try:
            service.acl().insert(
                calendarId=calendar_id,
                body=acl_body
            ).execute()
            logger.info(f"Calendar {calendar_id} is now public")
        except HttpError as e:
            if "duplicate" not in str(e).lower():
                logger.error(f"Failed to share calendar: {e}")
                return None
            # Already shared, that's fine

    # Build the subscription URL
    ical_url = f"https://calendar.google.com/calendar/ical/{calendar_id}/public/basic.ics"

    # Also store it in config
    config = _load_config()
    config["public_ical_url"] = ical_url
    config["shared_at"] = datetime.now(timezone.utc).isoformat()
    _save_config(config)

    return ical_url


def list_events(
    calendar_id: str,
    time_min: Optional[datetime] = None,
    time_max: Optional[datetime] = None,
    max_results: int = 50,
) -> list[dict]:
    """List events from the school calendar.

    Args:
        calendar_id: Google Calendar ID
        time_min: Start of time range (defaults to now)
        time_max: End of time range (defaults to 30 days from now)
        max_results: Max events to return

    Returns:
        List of event dicts with summary, start, end, description, location
    """
    from datetime import timedelta

    service = get_calendar_service()

    if time_min is None:
        time_min = datetime.now(timezone.utc)
    if time_max is None:
        time_max = time_min + timedelta(days=30)

    # Ensure timezone-aware
    if time_min.tzinfo is None:
        time_min = time_min.replace(tzinfo=timezone.utc)
    if time_max.tzinfo is None:
        time_max = time_max.replace(tzinfo=timezone.utc)

    try:
        result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min.isoformat(),
            timeMax=time_max.isoformat(),
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        return result.get("items", [])
    except HttpError as e:
        logger.error(f"Failed to list events: {e}")
        return []


def list_all_source_uids(calendar_id: str) -> dict[str, str]:
    """List all source_uids currently in the calendar.

    Returns:
        Dict mapping source_uid -> event_id
    """
    service = get_calendar_service()
    uid_map = {}
    page_token = None

    while True:
        try:
            result = service.events().list(
                calendarId=calendar_id,
                maxResults=250,
                singleEvents=False,
                pageToken=page_token,
            ).execute()

            for event in result.get("items", []):
                props = event.get("extendedProperties", {}).get("private", {})
                uid = props.get("source_uid")
                if uid:
                    uid_map[uid] = event["id"]

            page_token = result.get("nextPageToken")
            if not page_token:
                break
        except HttpError as e:
            logger.error(f"Failed to list source_uids: {e}")
            break

    return uid_map
