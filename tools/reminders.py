import subprocess
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from security.sanitize import sanitize_subprocess_arg, validate_id

REMINDERS_CLI = Path.home() / "Projects/doris-reminders/.build/release/doris-reminders"

# Known locations - coordinates for location-based reminders
# Configure via environment variables or override KNOWN_LOCATIONS directly
# e.g., DORIS_HOME_LAT=40.7128, DORIS_HOME_LON=-74.0060
_home_lat = float(os.getenv("DORIS_HOME_LAT", "0"))
_home_lon = float(os.getenv("DORIS_HOME_LON", "0"))
_secondary_lat = float(os.getenv("DORIS_SECONDARY_LAT", "0"))
_secondary_lon = float(os.getenv("DORIS_SECONDARY_LON", "0"))

KNOWN_LOCATIONS = {
    "home": {"lat": _home_lat, "lon": _home_lon, "name": "Home"},
    "apartment": {"lat": _home_lat, "lon": _home_lon, "name": "Home"},
    "the house": {"lat": _secondary_lat, "lon": _secondary_lon, "name": "Secondary Home"},
    "secondary": {"lat": _secondary_lat, "lon": _secondary_lon, "name": "Secondary Home"},
}


def list_reminders(include_completed: bool = False) -> list[dict]:
    """List all reminders."""
    args = [str(REMINDERS_CLI), "list"]
    if include_completed:
        args.append("--completed")

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    data = json.loads(result.stdout)
    return data.get("reminders", [])


def create_reminder(
    title: str,
    list_name: Optional[str] = None,
    due: Optional[datetime] = None,
    notes: Optional[str] = None,
    location: Optional[str] = None,
    trigger: str = "arrive",
    radius: float = 100
) -> dict:
    """
    Create a reminder with optional location-based trigger.

    Args:
        title: Reminder text
        list_name: Reminders list name (optional)
        due: Due date/time (optional)
        notes: Additional notes (optional)
        location: Location name from KNOWN_LOCATIONS or "lat,lon" string
        trigger: "arrive" or "depart"
        radius: Geofence radius in meters (default 100)

    Returns:
        dict with success status and reminder id
    """
    # Sanitize user-controlled strings before passing to subprocess
    safe_title = sanitize_subprocess_arg(title, "title")
    args = [str(REMINDERS_CLI), "create", safe_title]

    args.append(sanitize_subprocess_arg(list_name or "", "list_name"))
    # Convert naive local datetime to UTC ISO8601 for Swift's parser
    if due:
        utc_dt = due.astimezone(timezone.utc)
        args.append(utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
    else:
        args.append("")
    if notes:
        args.append(sanitize_subprocess_arg(notes, "notes"))

    # Add location params if provided
    if location:
        loc_data = resolve_location(location)
        if loc_data:
            args.extend(["--location", f"{loc_data['lat']},{loc_data['lon']}"])
            args.extend(["--location-name", loc_data['name']])
            args.extend(["--trigger", trigger])
            args.extend(["--radius", str(radius)])

    result = subprocess.run(args, capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": result.stderr or "Unknown error"}


def resolve_location(location: str) -> Optional[dict]:
    """
    Resolve a location name or coordinates to lat/lon/name dict.

    Args:
        location: Either a known location name or "lat,lon" string

    Returns:
        dict with lat, lon, name keys or None if not found
    """
    # Check known locations first
    loc_lower = location.lower().strip()
    if loc_lower in KNOWN_LOCATIONS:
        return KNOWN_LOCATIONS[loc_lower]

    # Try parsing as coordinates
    if "," in location:
        try:
            parts = location.split(",")
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            return {"lat": lat, "lon": lon, "name": "Custom Location"}
        except (ValueError, IndexError):
            pass

    return None


def get_known_location_names() -> list[str]:
    """Get list of known location names."""
    return list(KNOWN_LOCATIONS.keys())

def complete_reminder(reminder_id: str) -> dict:
    """Mark a reminder as complete."""
    safe_id = validate_id(reminder_id, "reminder_id")
    args = [str(REMINDERS_CLI), "complete", safe_id]
    result = subprocess.run(args, capture_output=True, text=True)
    return json.loads(result.stdout)

def get_shopping_list() -> list[dict]:
    """Get items from the Shopping List."""
    reminders = list_reminders()
    return [r for r in reminders if r.get("list") == "Shopping List"]

def get_family_reminders() -> list[dict]:
    """Get items from the Family list."""
    reminders = list_reminders()
    return [r for r in reminders if r.get("list") == "Family"]
