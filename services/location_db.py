"""
Location tracking and context.

Stores location history from iOS app and provides location-aware context for Doris.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from math import radians, sin, cos, sqrt, atan2

DB_PATH = Path(__file__).parent.parent / "data" / "doris.db"

# Known locations (for context)
# NOTE: Configure your own known locations here or via add_known_location().
# Example entries shown with placeholder coordinates.
KNOWN_LOCATIONS = {
    # "home": {
    #     "lat": 0.0,
    #     "lon": 0.0,
    #     "radius_meters": 200,
    #     "name": "Home",
    #     "type": "home"
    # },
}


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_location_db():
    """Create location tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS location_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            accuracy REAL,
            speed REAL,
            source TEXT DEFAULT 'ios',
            resolved_name TEXT,
            location_type TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_location_timestamp
        ON location_history(timestamp DESC)
    """)

    conn.commit()
    conn.close()


def store_location(lat: float, lon: float, accuracy: float = None, speed: float = None,
                   source: str = "ios") -> dict:
    """
    Store a location update.

    Returns dict with location info including resolved known location if applicable.
    """
    init_location_db()

    conn = get_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    # Check if location matches a known place
    resolved_name = None
    location_type = None
    for loc_id, loc in KNOWN_LOCATIONS.items():
        distance = haversine_distance(lat, lon, loc["lat"], loc["lon"])
        if distance <= loc["radius_meters"]:
            resolved_name = loc["name"]
            location_type = loc["type"]
            break

    cursor.execute("""
        INSERT INTO location_history (timestamp, lat, lon, accuracy, speed, source, resolved_name, location_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, lat, lon, accuracy, speed, source, resolved_name, location_type))

    conn.commit()
    conn.close()

    return {
        "timestamp": timestamp,
        "lat": lat,
        "lon": lon,
        "resolved_name": resolved_name,
        "location_type": location_type
    }


def get_current_location() -> Optional[dict]:
    """Get the most recent location."""
    init_location_db()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, lat, lon, accuracy, speed, resolved_name, location_type
        FROM location_history
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "timestamp": row[0],
            "lat": round(row[1], 2),   # ~1.1km precision — avoid exposing exact GPS
            "lon": round(row[2], 2),
            "accuracy": row[3],
            "speed": row[4],
            "resolved_name": row[5],
            "location_type": row[6]
        }
    return None


def get_location_history(hours: int = 24) -> list[dict]:
    """Get location history for the last N hours."""
    init_location_db()

    conn = get_connection()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

    cursor.execute("""
        SELECT timestamp, lat, lon, accuracy, speed, resolved_name, location_type
        FROM location_history
        WHERE timestamp > ?
        ORDER BY timestamp DESC
    """, (cutoff,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "timestamp": row[0],
            "lat": round(row[1], 2),   # ~1.1km precision
            "lon": round(row[2], 2),
            "accuracy": row[3],
            "speed": row[4],
            "resolved_name": row[5],
            "location_type": row[6]
        }
        for row in rows
    ]


def get_location_context() -> dict:
    """
    Get current location context for Doris.

    Returns:
        - current_location: Where the user is now
        - is_home: Whether at a known home location
        - is_traveling: Whether movement suggests traveling
        - last_transition: Last significant location change
    """
    current = get_current_location()
    if not current:
        return {"available": False}

    history = get_location_history(hours=4)

    # Determine if at home
    is_home = current.get("location_type") == "home"

    # Check for recent movement (traveling indicator)
    is_traveling = False
    if len(history) >= 3:
        # Check if locations vary significantly
        unique_locations = set()
        for loc in history[:10]:
            if loc.get("resolved_name"):
                unique_locations.add(loc["resolved_name"])
            else:
                # Round to ~100m precision for comparison
                unique_locations.add((round(loc["lat"], 3), round(loc["lon"], 3)))

        if len(unique_locations) >= 3:
            is_traveling = True

    # Find last transition (when location type changed)
    last_transition = None
    if len(history) >= 2:
        current_type = history[0].get("resolved_name") or "unknown"
        for loc in history[1:]:
            loc_type = loc.get("resolved_name") or "unknown"
            if loc_type != current_type:
                last_transition = {
                    "from": loc_type,
                    "to": current_type,
                    "timestamp": history[0]["timestamp"]
                }
                break

    return {
        "available": True,
        "current_location": current.get("resolved_name") or f"{current['lat']:.2f}, {current['lon']:.2f}",
        "is_home": is_home,
        "is_traveling": is_traveling,
        "last_transition": last_transition,
        "timestamp": current["timestamp"]
    }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points in meters using Haversine formula.
    """
    R = 6371000  # Earth's radius in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def add_known_location(name: str, lat: float, lon: float, radius_meters: float = 200,
                       location_type: str = "place") -> None:
    """Add a new known location (runtime only, not persisted)."""
    loc_id = name.lower().replace(" ", "_")
    KNOWN_LOCATIONS[loc_id] = {
        "lat": lat,
        "lon": lon,
        "radius_meters": radius_meters,
        "name": name,
        "type": location_type
    }
