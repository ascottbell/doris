"""
Photos metadata extraction for Doris memory.

Extracts metadata from Apple Photos library:
- Location patterns (where photos are taken)
- Time patterns (when photos are taken)
- Travel detection (clusters of photos away from home)
"""

import os
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import tempfile

PHOTOS_DB = Path.home() / "Pictures/Photos Library.photoslibrary/database/Photos.sqlite"


def get_photos_db_copy():
    """Copy the Photos database to a temp location (can't read while Photos has lock)."""
    if not PHOTOS_DB.exists():
        return None

    temp_db = Path(tempfile.gettempdir()) / "photos_copy.sqlite"
    shutil.copy2(PHOTOS_DB, temp_db)
    return temp_db


def extract_photo_metadata(limit: int = None) -> list[dict]:
    """Extract metadata from all photos."""
    db_path = get_photos_db_copy()
    if not db_path:
        print("[Photos] Database not found")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query photos with location and date info
    query = """
        SELECT
            ZASSET.ZDATECREATED,
            ZASSET.ZLATITUDE,
            ZASSET.ZLONGITUDE,
            ZASSET.ZDIRECTORY,
            ZADDITIONALASSETATTRIBUTES.ZORIGINALFILENAME
        FROM ZASSET
        LEFT JOIN ZADDITIONALASSETATTRIBUTES
            ON ZASSET.Z_PK = ZADDITIONALASSETATTRIBUTES.ZASSET
        WHERE ZASSET.ZTRASHEDSTATE = 0
        ORDER BY ZASSET.ZDATECREATED DESC
    """

    params = []
    if limit:
        query += " LIMIT ?"
        params.append(int(limit))

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Clean up temp file
    db_path.unlink(missing_ok=True)

    photos = []
    # Apple's Core Data timestamp epoch is Jan 1, 2001
    apple_epoch = datetime(2001, 1, 1)

    for row in rows:
        date_val, lat, lon, directory, filename = row

        if date_val:
            try:
                photo_date = apple_epoch + timedelta(seconds=date_val)
            except:
                photo_date = None
        else:
            photo_date = None

        photos.append({
            "date": photo_date,
            "latitude": lat,
            "longitude": lon,
            "filename": filename,
            "has_location": lat is not None and lon is not None
        })

    return photos


def analyze_photo_patterns(photos: list[dict]) -> dict:
    """Analyze photos for patterns."""

    # Home location — configure with your own coordinates
    # Example: HOME_LAT, HOME_LON = 40.7128, -74.0060  # NYC
    HOME_LAT, HOME_LON = float(os.environ.get("DORIS_HOME_LAT", "0")), float(os.environ.get("DORIS_HOME_LON", "0"))
    HOME_RADIUS = 0.02  # ~1.4 miles

    # Second home / vacation home — configure with your own coordinates
    HV_LAT, HV_LON = float(os.environ.get("DORIS_SECONDARY_LAT", "0")), float(os.environ.get("DORIS_SECONDARY_LON", "0"))
    HV_RADIUS = 0.07  # ~5 miles

    def is_near(lat, lon, target_lat, target_lon, radius):
        if lat is None or lon is None:
            return False
        return abs(lat - target_lat) < radius and abs(lon - target_lon) < radius

    # Counts
    total = len(photos)
    with_location = sum(1 for p in photos if p["has_location"])

    # Location breakdown
    at_home = 0
    at_second_home = 0
    elsewhere = 0

    # Time patterns
    by_month = defaultdict(int)
    by_day_of_week = defaultdict(int)
    by_year = defaultdict(int)

    # Travel clusters (photos not at home/HV)
    travel_locations = []

    for p in photos:
        lat, lon = p.get("latitude"), p.get("longitude")
        date = p.get("date")

        if lat and lon:
            if is_near(lat, lon, HOME_LAT, HOME_LON, HOME_RADIUS):
                at_home += 1
            elif is_near(lat, lon, HV_LAT, HV_LON, HV_RADIUS):
                at_second_home += 1
            else:
                elsewhere += 1
                if date:
                    travel_locations.append({
                        "date": date,
                        "lat": lat,
                        "lon": lon
                    })

        if date:
            by_month[date.strftime("%B")] += 1
            by_day_of_week[date.strftime("%A")] += 1
            by_year[date.year] += 1

    # Find travel clusters (multiple photos in same area on same trip)
    travel_clusters = []
    if travel_locations:
        # Sort by date
        travel_locations.sort(key=lambda x: x["date"])

        # Simple clustering: group photos within 3 days and similar location
        current_cluster = [travel_locations[0]] if travel_locations else []

        for loc in travel_locations[1:]:
            last = current_cluster[-1] if current_cluster else None
            if last:
                days_apart = (loc["date"] - last["date"]).days
                dist = ((loc["lat"] - last["lat"])**2 + (loc["lon"] - last["lon"])**2)**0.5

                if days_apart <= 3 and dist < 0.5:  # Same trip
                    current_cluster.append(loc)
                else:
                    if len(current_cluster) >= 5:  # Significant trip
                        travel_clusters.append({
                            "start": current_cluster[0]["date"],
                            "end": current_cluster[-1]["date"],
                            "photos": len(current_cluster),
                            "lat": sum(l["lat"] for l in current_cluster) / len(current_cluster),
                            "lon": sum(l["lon"] for l in current_cluster) / len(current_cluster)
                        })
                    current_cluster = [loc]

        # Don't forget last cluster
        if len(current_cluster) >= 5:
            travel_clusters.append({
                "start": current_cluster[0]["date"],
                "end": current_cluster[-1]["date"],
                "photos": len(current_cluster),
                "lat": sum(l["lat"] for l in current_cluster) / len(current_cluster),
                "lon": sum(l["lon"] for l in current_cluster) / len(current_cluster)
            })

        # Sort by most recent first
        travel_clusters.sort(key=lambda x: x["end"], reverse=True)

    return {
        "total_photos": total,
        "with_location": with_location,
        "location_rate": round(with_location / total * 100, 1) if total else 0,
        "at_home": at_home,
        "at_second_home": at_second_home,
        "elsewhere": elsewhere,
        "by_month": dict(by_month),
        "by_day_of_week": dict(by_day_of_week),
        "by_year": dict(by_year),
        "travel_clusters": travel_clusters[:10]  # Top 10 trips
    }


def store_photo_insights():
    """Extract photos and store insights in memory."""
    from memory.store import store_memory

    print("[Photos] Extracting metadata from Photos library...")
    photos = extract_photo_metadata()

    if not photos:
        print("[Photos] No photos found")
        return

    print(f"[Photos] Analyzing {len(photos)} photos...")
    analysis = analyze_photo_patterns(photos)

    stored = 0

    # Store basic stats
    content = f"User has {analysis['total_photos']:,} photos, {analysis['location_rate']}% have location data"
    store_memory(content=content, category="preference", subject="Photos", source="photos:metadata", confidence=0.95)
    stored += 1
    print(f"[Photos] Stored: {content}")

    # Location breakdown
    if analysis["at_home"] or analysis["at_second_home"]:
        home_pct = round(analysis["at_home"] / max(analysis["with_location"], 1) * 100)
        hv_pct = round(analysis["at_second_home"] / max(analysis["with_location"], 1) * 100)
        content = f"Photo locations: {home_pct}% at home, {hv_pct}% at second home, rest traveling"
        store_memory(content=content, category="preference", subject="Photos", source="photos:metadata", confidence=0.9)
        stored += 1
        print(f"[Photos] Stored: {content}")

    # Most active months
    if analysis["by_month"]:
        top_months = sorted(analysis["by_month"].items(), key=lambda x: -x[1])[:3]
        content = f"Most photos taken in: {', '.join(m[0] for m in top_months)}"
        store_memory(content=content, category="preference", subject="Photos", source="photos:metadata", confidence=0.85)
        stored += 1
        print(f"[Photos] Stored: {content}")

    # Travel clusters (trips)
    for i, trip in enumerate(analysis["travel_clusters"][:5]):
        start = trip["start"].strftime("%B %Y")
        photos_count = trip["photos"]
        content = f"Family trip in {start} ({photos_count} photos taken)"
        store_memory(content=content, category="event", subject="Family", source="photos:metadata", confidence=0.8)
        stored += 1
        print(f"[Photos] Stored: {content}")

    print(f"[Photos] Complete! Analyzed {analysis['total_photos']:,} photos, stored {stored} insights")
    return analysis


if __name__ == "__main__":
    store_photo_insights()
