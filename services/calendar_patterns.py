"""
Calendar pattern analysis for Doris memory.

Analyzes calendar history to extract recurring patterns like:
- Kids' weekly activities
- Regular meetings
- Travel patterns
"""

import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

CALENDAR_CLI = Path.home() / "Projects/doris-calendar/.build/release/doris-calendar"


def list_all_events(start: datetime, end: datetime) -> list[dict]:
    """List events from ALL calendars (read-only)."""
    args = [str(CALENDAR_CLI), "list-all", start.isoformat(), end.isoformat()]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    data = json.loads(result.stdout)
    return data.get("events", [])


def analyze_patterns(days: int = 90) -> dict:
    """Analyze calendar patterns over the last N days."""
    end = datetime.now() + timedelta(days=30)  # Include upcoming
    start = datetime.now() - timedelta(days=days)

    events = list_all_events(start, end)
    if not events:
        return {"patterns": [], "insights": []}

    # Group events by various dimensions
    by_dow = defaultdict(list)  # Day of week
    by_title = defaultdict(list)  # Same title recurring
    by_person = defaultdict(list)  # Events mentioning family members

    family_names = []  # Configure with your family member names (lowercase)

    for e in events:
        title = e.get("title", "").lower()
        start_str = e.get("start", "")
        calendar = e.get("calendar", "")

        # Skip birthday/holiday calendars for pattern analysis
        if calendar in ["Birthdays", "Holidays in United States", "US Holidays"]:
            continue

        try:
            dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            dow = dt.strftime("%A")
            time_str = dt.strftime("%-I:%M %p")

            by_dow[dow].append({
                "title": e.get("title", ""),
                "time": time_str,
                "calendar": calendar
            })

            by_title[title].append({
                "dow": dow,
                "time": time_str,
                "date": dt.date().isoformat()
            })

            # Check for family member mentions
            for name in family_names:
                if name in title:
                    by_person[name].append({
                        "title": e.get("title", ""),
                        "dow": dow,
                        "time": time_str
                    })
        except Exception:
            pass

    patterns = []
    insights = []

    # Find weekly recurring activities (same title, same day of week)
    for title, occurrences in by_title.items():
        if len(occurrences) >= 2:
            # Check if they're on the same day of week
            dows = [o["dow"] for o in occurrences]
            times = [o["time"] for o in occurrences]

            if len(set(dows)) == 1:  # All on same day
                dow = dows[0]
                time = times[0]  # Take first time

                # Clean up title for insight
                clean_title = title.title()
                patterns.append({
                    "type": "weekly_recurring",
                    "title": clean_title,
                    "day": dow,
                    "time": time,
                    "count": len(occurrences)
                })

    # Extract family activity patterns
    for name, activities in by_person.items():
        if activities:
            # Group by day of week
            by_day = defaultdict(list)
            for a in activities:
                by_day[a["dow"]].append(a)

            for dow, acts in by_day.items():
                if len(acts) >= 1:
                    # Get most common activity for this person on this day
                    titles = [a["title"] for a in acts]
                    most_common = max(set(titles), key=titles.count)
                    time = acts[0]["time"]

                    insight = f"{name.title()} has {most_common} on {dow}s at {time}"
                    insights.append({
                        "person": name,
                        "activity": most_common,
                        "day": dow,
                        "time": time,
                        "insight": insight
                    })

    return {
        "patterns": patterns,
        "insights": insights,
        "events_analyzed": len(events),
        "date_range": {
            "start": start.date().isoformat(),
            "end": end.date().isoformat()
        }
    }


def store_calendar_patterns():
    """Analyze calendar and store patterns in memory."""
    from memory.store import store_memory

    analysis = analyze_patterns(days=90)

    stored = 0

    # Store family activity insights
    for insight in analysis.get("insights", []):
        store_memory(
            content=insight["insight"],
            category="schedule",
            subject=insight["person"].title(),
            source="calendar:pattern_analysis",
            confidence=0.9
        )
        stored += 1
        print(f"[CalendarPatterns] Stored: {insight['insight']}")

    # Store weekly recurring patterns
    for pattern in analysis.get("patterns", []):
        if pattern["type"] == "weekly_recurring":
            content = f"{pattern['title']} happens every {pattern['day']} at {pattern['time']}"
            store_memory(
                content=content,
                category="schedule",
                subject="recurring",
                source="calendar:pattern_analysis",
                confidence=0.85
            )
            stored += 1
            print(f"[CalendarPatterns] Stored: {content}")

    print(f"[CalendarPatterns] Analyzed {analysis['events_analyzed']} events, stored {stored} patterns")
    return analysis


if __name__ == "__main__":
    # Run analysis
    result = store_calendar_patterns()
    print(f"\nFound {len(result['patterns'])} recurring patterns")
    print(f"Found {len(result['insights'])} family activity insights")
