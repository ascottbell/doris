"""
Calendar Scout

Monitors Apple Calendar for upcoming events, changes, and conflicts.
Runs every hour via the daemon scheduler.

Uses existing tools/cal.py for calendar access.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
import sys
from pathlib import Path
import re
import math

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scouts.base import HaikuScout, Observation, Relevance
from tools.cal import (
    get_todays_events,
    get_tomorrows_events,
    get_weeks_events,
    list_events,
)
from security.prompt_safety import wrap_with_scan
from memory.store import (
    find_entity_by_name,
    get_entity_relationships,
    find_or_create_entity,
    add_relationship,
)


class CalendarScout(HaikuScout):
    """
    Scout that monitors Apple Calendar.

    Checks:
    - Upcoming events in the next few hours
    - New events added since last check
    - Events starting soon (30 min, 1 hour warnings)
    - Calendar conflicts

    Runs every hour by default.
    """

    name = "calendar-scout"

    # Max entries before pruning (typical day: <50 events)
    _MAX_KNOWN_IDS = 500

    def __init__(self):
        super().__init__()
        self._known_event_ids: set[str] = set()
        self._alerted_event_ids: set[str] = set()  # Events we've already alerted about
        self._last_prune_date: str = ""  # YYYY-MM-DD of last prune
        # Home coordinates — configure via DORIS_HOME_LAT / DORIS_HOME_LON env vars
        import os
        self._home_coords = (
            float(os.getenv("DORIS_HOME_LAT", "40.7128")),
            float(os.getenv("DORIS_HOME_LON", "-74.0060")),
        )

    def _parse_location_coords(self, location: str) -> Optional[Tuple[float, float]]:
        """
        Parse coordinates from location string if available.

        Apple Calendar sometimes includes coordinates in location data.
        Format examples: "Venue Name, 123 Main St, New York, NY (40.7128,-74.0060)"

        Returns:
            (latitude, longitude) tuple if found, None otherwise
        """
        if not location:
            return None

        # Look for coordinate patterns like (lat, lon) or lat,lon
        coord_pattern = r'\(?\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)?'
        match = re.search(coord_pattern, location)

        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                # Sanity check: NYC area roughly 40.5-41.0, -74.3 to -73.7
                if 40.0 <= lat <= 42.0 and -75.0 <= lon <= -73.0:
                    return (lat, lon)
            except ValueError:
                pass

        return None

    def _calculate_distance(self, coords1: Tuple[float, float], coords2: Tuple[float, float]) -> float:
        """
        Calculate distance between two coordinates using Haversine formula.

        Args:
            coords1: (latitude, longitude) tuple
            coords2: (latitude, longitude) tuple

        Returns:
            Distance in miles
        """
        lat1, lon1 = coords1
        lat2, lon2 = coords2

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Earth radius in miles
        radius = 3956

        return radius * c

    def _detect_transit_mode(self, event: dict) -> str:
        """
        Detect transit mode from event title, notes, or location.

        Returns one of: 'walk', 'subway', 'car', 'bike', or 'default'.
        """
        title = event.get('title', '').lower()
        notes = event.get('notes', '').lower() if event.get('notes') else ''
        location = event.get('location', '').lower()

        text = f"{title} {notes} {location}"

        # Explicit keywords
        if re.search(r'\b(walk|walking)\b', text):
            return 'walk'
        if re.search(r'\b(subway|train|transit|mta)\b', text):
            return 'subway'
        if re.search(r'\b(car|drive|driving|uber|taxi|lyft)\b', text):
            return 'car'
        if re.search(r'\b(bike|cycling|citibike)\b', text):
            return 'bike'

        return 'default'

    def _get_transit_mode_for_location(self, location: str) -> Optional[str]:
        """
        Check knowledge graph for typical transit mode to a location.

        Returns transit mode string if found, None otherwise.
        """
        if not location:
            return None

        try:
            # Look for a place entity matching this location
            entity = find_entity_by_name(location, entity_type='place')
            if not entity:
                return None

            # Check for typical_transit_mode relationship
            relationships = get_entity_relationships(entity['id'])
            for rel in relationships:
                if rel.get('predicate') == 'typical_transit_mode' and not rel.get('expired'):
                    return rel.get('object_value')

            return None
        except Exception as e:
            print(f"[{self.name}] Error looking up transit mode: {e}")
            return None

    def store_transit_preference(self, location: str, transit_mode: str) -> bool:
        """
        Store a typical transit mode preference for a location in the knowledge graph.

        This allows Doris to learn patterns like "Code Academy is always walking distance"
        or "Upstate always requires driving".

        Args:
            location: Location name (e.g., "Code Academy", "Riverside Park", "Upstate")
            transit_mode: One of 'walk', 'subway', 'car', 'bike'

        Returns:
            True if stored successfully, False otherwise

        Example usage:
            scout.store_transit_preference("Code Academy", "walk")
            scout.store_transit_preference("Upstate", "car")
        """
        if not location or transit_mode not in ['walk', 'subway', 'car', 'bike']:
            print(f"[{self.name}] Invalid location or transit mode")
            return False

        try:
            # Find or create the place entity (returns entity_id string)
            entity_id = find_or_create_entity(location, entity_type='place')
            if not entity_id:
                print(f"[{self.name}] Failed to create/find entity for {location}")
                return False

            # Add the relationship (this will update if one already exists)
            add_relationship(
                entity_id,
                'typical_transit_mode',
                object_value=transit_mode,
                source='calendar-scout'
            )

            print(f"[{self.name}] Stored transit preference: {location} -> {transit_mode}")
            return True

        except Exception as e:
            print(f"[{self.name}] Error storing transit preference: {e}")
            return False

    def _estimate_travel_time(self, location: str, transit_mode: str) -> int:
        """
        Estimate travel time in minutes based on location and transit mode.

        Uses actual distance calculation when coordinates are available,
        otherwise falls back to location name heuristics.

        Args:
            location: Event location string
            transit_mode: One of 'walk', 'subway', 'car', 'bike', 'default'

        Returns:
            Estimated travel time in minutes (includes buffer)
        """
        location_lower = location.lower() if location else ''

        # Try to get actual distance if coordinates are available
        coords = self._parse_location_coords(location)
        if coords:
            distance_miles = self._calculate_distance(self._home_coords, coords)

            # Calculate travel time based on distance and mode
            if distance_miles < 0.1:
                return 5  # Very close, minimal travel time

            # Speed estimates (mph) for different transit modes
            speed = {
                'walk': 3.0,     # Walking speed
                'subway': 10.0,  # Average including waiting and walking to/from stations
                'car': 12.0,     # NYC car speed (accounting for traffic)
                'bike': 10.0,    # Cycling speed in NYC
                'default': 10.0, # Assume transit/subway
            }.get(transit_mode, 10.0)

            # Base travel time
            travel_minutes = int((distance_miles / speed) * 60)

            # Add buffers for prep/waiting time
            buffer = {
                'walk': 5,
                'subway': 10,    # Extra for waiting
                'car': 10,       # Getting car/waiting for ride
                'bike': 5,
                'default': 10,
            }.get(transit_mode, 10)

            # Special case: walking > 1 mile probably means subway is better
            if transit_mode == 'walk' and distance_miles > 1.0:
                # Suggest this might take a while
                travel_minutes = int((distance_miles / 3.0) * 60) + buffer
                print(f"[{self.name}] Walk distance is {distance_miles:.1f} mi, estimated {travel_minutes} min")
            else:
                travel_minutes += buffer

            print(f"[{self.name}] Distance to {location}: {distance_miles:.1f} mi, "
                  f"mode: {transit_mode}, estimated: {travel_minutes} min")

            return travel_minutes

        # Fallback: Use location name heuristics if no coordinates

        # Recognize home/nearby locations
        if any(x in location_lower for x in ['home', 'apartment']):
            return 5  # Already home or very close

        # Default NYC locations by transit mode (no coordinates available)
        base_time = {
            'walk': 20,      # ~1 mile at 3 mph
            'subway': 30,    # Typical subway trip + waiting
            'car': 20,       # Typical NYC car trip
            'bike': 15,      # ~2 miles at ~8 mph
            'default': 30,   # Conservative default
        }.get(transit_mode, 30)

        # Add buffer for prep time
        buffer = {
            'walk': 5,
            'subway': 10,    # Extra for waiting
            'car': 10,       # Getting car/waiting for ride
            'bike': 5,
            'default': 10,
        }.get(transit_mode, 10)

        return base_time + buffer

    def _is_pickup_event(self, event: dict, now: datetime) -> bool:
        """
        Determine if this is a pickup event (vs drop-off).

        Heuristics:
        1. Explicit keywords in title: "pickup", "pick up"
        2. Current time is closer to event END than START (likely going to pick up)
        3. Known kids' activities where we typically do pickup

        Returns:
            True if this is likely a pickup event, False for drop-off
        """
        title = event.get('title', '').lower()

        # Explicit pickup keywords
        if re.search(r'\b(pickup|pick up|pick-up)\b', title):
            return True

        # Drop-off keywords override
        if re.search(r'\b(drop off|drop-off|dropoff|bring)\b', title):
            return False

        # Known kids' activities where we typically do pickups
        kids_activities = [
            'launch', 'coding', 'scratch',
            'gymnastics', 'gym class',
            'soccer', 'karate', 'dance', 'ballet',
            'school', 'afterschool', 'after school',
            'playdate', 'play date',
            'camp', 'day camp',
        ]

        is_kids_activity = any(activity in title for activity in kids_activities)

        if not is_kids_activity:
            return False  # Not a kids' activity, default to drop-off timing

        # For kids' activities, check if we're closer to end time (pickup) or start (drop-off)
        start_str = event.get('start', '')
        end_str = event.get('end', '')

        if not start_str or not end_str:
            return False  # Can't determine, default to drop-off

        try:
            start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))

            if start.tzinfo:
                start = start.replace(tzinfo=None)
            if end.tzinfo:
                end = end.replace(tzinfo=None)

            # If current time is in the second half of the event, it's likely pickup
            time_to_start = (start - now).total_seconds()
            time_to_end = (end - now).total_seconds()

            # If we're past the start time but before end, definitely pickup
            if time_to_start < 0 < time_to_end:
                return True

            # If we're before the event, check which is closer
            if time_to_start > 0 and time_to_end > 0:
                return time_to_end < time_to_start

        except (ValueError, TypeError):
            pass

        return False  # Default to drop-off

    def _prune_stale_ids(self, current_ids: set[str]) -> None:
        """Reset tracking sets daily and enforce size cap.

        Called once per observe() cycle. Clears stale IDs that no longer
        correspond to today's events, preventing unbounded memory growth.
        """
        today = datetime.now().strftime("%Y-%m-%d")

        if today != self._last_prune_date:
            # New day: keep only IDs that are still on today's calendar
            self._known_event_ids &= current_ids
            self._alerted_event_ids &= current_ids
            self._last_prune_date = today
            return

        # Mid-day safety cap: if sets grow beyond threshold, trim to current
        if len(self._known_event_ids) > self._MAX_KNOWN_IDS:
            self._known_event_ids &= current_ids
        if len(self._alerted_event_ids) > self._MAX_KNOWN_IDS:
            self._alerted_event_ids &= current_ids

    async def observe(self) -> list[Observation]:
        """
        Check calendar for noteworthy events.

        Returns observations for:
        - Events starting in the next 2 hours
        - New events added to calendar
        - Potential conflicts
        """
        observations = []
        now = datetime.now()

        try:
            # Get today's events
            today_events = get_todays_events()

            # Get events for next 2 hours for immediate awareness
            upcoming_events = self._get_upcoming_events(today_events, hours=2)

            # Check for events starting soon
            for event in upcoming_events:
                obs = self._check_event_timing(event, now)
                if obs:
                    observations.append(obs)

            # Check for new events we haven't seen
            current_ids = {e.get('id') for e in today_events if e.get('id')}
            self._prune_stale_ids(current_ids)
            new_ids = current_ids - self._known_event_ids

            for event in today_events:
                if event.get('id') in new_ids:
                    obs = await self._create_new_event_observation(event, now)
                    if obs:
                        observations.append(obs)

            # Update known events
            self._known_event_ids.update(current_ids)

            # Check for conflicts in today's schedule
            conflicts = self._find_conflicts(today_events)
            for conflict in conflicts:
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation=f"Calendar conflict: {conflict}",
                    relevance=Relevance.MEDIUM,
                    escalate=False,
                    context_tags=["calendar", "conflict"],
                ))

            # Tomorrow preview (useful for evening check)
            if now.hour >= 20:  # After 8 PM
                tomorrow_events = get_tomorrows_events()
                if tomorrow_events:
                    event_count = len(tomorrow_events)
                    first_event = tomorrow_events[0] if tomorrow_events else None
                    if first_event:
                        first_title = first_event.get('title', 'Untitled')
                        first_time = self._format_time(first_event.get('start', ''))
                        observations.append(Observation(
                            scout=self.name,
                            timestamp=now,
                            observation=f"Tomorrow: {event_count} event(s). First is '{first_title}' at {first_time}",
                            relevance=Relevance.LOW,
                            escalate=False,
                            context_tags=["calendar", "tomorrow", "preview"],
                            raw_data={"events": tomorrow_events}
                        ))

        except Exception as e:
            print(f"[{self.name}] Error checking calendar: {e}")
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation=f"Error accessing calendar: {str(e)[:100]}",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["error", "calendar"],
            ))

        return observations

    def _get_upcoming_events(self, events: list[dict], hours: int = 2) -> list[dict]:
        """
        Filter to events starting OR ending within the next N hours.

        Includes events based on end time to catch pickup scenarios.
        """
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)
        upcoming = []

        for event in events:
            start_str = event.get('start', '')
            end_str = event.get('end', '')

            if not start_str:
                continue

            try:
                # Parse ISO datetime for start
                start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                if start.tzinfo:
                    start = start.replace(tzinfo=None)

                # Check if start time is in range
                if now <= start <= cutoff:
                    upcoming.append(event)
                    continue

                # Also check end time for pickup scenarios
                if end_str:
                    end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    if end.tzinfo:
                        end = end.replace(tzinfo=None)

                    if now <= end <= cutoff:
                        upcoming.append(event)

            except (ValueError, TypeError):
                continue

        return upcoming

    def _check_event_timing(self, event: dict, now: datetime) -> Optional[Observation]:
        """
        Check if we should alert about an event's timing.

        Uses smart detection for:
        - Pickup vs drop-off (alerts based on END time for pickups)
        - Transit mode (from event or knowledge graph)
        - Travel time estimation with buffer

        Returns observation if it's time to leave and we haven't alerted yet.
        """
        event_id = event.get('id')
        if not event_id or event_id in self._alerted_event_ids:
            return None

        title = event.get('title', 'Untitled event')
        location = event.get('location', '')

        # Determine if this is a pickup or drop-off
        is_pickup = self._is_pickup_event(event, now)

        # Get the target time (when we need to arrive)
        target_time_str = event.get('end' if is_pickup else 'start', '')
        if not target_time_str:
            return None

        try:
            target_time = datetime.fromisoformat(target_time_str.replace('Z', '+00:00'))
            if target_time.tzinfo:
                target_time = target_time.replace(tzinfo=None)
        except (ValueError, TypeError):
            return None

        # Detect transit mode
        transit_mode = self._detect_transit_mode(event)

        # If default, check knowledge graph for location preference
        if transit_mode == 'default' and location:
            graph_mode = self._get_transit_mode_for_location(location)
            if graph_mode:
                transit_mode = graph_mode
                print(f"[{self.name}] Using knowledge graph transit mode '{transit_mode}' for {location}")

        # Estimate travel time (includes buffer)
        travel_time_minutes = self._estimate_travel_time(location, transit_mode)

        # Calculate when we need to leave
        leave_time = target_time - timedelta(minutes=travel_time_minutes)
        minutes_until_leave = (leave_time - now).total_seconds() / 60

        # Log for debugging
        target_type = "end" if is_pickup else "start"
        alert_time_str = leave_time.strftime('%I:%M %p').lstrip('0')
        print(f"[{self.name}] Event: {title}, detected as: {'pickup' if is_pickup else 'drop-off'}, "
              f"transit mode: {transit_mode}, estimated travel: {travel_time_minutes} min, "
              f"alerting at: {alert_time_str} (targeting {target_type} time)")

        # Alert when it's time to leave (within 5-minute window)
        if -2 <= minutes_until_leave <= 5:
            self._alerted_event_ids.add(event_id)

            # Build helpful message
            action = "pickup" if is_pickup else "get to"
            loc_str = f" at {location}" if location else ""
            mode_str = f" by {transit_mode}" if transit_mode != 'default' else ""

            # Create context-aware message
            if minutes_until_leave <= 0:
                urgency = "Leave now"
            elif minutes_until_leave <= 2:
                urgency = "Leave in a few minutes"
            else:
                urgency = f"Leave in {int(minutes_until_leave)} minutes"

            observation_text = (
                f"{urgency} to {action} '{title}'{loc_str}{mode_str}. "
                f"Estimated travel time: {travel_time_minutes} minutes."
            )

            return Observation(
                scout=self.name,
                timestamp=now,
                observation=observation_text,
                relevance=Relevance.HIGH,
                escalate=True,  # Departure reminders should interrupt
                context_tags=["calendar", "departure", "reminder", transit_mode],
                raw_data={
                    **event,
                    "is_pickup": is_pickup,
                    "transit_mode": transit_mode,
                    "travel_time_minutes": travel_time_minutes,
                }
            )

        # Also provide advance warning (30 min before leave time for longer trips)
        if travel_time_minutes >= 30 and 25 <= minutes_until_leave <= 35:
            self._alerted_event_ids.add(event_id)

            action = "pickup" if is_pickup else "event"
            loc_str = f" at {location}" if location else ""

            return Observation(
                scout=self.name,
                timestamp=now,
                observation=f"Heads up: '{title}'{loc_str} {action} in about {int((target_time - now).total_seconds() / 60)} minutes. Plan to leave in ~30 min.",
                relevance=Relevance.MEDIUM,
                escalate=False,
                context_tags=["calendar", "reminder", "advance-warning"],
                raw_data=event
            )

        return None

    async def _create_new_event_observation(
        self,
        event: dict,
        timestamp: datetime
    ) -> Optional[Observation]:
        """Create observation for a newly added event."""
        title = event.get('title', 'Untitled')
        start_str = event.get('start', '')
        location = event.get('location', '')

        time_str = self._format_time(start_str)

        # Wrap + scan untrusted calendar content to detect prompt injection
        wrapped_title = wrap_with_scan(title, 'calendar_title')
        wrapped_location = wrap_with_scan(location, 'calendar_location') if location else ""

        loc_str = f" at {wrapped_location}" if wrapped_location else ""
        description = f"New calendar event: {wrapped_title} at {time_str}{loc_str}"

        # Use Haiku to classify
        relevance, escalate, tags = await self.classify_relevance(
            description,
            context="This is the user's personal/family calendar. "
                    "School events, kids' activities, and family appointments are important."
        )

        tags = list(set(tags + ["calendar", "new"]))

        return Observation(
            scout=self.name,
            timestamp=timestamp,
            observation=description,
            relevance=relevance,
            escalate=escalate,
            context_tags=tags,
            raw_data=event
        )

    def _find_conflicts(self, events: list[dict]) -> list[str]:
        """Find overlapping events (potential conflicts)."""
        conflicts = []

        # Parse all events with start/end times
        timed_events = []
        for event in events:
            if event.get('isAllDay'):
                continue
            start_str = event.get('start', '')
            end_str = event.get('end', '')
            if not start_str or not end_str:
                continue

            try:
                start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                if start.tzinfo:
                    start = start.replace(tzinfo=None)
                if end.tzinfo:
                    end = end.replace(tzinfo=None)
                timed_events.append({
                    'title': event.get('title', 'Untitled'),
                    'start': start,
                    'end': end
                })
            except (ValueError, TypeError):
                continue

        # Check for overlaps
        for i, e1 in enumerate(timed_events):
            for e2 in timed_events[i+1:]:
                # Check if they overlap
                if e1['start'] < e2['end'] and e2['start'] < e1['end']:
                    conflicts.append(
                        f"'{e1['title']}' and '{e2['title']}' overlap"
                    )

        return conflicts

    def _format_time(self, iso_str: str) -> str:
        """Format ISO datetime string for display in Eastern time."""
        if not iso_str:
            return "unknown time"
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
            # Convert to Eastern time
            eastern = ZoneInfo('America/New_York')
            if dt.tzinfo:
                dt = dt.astimezone(eastern)
            return dt.strftime('%I:%M %p').lstrip('0')
        except (ValueError, TypeError):
            return "unknown time"


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        scout = CalendarScout()
        print(f"Running {scout.name}...")

        observations = await scout.run()

        if not observations:
            print("No observations")
        else:
            print(f"Found {len(observations)} observations:\n")
            for obs in observations:
                print(f"[{obs.relevance.value.upper()}] {obs.observation}")
                print(f"  Escalate: {obs.escalate}")
                print(f"  Tags: {obs.context_tags}")
                print()

    asyncio.run(test())
