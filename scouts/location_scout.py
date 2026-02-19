"""
Location Scout

Monitors location patterns and provides context-aware observations.
Runs every 15 minutes via the daemon scheduler.

Uses services/location_db.py for location data access.
"""

from datetime import datetime, timedelta
from typing import Optional
import sys
import logging
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scouts.base import HaikuScout, Observation, Relevance
from services.location_db import (
    get_current_location,
    get_location_history,
    get_location_context,
    haversine_distance
)
from security.injection_scanner import scan_for_injection

logger = logging.getLogger(__name__)


class LocationScout(HaikuScout):
    """
    Scout that monitors location patterns.

    Checks:
    - Departures from home
    - Arrivals at new locations
    - Travel patterns (commuting vs. trips)
    - Long periods away from known locations

    Runs every 15 minutes.
    """

    name = "location-scout"

    def __init__(self):
        super().__init__()
        self._last_known_location: Optional[str] = None
        self._last_home_departure: Optional[datetime] = None
        self._notified_away: bool = False

    async def observe(self) -> list[Observation]:
        """
        Check location for noteworthy changes.

        Returns observations for:
        - Leaving home
        - Arriving somewhere new
        - Extended time away from home
        """
        observations = []
        now = datetime.now()

        try:
            context = get_location_context()

            if not context.get("available"):
                # No location data yet
                return []

            current_location = context.get("current_location")
            is_home = context.get("is_home", False)
            last_transition = context.get("last_transition")

            # Scan location description for injection patterns (external data)
            if current_location:
                scan = scan_for_injection(str(current_location), source="location-scout")
                if scan.is_suspicious:
                    logger.warning(
                        f"[{self.name}] Suspicious location data "
                        f"(risk={scan.risk_level}): {str(current_location)[:100]!r}"
                    )

            # Check for home departure
            if self._last_known_location and is_home != (self._last_known_location == "home"):
                if not is_home and self._last_known_location == "home":
                    # Just left home
                    self._last_home_departure = now
                    self._notified_away = False
                    observations.append(Observation(
                        scout=self.name,
                        timestamp=now,
                        observation=f"User left home. Currently at: {current_location}",
                        relevance=Relevance.LOW,
                        escalate=False,
                        context_tags=["location", "home", "departure"],
                        raw_data={"current": current_location, "is_home": is_home}
                    ))
                elif is_home and self._last_known_location != "home":
                    # Just arrived home
                    if self._last_home_departure:
                        away_duration = now - self._last_home_departure
                        hours_away = away_duration.total_seconds() / 3600
                        observations.append(Observation(
                            scout=self.name,
                            timestamp=now,
                            observation=f"User is back home after {hours_away:.1f} hours away.",
                            relevance=Relevance.LOW,
                            escalate=False,
                            context_tags=["location", "home", "arrival"],
                            raw_data={"hours_away": hours_away}
                        ))
                    self._last_home_departure = None
                    self._notified_away = False

            # Check for extended time away (notify once at 8+ hours)
            if not is_home and self._last_home_departure and not self._notified_away:
                hours_away = (now - self._last_home_departure).total_seconds() / 3600
                if hours_away >= 8:
                    self._notified_away = True
                    observations.append(Observation(
                        scout=self.name,
                        timestamp=now,
                        observation=f"User has been away from home for {hours_away:.1f} hours. Currently at: {current_location}",
                        relevance=Relevance.LOW,
                        escalate=False,
                        context_tags=["location", "away", "extended"],
                        raw_data={"hours_away": hours_away, "current": current_location}
                    ))

            # Detect travel (significant movement)
            if context.get("is_traveling"):
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation="User appears to be traveling (multiple location changes detected).",
                    relevance=Relevance.MEDIUM,
                    escalate=False,
                    context_tags=["location", "travel"],
                    raw_data={"current": current_location}
                ))

            # Update state
            self._last_known_location = "home" if is_home else current_location

        except Exception as e:
            print(f"[{self.name}] Error checking location: {e}")
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation=f"Error accessing location data: {str(e)[:100]}",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["error", "location"],
            ))

        return observations


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        scout = LocationScout()
        print(f"Running {scout.name}...")

        # Get current context first
        context = get_location_context()
        print(f"Current context: {context}")

        observations = await scout.run()

        if not observations:
            print("No observations (possibly no location data or no changes)")
        else:
            print(f"Found {len(observations)} observations:")
            for obs in observations:
                print(f"  [{obs.relevance.value.upper()}] {obs.observation}")

    asyncio.run(test())
