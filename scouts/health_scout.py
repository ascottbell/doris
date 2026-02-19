"""
Health Scout

Monitors Apple Health data for trends and patterns worth noting.
Runs daily via the daemon scheduler.

Uses services/health_db.py for health data access.
"""

from datetime import datetime, timedelta
from typing import Optional
import sys
import logging
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scouts.base import HaikuScout, Observation, Relevance
from services.health_db import get_health_data, get_health_summary
from security.injection_scanner import scan_for_injection

logger = logging.getLogger(__name__)


class HealthScout(HaikuScout):
    """
    Scout that monitors Apple Health data for trends.

    Checks:
    - Sleep patterns (poor sleep, sleep debt)
    - Activity levels (low step days, missed goals)
    - Workout consistency
    - Recovery indicators (HRV trends)

    Runs once daily, typically in the morning.
    """

    name = "health-scout"

    def __init__(self):
        super().__init__()
        self._last_check_date: Optional[str] = None

    async def observe(self) -> list[Observation]:
        """
        Check health data for noteworthy trends.

        Returns observations for:
        - Sleep issues (< 6 hours, poor quality)
        - Low activity streaks (< 5000 steps for multiple days)
        - Workout patterns (consistency, missed days)
        - Recovery concerns (low HRV trend)
        """
        observations = []
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Don't run more than once per day
        if self._last_check_date == today:
            return []

        self._last_check_date = today

        try:
            # Get last 7 days of data
            health_data = get_health_data(days=7)

            if not health_data:
                # No data yet - this is expected if iOS app hasn't synced
                return []

            # Check various health indicators
            observations.extend(await self._check_sleep_patterns(health_data, now))
            observations.extend(await self._check_activity_levels(health_data, now))
            observations.extend(await self._check_workout_consistency(health_data, now))
            observations.extend(await self._check_recovery(health_data, now))

        except Exception as e:
            print(f"[{self.name}] Error checking health data: {e}")
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation=f"Error accessing health data: {str(e)[:100]}",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["error", "health"],
            ))

        return observations

    async def _check_sleep_patterns(
        self,
        data: list[dict],
        timestamp: datetime
    ) -> list[Observation]:
        """Check for sleep issues."""
        observations = []

        # Get sleep data from recent days
        sleep_data = [(d['date'], d.get('sleep_hours')) for d in data if d.get('sleep_hours')]

        if not sleep_data:
            return []

        # Check for sleep debt (multiple nights under 6 hours)
        poor_sleep_nights = [d for d in sleep_data if d[1] < 6]
        if len(poor_sleep_nights) >= 3:
            avg_sleep = sum(d[1] for d in poor_sleep_nights) / len(poor_sleep_nights)
            observations.append(Observation(
                scout=self.name,
                timestamp=timestamp,
                observation=f"Sleep debt building: {len(poor_sleep_nights)} nights under 6 hours this week (avg {avg_sleep:.1f}h). Consider prioritizing rest.",
                relevance=Relevance.MEDIUM,
                escalate=False,
                context_tags=["health", "sleep", "wellness"],
                raw_data={"poor_nights": poor_sleep_nights}
            ))

        # Check last night's sleep specifically
        if sleep_data:
            last_night = sleep_data[0]  # Most recent
            if last_night[1] < 5:
                observations.append(Observation(
                    scout=self.name,
                    timestamp=timestamp,
                    observation=f"Rough night: only {last_night[1]:.1f} hours of sleep. Maybe take it easy today?",
                    relevance=Relevance.MEDIUM,
                    escalate=False,
                    context_tags=["health", "sleep", "today"],
                    raw_data={"date": last_night[0], "hours": last_night[1]}
                ))

        # Check for great sleep streak
        good_sleep_nights = [d for d in sleep_data if d[1] >= 7.5]
        if len(good_sleep_nights) >= 5:
            observations.append(Observation(
                scout=self.name,
                timestamp=timestamp,
                observation=f"Great sleep streak: {len(good_sleep_nights)} nights of 7.5+ hours this week. Keep it up!",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["health", "sleep", "positive"],
            ))

        return observations

    async def _check_activity_levels(
        self,
        data: list[dict],
        timestamp: datetime
    ) -> list[Observation]:
        """Check for low activity patterns."""
        observations = []

        # Get step data
        step_data = [(d['date'], d.get('steps')) for d in data if d.get('steps')]

        if not step_data:
            return []

        # Check for low activity streak
        low_activity_days = [d for d in step_data if d[1] < 5000]
        if len(low_activity_days) >= 3:
            observations.append(Observation(
                scout=self.name,
                timestamp=timestamp,
                observation=f"Activity dip: {len(low_activity_days)} days under 5,000 steps this week. A short walk might feel good.",
                relevance=Relevance.MEDIUM,
                escalate=False,
                context_tags=["health", "activity", "steps"],
                raw_data={"low_days": low_activity_days}
            ))

        # Check for great activity
        active_days = [d for d in step_data if d[1] >= 10000]
        if len(active_days) >= 5:
            observations.append(Observation(
                scout=self.name,
                timestamp=timestamp,
                observation=f"Crushing it: {len(active_days)} days hitting 10k+ steps this week!",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["health", "activity", "positive"],
            ))

        return observations

    async def _check_workout_consistency(
        self,
        data: list[dict],
        timestamp: datetime
    ) -> list[Observation]:
        """Check workout patterns."""
        observations = []

        # Count workouts
        total_workouts = 0
        workout_types = []
        for d in data:
            workouts = d.get('workouts') or []
            total_workouts += len(workouts)
            for w in workouts:
                wtype = w.get('type', 'workout')
                # Scan workout type string for injection (HealthKit data, low risk)
                scan = scan_for_injection(wtype, source="health-scout")
                if scan.is_suspicious:
                    logger.warning(
                        f"[{self.name}] Suspicious workout type "
                        f"(risk={scan.risk_level}): {wtype[:50]!r}"
                    )
                    wtype = "workout"  # Replace suspicious type with safe default
                workout_types.append(wtype)

        if total_workouts == 0 and len(data) >= 5:
            # No workouts in a week
            observations.append(Observation(
                scout=self.name,
                timestamp=timestamp,
                observation="No logged workouts this week. Even a short workout can boost your mood.",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["health", "workout", "nudge"],
            ))
        elif total_workouts >= 5:
            # Great workout week
            type_summary = ", ".join(set(workout_types[:5]))
            observations.append(Observation(
                scout=self.name,
                timestamp=timestamp,
                observation=f"Strong workout week: {total_workouts} sessions ({type_summary}). Nice work!",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["health", "workout", "positive"],
            ))

        return observations

    async def _check_recovery(
        self,
        data: list[dict],
        timestamp: datetime
    ) -> list[Observation]:
        """Check recovery indicators like HRV."""
        observations = []

        # Get HRV data
        hrv_data = [(d['date'], d.get('hrv')) for d in data if d.get('hrv')]

        if len(hrv_data) < 3:
            return []

        # Check for downward HRV trend (stress/poor recovery)
        recent_hrv = [d[1] for d in hrv_data[:3]]
        avg_recent = sum(recent_hrv) / len(recent_hrv)

        if avg_recent < 30:
            observations.append(Observation(
                scout=self.name,
                timestamp=timestamp,
                observation=f"HRV has been low lately (avg {avg_recent:.0f}ms). Your body might need more recovery time.",
                relevance=Relevance.MEDIUM,
                escalate=False,
                context_tags=["health", "hrv", "recovery", "stress"],
                raw_data={"recent_hrv": recent_hrv}
            ))
        elif avg_recent > 60:
            observations.append(Observation(
                scout=self.name,
                timestamp=timestamp,
                observation=f"HRV looking great (avg {avg_recent:.0f}ms). You're well-recovered!",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["health", "hrv", "positive"],
            ))

        return observations


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        scout = HealthScout()
        print(f"Running {scout.name}...")

        # Reset last check to force run
        scout._last_check_date = None

        observations = await scout.run()

        if not observations:
            print("No observations (possibly no health data synced yet)")
        else:
            print(f"Found {len(observations)} observations:\n")
            for obs in observations:
                print(f"[{obs.relevance.value.upper()}] {obs.observation}")
                print(f"  Escalate: {obs.escalate}")
                print(f"  Tags: {obs.context_tags}")
                print()

    asyncio.run(test())
