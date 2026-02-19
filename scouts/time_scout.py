"""
Time Scout

Rule-based scout that monitors the clock and triggers based on:
- Time of day (morning, evening transitions)
- Approaching calendar events
- Daily rhythms

Runs every minute via the daemon scheduler.

This scout is entirely rule-based (no LLM) for speed and cost.
"""

from datetime import datetime, timedelta
from typing import Optional
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scouts.base import Scout, Observation, Relevance


class TimeScout(Scout):
    """
    Rule-based scout that monitors time-based triggers.

    Triggers:
    - Morning wake-up time (7:00 AM)
    - Evening wind-down time (10:00 PM)
    - Pre-commute warning (configurable)
    - End of work day (6:00 PM)
    - Weekend transitions

    Runs every minute but only generates observations at key times.
    """

    name = "time-scout"

    # Configurable triggers (hour, minute)
    MORNING_WAKE = (7, 0)
    EVENING_WIND_DOWN = (22, 0)  # 10 PM
    END_OF_WORKDAY = (18, 0)  # 6 PM

    def __init__(self):
        super().__init__()
        self._triggered_today: set[str] = set()
        self._last_date: Optional[str] = None

    async def observe(self) -> list[Observation]:
        """
        Check for time-based triggers.

        Only fires each trigger once per day.
        """
        observations = []
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')

        # Reset triggers at midnight
        if self._last_date != today:
            self._triggered_today = set()
            self._last_date = today

        current_time = (now.hour, now.minute)

        # Morning wake trigger — context only, NOT escalation.
        # The actual morning briefing is handled by the cron job in scheduler.py.
        # Escalating here causes duplicate briefings via the escalation buffer.
        if self._should_trigger('morning', current_time, self.MORNING_WAKE):
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation="Good morning - time for daily briefing",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["time", "morning", "briefing"],
                raw_data={"trigger": "morning_wake", "time": now.isoformat()}
            ))
            self._triggered_today.add('morning')

        # Evening wind-down trigger — context only, NOT escalation.
        # The actual evening reflection is handled by the cron job in scheduler.py.
        if self._should_trigger('evening', current_time, self.EVENING_WIND_DOWN):
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation="Evening wind-down - time for daily reflection",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["time", "evening", "reflection"],
                raw_data={"trigger": "evening_wind_down", "time": now.isoformat()}
            ))
            self._triggered_today.add('evening')

        # End of workday (weekdays only)
        if now.weekday() < 5:  # Monday-Friday
            if self._should_trigger('workday_end', current_time, self.END_OF_WORKDAY):
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation="End of standard work hours",
                    relevance=Relevance.LOW,
                    escalate=False,
                    context_tags=["time", "workday", "transition"],
                    raw_data={"trigger": "workday_end", "time": now.isoformat()}
                ))
                self._triggered_today.add('workday_end')

        # Weekend start (Friday 6 PM)
        if now.weekday() == 4:  # Friday
            if self._should_trigger('weekend_start', current_time, (18, 0)):
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation="Weekend is here!",
                    relevance=Relevance.LOW,
                    escalate=False,
                    context_tags=["time", "weekend", "transition"],
                    raw_data={"trigger": "weekend_start", "time": now.isoformat()}
                ))
                self._triggered_today.add('weekend_start')

        # Sunday evening prep (Sunday 6 PM)
        if now.weekday() == 6:  # Sunday
            if self._should_trigger('week_prep', current_time, (18, 0)):
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation="Sunday evening - good time to prep for the week",
                    relevance=Relevance.LOW,
                    escalate=False,
                    context_tags=["time", "sunday", "prep"],
                    raw_data={"trigger": "week_prep", "time": now.isoformat()}
                ))
                self._triggered_today.add('week_prep')

        return observations

    def _should_trigger(
        self,
        trigger_name: str,
        current_time: tuple[int, int],
        trigger_time: tuple[int, int],
        window_minutes: int = 2
    ) -> bool:
        """
        Check if a trigger should fire.

        Args:
            trigger_name: Unique identifier for this trigger
            current_time: (hour, minute) tuple of current time
            trigger_time: (hour, minute) tuple of target time
            window_minutes: How many minutes after target time to still trigger

        Returns:
            True if trigger should fire (within window and not already fired today)
        """
        if trigger_name in self._triggered_today:
            return False

        target_hour, target_minute = trigger_time
        current_hour, current_minute = current_time

        # Check if we're within the trigger window
        target_minutes = target_hour * 60 + target_minute
        current_minutes = current_hour * 60 + current_minute

        return target_minutes <= current_minutes < target_minutes + window_minutes


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        scout = TimeScout()
        print(f"Running {scout.name}...")
        print(f"Current time: {datetime.now().strftime('%H:%M')}")

        observations = await scout.run()

        if not observations:
            print("No time-based triggers right now")
            print("\nConfigured triggers:")
            print(f"  Morning wake: {scout.MORNING_WAKE[0]:02d}:{scout.MORNING_WAKE[1]:02d}")
            print(f"  Evening wind-down: {scout.EVENING_WIND_DOWN[0]:02d}:{scout.EVENING_WIND_DOWN[1]:02d}")
            print(f"  End of workday: {scout.END_OF_WORKDAY[0]:02d}:{scout.END_OF_WORKDAY[1]:02d}")
        else:
            print(f"Found {len(observations)} observations:\n")
            for obs in observations:
                print(f"[{obs.relevance.value.upper()}] {obs.observation}")
                print(f"  Escalate: {obs.escalate}")
                print(f"  Tags: {obs.context_tags}")
                print()

    asyncio.run(test())
