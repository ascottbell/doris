"""
Daemon Scheduler

APScheduler-based job scheduling for scouts and Doris wake-ups.

Jobs:
- email_scout: Every 30 minutes
- calendar_scout: Every hour
- weather_scout: Every hour
- time_scout: Every minute
- health_scout: Daily at 8 AM
- location_scout: Every 15 minutes
- memory_scout: Every hour (monitors extraction/compaction health)
- system_scout: Every 15 minutes (errors, latency, disk, logs, circuit breakers)
- reminders_scout: Every 15 minutes (due/overdue reminders)
- behavioral_patterns: Daily at 3:00 AM (query pattern analysis)
- calendar_patterns: Daily at 3:15 AM (calendar pattern analysis)
- email_patterns: Daily at 3:30 AM (email sender analysis)
- morning_wake: 7:00 AM daily
- evening_wake: 10:00 PM daily
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

# State file for heartbeat updates
STATE_FILE = Path(__file__).parent.parent / "data" / "daemon_state.json"
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from scouts import (
    EmailScout,
    CalendarScout,
    WeatherScout,
    TimeScout,
    HealthScout,
    LocationScout,
    MemoryScout,
    SystemScout,
    RemindersScout,
    Observation,
    Relevance,
)
from daemon.digest import get_digest, save_digest
from daemon.scout_health import get_health_tracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doris.daemon")


class DorisScheduler:
    """
    Manages scheduled jobs for the Doris daemon.

    Handles:
    - Scout scheduling (email, calendar, weather, time)
    - Doris wake-up triggers (morning, evening)
    - Escalation handling
    """

    def __init__(
        self,
        on_escalation: Optional[Callable[[list[Observation]], None]] = None,
        on_wake: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize scheduler.

        Args:
            on_escalation: Callback when escalations need attention
            on_wake: Callback for scheduled wake-ups (reason, prompt)
        """
        self.scheduler = AsyncIOScheduler()
        self.on_escalation = on_escalation
        self.on_wake = on_wake

        # Scout instances
        self.email_scout = EmailScout()
        self.calendar_scout = CalendarScout()
        self.weather_scout = WeatherScout()
        self.time_scout = TimeScout()
        self.health_scout = HealthScout()
        self.location_scout = LocationScout()
        self.memory_scout = MemoryScout()
        self.system_scout = SystemScout()
        self.reminders_scout = RemindersScout()

        # Track state
        self._running = False

    def setup_jobs(self) -> None:
        """Configure all scheduled jobs."""

        # Scout jobs
        self.scheduler.add_job(
            self._run_scout,
            IntervalTrigger(minutes=30),
            args=[self.email_scout],
            id="email_scout",
            name="Email Scout (30 min)",
        )

        self.scheduler.add_job(
            self._run_scout,
            IntervalTrigger(hours=1),
            args=[self.calendar_scout],
            id="calendar_scout",
            name="Calendar Scout (1 hour)",
        )

        self.scheduler.add_job(
            self._run_scout,
            IntervalTrigger(hours=1),
            args=[self.weather_scout],
            id="weather_scout",
            name="Weather Scout (1 hour)",
        )

        self.scheduler.add_job(
            self._run_scout,
            IntervalTrigger(minutes=1),
            args=[self.time_scout],
            id="time_scout",
            name="Time Scout (1 min)",
        )

        # Health scout - runs daily at 8 AM (after morning health sync)
        self.scheduler.add_job(
            self._run_scout,
            CronTrigger(hour=8, minute=0),
            args=[self.health_scout],
            id="health_scout",
            name="Health Scout (daily 8 AM)",
        )

        # Location scout - runs every 15 minutes
        self.scheduler.add_job(
            self._run_scout,
            IntervalTrigger(minutes=15),
            args=[self.location_scout],
            id="location_scout",
            name="Location Scout (15 min)",
        )

        # Doris wake-ups
        self.scheduler.add_job(
            self._morning_wake,
            CronTrigger(hour=7, minute=0),
            id="morning_wake",
            name="Morning Briefing (7 AM)",
        )

        self.scheduler.add_job(
            self._evening_wake,
            CronTrigger(hour=22, minute=0),
            id="evening_wake",
            name="Evening Reflection (10 PM)",
        )

        # Periodic digest save
        self.scheduler.add_job(
            self._save_digest,
            IntervalTrigger(minutes=5),
            id="save_digest",
            name="Save Digest (5 min)",
        )

        # Memory scout (every hour - monitors extraction/compaction health)
        self.scheduler.add_job(
            self._run_scout,
            IntervalTrigger(hours=1),
            args=[self.memory_scout],
            id="memory_scout",
            name="Memory Scout (1 hour)",
        )

        # System health scout (every 15 min - errors, latency, disk, logs, circuits)
        self.scheduler.add_job(
            self._run_scout,
            IntervalTrigger(minutes=15),
            args=[self.system_scout],
            id="system_scout",
            name="System Scout (15 min)",
        )

        # Reminders scout (every 15 min - due/overdue reminders)
        self.scheduler.add_job(
            self._run_scout,
            IntervalTrigger(minutes=15),
            args=[self.reminders_scout],
            id="reminders_scout",
            name="Reminders Scout (15 min)",
        )

        # Pattern analysis jobs (daily at 3 AM — store insights to memory)
        self.scheduler.add_job(
            self._run_pattern_analysis,
            CronTrigger(hour=3, minute=0),
            args=["behavioral"],
            id="behavioral_patterns",
            name="Behavioral Patterns (daily 3 AM)",
        )

        self.scheduler.add_job(
            self._run_pattern_analysis,
            CronTrigger(hour=3, minute=15),
            args=["calendar"],
            id="calendar_patterns",
            name="Calendar Patterns (daily 3:15 AM)",
        )

        self.scheduler.add_job(
            self._run_pattern_analysis,
            CronTrigger(hour=3, minute=30),
            args=["email"],
            id="email_patterns",
            name="Email Patterns (daily 3:30 AM)",
        )

        logger.info("Scheduled jobs configured")

    async def _run_scout(self, scout) -> None:
        """Run a scout and process observations."""
        health = get_health_tracker()

        try:
            observations = await scout.run()

            # Record success
            health.record_success(scout.name)

            if observations:
                digest = get_digest()
                digest.add_many(observations)

                # Log observations
                for obs in observations:
                    level = logging.WARNING if obs.escalate else logging.INFO
                    logger.log(level, f"[{scout.name}] {obs.observation}")

                # Check for escalations
                if digest.has_escalation() and self.on_escalation:
                    escalations = digest.get_escalations()
                    await self._handle_escalations(escalations)

            # Only update heartbeat on success
            self._update_heartbeat()

        except Exception as e:
            logger.error(f"[{scout.name}] Error: {e}")

            # Record failure
            health.record_failure(scout.name, str(e))

            # Auto-escalate if this scout has persistent failures
            if scout.name in health.get_failing_scouts(threshold=3):
                consecutive = health.get_scout_status(scout.name)["consecutive_failures"]
                logger.warning(
                    f"[{scout.name}] {consecutive} consecutive failures — auto-escalating"
                )
                error_obs = Observation(
                    scout=scout.name,
                    timestamp=datetime.now(),
                    observation=f"Scout '{scout.name}' has failed {consecutive} times in a row: {str(e)[:100]}",
                    relevance=Relevance.HIGH,
                    escalate=True,
                    context_tags=["scout_error", "persistent_failure"],
                )
                digest = get_digest()
                digest.add(error_obs)

                if self.on_escalation:
                    await self._handle_escalations([error_obs])

    async def _run_pattern_analysis(self, pattern_type: str) -> None:
        """Run a pattern analysis job (behavioral, calendar, or email)."""
        try:
            if pattern_type == "behavioral":
                from services.behavioral_patterns import store_behavioral_patterns
                store_behavioral_patterns()
            elif pattern_type == "calendar":
                from services.calendar_patterns import store_calendar_patterns
                store_calendar_patterns()
            elif pattern_type == "email":
                from services.email_patterns import store_email_patterns
                store_email_patterns()
            else:
                logger.error(f"Unknown pattern type: {pattern_type}")
                return

            logger.info(f"Pattern analysis complete: {pattern_type}")
        except Exception as e:
            logger.error(f"Pattern analysis failed ({pattern_type}): {e}")

    def _update_heartbeat(self) -> None:
        """Update heartbeat timestamp to indicate daemon is alive."""
        try:
            existing = {}
            if STATE_FILE.exists():
                existing = json.loads(STATE_FILE.read_text())

            existing["last_scout_run"] = datetime.now().isoformat()
            existing["updated_at"] = datetime.now().isoformat()
            STATE_FILE.write_text(json.dumps(existing, indent=2))
        except Exception as e:
            logger.error(f"Failed to update heartbeat: {e}")

    async def _handle_escalations(self, escalations: list[Observation]) -> None:
        """Handle pending escalations."""
        if not escalations:
            return

        logger.warning(f"Processing {len(escalations)} escalation(s)")

        if self.on_escalation:
            try:
                # Call the escalation handler (might be sync or async)
                result = self.on_escalation(escalations)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Escalation handler error: {e}")

    async def _morning_wake(self) -> None:
        """7 AM morning briefing trigger."""
        logger.info("Morning wake-up triggered")

        if self.on_wake:
            prompt = self._build_morning_prompt()
            try:
                result = self.on_wake("morning_briefing", prompt)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Morning wake error: {e}")

    async def _evening_wake(self) -> None:
        """10 PM evening reflection trigger."""
        logger.info("Evening wake-up triggered")

        if self.on_wake:
            prompt = self._build_evening_prompt()
            try:
                result = self.on_wake("evening_reflection", prompt)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Evening wake error: {e}")

    def _build_morning_prompt(self) -> str:
        """Build the morning briefing prompt."""
        digest = get_digest()
        digest_text = digest.format_for_prompt()

        return f"""Good morning! Time for your daily briefing.

{digest_text}

Please provide:
1. Weather summary and recommendations (what to wear, umbrella needed?)
2. Today's calendar overview
3. Important emails that need attention
4. Any reminders or notes from yesterday

Keep it conversational and concise."""

    def _build_evening_prompt(self) -> str:
        """Build the evening reflection prompt."""
        digest = get_digest()
        digest_text = digest.format_for_prompt()

        return f"""Good evening! Time for daily reflection.

{digest_text}

Please:
1. Summarize what happened today
2. Note any open items or follow-ups
3. Preview tomorrow's schedule
4. Consolidate any memories worth keeping

"""

    async def _save_digest(self) -> None:
        """Periodically save digest to disk."""
        try:
            save_digest()
        except Exception as e:
            logger.error(f"Error saving digest: {e}")

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self.setup_jobs()
        self.scheduler.start()
        self._running = True
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self.scheduler.shutdown()
        self._running = False
        save_digest()
        logger.info("Scheduler stopped")

    async def run_all_scouts_now(self) -> list[Observation]:
        """Run all scouts immediately (for testing/manual trigger)."""
        all_observations = []

        for scout in [self.email_scout, self.calendar_scout,
                      self.weather_scout, self.time_scout,
                      self.health_scout, self.location_scout,
                      self.memory_scout, self.system_scout, self.reminders_scout]:
            try:
                observations = await scout.run()
                all_observations.extend(observations)
            except Exception as e:
                logger.error(f"[{scout.name}] Error: {e}")

        if all_observations:
            digest = get_digest()
            digest.add_many(all_observations)

        # Update heartbeat after sweep
        self._update_heartbeat()

        return all_observations


# For testing
if __name__ == "__main__":
    import asyncio

    async def test_escalation(escalations):
        print(f"\n⚠️ ESCALATION: {len(escalations)} item(s)")
        for esc in escalations:
            print(f"  - [{esc.scout}] {esc.observation}")

    async def test_wake(reason, prompt):
        print(f"\n🔔 WAKE: {reason}")
        print(f"Prompt preview:\n{prompt[:500]}...")

    async def main():
        scheduler = DorisScheduler(
            on_escalation=test_escalation,
            on_wake=test_wake,
        )

        print("Running all scouts once...")
        observations = await scheduler.run_all_scouts_now()

        print(f"\nCollected {len(observations)} observations")
        for obs in observations:
            marker = "⚠️" if obs.escalate else "📌"
            print(f"  {marker} [{obs.scout}] {obs.observation}")

        # Show digest
        digest = get_digest()
        print("\n" + digest.format_for_prompt())

    asyncio.run(main())
