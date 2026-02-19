"""
Reminders Scout

Rule-based scout that monitors Apple Reminders for due and overdue items.
Runs every 15 minutes via the daemon scheduler.

Uses the existing tools/reminders.py CLI integration (no LLM needed).
"""

from datetime import datetime, timedelta
from typing import Optional
import sys
import logging
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scouts.base import Scout, Observation, Relevance
from security.injection_scanner import scan_for_injection

logger = logging.getLogger(__name__)


class RemindersScout(Scout):
    """
    Scout that monitors reminders for due and overdue items.

    Checks:
    - Reminders due within the next hour
    - Overdue reminders (summarized, debounced to once per hour)

    Entirely rule-based (no LLM) — calls the local Swift CLI via
    tools/reminders.py to get current reminders.
    """

    name = "reminders-scout"

    # Cap to prevent unbounded growth
    MAX_TRACKED_IDS = 500

    def __init__(self):
        super().__init__()
        self._alerted_reminder_ids: set[str] = set()
        self._last_overdue_alert: Optional[datetime] = None

    async def observe(self) -> list[Observation]:
        """
        Check reminders for noteworthy items.

        Returns observations for:
        - Reminders due in the next hour (per-item, deduplicated)
        - Overdue reminders (summary, once per hour)
        """
        observations = []
        now = datetime.now()

        try:
            from tools.reminders import list_reminders

            reminders = list_reminders(include_completed=False)

            if not reminders:
                return []

            # Check each reminder
            for reminder in reminders:
                due_str = reminder.get("dueDate")
                if not due_str:
                    continue

                try:
                    due_date = datetime.fromisoformat(due_str.replace("Z", "+00:00"))
                    # Make naive for comparison with now
                    if due_date.tzinfo:
                        due_date = due_date.replace(tzinfo=None)
                except (ValueError, AttributeError):
                    continue

                reminder_id = reminder.get("id", due_str)
                title = reminder.get("title", "Untitled")
                priority = reminder.get("priority", 0)

                minutes_until = (due_date - now).total_seconds() / 60

                # Due within 1 hour — individual alert
                if -5 <= minutes_until <= 60:
                    obs = self._create_due_soon_observation(
                        reminder_id, title, priority, minutes_until, now
                    )
                    if obs:
                        observations.append(obs)

                # Overdue (more than 5 minutes past due)
                elif minutes_until < -5:
                    obs = self._create_overdue_observation(
                        reminder_id, title, minutes_until, now
                    )
                    if obs:
                        observations.append(obs)

        except FileNotFoundError:
            logger.error(f"[{self.name}] Reminders CLI binary not found")
        except Exception as e:
            logger.error(f"[{self.name}] Error checking reminders: {e}")

        # Prune alerted IDs if they've grown too large
        if len(self._alerted_reminder_ids) > self.MAX_TRACKED_IDS:
            self._alerted_reminder_ids = set(
                list(self._alerted_reminder_ids)[-self.MAX_TRACKED_IDS // 2:]
            )

        return observations

    def _create_due_soon_observation(
        self,
        reminder_id: str,
        title: str,
        priority: int,
        minutes_until: float,
        now: datetime,
    ) -> Optional[Observation]:
        """Create observation for a reminder due within the next hour."""
        if reminder_id in self._alerted_reminder_ids:
            return None

        self._alerted_reminder_ids.add(reminder_id)

        # Scan reminder title for injection patterns (defense-in-depth)
        scan = scan_for_injection(title, source="reminders-scout")
        if scan.is_suspicious:
            logger.warning(
                f"[{self.name}] Suspicious reminder title detected "
                f"(risk={scan.risk_level}): {title[:100]!r}"
            )

        # Format urgency
        mins = int(minutes_until)
        if mins <= 0:
            urgency = "now"
        elif mins <= 5:
            urgency = "in a few minutes"
        elif mins <= 15:
            urgency = f"in {mins} minutes"
        else:
            urgency = f"in about {mins} minutes"

        # High-priority reminders escalate
        is_high_priority = priority >= 2
        relevance = Relevance.HIGH if is_high_priority else Relevance.MEDIUM
        escalate = is_high_priority

        return Observation(
            scout=self.name,
            timestamp=now,
            observation=f"Reminder due {urgency}: '{title}'",
            relevance=relevance,
            escalate=escalate,
            context_tags=["reminders", "due-soon"],
            raw_data={
                "id": reminder_id,
                "title": title,
                "priority": priority,
            },
        )

    def _create_overdue_observation(
        self,
        reminder_id: str,
        title: str,
        minutes_until: float,
        now: datetime,
    ) -> Optional[Observation]:
        """
        Create observation for an overdue reminder.

        Only sends an overdue summary once per hour to avoid spam.
        Individual overdue items are tracked by ID to avoid repeats.
        """
        if reminder_id in self._alerted_reminder_ids:
            return None

        # Debounce: only alert about overdue reminders once per hour
        if self._last_overdue_alert:
            if (now - self._last_overdue_alert).total_seconds() < 3600:
                return None

        self._alerted_reminder_ids.add(reminder_id)
        self._last_overdue_alert = now

        # Scan reminder title for injection patterns (defense-in-depth)
        scan = scan_for_injection(title, source="reminders-scout")
        if scan.is_suspicious:
            logger.warning(
                f"[{self.name}] Suspicious overdue reminder title detected "
                f"(risk={scan.risk_level}): {title[:100]!r}"
            )

        hours_overdue = abs(int(minutes_until)) // 60
        if hours_overdue < 1:
            time_str = f"{abs(int(minutes_until))} minutes"
        elif hours_overdue < 24:
            time_str = f"{hours_overdue} hour{'s' if hours_overdue != 1 else ''}"
        else:
            days = hours_overdue // 24
            time_str = f"{days} day{'s' if days != 1 else ''}"

        return Observation(
            scout=self.name,
            timestamp=now,
            observation=f"Overdue reminder ({time_str} past due): '{title}'",
            relevance=Relevance.MEDIUM,
            escalate=False,
            context_tags=["reminders", "overdue"],
            raw_data={
                "id": reminder_id,
                "title": title,
                "overdue_minutes": abs(int(minutes_until)),
            },
        )


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        scout = RemindersScout()
        print(f"Running {scout.name}...")

        observations = await scout.run()

        if not observations:
            print("No reminder alerts right now")
        else:
            print(f"Found {len(observations)} observations:\n")
            for obs in observations:
                marker = "⚠️ ESCALATE" if obs.escalate else f"[{obs.relevance.value}]"
                print(f"{marker} {obs.observation}")
                print(f"  Tags: {obs.context_tags}")
                print()

    asyncio.run(test())
