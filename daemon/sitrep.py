"""
Sitrep Engine — Intelligent Escalation for Doris

Replaces hash-based escalation dedup with consolidated situation reports.
Scouts observe; Doris decides.

Two lanes:
- Instant: Life safety emergencies bypass review (memory-scout critical data loss).
  Notify immediately.
- Sitrep: Everything else accumulates. Every 30 minutes, Doris reviews a consolidated
  sitrep with full context (notification ledger, ongoing conditions, time of day)
  and makes editorial decisions: NOTIFY, HOLD, or DISMISS.

Persistence (in data/sitrep/):
- observation_buffer.json — pending observations
- notification_ledger.json — what was sent (pruned at 72h)
- ongoing_conditions.json — Doris-managed conditions
- review_log.json — audit trail (500 entries max)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from scouts.base import Observation, Relevance
from security.file_io import locked_json_read, locked_json_update, atomic_json_write
from config import settings

logger = logging.getLogger("doris.sitrep")

# Persistence directory
SITREP_DIR = settings.data_dir / "sitrep"

# Instant lane: these scouts bypass sitrep review entirely
# (No HomeAssistant in open-source version — add your own if needed)
INSTANT_LANE_SCOUTS: set[str] = set()
INSTANT_LANE_TAGS = {"emergency", "alarm", "smoke", "co_detector", "water_leak"}


@dataclass
class LedgerEntry:
    """Record of a notification sent to the user."""
    timestamp: str  # ISO format
    scout: str
    summary: str
    priority: str  # "proactive" or "emergency"
    decision: str = "NOTIFY"  # from sitrep review

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LedgerEntry":
        return cls(
            timestamp=data["timestamp"],
            scout=data["scout"],
            summary=data["summary"],
            priority=data.get("priority", "proactive"),
            decision=data.get("decision", "NOTIFY"),
        )


@dataclass
class ReviewLogEntry:
    """Audit trail entry for a sitrep review."""
    timestamp: str
    observation_count: int
    scout_count: int
    decisions: list[dict]
    conditions: list[dict]
    summary: str
    notifications_sent: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewLogEntry":
        return cls(
            timestamp=data["timestamp"],
            observation_count=data["observation_count"],
            scout_count=data["scout_count"],
            decisions=data.get("decisions", []),
            conditions=data.get("conditions", []),
            summary=data.get("summary", ""),
            notifications_sent=data.get("notifications_sent", 0),
            error=data.get("error"),
        )


class SitrepEngine:
    """
    Consolidated situation report engine.

    Buffers all scout observations and periodically builds a sitrep
    for Doris to review. Doris makes the editorial decision about
    what warrants notifying the user.
    """

    def __init__(self, timezone: str = "America/New_York"):
        self._buffer: list[Observation] = []
        self._ledger: list[LedgerEntry] = []
        self._conditions: list[dict] = []
        self._last_review_summary: str = ""
        self._tz = ZoneInfo(timezone)

        # Load persisted state
        self._load_state()

    # --- Core API ---

    def ingest(self, observation: Observation) -> Optional[Observation]:
        """
        Route an observation to the appropriate lane.

        Returns the observation if it's instant lane (caller should notify
        immediately). Returns None if buffered for sitrep review.
        """
        if self._is_instant_lane(observation):
            logger.warning(f"Instant lane: [{observation.scout}] {observation.observation}")
            return observation

        # Sitrep lane — buffer it
        self._buffer.append(observation)

        # Cap buffer to prevent unbounded growth
        if len(self._buffer) > 200:
            dropped = len(self._buffer) - 200
            self._buffer = self._buffer[-200:]
            logger.warning(f"Sitrep buffer exceeded 200 — dropped {dropped} oldest")

        self._save_buffer()
        return None

    def should_review(self) -> bool:
        """True if there are buffered observations to review."""
        return len(self._buffer) > 0

    def build_sitrep(self) -> str:
        """
        Build the consolidated sitrep prompt for Doris.

        Includes time context, notification ledger, ongoing conditions,
        grouped observations, and previous review summary.
        """
        now = datetime.now(self._tz)

        # Time context
        time_ctx = self._build_time_context(now)

        # Notification ledger (last 48h for the prompt, we store 72h)
        ledger_text = self._build_ledger_text(now)

        # Ongoing conditions
        conditions_text = self._build_conditions_text()

        # Group observations by scout
        obs_text = self._build_observations_text()

        # Previous review
        prev_text = self._last_review_summary or "First review of the session."

        return f"""[SITREP REVIEW] Review the consolidated situation report and decide what warrants notifying the user.

=== TIME CONTEXT ===
{time_ctx}

=== NOTIFICATION LEDGER (last 48h) ===
{ledger_text}

=== ONGOING CONDITIONS ===
{conditions_text}

=== NEW OBSERVATIONS ({len(self._buffer)} from {self._count_scouts()} scouts, since last review) ===
{obs_text}

=== PREVIOUS REVIEW ===
{prev_text}

=== YOUR DECISION ===
For each observation or group, decide:
- NOTIFY: Send via notify_user now. Specify priority (proactive/emergency).
- HOLD: Include in next morning/evening brief. Not urgent enough to interrupt.
- DISMISS: Already covered, noise, or not actionable.

Also update your ongoing conditions list — add new ones, update existing, resolve any that are no longer active.

Respond with JSON then take action:
{{"decisions": [{{"scout": "...", "observation": "...", "action": "NOTIFY|HOLD|DISMISS", "priority": "proactive|emergency", "reason": "..."}}], "conditions": [{{"condition": "...", "status": "active|resolved", "since": "...", "notes": "..."}}], "summary": "Brief summary of your review for next time"}}
Then call notify_user for any NOTIFY decisions."""

    def record_notification(self, entry: LedgerEntry) -> None:
        """Record a notification that was sent to the user."""
        self._ledger.append(entry)
        self._prune_ledger()
        self._save_ledger()
        logger.info(f"Ledger: recorded notification [{entry.scout}] {entry.summary}")

    def update_conditions(self, conditions: list[dict]) -> None:
        """Update Doris-managed ongoing conditions."""
        self._conditions = conditions
        self._save_conditions()
        logger.info(f"Conditions updated: {len(conditions)} active")

    def record_review_summary(self, summary: str) -> None:
        """Store Doris's summary from the latest review."""
        self._last_review_summary = summary

    def clear_buffer(self) -> list[Observation]:
        """Clear the observation buffer after a successful review. Returns cleared observations."""
        cleared = list(self._buffer)
        self._buffer.clear()
        self._save_buffer()
        return cleared

    def log_review(self, entry: ReviewLogEntry) -> None:
        """Append to the review audit log."""
        try:
            log_file = SITREP_DIR / "review_log.json"
            existing = self._read_json(log_file, default=[])
            existing.append(entry.to_dict())
            # Keep last 500 entries
            if len(existing) > 500:
                existing = existing[-500:]
            self._write_json(log_file, existing)
        except Exception as e:
            logger.error(f"Failed to write review log: {e}")

    def get_overnight_summary(self) -> str:
        """
        Build summary of overnight activity for morning brief injection.

        Includes:
        - Items marked HOLD from overnight sitrep reviews
        - Active ongoing conditions
        - Notification count from overnight
        """
        now = datetime.now(self._tz)

        # "Overnight" = since 10 PM yesterday (evening reflection)
        overnight_start = now.replace(hour=22, minute=0, second=0, microsecond=0) - timedelta(days=1)
        if now.hour >= 22:
            # It's after 10 PM today, overnight started today
            overnight_start = now.replace(hour=22, minute=0, second=0, microsecond=0)

        # Use naive datetime for comparisons (ledger timestamps are naive ISO)
        overnight_start_naive = overnight_start.replace(tzinfo=None)

        # Count overnight notifications
        overnight_notifications = [
            e for e in self._ledger
            if datetime.fromisoformat(e.timestamp) > overnight_start_naive
        ]

        # Get HOLD items from review log
        hold_items = self._get_hold_items_since(overnight_start_naive)

        # Active conditions
        active_conditions = [c for c in self._conditions if c.get("status") == "active"]

        parts = []

        if overnight_notifications:
            parts.append(f"Overnight: {len(overnight_notifications)} notification(s) sent")
            for n in overnight_notifications:
                parts.append(f"  - [{n.scout}] {n.summary}")

        if hold_items:
            parts.append(f"\nHeld for morning ({len(hold_items)} items):")
            for item in hold_items:
                parts.append(f"  - {item}")

        if active_conditions:
            parts.append(f"\nOngoing conditions ({len(active_conditions)}):")
            for c in active_conditions:
                parts.append(f"  - {c.get('condition', 'unknown')} (since {c.get('since', '?')})")

        if not parts:
            return "No overnight activity to report."

        return "\n".join(parts)

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    @property
    def ledger_count(self) -> int:
        return len(self._ledger)

    @property
    def conditions(self) -> list[dict]:
        return list(self._conditions)

    # --- Instant Lane Detection ---

    def _is_instant_lane(self, obs: Observation) -> bool:
        """
        Determine if an observation should bypass sitrep review.

        Instant lane criteria:
        1. Scout in INSTANT_LANE_SCOUTS with emergency tags
        2. Memory scout with HIGH relevance + escalate (data loss)
        """
        if obs.scout in INSTANT_LANE_SCOUTS:
            if obs.escalate or any(tag in INSTANT_LANE_TAGS for tag in obs.context_tags):
                return True

        # Memory scout critical — data loss risk
        if obs.scout == "memory-scout" and obs.relevance == Relevance.HIGH and obs.escalate:
            return True

        return False

    # --- Prompt Building Helpers ---

    def _build_time_context(self, now: datetime) -> str:
        """Build time context section of the sitrep."""
        day = now.strftime("%A")
        date = now.strftime("%B %d, %Y")
        time_str = now.strftime("%I:%M %p")

        # Generic state estimation based on time
        hour = now.hour
        if hour < 6:
            state = "sleeping"
        elif hour < 9:
            state = "morning"
        elif hour < 12:
            state = "working"
        elif hour < 13:
            state = "lunch"
        elif hour < 17:
            state = "working"
        elif hour < 21:
            state = "evening"
        else:
            state = "winding down"

        if now.weekday() >= 5:
            if 9 <= hour < 17:
                state = "weekend"

        return f"Current: {day}, {date} at {time_str}\nUser's likely state: {state}"

    def _build_ledger_text(self, now: datetime) -> str:
        """Build notification ledger section (last 48h)."""
        # Use naive datetime for comparison (ledger timestamps are naive ISO)
        cutoff = (now - timedelta(hours=48)).replace(tzinfo=None)
        recent = [
            e for e in self._ledger
            if datetime.fromisoformat(e.timestamp) > cutoff
        ]

        if not recent:
            return "No notifications sent in the last 48 hours."

        lines = [f"{len(recent)} notification(s) sent:"]
        for e in recent:
            try:
                ts = datetime.fromisoformat(e.timestamp)
                ts_str = ts.strftime("%b %d %I:%M %p")
            except Exception:
                ts_str = e.timestamp
            lines.append(f"  [{ts_str}] ({e.priority}) [{e.scout}] {e.summary}")
        return "\n".join(lines)

    def _build_conditions_text(self) -> str:
        """Build ongoing conditions section."""
        active = [c for c in self._conditions if c.get("status") == "active"]
        if not active:
            return "No ongoing conditions being tracked."

        lines = []
        for c in active:
            lines.append(f"- {c.get('condition', 'unknown')} (since {c.get('since', '?')})")
            if c.get("notes"):
                lines.append(f"  Notes: {c['notes']}")
        return "\n".join(lines)

    def _build_observations_text(self) -> str:
        """Build observations section, grouped by scout."""
        if not self._buffer:
            return "No new observations."

        # Group by scout
        by_scout: dict[str, list[Observation]] = {}
        for obs in self._buffer:
            by_scout.setdefault(obs.scout, []).append(obs)

        lines = []
        for scout, observations in sorted(by_scout.items()):
            lines.append(f"[{scout}] ({len(observations)} observation(s))")
            for obs in observations:
                escalate_marker = " [flagged]" if obs.escalate else ""
                lines.append(f"  - {obs.observation}{escalate_marker}")
                if obs.context_tags:
                    lines.append(f"    tags: {', '.join(obs.context_tags)}")
            lines.append("")  # blank line between scouts
        return "\n".join(lines)

    def _count_scouts(self) -> int:
        """Count distinct scouts in the buffer."""
        return len(set(obs.scout for obs in self._buffer))

    def _get_hold_items_since(self, since: datetime) -> list[str]:
        """Get HOLD decisions from review log since a given time."""
        log_file = SITREP_DIR / "review_log.json"
        entries = self._read_json(log_file, default=[])
        hold_items = []
        for entry in entries:
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time > since.replace(tzinfo=None):
                    for d in entry.get("decisions", []):
                        if d.get("action") == "HOLD":
                            scout = d.get("scout", "?")
                            obs = d.get("observation", d.get("reason", "?"))
                            hold_items.append(f"[{scout}] {obs}")
            except (KeyError, ValueError):
                continue
        return hold_items

    # --- Persistence (using locked file I/O for safety) ---

    def _load_state(self) -> None:
        """Load all persisted state from disk."""
        SITREP_DIR.mkdir(parents=True, exist_ok=True)

        # Buffer
        buffer_data = self._read_json(SITREP_DIR / "observation_buffer.json", default=[])
        self._buffer = []
        for item in buffer_data:
            try:
                self._buffer.append(Observation.from_dict(item))
            except Exception as e:
                logger.warning(f"Skipping invalid buffered observation: {e}")
        if self._buffer:
            logger.info(f"Loaded {len(self._buffer)} buffered observations from disk")

        # Ledger
        ledger_data = self._read_json(SITREP_DIR / "notification_ledger.json", default=[])
        self._ledger = []
        for item in ledger_data:
            try:
                self._ledger.append(LedgerEntry.from_dict(item))
            except Exception as e:
                logger.warning(f"Skipping invalid ledger entry: {e}")
        self._prune_ledger()
        if self._ledger:
            logger.info(f"Loaded {len(self._ledger)} ledger entries from disk")

        # Conditions
        self._conditions = self._read_json(SITREP_DIR / "ongoing_conditions.json", default=[])
        if self._conditions:
            logger.info(f"Loaded {len(self._conditions)} ongoing conditions from disk")

    def _save_buffer(self) -> None:
        """Persist observation buffer to disk."""
        data = [obs.to_dict() for obs in self._buffer]
        self._write_json(SITREP_DIR / "observation_buffer.json", data)

    def _save_ledger(self) -> None:
        """Persist notification ledger to disk."""
        data = [e.to_dict() for e in self._ledger]
        self._write_json(SITREP_DIR / "notification_ledger.json", data)

    def _save_conditions(self) -> None:
        """Persist ongoing conditions to disk."""
        self._write_json(SITREP_DIR / "ongoing_conditions.json", self._conditions)

    def _prune_ledger(self) -> None:
        """Remove ledger entries older than 72 hours."""
        cutoff = datetime.now() - timedelta(hours=72)
        cutoff_iso = cutoff.isoformat()
        before = len(self._ledger)
        self._ledger = [e for e in self._ledger if e.timestamp > cutoff_iso]
        pruned = before - len(self._ledger)
        if pruned:
            logger.info(f"Pruned {pruned} ledger entries older than 72h")

    @staticmethod
    def _read_json(path: Path, default=None):
        """Read JSON file with locking, returning default on any error."""
        try:
            return locked_json_read(path, default=default if default is not None else {})
        except Exception as e:
            logger.warning(f"Failed to read {path.name}: {e}")
            return default if default is not None else {}

    @staticmethod
    def _write_json(path: Path, data) -> None:
        """Write JSON file atomically with error handling."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_write(path, data)
        except Exception as e:
            logger.error(f"Failed to write {path.name}: {e}")
