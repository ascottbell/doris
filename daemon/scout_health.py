"""
Scout Health Tracker

Tracks per-scout success/failure state, persisted to data/scout_health.json.
Used by the scheduler to detect persistent failures and by /status to report
honest health information.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from security.file_io import atomic_json_write, locked_json_read
from config import settings

logger = logging.getLogger("doris.daemon.health")

HEALTH_FILE = settings.data_dir / "scout_health.json"


class ScoutHealthTracker:
    """
    Tracks per-scout health state with disk persistence.

    Records successes and failures for each scout, persists to JSON,
    and provides queries for failing scouts.
    """

    def __init__(self):
        self._state: dict[str, dict] = {}
        self._meta: dict[str, any] = {}
        self._load()

    def _load(self) -> None:
        """Load state from disk under a shared lock."""
        data = locked_json_read(HEALTH_FILE, default={})
        self._meta = data.get("_meta", {})
        self._state = {k: v for k, v in data.items() if k != "_meta"}

    def _save(self) -> None:
        """Persist state to disk atomically."""
        try:
            data = {**self._state, "_meta": self._meta}
            atomic_json_write(HEALTH_FILE, data)
        except Exception as e:
            logger.error(f"Failed to save scout health: {e}")

    def _ensure_scout(self, scout_name: str) -> None:
        """Initialize scout entry if it doesn't exist."""
        if scout_name not in self._state:
            self._state[scout_name] = {
                "last_success": None,
                "last_failure": None,
                "last_error": None,
                "consecutive_failures": 0,
                "total_runs": 0,
                "total_failures": 0,
            }

    def record_success(self, scout_name: str) -> None:
        """Record a successful scout run."""
        self._ensure_scout(scout_name)
        entry = self._state[scout_name]
        entry["last_success"] = datetime.now().isoformat()
        entry["consecutive_failures"] = 0
        entry["total_runs"] += 1
        self._save()

    def record_failure(self, scout_name: str, error: str) -> None:
        """Record a failed scout run."""
        self._ensure_scout(scout_name)
        entry = self._state[scout_name]
        entry["last_failure"] = datetime.now().isoformat()
        entry["last_error"] = error[:200]
        entry["consecutive_failures"] += 1
        entry["total_runs"] += 1
        entry["total_failures"] += 1
        self._save()

    def get_health(self) -> dict[str, dict]:
        """Get all scout health state."""
        return dict(self._state)

    def get_failing_scouts(self, threshold: int = 3) -> list[str]:
        """Get scouts with consecutive failures >= threshold."""
        return [
            name for name, entry in self._state.items()
            if entry.get("consecutive_failures", 0) >= threshold
        ]

    def get_scout_status(self, scout_name: str) -> Optional[dict]:
        """Get health status for a specific scout."""
        return self._state.get(scout_name)

    def set_meta(self, key: str, value) -> None:
        """Store arbitrary metadata (e.g., last_nudge for memoir scout)."""
        self._meta[key] = value
        self._save()

    def get_meta(self, key: str, default=None):
        """Retrieve metadata."""
        return self._meta.get(key, default)


# Singleton
_tracker: Optional[ScoutHealthTracker] = None


def get_health_tracker() -> ScoutHealthTracker:
    """Get the global health tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ScoutHealthTracker()
    return _tracker
