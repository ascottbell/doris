"""
Token budget circuit breaker for Claude API spending.

Prevents runaway token usage by enforcing budgets at hourly and daily windows.
Soft warning at 80%, hard block at 100%.

State persists to disk so budgets survive daemon/server restarts.

Usage:
    from llm.token_budget import check_budget, record_usage, BudgetExceeded

    # Before making API call (optional pre-check):
    check_budget(estimated_tokens)  # raises BudgetExceeded if over

    # After API call succeeds:
    record_usage(actual_tokens, source="chat")
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from security.file_io import atomic_json_write, locked_json_read, locked_json_update
from config import settings

logger = logging.getLogger("doris.token_budget")

# Budget thresholds (in tokens)
# Based on actual usage data: peak hour 2.1M, peak day 6.2M
HOURLY_BUDGET = 3_000_000   # ~$9/hr at Opus pricing
DAILY_BUDGET = 8_000_000    # ~$24/day at Opus pricing

SOFT_LIMIT_PCT = 0.80  # Warn at 80%

STATE_FILE = settings.data_dir / "token_budget_state.json"


class BudgetExceeded(Exception):
    """Raised when a token budget window is exhausted."""
    def __init__(self, window: str, used: int, budget: int):
        self.window = window
        self.used = used
        self.budget = budget
        super().__init__(f"{window} budget exceeded: {used:,}/{budget:,} tokens")


class _BudgetState:
    """Tracks token usage across time windows. Singleton."""

    def __init__(self):
        self.hourly_start: Optional[str] = None
        self.hourly_tokens: int = 0
        self.daily_start: Optional[str] = None
        self.daily_tokens: int = 0
        self._load()

    def _load(self) -> None:
        if not STATE_FILE.exists():
            self._reset_all()
            return
        try:
            data = locked_json_read(STATE_FILE, default=None)
            if data is None:
                self._reset_all()
                return
            self.hourly_start = data.get("hourly_start")
            self.hourly_tokens = data.get("hourly_tokens", 0)
            self.daily_start = data.get("daily_start")
            self.daily_tokens = data.get("daily_tokens", 0)
            self._maybe_reset_windows()
        except Exception as e:
            logger.error(f"Failed to load budget state: {e}")
            self._reset_all()

    def _save(self) -> None:
        try:
            atomic_json_write(STATE_FILE, {
                "hourly_start": self.hourly_start,
                "hourly_tokens": self.hourly_tokens,
                "daily_start": self.daily_start,
                "daily_tokens": self.daily_tokens,
            })
        except Exception as e:
            logger.error(f"Failed to save budget state: {e}")

    def _reset_all(self) -> None:
        now = datetime.now().isoformat()
        self.hourly_start = now
        self.hourly_tokens = 0
        self.daily_start = now
        self.daily_tokens = 0

    def _maybe_reset_windows(self) -> None:
        now = datetime.now()
        if self.hourly_start:
            try:
                if now - datetime.fromisoformat(self.hourly_start) >= timedelta(hours=1):
                    self.hourly_start = now.isoformat()
                    self.hourly_tokens = 0
            except ValueError:
                self.hourly_start = now.isoformat()
                self.hourly_tokens = 0

        if self.daily_start:
            try:
                if now - datetime.fromisoformat(self.daily_start) >= timedelta(days=1):
                    self.daily_start = now.isoformat()
                    self.daily_tokens = 0
            except ValueError:
                self.daily_start = now.isoformat()
                self.daily_tokens = 0

    def add(self, tokens: int) -> None:
        self._maybe_reset_windows()
        self.hourly_tokens += tokens
        self.daily_tokens += tokens
        self._save()

    def check(self, tokens: int) -> Optional[str]:
        """
        Check if adding tokens would exceed budget.

        Returns:
            None if OK, "soft:<window>" if at soft limit,
            raises BudgetExceeded if over hard limit.
        """
        self._maybe_reset_windows()
        warning = None

        # Check hourly
        projected = self.hourly_tokens + tokens
        if projected > HOURLY_BUDGET:
            raise BudgetExceeded("hourly", self.hourly_tokens, HOURLY_BUDGET)
        if projected > HOURLY_BUDGET * SOFT_LIMIT_PCT:
            warning = f"hourly budget at {self.hourly_tokens / HOURLY_BUDGET * 100:.0f}%"

        # Check daily
        projected = self.daily_tokens + tokens
        if projected > DAILY_BUDGET:
            raise BudgetExceeded("daily", self.daily_tokens, DAILY_BUDGET)
        if projected > DAILY_BUDGET * SOFT_LIMIT_PCT:
            warning = f"daily budget at {self.daily_tokens / DAILY_BUDGET * 100:.0f}%"

        return warning


# Module-level singleton
_state: Optional[_BudgetState] = None


def _get_state() -> _BudgetState:
    global _state
    if _state is None:
        _state = _BudgetState()
    return _state


def check_budget(tokens: int) -> Optional[str]:
    """
    Check if a request of `tokens` size would exceed budget.

    Returns warning string at soft limit, None if OK.
    Raises BudgetExceeded at hard limit.
    """
    return _get_state().check(tokens)


def record_usage(tokens: int, source: str = "chat") -> None:
    """Record that tokens were consumed."""
    state = _get_state()
    state.add(tokens)
    logger.debug(f"Recorded {tokens:,} tokens ({source}). "
                 f"Hourly: {state.hourly_tokens:,}/{HOURLY_BUDGET:,}, "
                 f"Daily: {state.daily_tokens:,}/{DAILY_BUDGET:,}")


def get_status() -> dict:
    """Get current budget utilization for dashboards/monitoring."""
    state = _get_state()
    state._maybe_reset_windows()
    return {
        "hourly": {
            "used": state.hourly_tokens,
            "budget": HOURLY_BUDGET,
            "pct": round(state.hourly_tokens / HOURLY_BUDGET * 100, 1),
            "window_start": state.hourly_start,
        },
        "daily": {
            "used": state.daily_tokens,
            "budget": DAILY_BUDGET,
            "pct": round(state.daily_tokens / DAILY_BUDGET * 100, 1),
            "window_start": state.daily_start,
        },
    }


def reset_budgets() -> None:
    """Admin reset — clears all budget counters."""
    state = _get_state()
    state._reset_all()
    state._save()
    logger.warning("All token budgets reset manually")
