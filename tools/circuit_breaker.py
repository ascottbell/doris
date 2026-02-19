"""
Circuit Breaker pattern for tool execution.

Prevents Doris from getting stuck in error loops when tools fail repeatedly.
Tracks failures per-tool and disables failing tools for a cooldown period.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from enum import Enum
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, calls pass through
    OPEN = "open"          # Circuit tripped, calls are blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Tracks circuit breaker statistics for a single tool."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_error: Optional[str] = None
    opened_at: float = 0.0
    total_calls: int = 0
    total_failures: int = 0
    total_blocked: int = 0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 3        # Consecutive failures before opening
    cooldown_seconds: float = 300.0   # 5 minutes cooldown
    half_open_max_calls: int = 1      # Calls allowed in half-open state


# Tool categories for grouping related tools
TOOL_CATEGORIES = {
    # Apple ecosystem
    "apple_ecosystem": [
        "get_calendar_events", "create_calendar_event",
        "create_reminder", "list_reminders", "complete_reminder",
        "send_imessage", "read_imessages",
        "search_notes", "read_note", "create_note",
        "run_shortcut", "list_shortcuts",
    ],
    # Gmail
    "gmail": [
        "check_email", "send_email", "read_email",
    ],
    # MCP servers
    "mcp_music": ["control_music"],
    "mcp_filesystem": ["read_file", "write_file", "list_directory", "search_files"],
    "mcp_web": ["web_search", "control_browser"],
    # Memory
    "memory": ["store_memory", "search_memory"],
    # Local tools (rarely fail)
    "local": [
        "get_current_time", "get_weather", "daily_briefing",
        "lookup_contact", "system_info", "query_documents",
    ],
    # Notifications (app messages + emergency push)
    "notify": ["notify_user"],
}


# Reverse mapping: tool -> category
def _build_tool_to_category() -> Dict[str, str]:
    mapping = {}
    for category, tools in TOOL_CATEGORIES.items():
        for tool in tools:
            mapping[tool] = category
    return mapping

TOOL_TO_CATEGORY = _build_tool_to_category()


class CircuitBreaker:
    """
    Circuit breaker for tool execution.

    Prevents cascading failures by disabling tools that fail repeatedly.
    """

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self._circuits: Dict[str, CircuitStats] = {}
        self._lock = threading.Lock()

    def get_circuit(self, tool_name: str) -> CircuitStats:
        """Get or create circuit stats for a tool."""
        with self._lock:
            if tool_name not in self._circuits:
                self._circuits[tool_name] = CircuitStats()
            return self._circuits[tool_name]

    def can_execute(self, tool_name: str) -> tuple[bool, Optional[str]]:
        """
        Check if a tool can be executed.

        Returns:
            (allowed, reason) - True if allowed, or False with reason why blocked
        """
        circuit = self.get_circuit(tool_name)
        now = time.time()

        with self._lock:
            if circuit.state == CircuitState.CLOSED:
                return True, None

            elif circuit.state == CircuitState.OPEN:
                # Check if cooldown has passed
                time_since_opened = now - circuit.opened_at
                if time_since_opened >= self.config.cooldown_seconds:
                    # Transition to half-open
                    circuit.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit for {tool_name} transitioning to HALF_OPEN")
                    return True, None
                else:
                    remaining = int(self.config.cooldown_seconds - time_since_opened)
                    circuit.total_blocked += 1
                    return False, f"Tool '{tool_name}' is temporarily disabled due to repeated failures. Will retry in {remaining}s."

            elif circuit.state == CircuitState.HALF_OPEN:
                # Allow limited calls to test recovery
                return True, None

        return True, None

    def record_success(self, tool_name: str):
        """Record a successful tool execution."""
        circuit = self.get_circuit(tool_name)

        with self._lock:
            circuit.success_count += 1
            circuit.total_calls += 1
            circuit.last_success_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                # Success in half-open state - close the circuit
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
                logger.info(f"Circuit for {tool_name} CLOSED after successful recovery")
            elif circuit.state == CircuitState.CLOSED:
                # Reset failure count on success
                circuit.failure_count = 0

    def record_failure(self, tool_name: str, error: str):
        """Record a failed tool execution."""
        circuit = self.get_circuit(tool_name)
        now = time.time()

        with self._lock:
            circuit.failure_count += 1
            circuit.total_failures += 1
            circuit.total_calls += 1
            circuit.last_failure_time = now
            circuit.last_error = error[:200]  # Truncate long errors

            if circuit.state == CircuitState.HALF_OPEN:
                # Failure in half-open state - back to open
                circuit.state = CircuitState.OPEN
                circuit.opened_at = now
                logger.warning(f"Circuit for {tool_name} back to OPEN after half-open failure")

            elif circuit.state == CircuitState.CLOSED:
                if circuit.failure_count >= self.config.failure_threshold:
                    circuit.state = CircuitState.OPEN
                    circuit.opened_at = now
                    logger.warning(f"Circuit for {tool_name} OPENED after {circuit.failure_count} consecutive failures")

    def get_status(self) -> Dict[str, dict]:
        """Get status of all circuits."""
        status = {}
        with self._lock:
            for name, circuit in self._circuits.items():
                status[name] = {
                    "state": circuit.state.value,
                    "failure_count": circuit.failure_count,
                    "last_error": circuit.last_error,
                    "total_calls": circuit.total_calls,
                    "total_failures": circuit.total_failures,
                    "total_blocked": circuit.total_blocked,
                }
        return status

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system health based on circuit states.

        Returns categories and their health status.
        """
        health = {
            "healthy": True,
            "degraded_tools": [],
            "disabled_tools": [],
            "category_status": {},
        }

        with self._lock:
            for tool_name, circuit in self._circuits.items():
                category = TOOL_TO_CATEGORY.get(tool_name, "unknown")

                if circuit.state == CircuitState.OPEN:
                    health["disabled_tools"].append(tool_name)
                    health["healthy"] = False
                    health["category_status"][category] = "disabled"
                elif circuit.state == CircuitState.HALF_OPEN:
                    health["degraded_tools"].append(tool_name)
                    if category not in health["category_status"]:
                        health["category_status"][category] = "degraded"

        return health

    def reset_circuit(self, tool_name: str):
        """Manually reset a circuit to closed state."""
        circuit = self.get_circuit(tool_name)
        with self._lock:
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            logger.info(f"Circuit for {tool_name} manually reset to CLOSED")

    def reset_all(self):
        """Reset all circuits to closed state."""
        with self._lock:
            for circuit in self._circuits.values():
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
            logger.info("All circuits reset to CLOSED")


# Global circuit breaker instance
_circuit_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get the global circuit breaker instance."""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker()
    return _circuit_breaker


def reset_circuit_breaker():
    """Reset the global circuit breaker (mainly for testing)."""
    global _circuit_breaker
    if _circuit_breaker is not None:
        _circuit_breaker.reset_all()


# User-friendly error messages for different failure types
ERROR_MESSAGES = {
    "gmail": "I'm having trouble with email access. Gmail might be having issues.",
    "apple_ecosystem": "Apple services aren't responding properly right now.",
    "mcp_music": "Music control isn't working at the moment.",
    "mcp_filesystem": "I can't access the file system right now.",
    "mcp_web": "Web access is having problems.",
    "memory": "My memory system isn't responding. I'll try again shortly.",
    "notify": "Notifications aren't working at the moment.",
    "local": "Something's wrong with a basic operation. That's unusual.",
    "unknown": "That tool isn't working right now.",
}


def get_friendly_error(tool_name: str, circuit: CircuitStats) -> str:
    """Get a user-friendly error message for a disabled tool."""
    category = TOOL_TO_CATEGORY.get(tool_name, "unknown")
    base_message = ERROR_MESSAGES.get(category, ERROR_MESSAGES["unknown"])

    # Calculate time remaining
    if circuit.opened_at > 0:
        remaining = int(300 - (time.time() - circuit.opened_at))
        if remaining > 60:
            time_str = f"{remaining // 60} minutes"
        else:
            time_str = f"{remaining} seconds"
        return f"{base_message} I'll try again in about {time_str}."

    return base_message


class HealthTracker:
    """
    Tracks overall system health and provides status for the voice loop.

    The voice loop can check this before attempting operations to give
    better error messages proactively.
    """

    def __init__(self, circuit_breaker: CircuitBreaker = None):
        self.circuit_breaker = circuit_breaker or get_circuit_breaker()
        self._custom_status: Dict[str, str] = {}
        self._last_claude_failure: float = 0.0
        self._claude_consecutive_failures: int = 0
        self._lock = threading.Lock()

    def set_status(self, component: str, status: str):
        """Set a custom status message for a component."""
        with self._lock:
            self._custom_status[component] = status

    def clear_status(self, component: str):
        """Clear a custom status message."""
        with self._lock:
            self._custom_status.pop(component, None)

    def record_claude_failure(self):
        """Record a Claude API failure."""
        with self._lock:
            self._last_claude_failure = time.time()
            self._claude_consecutive_failures += 1

    def record_claude_success(self):
        """Record a Claude API success."""
        with self._lock:
            self._claude_consecutive_failures = 0

    def _is_claude_degraded_unlocked(self) -> bool:
        """Check if Claude is degraded (call with lock already held)."""
        # Degraded if we've had multiple failures recently
        if self._claude_consecutive_failures >= 2:
            return True
        # Or if we failed within the last minute
        if time.time() - self._last_claude_failure < 60:
            return self._claude_consecutive_failures > 0
        return False

    def is_claude_degraded(self) -> bool:
        """Check if Claude API is experiencing issues."""
        with self._lock:
            return self._is_claude_degraded_unlocked()

    def get_full_status(self) -> Dict[str, Any]:
        """Get full health status for dashboard."""
        circuit_health = self.circuit_breaker.get_health_summary()

        with self._lock:
            claude_degraded = self._is_claude_degraded_unlocked()
            return {
                "healthy": circuit_health["healthy"] and not claude_degraded,
                "claude_status": "degraded" if claude_degraded else "healthy",
                "disabled_tools": circuit_health["disabled_tools"],
                "degraded_tools": circuit_health["degraded_tools"],
                "custom_status": dict(self._custom_status),
                "circuit_details": self.circuit_breaker.get_status(),
            }

    def get_voice_status_message(self) -> Optional[str]:
        """
        Get a status message for the voice loop to optionally speak.

        Returns None if everything is healthy.
        """
        if self.is_claude_degraded():
            return "I'm having trouble thinking clearly right now. My brain might need a moment."

        health = self.circuit_breaker.get_health_summary()
        if health["disabled_tools"]:
            # Group by category
            categories = set()
            for tool in health["disabled_tools"]:
                cat = TOOL_TO_CATEGORY.get(tool, "unknown")
                categories.add(cat)

            if "home_assistant" in categories:
                return "Just so you know, I can't control the smart home right now."
            if len(categories) > 2:
                return "I'm experiencing some technical difficulties with a few of my tools."

        return None


# Global health tracker instance
_health_tracker: Optional[HealthTracker] = None


def get_health_tracker() -> HealthTracker:
    """Get the global health tracker instance."""
    global _health_tracker
    if _health_tracker is None:
        _health_tracker = HealthTracker()
    return _health_tracker
