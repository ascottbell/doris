"""
Proactive sources - monitors that detect events from various inputs.

Each source module provides:
- A monitor function that checks for new events
- Default interval for how often to check
"""

from typing import Callable


def get_all_monitors() -> list[tuple[str, Callable, int]]:
    """
    Get all registered source monitors.

    Returns list of (name, monitor_func, interval_minutes)
    """
    from .email import monitor as email_monitor
    from .calendar import monitor as calendar_monitor
    from .weather import monitor as weather_monitor
    from .checkin import monitor_checkin

    monitors = [
        ("email", email_monitor, 30),           # Check email every 30 minutes
        ("calendar", calendar_monitor, 30),     # Check calendar every 30 minutes
        ("weather", weather_monitor, 60),       # Check weather hourly
        # ("checkin", monitor_checkin, 60),     # DISABLED - daemon scouts handle this now (2026-01-22)
    ]

    # Future sources:
    # ("imessage", imessage_monitor, 5),   # Check messages every 5 min
    # ("followup", followup_monitor, 120), # Check pending follow-ups every 2 hours

    return monitors
