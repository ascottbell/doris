"""
Doris Scouts

Lightweight agents that monitor the environment and report observations.

Available scouts:
- EmailScout: Monitors Gmail (every 30 min)
- CalendarScout: Monitors Apple Calendar (every hour)
- WeatherScout: Monitors weather conditions (every hour)
- TimeScout: Rule-based time triggers (every minute)
- HealthScout: Monitors Apple Health trends (daily)

Usage:
    from scouts import EmailScout, CalendarScout, WeatherScout, TimeScout

    # Run a scout
    scout = EmailScout()
    observations = await scout.run()

    # Check for escalations
    for obs in observations:
        if scout.should_escalate(obs):
            # Wake Doris
            pass
"""

from scouts.base import Scout, HaikuScout, Observation, Relevance
from scouts.email_scout import EmailScout
from scouts.calendar_scout import CalendarScout
from scouts.weather_scout import WeatherScout
from scouts.time_scout import TimeScout
from scouts.health_scout import HealthScout
from scouts.location_scout import LocationScout
from scouts.memory_scout import MemoryScout
from scouts.system_scout import SystemScout
from scouts.reminders_scout import RemindersScout

__all__ = [
    # Base classes
    "Scout",
    "HaikuScout",
    "Observation",
    "Relevance",
    # Scout implementations
    "EmailScout",
    "CalendarScout",
    "WeatherScout",
    "TimeScout",
    "HealthScout",
    "LocationScout",
    "MemoryScout",
    "SystemScout",
    "RemindersScout",
]


def get_all_scouts() -> list[Scout]:
    """Get instances of all available scouts."""
    return [
        EmailScout(),
        CalendarScout(),
        WeatherScout(),
        TimeScout(),
        HealthScout(),
        LocationScout(),
        MemoryScout(),
        SystemScout(),
        RemindersScout(),
    ]
