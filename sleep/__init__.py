"""
Sleep-Time Compute — thin wrapper over maasv.lifecycle.

All sleep-time compute logic now lives in the maasv package.
This module re-exports everything for backward compatibility.
"""

from maasv.lifecycle.worker import (  # noqa: F401
    SleepWorker,
    SleepJob,
    JobType,
    get_sleep_worker,
    start_idle_monitor,
    stop_idle_monitor,
)
from maasv.lifecycle.memory_hygiene import run_hygiene as run_memory_hygiene  # noqa: F401

__all__ = [
    "SleepWorker",
    "SleepJob",
    "JobType",
    "get_sleep_worker",
    "start_idle_monitor",
    "stop_idle_monitor",
    "run_memory_hygiene",
]
