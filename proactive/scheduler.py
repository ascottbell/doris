"""
Scheduler - Runs proactive monitors on intervals.

Simple threading-based scheduler for background jobs.
"""

import threading
import time
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


class ProactiveScheduler:
    """
    Background scheduler for proactive monitoring jobs.

    Usage:
        scheduler = ProactiveScheduler()
        scheduler.add_job("email", email_monitor, interval_minutes=15)
        scheduler.add_job("weather", weather_monitor, interval_minutes=60)
        scheduler.start()
        ...
        scheduler.stop()
    """

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._running = False

    def add_job(
        self,
        name: str,
        func: Callable,
        interval_minutes: int = 15,
        run_immediately: bool = True
    ):
        """
        Add a monitoring job.

        Args:
            name: Unique job identifier
            func: Function to call (should take no arguments)
            interval_minutes: How often to run (default 15 min)
            run_immediately: If True, run once immediately on start
        """
        self._jobs[name] = {
            "func": func,
            "interval": interval_minutes * 60,  # Convert to seconds
            "run_immediately": run_immediately,
            "last_run": None,
            "run_count": 0,
            "error_count": 0
        }

    def remove_job(self, name: str):
        """Remove a job (stops it if running)."""
        if name in self._stop_events:
            self._stop_events[name].set()
        if name in self._jobs:
            del self._jobs[name]

    def start(self):
        """Start all registered jobs."""
        if self._running:
            print("[scheduler] Already running")
            return

        self._running = True
        print(f"[scheduler] Starting {len(self._jobs)} jobs...")

        for name, job in self._jobs.items():
            self._start_job(name, job)

    def _start_job(self, name: str, job: dict):
        """Start a single job in its own thread."""
        stop_event = threading.Event()
        self._stop_events[name] = stop_event

        def job_runner():
            func = job["func"]
            interval = job["interval"]

            # Run immediately if configured
            if job["run_immediately"]:
                self._run_job_safely(name, func)

            # Then run on interval
            while not stop_event.is_set():
                # Wait for interval, but check stop_event periodically
                for _ in range(int(interval)):
                    if stop_event.is_set():
                        break
                    time.sleep(1)

                if not stop_event.is_set():
                    self._run_job_safely(name, func)

        thread = threading.Thread(target=job_runner, name=f"proactive-{name}", daemon=True)
        thread.start()
        self._threads[name] = thread
        print(f"[scheduler] Started job: {name} (every {job['interval']//60} min)")

    def _run_job_safely(self, name: str, func: Callable):
        """Run a job function with error handling."""
        try:
            now = datetime.now(EASTERN)
            print(f"[scheduler] Running {name} at {now.strftime('%I:%M %p')}")
            func()
            self._jobs[name]["last_run"] = now
            self._jobs[name]["run_count"] += 1
        except Exception as e:
            print(f"[scheduler] Error in {name}: {e}")
            self._jobs[name]["error_count"] += 1

    def stop(self):
        """Stop all jobs."""
        if not self._running:
            return

        print("[scheduler] Stopping all jobs...")
        self._running = False

        # Signal all jobs to stop
        for stop_event in self._stop_events.values():
            stop_event.set()

        # Wait for threads to finish (with timeout)
        for name, thread in self._threads.items():
            thread.join(timeout=2)

        self._threads.clear()
        self._stop_events.clear()
        print("[scheduler] All jobs stopped")

    def status(self) -> dict:
        """Get status of all jobs."""
        result = {
            "running": self._running,
            "jobs": {}
        }

        for name, job in self._jobs.items():
            result["jobs"][name] = {
                "interval_minutes": job["interval"] // 60,
                "last_run": job["last_run"].isoformat() if job["last_run"] else None,
                "run_count": job["run_count"],
                "error_count": job["error_count"]
            }

        return result

    def run_now(self, name: str) -> bool:
        """Manually trigger a job to run immediately."""
        if name not in self._jobs:
            print(f"[scheduler] Unknown job: {name}")
            return False

        func = self._jobs[name]["func"]
        self._run_job_safely(name, func)
        return True


# Global scheduler instance
_scheduler: Optional[ProactiveScheduler] = None


def get_scheduler() -> ProactiveScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ProactiveScheduler()
    return _scheduler


def init_scheduler():
    """Initialize and start the proactive scheduler with default jobs."""
    from .sources import get_all_monitors

    scheduler = get_scheduler()

    # Register all source monitors
    for name, monitor_func, interval in get_all_monitors():
        scheduler.add_job(name, monitor_func, interval_minutes=interval)

    scheduler.start()
    return scheduler


def shutdown_scheduler():
    """Stop the scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
        _scheduler = None
