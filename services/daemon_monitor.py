"""
Daemon Health Monitor

Runs as a background task in the FastAPI server to monitor daemon health.
Auto-restarts daemon if it's unresponsive and notifies Doris.
"""

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Configuration
STATE_FILE = Path(__file__).parent.parent / "data" / "daemon_state.json"
DAEMON_LABEL = "com.doris.daemon"
CHECK_INTERVAL_SECONDS = 120  # Check every 2 minutes
HEARTBEAT_STALE_SECONDS = 300  # Consider stale after 5 minutes
DORIS_URL = "http://localhost:8000/chat/text"


class DaemonMonitor:
    """
    Monitors daemon health and auto-restarts if needed.

    Checks:
    1. Is the daemon process running? (via launchctl)
    2. Is the heartbeat fresh? (last_scout_run < 5 min ago)

    If either check fails, restarts the daemon and notifies Doris.
    """

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_restart: Optional[datetime] = None
        self._restart_cooldown = timedelta(minutes=5)  # Don't restart more than once per 5 min

    def start(self) -> None:
        """Start the monitor background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Daemon monitor started")

    def stop(self) -> None:
        """Stop the monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Daemon monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        # Wait a bit on startup to let daemon initialize
        await asyncio.sleep(30)

        while self._running:
            try:
                await self._check_daemon_health()
            except Exception as e:
                logger.error(f"Daemon monitor error: {e}")

            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

    async def _check_daemon_health(self) -> None:
        """Check daemon health and restart if needed."""
        is_running = self._is_daemon_running()
        heartbeat_fresh = self._is_heartbeat_fresh()

        if is_running and heartbeat_fresh:
            # All good
            return

        # Something's wrong
        issues = []
        if not is_running:
            issues.append("process not running")
        if not heartbeat_fresh:
            issues.append("heartbeat stale (no scout activity)")

        logger.warning(f"Daemon unhealthy: {', '.join(issues)}")

        # Check restart cooldown
        if self._last_restart:
            time_since_restart = datetime.now() - self._last_restart
            if time_since_restart < self._restart_cooldown:
                logger.warning(f"Skipping restart (cooldown active, {time_since_restart.seconds}s since last)")
                return

        # Restart daemon
        success = self._restart_daemon()

        if success:
            self._last_restart = datetime.now()
            # Notify Doris
            await self._notify_doris(issues)

    def _is_daemon_running(self) -> bool:
        """Check if daemon process is running via launchctl."""
        try:
            result = subprocess.run(
                ["launchctl", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )

            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 3 and parts[2] == DAEMON_LABEL:
                    pid = parts[0].strip()
                    # Running if PID is a number (not "-")
                    return pid != "-" and pid.isdigit()

            return False
        except Exception as e:
            logger.error(f"Failed to check daemon status: {e}")
            return False

    def _is_heartbeat_fresh(self) -> bool:
        """Check if daemon heartbeat is recent."""
        try:
            if not STATE_FILE.exists():
                return False

            state = json.loads(STATE_FILE.read_text())
            last_run = state.get("last_scout_run")

            if not last_run:
                # No heartbeat recorded yet - give benefit of doubt if just started
                started_at = state.get("started_at")
                if started_at:
                    started = datetime.fromisoformat(started_at)
                    # Allow 5 min grace period after startup
                    if datetime.now() - started < timedelta(minutes=5):
                        return True
                return False

            last_run_dt = datetime.fromisoformat(last_run)
            age = datetime.now() - last_run_dt

            return age.total_seconds() < HEARTBEAT_STALE_SECONDS

        except Exception as e:
            logger.error(f"Failed to check heartbeat: {e}")
            return False

    def _restart_daemon(self) -> bool:
        """Restart the daemon via launchctl."""
        try:
            # Get user ID for launchctl
            import os
            uid = os.getuid()

            # Use kickstart -k for atomic restart
            result = subprocess.run(
                ["launchctl", "kickstart", "-k", f"gui/{uid}/{DAEMON_LABEL}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info("Daemon restarted successfully")
                return True
            else:
                # Try stop then start
                subprocess.run(["launchctl", "stop", DAEMON_LABEL], timeout=5)
                import time
                time.sleep(1)
                subprocess.run(["launchctl", "start", DAEMON_LABEL], timeout=5)
                logger.info("Daemon restarted via stop/start")
                return True

        except Exception as e:
            logger.error(f"Failed to restart daemon: {e}")
            return False

    async def _notify_doris(self, issues: list[str]) -> None:
        """Notify Doris about the daemon restart."""
        message = (
            f"[SELF-HEALING] I detected my daemon was unhealthy ({', '.join(issues)}) "
            f"and restarted it automatically. Check the daemon logs at "
            f"logs/daemon.error.log to see what went wrong."
        )

        try:
            from config import settings
            headers = {"Content-Type": "application/json"}
            if settings.doris_api_token:
                headers["Authorization"] = f"Bearer {settings.doris_api_token}"

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    DORIS_URL,
                    json={"message": message, "daemon": True, "self_healing": True},
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        logger.info("Notified Doris about daemon restart")
                    else:
                        logger.warning(f"Failed to notify Doris: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"Failed to notify Doris: {e}")


# Singleton instance
_monitor: Optional[DaemonMonitor] = None


def get_daemon_monitor() -> DaemonMonitor:
    """Get or create the daemon monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = DaemonMonitor()
    return _monitor


def start_daemon_monitor() -> None:
    """Start the daemon monitor."""
    get_daemon_monitor().start()


def stop_daemon_monitor() -> None:
    """Stop the daemon monitor."""
    if _monitor:
        _monitor.stop()
