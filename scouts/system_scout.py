"""
System Health Scout

Monitors Doris infrastructure health: disk usage,
log file sizes, service availability, and circuit breaker states.

Runs every 15 minutes. Escalates critical issues via push notification.
Non-urgent issues are injected into conversation context.
"""

import shutil

import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Optional

from scouts.base import Scout, Observation, Relevance

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"


class SystemScout(Scout):
    """
    Scout that monitors Doris system health.

    No LLM calls — purely rule-based checks for speed and cost.

    Checks:
    - Disk usage
    - Log file sizes
    - Circuit breaker states
    - Daemon health
    - MCP server connectivity

    Escalates critical issues. Adds non-urgent warnings to digest.
    """

    name = "system-scout"

    def __init__(self):
        super().__init__()
        # MCP escalation cooldown (4 hours)
        self._last_mcp_escalation_time: Optional[datetime] = None
        self._last_mcp_failed_set: Optional[frozenset] = None
        self._mcp_cooldown_hours: float = 4.0

    async def observe(self) -> list[Observation]:
        observations = []
        now = datetime.now()

        observations.extend(self._check_disk(now))
        observations.extend(self._check_log_sizes(now))
        observations.extend(self._check_circuit_breakers(now))
        observations.extend(self._check_daemon(now))
        observations.extend(await self._check_mcp(now))

        # Run log rotation if needed
        self._maybe_rotate_logs()

        return observations

    def _check_disk(self, now: datetime) -> list[Observation]:
        """Check disk usage using stdlib shutil."""
        try:
            disk = shutil.disk_usage("/")
            pct = round(disk.used / disk.total * 100, 1)
            free_gb = round(disk.free / (1024 ** 3), 1)

            if pct > 95:
                return [Observation(
                    scout=self.name, timestamp=now,
                    observation=f"CRITICAL: Disk {pct}% full, only {free_gb}GB free. Immediate action needed.",
                    relevance=Relevance.HIGH, escalate=True,
                    context_tags=["system", "disk", "critical"],
                )]
            elif pct > 90:
                return [Observation(
                    scout=self.name, timestamp=now,
                    observation=f"Disk usage at {pct}% ({free_gb}GB free). Consider cleaning up.",
                    relevance=Relevance.MEDIUM, escalate=False,
                    context_tags=["system", "disk"],
                )]

        except Exception:
            pass
        return []

    def _check_log_sizes(self, now: datetime) -> list[Observation]:
        """Check for oversized log files."""
        observations = []

        for log_file in LOGS_DIR.glob("*.log"):
            try:
                size_mb = log_file.stat().st_size / (1024 * 1024)
                if size_mb > 100:
                    observations.append(Observation(
                        scout=self.name, timestamp=now,
                        observation=f"Log file {log_file.name} is {size_mb:.0f}MB. Rotation triggered.",
                        relevance=Relevance.MEDIUM, escalate=False,
                        context_tags=["system", "logs", "rotation"],
                    ))
            except Exception:
                continue

        return observations

    def _check_circuit_breakers(self, now: datetime) -> list[Observation]:
        """Check for open circuit breakers."""
        try:
            from tools.circuit_breaker import get_circuit_breaker

            cb = get_circuit_breaker()
            status = cb.get_status()

            open_circuits = [
                name for name, info in status.items()
                if info["state"] == "open"
            ]

            if len(open_circuits) > 3:
                return [Observation(
                    scout=self.name, timestamp=now,
                    observation=f"Multiple tools disabled: {', '.join(open_circuits[:5])}. Something may be systematically wrong.",
                    relevance=Relevance.HIGH, escalate=True,
                    context_tags=["system", "circuit-breaker", "tools"],
                    raw_data={"open_circuits": open_circuits},
                )]
            elif open_circuits:
                return [Observation(
                    scout=self.name, timestamp=now,
                    observation=f"Circuit breaker open for: {', '.join(open_circuits)}",
                    relevance=Relevance.MEDIUM, escalate=False,
                    context_tags=["system", "circuit-breaker"],
                )]

        except Exception:
            pass
        return []

    def _check_daemon(self, now: datetime) -> list[Observation]:
        """Check daemon heartbeat freshness."""
        import json

        state_file = DATA_DIR / "daemon_state.json"
        try:
            with open(state_file) as f:
                state = json.load(f)

            last_run = state.get("last_scout_run", "")
            if not last_run:
                return []

            last_dt = datetime.fromisoformat(last_run)
            age_minutes = (now - last_dt).total_seconds() / 60

            if age_minutes > 30:
                return [Observation(
                    scout=self.name, timestamp=now,
                    observation=f"Daemon hasn't run scouts in {age_minutes:.0f} minutes. May be stuck.",
                    relevance=Relevance.HIGH, escalate=True,
                    context_tags=["system", "daemon", "stale"],
                )]
            elif age_minutes > 15:
                return [Observation(
                    scout=self.name, timestamp=now,
                    observation=f"Daemon last ran {age_minutes:.0f} minutes ago (usually every 1-2 min).",
                    relevance=Relevance.MEDIUM, escalate=False,
                    context_tags=["system", "daemon"],
                )]

        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass
        return []

    async def _check_mcp(self, now: datetime) -> list[Observation]:
        """Check MCP server connectivity via the server's health endpoint.

        The daemon runs in a separate process from the server, so we can't
        use get_mcp_manager() directly — that would create a disconnected
        manager with no connections. Instead, we call the server's
        /mcp/health endpoint which checks the actual MCP connections.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get("http://localhost:8000/mcp/health") as resp:
                    if resp.status != 200:
                        return [Observation(
                            scout=self.name, timestamp=now,
                            observation=f"MCP health endpoint returned HTTP {resp.status}",
                            relevance=Relevance.MEDIUM, escalate=False,
                            context_tags=["system", "mcp", "meta-error"],
                        )]
                    statuses = await resp.json()

            if not statuses:
                return []

            observations = []

            # Report reconnections as low-priority info
            reconnected = [n for n, s in statuses.items() if s == "reconnected"]
            if reconnected:
                observations.append(Observation(
                    scout=self.name, timestamp=now,
                    observation=f"MCP servers auto-reconnected: {', '.join(reconnected)}",
                    relevance=Relevance.LOW, escalate=False,
                    context_tags=["system", "mcp", "reconnected"],
                ))

            # Dead or missing servers are the real problem
            failed = [n for n, s in statuses.items() if s in ("dead", "missing")]
            if len(failed) > len(statuses) // 2:
                failed_set = frozenset(failed)
                should_escalate = False

                if self._last_mcp_escalation_time is None:
                    should_escalate = True
                else:
                    hours_since = (now - self._last_mcp_escalation_time).total_seconds() / 3600
                    if hours_since >= self._mcp_cooldown_hours:
                        should_escalate = True
                    elif failed_set != self._last_mcp_failed_set:
                        should_escalate = True

                if should_escalate:
                    self._last_mcp_escalation_time = now
                    self._last_mcp_failed_set = failed_set

                observations.append(Observation(
                    scout=self.name, timestamp=now,
                    observation=f"Most MCP servers disconnected: {', '.join(failed)}",
                    relevance=Relevance.HIGH if should_escalate else Relevance.MEDIUM,
                    escalate=should_escalate,
                    context_tags=["system", "mcp", "disconnected"],
                    raw_data=statuses,
                ))
            elif failed:
                observations.append(Observation(
                    scout=self.name, timestamp=now,
                    observation=f"MCP servers unreachable: {', '.join(failed)}",
                    relevance=Relevance.MEDIUM, escalate=False,
                    context_tags=["system", "mcp"],
                    raw_data=statuses,
                ))

            return observations

        except aiohttp.ClientError as e:
            # Server not reachable — could be restarting, don't spam
            return [Observation(
                scout=self.name, timestamp=now,
                observation=f"Cannot reach Doris server for MCP health check: {type(e).__name__}",
                relevance=Relevance.LOW, escalate=False,
                context_tags=["system", "mcp", "meta-error"],
            )]
        except Exception as e:
            return [Observation(
                scout=self.name, timestamp=now,
                observation=f"MCP health check failed: {e}",
                relevance=Relevance.LOW, escalate=False,
                context_tags=["system", "mcp", "meta-error"],
            )]

    def _maybe_rotate_logs(self):
        """Trigger log rotation if any files are oversized."""
        try:
            from services.log_rotation import rotate_logs
            actions = rotate_logs()
            for action in actions:
                print(f"[{self.name}] {action}")
        except Exception as e:
            print(f"[{self.name}] Log rotation error: {e}")
