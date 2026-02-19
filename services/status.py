"""
Consolidated Status Endpoint

Single /status view composing data from all Doris subsystems.
Moved from dashboard/status.py during dashboard removal (security hardening #48).
"""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# Server start time — set by main.py lifespan on startup
server_start_time: float = time.time()

DATA_DIR = Path(__file__).parent.parent / "data"
LOGS_DIR = Path(__file__).parent.parent / "logs"
SCOUT_HEALTH_FILE = DATA_DIR / "scout_health.json"


def _relative_time(iso_str: str) -> str:
    """Convert ISO timestamp to relative time string like '2m ago'."""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
        now = datetime.now(ZoneInfo("America/New_York"))
        delta = now - dt
        seconds = int(delta.total_seconds())

        if seconds < 0:
            return "just now"
        if seconds < 60:
            return f"{seconds}s ago"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        return f"{seconds // 86400}d ago"
    except Exception:
        return "unknown"


def _format_uptime(seconds: float) -> str:
    """Format seconds into a human-readable uptime string."""
    hours = seconds / 3600
    if hours < 1:
        return f"{int(seconds / 60)}m"
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"


def check_ollama() -> dict:
    """Check Ollama server status."""
    try:
        import httpx
    except ImportError:
        import urllib.request
        import urllib.error

        start = time.time()
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                latency_ms = int((time.time() - start) * 1000)
                if response.status == 200:
                    data = json.loads(response.read())
                    models = [m["name"] for m in data.get("models", [])]
                    return {
                        "status": "online",
                        "latency_ms": latency_ms,
                        "details": f"{len(models)} models loaded",
                    }
        except urllib.error.URLError:
            return {"status": "offline", "latency_ms": None, "details": "Connection refused"}
        except Exception as e:
            return {"status": "error", "latency_ms": None, "details": str(e)}

    start = time.time()
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get("http://localhost:11434/api/tags")
            latency_ms = int((time.time() - start) * 1000)

            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return {
                    "status": "online",
                    "latency_ms": latency_ms,
                    "details": f"{len(models)} models loaded",
                }
            else:
                return {
                    "status": "error",
                    "latency_ms": latency_ms,
                    "details": f"HTTP {response.status_code}",
                }
    except Exception:
        return {"status": "offline", "latency_ms": None, "details": "Connection refused"}


def check_llm_provider() -> dict:
    """Check LLM provider configuration (no API call, just config check)."""
    try:
        from config import settings
        provider = getattr(settings, "llm_provider", "claude")
    except Exception:
        provider = os.environ.get("LLM_PROVIDER", "claude")

    if provider == "claude":
        try:
            from config import settings
            api_key = settings.anthropic_api_key
        except Exception:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if not api_key:
            return {"status": "offline", "details": "API key not configured"}
        return {"status": "online", "details": "configured"}

    elif provider == "openai":
        try:
            from config import settings
            api_key = settings.openai_api_key
        except Exception:
            api_key = os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            return {"status": "offline", "details": "API key not configured"}
        return {"status": "online", "details": "configured"}

    elif provider == "ollama":
        ollama_status = check_ollama()
        return {"status": ollama_status["status"], "details": ollama_status["details"]}

    else:
        return {"status": "error", "details": f"Unknown provider: {provider}"}


def _get_daemon_status() -> dict:
    """Read daemon state from data/daemon_state.json."""
    state_file = DATA_DIR / "daemon_state.json"
    try:
        with open(state_file) as f:
            state = json.load(f)

        last_run = state.get("last_scout_run", "")
        return {
            "status": "up" if state.get("status") == "running" else "down",
            "last_scout_run": _relative_time(last_run) if last_run else "never",
        }
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "unknown", "last_scout_run": "unknown"}


def _get_memory_stats() -> dict:
    """Get memory system stats from SQLite."""
    try:
        from memory.store import get_db
        db = get_db()

        row = db.execute("SELECT COUNT(*) as cnt FROM memories WHERE superseded_by IS NULL").fetchone()
        active = row["cnt"] if row else 0

        row = db.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()
        entities = row["cnt"] if row else 0

        db.close()

        return {"status": "healthy", "active_memories": active, "entities": entities}
    except Exception as e:
        return {"status": "error", "error": str(e)[:80]}


def _get_mcp_status() -> dict:
    """Get MCP server connection status."""
    try:
        from mcp_client import get_mcp_manager
        from mcp_client.config import get_enabled_servers

        manager = get_mcp_manager()
        enabled = get_enabled_servers()
        connected = list(manager.connected_servers)
        failing = [s for s in enabled if s not in connected]

        return {
            "connected": len(connected),
            "total": len(enabled),
            "failing": failing if failing else [],
        }
    except Exception:
        return {"connected": 0, "total": 0, "failing": []}


def _get_tool_health() -> dict:
    """Get circuit breaker status for all tools."""
    try:
        from tools.circuit_breaker import get_circuit_breaker

        cb = get_circuit_breaker()
        status = cb.get_status()

        healthy = 0
        open_circuits = []

        for name, info in status.items():
            if info["state"] == "open":
                open_circuits.append({
                    "name": name,
                    "state": "open",
                    "last_error": info.get("last_error", ""),
                })
            else:
                healthy += 1

        return {
            "healthy": healthy,
            "circuit_open": len(open_circuits),
            "details": open_circuits if open_circuits else [],
        }
    except Exception:
        return {"healthy": 0, "circuit_open": 0, "details": []}


def _get_alerts() -> list[dict]:
    """Detect current alerts/warnings."""
    alerts = []

    # Check log file sizes
    for log_name in ["server.error.log", "server.log", "daemon.error.log"]:
        log_path = LOGS_DIR / log_name
        if log_path.exists():
            size_mb = log_path.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                alerts.append({
                    "level": "warn",
                    "message": f"{log_name} is {size_mb:.0f}MB — consider rotation",
                    "since": datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y-%m-%d"),
                })

    # Check disk usage (stdlib shutil, no psutil needed)
    try:
        disk = shutil.disk_usage("/")
        pct = round(disk.used / disk.total * 100, 1)
        if pct > 90:
            free_gb = round(disk.free / (1024**3), 1)
            level = "critical" if pct > 95 else "warn"
            alerts.append({"level": level, "message": f"Disk usage at {pct}% ({free_gb}GB free)"})
    except Exception:
        pass

    return alerts


def _get_scout_health(threshold: int = 3) -> dict:
    """Read scout health from data/scout_health.json."""
    try:
        if not SCOUT_HEALTH_FILE.exists():
            return {"healthy": 0, "failing": 0, "details": []}

        data = json.load(open(SCOUT_HEALTH_FILE))
        scouts = {k: v for k, v in data.items() if k != "_meta"}

        healthy = 0
        failing_details = []

        for name, info in scouts.items():
            consecutive = info.get("consecutive_failures", 0)
            if consecutive >= threshold:
                detail = {
                    "name": name,
                    "consecutive_failures": consecutive,
                    "last_error": info.get("last_error", ""),
                }
                last_success = info.get("last_success")
                detail["last_success"] = _relative_time(last_success) if last_success else "never"
                failing_details.append(detail)
            else:
                healthy += 1

        return {"healthy": healthy, "failing": len(failing_details), "details": failing_details}
    except Exception:
        return {"healthy": 0, "failing": 0, "details": []}


def _get_push_status() -> dict:
    """Get push notification / device token health."""
    try:
        from services.push import get_token_status
        return get_token_status()
    except Exception as e:
        return {"health": "unknown", "active_tokens": 0, "error": str(e)[:80]}


def get_consolidated_status() -> dict:
    """Build the consolidated status response.

    Composes data from all Doris subsystems into a single view.
    """
    now = time.time()
    uptime_seconds = now - server_start_time

    # Gather subsystem data
    ollama = check_ollama()
    claude = check_llm_provider()
    daemon = _get_daemon_status()
    memory = _get_memory_stats()
    mcp = _get_mcp_status()
    tools = _get_tool_health()
    scouts = _get_scout_health()
    push = _get_push_status()
    alerts = _get_alerts()

    # Determine overall status
    overall = "healthy"
    if tools["circuit_open"] > 0 or mcp.get("failing"):
        overall = "degraded"
    if ollama["status"] != "online" or daemon["status"] == "down":
        overall = "degraded"
    if scouts["failing"] > 0:
        overall = "degraded"
    if push["health"] == "critical":
        overall = "degraded"
        alerts.append({
            "level": "warn",
            "message": "No active device tokens — push notifications will fail",
        })
    if memory.get("status") == "error":
        overall = "critical"
    if any(a.get("level") == "critical" for a in alerts):
        overall = "critical"

    return {
        "overall": overall,
        "uptime": _format_uptime(uptime_seconds),
        "uptime_seconds": round(uptime_seconds),
        "subsystems": {
            "server": {
                "status": "up",
                "uptime": _format_uptime(uptime_seconds),
            },
            "daemon": daemon,
            "memory": memory,
            "ollama": {
                "status": ollama["status"],
                "latency_ms": ollama.get("latency_ms"),
                "details": ollama.get("details", ""),
            },
            "claude_api": {
                "status": claude["status"],
                "details": claude.get("details", ""),
            },
            "mcp_servers": mcp,
            "tools": tools,
            "push": push,
        },
        "scouts": scouts,
        "alerts": alerts,
    }
