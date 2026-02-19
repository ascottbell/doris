"""
Service control tool for Doris.

Manages launchd services (server, daemon) using launchctl.
This allows Doris to restart her own services when they're unresponsive.
"""

import subprocess
import time
from typing import Optional


# Service label mappings
SERVICE_LABELS = {
    "server": "com.doris.server",
    "daemon": "com.doris.daemon",
}

# Health check endpoints
HEALTH_CHECKS = {
    "server": ("http://localhost:8000/health", 5),
    # daemon doesn't have an HTTP endpoint
}


def _run_launchctl(command: str, label: str, timeout: int = 10) -> tuple[bool, str]:
    """
    Run a launchctl command.

    Args:
        command: start, stop, or kickstart
        label: Service label (e.g., com.doris.server)
        timeout: Command timeout in seconds

    Returns:
        (success, message) tuple
    """
    try:
        if command == "kickstart":
            # kickstart -k forces restart even if running
            cmd = ["launchctl", "kickstart", "-k", f"gui/{_get_uid()}/{label}"]
        else:
            cmd = ["launchctl", command, label]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            return True, f"Command succeeded: {' '.join(cmd)}"
        else:
            # launchctl often returns non-zero for benign reasons
            error = result.stderr.strip() or result.stdout.strip()
            if "No such process" in error and command == "stop":
                return True, "Service was not running"
            return False, f"Command failed: {error}"

    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s"
    except Exception as e:
        return False, f"Error: {str(e)}"


def _get_uid() -> int:
    """Get current user's UID for launchctl commands."""
    import os
    return os.getuid()


def _get_service_status(label: str) -> dict:
    """
    Get status of a launchd service.

    Returns:
        dict with: running (bool), pid (int or None), exit_code (int or None)
    """
    try:
        result = subprocess.run(
            ["launchctl", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return {"running": False, "pid": None, "exit_code": None, "error": "launchctl list failed"}

        # Parse output: PID\tStatus\tLabel
        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2] == label:
                pid_str = parts[0].strip()
                status_str = parts[1].strip()

                pid = int(pid_str) if pid_str != "-" else None
                exit_code = int(status_str) if status_str != "-" else None

                # Running if we have a PID
                running = pid is not None

                return {
                    "running": running,
                    "pid": pid,
                    "exit_code": exit_code
                }

        return {"running": False, "pid": None, "exit_code": None, "error": "Service not found"}

    except Exception as e:
        return {"running": False, "pid": None, "exit_code": None, "error": str(e)}


def _check_health(service: str, timeout: int = 5) -> tuple[bool, str]:
    """
    Check if a service is healthy (responds to health check).

    Args:
        service: Service name (server, daemon)
        timeout: HTTP timeout

    Returns:
        (healthy, message) tuple
    """
    if service not in HEALTH_CHECKS:
        # No health check available, just check if running
        label = SERVICE_LABELS.get(service)
        if not label:
            return False, "Unknown service"
        status = _get_service_status(label)
        if status.get("running"):
            return True, f"Running (PID {status['pid']})"
        return False, "Not running"

    url, check_timeout = HEALTH_CHECKS[service]

    try:
        import urllib.request
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=check_timeout) as response:
            if response.status == 200:
                return True, "Healthy"
            return False, f"Unhealthy (HTTP {response.status})"
    except Exception as e:
        return False, f"Health check failed: {str(e)}"


def control_service(service: str, action: str) -> dict:
    """
    Control a Doris service.

    Args:
        service: server or daemon
        action: restart, stop, start, or status

    Returns:
        dict with success, message, and optional details
    """
    # Validate service
    label = SERVICE_LABELS.get(service)
    if not label:
        return {
            "success": False,
            "error": f"Unknown service: {service}. Valid services: {', '.join(SERVICE_LABELS.keys())}"
        }

    # Handle status action
    if action == "status":
        status = _get_service_status(label)
        healthy, health_msg = _check_health(service)

        return {
            "success": True,
            "message": f"Service {service}: {'running' if status.get('running') else 'stopped'}",
            "details": {
                "running": status.get("running", False),
                "pid": status.get("pid"),
                "exit_code": status.get("exit_code"),
                "health": health_msg
            }
        }

    # Handle stop action
    if action == "stop":
        success, msg = _run_launchctl("stop", label)
        return {
            "success": success,
            "message": f"Stopped {service}" if success else f"Failed to stop {service}: {msg}"
        }

    # Handle start action
    if action == "start":
        success, msg = _run_launchctl("start", label)
        if success:
            # Wait a moment and check if it's running
            time.sleep(2)
            status = _get_service_status(label)
            if status.get("running"):
                return {
                    "success": True,
                    "message": f"Started {service} (PID {status['pid']})"
                }
            else:
                return {
                    "success": False,
                    "error": f"Service started but exited immediately (exit code: {status.get('exit_code')})"
                }
        return {
            "success": False,
            "error": f"Failed to start {service}: {msg}"
        }

    # Handle restart action
    if action == "restart":
        # Use kickstart -k for atomic restart
        success, msg = _run_launchctl("kickstart", label)

        if not success:
            # Fallback: stop then start
            _run_launchctl("stop", label)
            time.sleep(1)
            success, msg = _run_launchctl("start", label)

        if success:
            # Wait and verify
            time.sleep(3)
            healthy, health_msg = _check_health(service)
            if healthy:
                return {
                    "success": True,
                    "message": f"Restarted {service} successfully. {health_msg}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Service restarted but health check failed: {health_msg}"
                }

        return {
            "success": False,
            "error": f"Failed to restart {service}: {msg}"
        }

    return {
        "success": False,
        "error": f"Unknown action: {action}. Valid actions: restart, stop, start, status"
    }


# For testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m tools.service_control <service> <action>")
        print("Services: server, daemon")
        print("Actions: status, start, stop, restart")
        sys.exit(1)

    service = sys.argv[1]
    action = sys.argv[2]

    result = control_service(service, action)
    print(f"Success: {result.get('success')}")
    print(f"Message: {result.get('message', result.get('error'))}")
    if result.get('details'):
        print(f"Details: {result['details']}")
