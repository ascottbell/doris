"""Structured audit logging for security-relevant events.

Writes JSON-formatted log entries to a dedicated audit log file with
automatic rotation. Covers authentication, rate limiting, and external
data ingestion events that need a durable trail.

Usage:
    from security.audit import audit

    audit.auth_failure(ip="1.2.3.4", reason="invalid_token", endpoint="/chat/text")
    audit.rate_limit(ip="1.2.3.4", path="/chat/text", limit=20)
    audit.auth_success(ip="1.2.3.4", endpoint="/chat/text")
"""

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        # Merge any extra fields passed via `extra={"audit": {...}}`
        audit_data = getattr(record, "audit", None)
        if audit_data and isinstance(audit_data, dict):
            entry.update(audit_data)
        return json.dumps(entry, default=str)


def _setup_audit_logger() -> logging.Logger:
    """Create the audit logger with file + console handlers."""
    logger = logging.getLogger("doris.audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't double-log to root

    if logger.handlers:
        return logger  # Already configured (module re-import)

    # File handler — logs/audit.log with rotation
    log_dir = os.getenv("DORIS_LOG_DIR", str(Path(__file__).parent.parent / "logs"))
    os.makedirs(log_dir, exist_ok=True)
    audit_path = os.path.join(log_dir, "audit.log")

    file_handler = RotatingFileHandler(
        audit_path, maxBytes=10 * 1024 * 1024, backupCount=5  # 10 MB, keep 5 old
    )
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    # Also log to stderr so it shows up in docker logs / journalctl
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_JsonFormatter())
    logger.addHandler(console_handler)

    return logger


_logger = _setup_audit_logger()


class AuditLog:
    """Convenience methods for common audit events."""

    @staticmethod
    def auth_failure(*, ip: str, reason: str, endpoint: str = "", component: str = "api") -> None:
        """Log a failed authentication attempt."""
        _logger.warning("auth_failure", extra={"audit": {
            "action": "auth_failure",
            "component": component,
            "ip": ip,
            "reason": reason,
            "endpoint": endpoint,
        }})

    @staticmethod
    def auth_success(*, ip: str, endpoint: str = "", component: str = "api") -> None:
        """Log a successful authentication (debug-level, off by default)."""
        _logger.debug("auth_success", extra={"audit": {
            "action": "auth_success",
            "component": component,
            "ip": ip,
            "endpoint": endpoint,
        }})

    @staticmethod
    def rate_limit(*, ip: str, path: str, limit: int) -> None:
        """Log a rate limit rejection."""
        _logger.warning("rate_limit", extra={"audit": {
            "action": "rate_limit",
            "component": "rate_limiter",
            "ip": ip,
            "path": path,
            "limit": limit,
        }})

    @staticmethod
    def startup(*, component: str, detail: str) -> None:
        """Log a security-relevant startup event."""
        _logger.info("startup", extra={"audit": {
            "action": "startup",
            "component": component,
            "detail": detail,
        }})

    @staticmethod
    def tool_action(*, tool: str, detail: str, blocked: bool = False, **kwargs) -> None:
        """Log a security-relevant tool action (send_email, send_imessage, etc.)."""
        level = logging.WARNING if blocked else logging.INFO
        _logger.log(level, "tool_action", extra={"audit": {
            "action": "tool_action",
            "tool": tool,
            "detail": detail,
            "blocked": blocked,
            **kwargs,
        }})


audit = AuditLog()
