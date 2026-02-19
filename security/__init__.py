"""
Security module for Doris.

Contains authentication, prompt safety, injection scanning, and input sanitization.
"""

from security.auth import verify_token, get_token_dependency
from security.prompt_safety import (
    wrap_untrusted,
    escape_for_prompt,
    wrap_email_content,
    wrap_calendar_content,
    wrap_reminder_content,
    wrap_weather_content,
    wrap_location_content,
    wrap_health_content,
    wrap_imessage_content,
    wrap_mcp_response,
    wrap_memory_content,
    wrap_with_scan,
)
from security.sanitize import escape_applescript_string, sanitize_subprocess_arg, validate_id
from security.injection_scanner import scan_for_injection, is_suspicious, ScanResult
from security.pii_scanner import scan_args_for_pii, PiiScanResult
from security.audit import audit, AuditLog
from security.crypto import token_matches_any, get_fernet

__all__ = [
    # Auth
    "verify_token",
    "get_token_dependency",
    # Prompt safety — wrapping
    "wrap_untrusted",
    "escape_for_prompt",
    "wrap_email_content",
    "wrap_calendar_content",
    "wrap_reminder_content",
    "wrap_weather_content",
    "wrap_location_content",
    "wrap_health_content",
    "wrap_imessage_content",
    "wrap_mcp_response",
    "wrap_memory_content",
    "wrap_with_scan",
    # Injection scanning
    "scan_for_injection",
    "is_suspicious",
    "ScanResult",
    # Input sanitization
    "escape_applescript_string",
    "sanitize_subprocess_arg",
    "validate_id",
    # PII scanning
    "scan_args_for_pii",
    "PiiScanResult",
    # Audit logging
    "audit",
    "AuditLog",
    # Crypto utilities (key rotation)
    "token_matches_any",
    "get_fernet",
]
