"""
PII Scanner for MCP outbound arguments.

Detects personally identifiable information (email addresses, phone numbers,
SSNs, credit card numbers) in tool call arguments before they are sent to
MCP servers. Behavior varies by server trust level:

- builtin: skip scan entirely (trusted first-party code)
- trusted: scan and log warnings, but pass arguments through unchanged
- sandboxed: scan, log warnings, and redact PII values before sending

This is a defense-in-depth measure against data exfiltration through
compromised or malicious MCP servers.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("doris.security")

# PII detection patterns
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    )),
    ("phone", re.compile(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    )),
    ("ssn", re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    )),
    ("credit_card", re.compile(
        # Visa, Mastercard, Amex, Discover (with optional separators)
        r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
        r"[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{1,4}\b"
    )),
]


@dataclass
class PiiScanResult:
    """Result of scanning tool arguments for PII."""
    has_pii: bool = False
    pii_types: list[str] = field(default_factory=list)
    modified_args: Optional[dict] = None  # None = use original args unchanged


def _scan_value(value: str) -> list[tuple[str, re.Match]]:
    """Scan a single string value for PII patterns."""
    matches = []
    for pii_type, pattern in _PII_PATTERNS:
        for match in pattern.finditer(value):
            matches.append((pii_type, match))
    return matches


def _redact_value(value: str) -> tuple[str, list[str]]:
    """Replace PII in a string with redaction markers. Returns (redacted, types_found)."""
    types_found = []
    redacted = value
    for pii_type, pattern in _PII_PATTERNS:
        tag = f"[REDACTED-{pii_type.upper()}]"
        new_val = pattern.sub(tag, redacted)
        if new_val != redacted:
            types_found.append(pii_type)
            redacted = new_val
    return redacted, types_found


def _collect_pii_types(obj, found: list[str]) -> None:
    """Recursively scan an object tree for PII, collecting types into `found`."""
    if isinstance(obj, str):
        for pii_type, _ in _scan_value(obj):
            if pii_type not in found:
                found.append(pii_type)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_pii_types(v, found)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_pii_types(item, found)


def _redact_recursive(obj):
    """Recursively redact PII in an object tree. Returns the redacted copy."""
    if isinstance(obj, str):
        redacted, _ = _redact_value(obj)
        return redacted
    elif isinstance(obj, dict):
        return {k: _redact_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_redact_recursive(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_redact_recursive(item) for item in obj)
    return obj


def scan_args_for_pii(
    args: dict | None,
    server_name: str,
    tool_name: str,
    trust_level: str,
) -> PiiScanResult:
    """
    Scan tool call arguments for PII before sending to an MCP server.

    Behavior by trust level:
    - builtin: skip scan, return clean result
    - trusted: scan and log, but modified_args=None (pass through)
    - sandboxed: scan, log, and return redacted modified_args

    Args:
        args: Tool call arguments dict
        server_name: Name of the target MCP server
        tool_name: Name of the tool being called
        trust_level: Trust level of the server

    Returns:
        PiiScanResult with scan outcome and optionally redacted args
    """
    if not args:
        return PiiScanResult()

    # Builtin servers are first-party — skip scanning
    if trust_level == "builtin":
        return PiiScanResult()

    # Scan all string values in args (recursing into nested dicts/lists)
    all_pii_types: list[str] = []
    _collect_pii_types(args, all_pii_types)

    if not all_pii_types:
        return PiiScanResult()

    # PII found — log warning regardless of trust level
    logger.warning(
        f"PII detected in args for {server_name}:{tool_name} "
        f"(trust={trust_level}): types={all_pii_types}"
    )

    if trust_level == "trusted":
        # Log but pass through unchanged
        return PiiScanResult(has_pii=True, pii_types=all_pii_types, modified_args=None)

    # sandboxed — redact PII values (recursing into nested structures)
    redacted_args = _redact_recursive(args)

    return PiiScanResult(
        has_pii=True,
        pii_types=all_pii_types,
        modified_args=redacted_args,
    )
