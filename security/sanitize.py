"""
Input sanitization for subprocess and AppleScript arguments.

Provides defense-in-depth for any user-controlled strings that flow into
subprocess calls (Swift CLIs for Calendar, Reminders) or AppleScript commands
(iMessage). Even when subprocess uses shell=False, the downstream executables
may construct AppleScript or other interpreted commands internally.

Key functions:
- escape_applescript_string: JSON-based escaping for AppleScript string literals
- sanitize_subprocess_arg: Defense-in-depth for CLI arguments
- validate_id: Format validation for event/reminder IDs
"""

import json
import re


# Maximum length for any single CLI argument (defense against abuse)
MAX_ARG_LENGTH = 10_000

# Maximum length for ID fields (UUIDs are 36 chars, leave room for other formats)
MAX_ID_LENGTH = 128

# Pattern for valid IDs: alphanumeric, hyphens, underscores, forward slashes, colons
# Covers UUIDs, Apple EventKit IDs (e.g., "E7A2B3C4-..."), and similar formats
_VALID_ID_PATTERN = re.compile(r'^[a-zA-Z0-9\-_/:\.]+$')

# Null bytes and other control characters that should never appear in CLI args
_DANGEROUS_CHARS = re.compile(r'[\x00-\x08\x0e-\x1f\x7f]')


def escape_applescript_string(s: str) -> str:
    """
    Escape a string for safe use in AppleScript string literals.

    Uses JSON encoding which handles all special characters properly:
    - Backslashes -> \\\\
    - Double quotes -> \\"
    - Newlines, tabs, carriage returns -> \\n, \\t, \\r
    - Unicode characters -> \\uXXXX

    The JSON encoding produces escape sequences that AppleScript accepts.

    Args:
        s: Raw string to escape

    Returns:
        Escaped string safe for AppleScript interpolation (without outer quotes)
    """
    # json.dumps produces: "content with \"escapes\""
    # [1:-1] strips the outer quotes to get: content with \"escapes\"
    return json.dumps(s)[1:-1]


def sanitize_subprocess_arg(value: str, field_name: str = "argument",
                            max_length: int = MAX_ARG_LENGTH,
                            context: str = "applescript") -> str:
    """
    Sanitize a string for use as a subprocess argument.

    Defense-in-depth: even with shell=False, downstream executables (Swift CLIs)
    may internally construct AppleScript or other interpreted commands.

    Sanitization steps:
    1. Strip null bytes and dangerous control characters
    2. Enforce maximum length
    3. Apply context-appropriate escaping

    Args:
        value: Raw user-controlled string
        field_name: Name of the field (for error messages)
        max_length: Maximum allowed length
        context: Escaping context — determines which escaping rules apply:
            - "applescript": JSON-style escaping for AppleScript string literals
              (quotes, backslashes, newlines). Use for Swift CLIs that may
              construct AppleScript internally (cal, reminders).
            - "shell": Strip control chars and enforce length only. No
              additional escaping — caller MUST use shell=False.
            - "raw": Strip control chars and enforce length only. For
              contexts where the value is never interpolated into a script.

    Returns:
        Sanitized string safe for subprocess argument use

    Raises:
        ValueError: If the value exceeds max_length after sanitization
    """
    if not value:
        return value

    if context not in ("applescript", "shell", "raw"):
        raise ValueError(f"Unknown sanitize context: {context!r}")

    # Strip dangerous control characters (null bytes, etc.)
    cleaned = _DANGEROUS_CHARS.sub('', value)

    # Enforce length limit
    if len(cleaned) > max_length:
        raise ValueError(
            f"{field_name} exceeds maximum length of {max_length} characters"
        )

    if context == "applescript":
        # Apply AppleScript escaping as defense-in-depth
        # This ensures that even if the downstream CLI constructs AppleScript,
        # the string won't break out of a quoted context
        return escape_applescript_string(cleaned)

    # "shell" and "raw": control chars stripped, length enforced — no
    # additional escaping needed (caller uses shell=False for "shell").
    return cleaned


def validate_id(value: str, field_name: str = "id") -> str:
    """
    Validate and return an ID string for use in subprocess arguments.

    IDs should be alphanumeric with limited special characters.
    This prevents injection through ID fields that might be interpolated
    into queries or scripts downstream.

    Args:
        value: The ID string to validate
        field_name: Name of the field (for error messages)

    Returns:
        The validated ID string (unchanged if valid)

    Raises:
        ValueError: If the ID is empty, too long, or contains invalid characters
    """
    if not value or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")

    value = value.strip()

    if len(value) > MAX_ID_LENGTH:
        raise ValueError(
            f"{field_name} exceeds maximum length of {MAX_ID_LENGTH} characters"
        )

    if not _VALID_ID_PATTERN.match(value):
        raise ValueError(
            f"{field_name} contains invalid characters. "
            f"Only alphanumeric, hyphens, underscores, slashes, colons, and dots are allowed."
        )

    return value
