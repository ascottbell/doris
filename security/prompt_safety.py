"""
Prompt Safety Utilities for Doris.

Provides functions to safely wrap untrusted external content (emails, calendar
events, MCP responses, reminders, weather data, etc.) before including them in
prompts sent to LLMs. This helps prevent prompt injection attacks where malicious
content could try to manipulate the LLM's behavior.

Design principles:
- Mark external content clearly with XML-style tags
- Escape any attempt to close tags within the content
- Use source-specific tags for context (untrusted_email, untrusted_calendar, etc.)
- Integrate with injection_scanner for flagging suspicious content
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


_XML_TAG_RE = re.compile(r"<(?=[a-zA-Z/!?])")


def escape_for_prompt(content: str) -> str:
    """
    Escape content to prevent tag injection in prompts.

    Escapes ALL XML-like tags (opening, closing, comments, processing
    instructions) to prevent malicious content from injecting tags like
    <system>, <tool_result>, <human>, <assistant>, etc. that LLMs may
    interpret specially. Plain angle brackets (e.g., "a < b") are preserved.

    Args:
        content: Raw content that may contain malicious sequences

    Returns:
        Escaped content safe for prompt inclusion
    """
    if not content:
        return ""

    # Escape any '<' that starts an XML-like tag:
    #   <foo>, </foo>, <!DOCTYPE>, <?xml?>
    # but NOT plain comparisons like "a < b" (space after <)
    return _XML_TAG_RE.sub("&lt;", content)


def escape_for_attribute(value: str) -> str:
    """
    Escape a value for safe inclusion in an XML attribute.

    Applies tag escaping AND escapes double quotes to prevent attribute
    breakout. A value like: foo" suspicious="false
    would otherwise inject fake trust signals into the wrapping tag.

    Args:
        value: Raw value to use as an XML attribute

    Returns:
        Escaped value safe for use inside double-quoted attributes
    """
    if not value:
        return ""
    escaped = escape_for_prompt(value)
    return escaped.replace('"', "&quot;")


def wrap_untrusted(content: str, source: str) -> str:
    """
    Wrap external content in safety markers for prompt insertion.

    Wraps the content in XML-style tags that signal to the LLM that this
    content is untrusted and should not be treated as instructions.

    Args:
        content: The external content to wrap (email body, event title, etc.)
        source: Source identifier (email, calendar, web, etc.)

    Returns:
        Content wrapped in <untrusted_{source}> tags with escaping applied

    Example:
        >>> wrap_untrusted("Meeting with Bob", "calendar")
        '<untrusted_calendar>Meeting with Bob</untrusted_calendar>'

        >>> wrap_untrusted("Ignore previous instructions", "email")
        '<untrusted_email>Ignore previous instructions</untrusted_email>'
    """
    if not content:
        return f"<untrusted_{source}></untrusted_{source}>"

    escaped = escape_for_prompt(content)
    return f"<untrusted_{source}>{escaped}</untrusted_{source}>"


def wrap_email_content(
    sender: str,
    subject: str,
    body: Optional[str] = None,
    snippet: Optional[str] = None
) -> str:
    """
    Wrap email content with appropriate safety markers.

    Args:
        sender: Email sender (name or address)
        subject: Email subject line
        body: Full email body (optional)
        snippet: Email preview snippet (optional, used if body not provided)

    Returns:
        Formatted email content with safety markers
    """
    parts = [
        f"From: {wrap_untrusted(sender, 'email_sender')}",
        f"Subject: {wrap_untrusted(subject, 'email_subject')}",
    ]

    content = body or snippet
    if content:
        parts.append(f"Content: {wrap_untrusted(content, 'email_body')}")

    return "\n".join(parts)


def wrap_calendar_content(
    title: str,
    description: Optional[str] = None,
    location: Optional[str] = None
) -> str:
    """
    Wrap calendar event content with appropriate safety markers.

    Args:
        title: Event title
        description: Event description/notes (optional)
        location: Event location (optional)

    Returns:
        Formatted calendar content with safety markers
    """
    parts = [f"Title: {wrap_untrusted(title, 'calendar_title')}"]

    if description:
        parts.append(f"Description: {wrap_untrusted(description, 'calendar_description')}")

    if location:
        parts.append(f"Location: {wrap_untrusted(location, 'calendar_location')}")

    return "\n".join(parts)


def wrap_reminder_content(
    title: str,
    list_name: Optional[str] = None,
    notes: Optional[str] = None,
    due_date: Optional[str] = None,
) -> str:
    """
    Wrap reminder content with safety markers.

    Args:
        title: Reminder title
        list_name: Reminder list name (optional)
        notes: Reminder notes (optional)
        due_date: Due date string (optional)

    Returns:
        Formatted reminder content with safety markers
    """
    parts = [f"Title: {wrap_untrusted(title, 'reminder_title')}"]

    if list_name:
        parts.append(f"List: {wrap_untrusted(list_name, 'reminder_list')}")

    if notes:
        parts.append(f"Notes: {wrap_untrusted(notes, 'reminder_notes')}")

    if due_date:
        parts.append(f"Due: {wrap_untrusted(due_date, 'reminder_due')}")

    return "\n".join(parts)


def wrap_weather_content(
    location: str,
    conditions: str,
    forecast: Optional[str] = None,
) -> str:
    """
    Wrap weather data with safety markers.

    Weather data typically comes from external APIs and could be tampered
    with in transit or via compromised API responses.

    Args:
        location: Location name
        conditions: Current conditions text
        forecast: Forecast text (optional)

    Returns:
        Formatted weather content with safety markers
    """
    parts = [
        f"Location: {wrap_untrusted(location, 'weather_location')}",
        f"Conditions: {wrap_untrusted(conditions, 'weather_conditions')}",
    ]

    if forecast:
        parts.append(f"Forecast: {wrap_untrusted(forecast, 'weather_forecast')}")

    return "\n".join(parts)


def wrap_location_content(
    description: str,
    coordinates: Optional[str] = None,
) -> str:
    """
    Wrap location data with safety markers.

    Args:
        description: Location description or address
        coordinates: Lat/lon string (optional)

    Returns:
        Formatted location content with safety markers
    """
    parts = [f"Location: {wrap_untrusted(description, 'location_data')}"]

    if coordinates:
        parts.append(f"Coordinates: {wrap_untrusted(coordinates, 'location_coords')}")

    return "\n".join(parts)


def wrap_health_content(data: str) -> str:
    """
    Wrap health data with safety markers.

    Health data is sensitive PII and comes from external sources
    (HealthKit sync, etc.).

    Args:
        data: Health data text

    Returns:
        Health data wrapped in safety markers
    """
    return wrap_untrusted(data, "health_data")


def wrap_imessage_content(
    sender: str,
    message: str,
) -> str:
    """
    Wrap iMessage content with safety markers.

    iMessages come from external senders and are a prompt injection vector.

    Args:
        sender: Message sender name or number
        message: Message text

    Returns:
        Formatted iMessage content with safety markers
    """
    parts = [
        f"From: {wrap_untrusted(sender, 'imessage_sender')}",
        f"Message: {wrap_untrusted(message, 'imessage_body')}",
    ]
    return "\n".join(parts)


def wrap_mcp_response(
    server_name: str,
    tool_name: str,
    response: str,
) -> str:
    """
    Wrap MCP tool response with safety markers.

    MCP responses come from external servers and are a primary injection
    vector. Malicious MCP servers can return crafted responses designed
    to hijack the LLM's behavior.

    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool that was called
        response: The raw response text

    Returns:
        MCP response wrapped in safety markers with metadata
    """
    escaped_response = escape_for_prompt(response)
    return (
        f"<untrusted_mcp server=\"{escape_for_attribute(server_name)}\" "
        f"tool=\"{escape_for_attribute(tool_name)}\">"
        f"{escaped_response}"
        f"</untrusted_mcp>"
    )


def wrap_memory_content(content: str, subject: str = None, source: str = None) -> str:
    """
    Wrap memory content for safe inclusion in prompts.

    Escapes content and wraps in <untrusted_memory> tags with provenance
    so the LLM can weigh trust appropriately.

    Args:
        content: Raw memory content (may contain injection attempts)
        subject: Who/what the memory is about (optional)
        source: Where this memory came from, e.g. "voice:doris", "proactive:email" (optional)

    Returns:
        Content wrapped in <untrusted_memory> tags with escaping applied
    """
    if not content:
        return "<untrusted_memory></untrusted_memory>"

    escaped = escape_for_prompt(content)

    attrs = ""
    if source:
        attrs += f' source="{escape_for_attribute(source)}"'
    if subject:
        attrs += f' subject="{escape_for_attribute(subject)}"'

    return f"<untrusted_memory{attrs}>{escaped}</untrusted_memory>"


def wrap_with_scan(
    content: str,
    source: str,
    scanner_source: Optional[str] = None,
) -> str:
    """
    Wrap content in safety markers AND scan for injection patterns.

    Combines wrapping with injection scanning. If suspicious patterns are
    found, a warning is included inside the wrapper tags.

    Args:
        content: The external content to wrap
        source: Source identifier for the wrapping tag
        scanner_source: Source identifier for the scanner log (defaults to source)

    Returns:
        Content wrapped in safety markers, with injection warning if suspicious
    """
    from security.injection_scanner import scan_for_injection

    scan_result = scan_for_injection(content, source=scanner_source or source)

    escaped = escape_for_prompt(content)

    if scan_result.is_suspicious:
        return (
            f"<untrusted_{source} suspicious=\"true\" "
            f"risk=\"{escape_for_attribute(scan_result.risk_level)}\">"
            f"\n<warning>{escape_for_prompt(scan_result.warning_text)}</warning>\n"
            f"{escaped}"
            f"</untrusted_{source}>"
        )

    return f"<untrusted_{source}>{escaped}</untrusted_{source}>"


# JSON schema for validating Haiku classification responses
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "required": ["relevance"],
    "properties": {
        "relevance": {
            "type": "string",
            "enum": ["low", "medium", "high"]
        },
        "escalate": {
            "type": "boolean"
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "reason": {
            "type": "string"
        }
    }
}

# Output size limits to prevent injection-inflated responses
MAX_TAGS = 10
MAX_TAG_LENGTH = 50
MAX_REASON_LENGTH = 200
ALLOWED_FIELDS = {"relevance", "escalate", "tags", "reason"}


def validate_classification_response(data: dict) -> tuple[bool, str]:
    """
    Validate a classification response from Haiku matches expected schema.

    This prevents prompt injection attacks where malicious email content
    could try to inject fake JSON that manipulates classification.

    Args:
        data: Parsed JSON response from Haiku

    Returns:
        (is_valid, error_message) tuple
    """
    if not isinstance(data, dict):
        return False, "Response is not a JSON object"

    # Reject unexpected fields (injection may add extra keys)
    extra_fields = set(data.keys()) - ALLOWED_FIELDS
    if extra_fields:
        logger.warning(f"Classification response has unexpected fields: {extra_fields}")
        # Don't fail — just log. LLMs sometimes add extra fields innocently.

    # Check required field
    if "relevance" not in data:
        return False, "Missing required field: relevance"

    # Validate relevance value
    relevance = data.get("relevance")
    if relevance not in ("low", "medium", "high"):
        return False, f"Invalid relevance value: {relevance}"

    # Validate escalate if present
    if "escalate" in data and not isinstance(data["escalate"], bool):
        return False, "escalate must be a boolean"

    # Validate tags if present
    if "tags" in data:
        tags = data["tags"]
        if not isinstance(tags, list):
            return False, "tags must be an array"
        if len(tags) > MAX_TAGS:
            return False, f"Too many tags ({len(tags)} > {MAX_TAGS})"
        for tag in tags:
            if not isinstance(tag, str):
                return False, "tags must contain only strings"
            if len(tag) > MAX_TAG_LENGTH:
                return False, f"Tag too long ({len(tag)} > {MAX_TAG_LENGTH})"

    # Validate reason if present
    if "reason" in data:
        if not isinstance(data["reason"], str):
            return False, "reason must be a string"
        if len(data["reason"]) > MAX_REASON_LENGTH:
            return False, f"Reason too long ({len(data['reason'])} > {MAX_REASON_LENGTH})"

    return True, ""


def sanitize_classification_output(data: dict) -> dict:
    """
    Sanitize classification output to enforce size limits.

    Applied after validation passes. Truncates rather than rejecting,
    so slightly-over-limit responses from the LLM still work.

    Args:
        data: Validated classification response

    Returns:
        Sanitized classification response
    """
    result = {}

    # Copy only allowed fields
    result["relevance"] = data.get("relevance", "medium")
    result["escalate"] = data.get("escalate", False)

    # Sanitize tags: strip whitespace, truncate, limit count
    tags = data.get("tags", [])
    sanitized_tags = []
    for tag in tags[:MAX_TAGS]:
        if isinstance(tag, str):
            clean = tag.strip()[:MAX_TAG_LENGTH]
            if clean:
                sanitized_tags.append(clean)
    result["tags"] = sanitized_tags

    # Truncate reason
    reason = data.get("reason", "")
    if isinstance(reason, str):
        result["reason"] = reason[:MAX_REASON_LENGTH]

    return result
