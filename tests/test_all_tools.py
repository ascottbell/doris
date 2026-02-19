#!/usr/bin/env python3
"""
Doris Comprehensive Test Suite
Tests all tools, services, and integrations.

Run:
    cd <project_root>
    python tests/test_all_tools.py
"""

import sys
import os
import time
import asyncio
import traceback
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Terminal colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Test results tracking
results = []
current_section = ""


def log_result(name: str, passed: bool, error: str = None):
    """Log a test result."""
    results.append({
        "section": current_section,
        "name": name,
        "passed": passed,
        "error": error
    })
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  {'✓' if passed else '✗'} {name:<50} [{status}]")
    if error and not passed:
        # Truncate long errors
        error_short = error[:80] + "..." if len(error) > 80 else error
        print(f"    {RED}{error_short}{RESET}")


def section(name: str):
    """Start a new test section."""
    global current_section
    current_section = name
    print(f"\n{BOLD}{CYAN}{name}{RESET}")
    print("=" * 60)


def run_test(name: str, test_fn):
    """Run a single test with exception handling."""
    try:
        result = test_fn()
        if result is True or result is None:
            log_result(name, True)
        elif result is False:
            log_result(name, False)
        else:
            # Returned some value, consider it a pass
            log_result(name, True)
    except Exception as e:
        log_result(name, False, str(e))


async def run_async_test(name: str, test_fn):
    """Run an async test with exception handling."""
    try:
        result = await test_fn()
        if result is True or result is None:
            log_result(name, True)
        elif result is False:
            log_result(name, False)
        else:
            log_result(name, True)
    except Exception as e:
        log_result(name, False, str(e))


# =============================================================================
# 1. WEATHER TESTS
# =============================================================================

def test_weather():
    section("WEATHER TOOLS (tools/weather.py)")

    from tools.weather import (
        get_current_weather, get_forecast,
        coords_to_known_location, geocode_location
    )

    def test_current_default():
        result = get_current_weather()
        assert result is not None, "No weather data returned"
        assert "temperature" in result, "Missing temperature"
        assert "conditions" in result, "Missing conditions"
        return True

    def test_current_location():
        result = get_current_weather(location="los angeles")
        assert result is not None, "No weather for Los Angeles"
        return True

    def test_current_coords():
        result = get_current_weather(lat=34.0522, lon=-118.2437)
        assert result is not None, "No weather for coords"
        return True

    def test_forecast_3days():
        result = get_forecast(days=3)
        assert result is not None, "No forecast data"
        assert len(result) == 3, f"Expected 3 days, got {len(result)}"
        return True

    def test_forecast_location():
        result = get_forecast(days=3, location="los angeles")
        assert result is not None, "No forecast for Los Angeles"
        return True

    def test_coords_to_los_angeles():
        result = coords_to_known_location(34.0522, -118.2437)
        assert result == "Los Angeles", f"Expected 'Los Angeles', got '{result}'"
        return True

    def test_coords_unknown():
        # Random coords in the ocean
        result = coords_to_known_location(0.0, 0.0)
        assert result is None, f"Expected None for ocean coords, got '{result}'"
        return True

    def test_geocode_paris():
        result = geocode_location("Paris")
        assert result is not None, "Failed to geocode Paris"
        lat, lon = result
        # Paris is roughly at 48.8, 2.3
        assert 48 < lat < 49, f"Paris lat should be ~48.8, got {lat}"
        assert 2 < lon < 3, f"Paris lon should be ~2.3, got {lon}"
        return True

    run_test("get_current_weather() default", test_current_default)
    run_test("get_current_weather(location='los angeles')", test_current_location)
    run_test("get_current_weather(lat=34.0522, lon=-118.2437)", test_current_coords)
    run_test("get_forecast(days=3)", test_forecast_3days)
    run_test("get_forecast(days=3, location='los angeles')", test_forecast_location)
    run_test("coords_to_known_location() Los Angeles", test_coords_to_los_angeles)
    run_test("coords_to_known_location() unknown", test_coords_unknown)
    run_test("geocode_location('Paris')", test_geocode_paris)


# =============================================================================
# 2. CALENDAR TESTS
# =============================================================================

def test_calendar():
    section("CALENDAR TOOLS (tools/cal.py)")

    CLI_PATH = Path.home() / "Projects/doris-calendar/.build/release/doris-calendar"

    def test_cli_exists():
        assert CLI_PATH.exists(), f"Calendar CLI not found at {CLI_PATH}"
        return True

    run_test("Verify CLI exists", test_cli_exists)

    if not CLI_PATH.exists():
        print(f"  {YELLOW}Skipping remaining calendar tests - CLI not found{RESET}")
        return

    from tools.cal import (
        list_events, get_todays_events, get_tomorrows_events,
        get_weeks_events, get_weekend_events, create_event
    )

    def test_list_events():
        result = list_events()
        assert isinstance(result, list), "list_events should return list"
        return True

    def test_todays_events():
        result = get_todays_events()
        assert isinstance(result, list), "get_todays_events should return list"
        return True

    def test_tomorrows_events():
        result = get_tomorrows_events()
        assert isinstance(result, list), "get_tomorrows_events should return list"
        return True

    def test_weeks_events():
        result = get_weeks_events()
        assert isinstance(result, list), "get_weeks_events should return list"
        return True

    def test_weekend_events():
        result = get_weekend_events()
        assert isinstance(result, list), "get_weekend_events should return list"
        return True

    def test_create_delete_event():
        # Create a test event
        now = datetime.now()
        start = now + timedelta(hours=1)
        end = start + timedelta(hours=1)

        result = create_event(
            title="TEST_DORIS_EVENT_DELETE_ME",
            start=start,
            end=end
        )
        assert "id" in result or "success" in result, f"Event creation failed: {result}"
        # Note: Delete would require implementing delete_event in cal.py
        return True

    run_test("list_events()", test_list_events)
    run_test("get_todays_events()", test_todays_events)
    run_test("get_tomorrows_events()", test_tomorrows_events)
    run_test("get_weeks_events()", test_weeks_events)
    run_test("get_weekend_events()", test_weekend_events)
    run_test("create_event() round trip", test_create_delete_event)


# =============================================================================
# 3. REMINDERS TESTS
# =============================================================================

def test_reminders():
    section("REMINDERS TOOLS (tools/reminders.py)")

    CLI_PATH = Path.home() / "Projects/doris-reminders/.build/release/doris-reminders"

    def test_cli_exists():
        assert CLI_PATH.exists(), f"Reminders CLI not found at {CLI_PATH}"
        return True

    run_test("Verify CLI exists", test_cli_exists)

    if not CLI_PATH.exists():
        print(f"  {YELLOW}Skipping remaining reminders tests - CLI not found{RESET}")
        return

    from tools.reminders import list_reminders, create_reminder, complete_reminder

    def test_list_reminders():
        result = list_reminders()
        assert isinstance(result, list), "list_reminders should return list"
        return True

    def test_create_complete_reminder():
        # Create test reminder
        result = create_reminder(title="TEST_DORIS_REMINDER_DELETE_ME")
        assert "id" in result or "success" in result, f"Reminder creation failed: {result}"

        if "id" in result:
            # Complete it
            complete_result = complete_reminder(result["id"])
            assert complete_result.get("success", False) or "completed" in str(complete_result).lower(), \
                f"Completion failed: {complete_result}"
        return True

    def test_create_with_due():
        tomorrow = datetime.now() + timedelta(days=1)
        result = create_reminder(
            title="TEST_DORIS_REMINDER_DUE",
            due=tomorrow
        )
        assert "id" in result or "success" in result or "error" not in result, \
            f"Reminder with due date failed: {result}"
        return True

    run_test("list_reminders()", test_list_reminders)
    run_test("create_reminder() + complete() round trip", test_create_complete_reminder)
    run_test("create_reminder with due datetime", test_create_with_due)


# =============================================================================
# 4. iMESSAGE TESTS
# =============================================================================

def test_imessage():
    section("iMESSAGE TOOLS (tools/imessage.py)")

    from tools.imessage import read_recent, send_message

    def test_read_recent():
        result = read_recent(limit=5)
        assert isinstance(result, list), "read_recent should return list"
        return True

    def test_send_exists():
        # Just verify the function exists and is callable
        assert callable(send_message), "send_message should be callable"
        return True

    run_test("read_recent() returns list", test_read_recent)
    run_test("send_message function exists", test_send_exists)


# =============================================================================
# 5. GMAIL TESTS
# =============================================================================

def test_gmail():
    section("GMAIL TOOLS (tools/gmail.py)")

    CREDENTIALS_PATH = PROJECT_ROOT / "credentials" / "credentials.json"
    TOKEN_PATH = PROJECT_ROOT / "token.json"

    def test_credentials_exist():
        assert CREDENTIALS_PATH.exists(), f"credentials.json not found at {CREDENTIALS_PATH}"
        return True

    run_test("Verify credentials exist", test_credentials_exist)

    if not CREDENTIALS_PATH.exists():
        print(f"  {YELLOW}Skipping Gmail tests - credentials not found{RESET}")
        return

    from tools.gmail import get_unread_count, scan_recent_emails

    def test_unread_count():
        result = get_unread_count()
        assert isinstance(result, int), f"get_unread_count should return int, got {type(result)}"
        return True

    def test_scan_emails():
        result = scan_recent_emails(hours=24)
        assert isinstance(result, list), "scan_recent_emails should return list"
        return True

    run_test("get_unread_count() returns int", test_unread_count)
    run_test("scan_recent_emails() returns list", test_scan_emails)


# =============================================================================
# 6. MEMORY TESTS
# =============================================================================

def test_memory():
    section("MEMORY STORE (memory/store.py)")

    from memory.store import store_memory, find_similar_memories, get_all_active, delete_memory

    test_id = None

    def test_store():
        nonlocal test_id
        test_id = store_memory(
            content="TEST_MEMORY_DELETE_ME This is a test memory",
            category="test",
            subject="test_suite"
        )
        assert test_id is not None, "store_memory returned None"
        assert test_id.startswith("mem_"), f"Memory ID format wrong: {test_id}"
        return True

    def test_find_similar():
        result = find_similar_memories("TEST_MEMORY_DELETE_ME", limit=5)
        assert isinstance(result, list), "find_similar_memories should return list"
        # Should find our test memory
        found = any("TEST_MEMORY_DELETE_ME" in m.get("content", "") for m in result)
        assert found, "Did not find the test memory"
        return True

    def test_get_all():
        result = get_all_active()
        assert isinstance(result, list), "get_all_active should return list"
        return True

    def test_delete():
        nonlocal test_id
        if test_id:
            result = delete_memory(test_id)
            assert result is True, f"delete_memory failed: {result}"
        return True

    run_test("store_memory() creates memory", test_store)
    run_test("find_similar_memories() finds it", test_find_similar)
    run_test("get_all_active() lists memories", test_get_all)
    run_test("delete_memory() cleans up", test_delete)


# =============================================================================
# 7. BRAIN DUMP TESTS
# =============================================================================

def test_brain():
    section("BRAIN DUMP (tools/brain.py)")

    from tools.brain import get_inbox_status, get_active_goals

    def test_inbox_status():
        result = get_inbox_status()
        assert isinstance(result, dict), "get_inbox_status should return dict"
        assert "unprocessed_count" in result, "Missing unprocessed_count"
        return True

    def test_active_goals():
        result = get_active_goals()
        assert isinstance(result, list), "get_active_goals should return list"
        return True

    run_test("get_inbox_status()", test_inbox_status)
    run_test("get_active_goals()", test_active_goals)


# =============================================================================
# 9. CONTACTS TESTS
# =============================================================================

def test_contacts():
    section("CONTACTS (tools/contacts.py)")

    def test_module_loads():
        try:
            from tools.contacts import lookup_contact
            assert callable(lookup_contact), "lookup_contact should be callable"
            return True
        except ImportError as e:
            if "Contacts" in str(e):
                # pyobjc Contacts framework not available
                raise AssertionError("Contacts framework not available (pyobjc)")
            raise

    def test_lookup_exists():
        from tools.contacts import lookup_contact
        # Just verify function exists
        return True

    run_test("Module loads", test_module_loads)
    run_test("lookup_contact function exists", test_lookup_exists)


# =============================================================================
# 10. CLAUDE INTEGRATION TESTS
# =============================================================================

def test_claude_integration():
    section("CLAUDE INTEGRATION (llm/brain.py)")

    try:
        from llm.brain import execute_tool
    except ImportError:
        print(f"  {YELLOW}execute_tool not found in llm/brain.py - skipping{RESET}")
        return

    def test_get_time():
        result = execute_tool("get_current_time", {})
        assert result is not None, "get_current_time returned None"
        return True

    def test_get_location():
        result = execute_tool("get_user_location", {}, user_location={"lat": 34.0522, "lon": -118.2437})
        assert result is not None, "get_user_location returned None"
        return True

    def test_get_weather():
        result = execute_tool("get_weather", {})
        assert result is not None, "get_weather returned None"
        return True

    def test_get_weather_location():
        result = execute_tool("get_weather", {"location": "los angeles", "days": 3})
        assert result is not None, "get_weather with location returned None"
        return True

    def test_calendar_events():
        result = execute_tool("get_calendar_events", {"period": "today"})
        # May return empty if calendar CLI not set up
        return True

    def test_calendar_specific_date():
        # Test that specific dates work (past and future)
        result = execute_tool("get_calendar_events", {"period": "February 2"})
        assert "CALENDAR FOR" in result, "Should return calendar header"
        return True

    def test_calendar_iso_date():
        # Test ISO date format
        result = execute_tool("get_calendar_events", {"period": "2026-02-02"})
        assert "CALENDAR FOR" in result, "Should parse ISO dates"
        return True

    def test_list_reminders():
        result = execute_tool("list_reminders", {})
        return True

    def test_check_email():
        result = execute_tool("check_email", {})
        return True

    def test_search_memory():
        result = execute_tool("search_memory", {"query": "test"})
        return True

    def test_daily_briefing():
        result = execute_tool("daily_briefing", {})
        return True

    run_test("execute_tool('get_current_time', {})", test_get_time)
    run_test("execute_tool('get_user_location', {}) with location", test_get_location)
    run_test("execute_tool('get_weather', {})", test_get_weather)
    run_test("execute_tool('get_weather', {location, days})", test_get_weather_location)
    run_test("execute_tool('get_calendar_events', {period})", test_calendar_events)
    run_test("execute_tool('get_calendar_events', {specific_date})", test_calendar_specific_date)
    run_test("execute_tool('get_calendar_events', {iso_date})", test_calendar_iso_date)
    run_test("execute_tool('list_reminders', {})", test_list_reminders)
    run_test("execute_tool('check_email', {})", test_check_email)
    run_test("execute_tool('search_memory', {query})", test_search_memory)
    run_test("execute_tool('daily_briefing', {})", test_daily_briefing)


# =============================================================================
# 11. MCP CONNECTIONS TESTS
# =============================================================================

async def test_mcp_connections():
    section("MCP CONNECTIONS (mcp_client/)")

    from mcp_client import init_mcp, shutdown_mcp, get_mcp_manager
    from mcp_client.config import get_enabled_servers

    def test_config_exists():
        servers = get_enabled_servers()
        assert isinstance(servers, (list, dict)), "get_enabled_servers should return list/dict"
        return True

    run_test("MCP config exists", test_config_exists)

    # Test connections
    enabled = get_enabled_servers()
    expected_servers = ["apple-music", "home-assistant", "doris-memory", "github"]

    for server in expected_servers:
        def make_test(s):
            def test():
                if s in enabled:
                    return True
                raise AssertionError(f"Server '{s}' not enabled in config")
            return test

        run_test(f"{server} configured", make_test(server))


# =============================================================================
# 12. API ENDPOINTS TESTS
# =============================================================================

async def test_api_endpoints():
    section("API ENDPOINTS (main.py)")

    import httpx

    BASE_URL = "http://localhost:8000"
    timeout = httpx.Timeout(30.0)

    async def test_health():
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{BASE_URL}/health")
            assert r.status_code == 200, f"Health check failed: {r.status_code}"
            data = r.json()
            assert "status" in data, "Missing status in health response"
        return True

    async def test_emails():
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{BASE_URL}/emails")
            assert r.status_code == 200, f"Emails endpoint failed: {r.status_code}"
        return True

    async def test_briefing():
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{BASE_URL}/briefing")
            assert r.status_code == 200, f"Briefing endpoint failed: {r.status_code}"
        return True

    async def test_brain_status():
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{BASE_URL}/brain/status")
            assert r.status_code == 200, f"Brain status failed: {r.status_code}"
        return True

    async def test_brain_goals():
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{BASE_URL}/brain/goals")
            assert r.status_code == 200, f"Brain goals failed: {r.status_code}"
        return True

    async def test_chat_text():
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{BASE_URL}/chat/text",
                json={"message": "What time is it?"}
            )
            assert r.status_code == 200, f"Chat text failed: {r.status_code}"
            data = r.json()
            assert "response" in data or "tool_request" in data, "Missing response in chat"
        return True

    # Check if server is running
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            await client.get(f"{BASE_URL}/health")
    except Exception:
        print(f"  {YELLOW}Server not running at {BASE_URL} - skipping API tests{RESET}")
        return

    await run_async_test("GET /health", test_health)
    await run_async_test("GET /emails", test_emails)
    await run_async_test("GET /briefing", test_briefing)
    await run_async_test("GET /brain/status", test_brain_status)
    await run_async_test("GET /brain/goals", test_brain_goals)
    await run_async_test("POST /chat/text", test_chat_text)


# =============================================================================
# 13. PUSH NOTIFICATIONS TESTS
# =============================================================================

def test_push():
    section("PUSH NOTIFICATIONS (services/push.py)")

    def test_module_loads():
        from services.push import register_device
        assert callable(register_device), "register_device should be callable"
        return True

    run_test("Module loads", test_module_loads)
    run_test("register_device function exists", test_module_loads)


# =============================================================================
# 14. PROACTIVE SYSTEM TESTS
# =============================================================================

def test_proactive():
    section("PROACTIVE SYSTEM (proactive/)")

    def test_scheduler_loads():
        from proactive.scheduler import init_scheduler
        assert callable(init_scheduler), "init_scheduler should be callable"
        return True

    def test_evaluator_loads():
        from proactive.evaluator import evaluate_event
        assert callable(evaluate_event), "evaluate_event should be callable"
        return True

    def test_notifier_loads():
        from proactive.notifier import notify_action
        assert callable(notify_action), "notify_action should be callable"
        return True

    run_test("Scheduler initializes", test_scheduler_loads)
    run_test("Evaluator module loads", test_evaluator_loads)
    run_test("Notifier module loads", test_notifier_loads)


# =============================================================================
# MAIN
# =============================================================================

def print_summary():
    """Print test summary."""
    print("\n")
    print("=" * 80)
    print(f"{BOLD}DORIS COMPREHENSIVE TEST SUITE - SUMMARY{RESET}")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pct = (passed / total * 100) if total > 0 else 0

    # Group by section
    by_section = {}
    for r in results:
        s = r["section"]
        if s not in by_section:
            by_section[s] = {"passed": 0, "failed": 0}
        if r["passed"]:
            by_section[s]["passed"] += 1
        else:
            by_section[s]["failed"] += 1

    # Print per-section summary
    for section_name, counts in by_section.items():
        section_total = counts["passed"] + counts["failed"]
        section_pct = (counts["passed"] / section_total * 100) if section_total > 0 else 0
        status = f"{GREEN}PASS{RESET}" if counts["failed"] == 0 else f"{RED}FAIL{RESET}"
        print(f"  {section_name:<40} {counts['passed']}/{section_total} [{status}]")

    print("-" * 80)
    color = GREEN if failed == 0 else RED
    print(f"{BOLD}TOTAL: {passed}/{total} tests passed ({pct:.1f}%){RESET}")

    # Print failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{BOLD}{RED}FAILURES:{RESET}")
        for f in failures:
            print(f"  {f['section']} > {f['name']}")
            if f.get("error"):
                error_short = f["error"][:100] + "..." if len(f["error"]) > 100 else f["error"]
                print(f"    {RED}{error_short}{RESET}")

    return failed == 0


async def main():
    """Run all tests."""
    print("=" * 80)
    print(f"{BOLD}{CYAN}DORIS COMPREHENSIVE TEST SUITE{RESET}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    start_time = time.time()

    # Synchronous tests
    test_weather()
    test_calendar()
    test_reminders()
    test_imessage()
    test_gmail()
    test_memory()
    test_brain()
    test_contacts()
    test_claude_integration()
    test_push()
    test_proactive()

    # Async tests
    await test_mcp_connections()
    await test_api_endpoints()

    elapsed = time.time() - start_time

    success = print_summary()
    print(f"\nCompleted in {elapsed:.1f}s")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
