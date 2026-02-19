#!/usr/bin/env python3
"""
Test all Doris tools via Claude text chat.

Tests are grouped:
- READ tests: safe, read-only operations
- WRITE tests: create events/reminders (marked skip by default)
- MCP tests: require MCP servers to be running
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.brain import chat

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def test_tool(name: str, prompt: str, expected_tool: str = None, skip: bool = False):
    """Run a test and report results."""
    if skip:
        print(f"{YELLOW}SKIP{RESET} {name}")
        return None

    print(f"\n{CYAN}TEST{RESET} {name}")
    print(f"  Prompt: {prompt[:60]}...")

    try:
        response = chat(prompt)
        print(f"  {GREEN}OK{RESET} Response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"  {RED}FAIL{RESET} Error: {e}")
        return False


def main():
    print("=" * 60)
    print("DORIS TOOL TEST SUITE")
    print("=" * 60)

    results = {"pass": 0, "fail": 0, "skip": 0}

    # ==================== READ-ONLY TESTS ====================
    print(f"\n{CYAN}--- READ-ONLY TOOLS ---{RESET}")

    # 1. Time
    r = test_tool(
        "get_current_time",
        "What time is it?"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 2. Weather
    r = test_tool(
        "get_weather",
        "What's the weather like?"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 3. Calendar - today
    r = test_tool(
        "get_calendar_events (today)",
        "What's on my calendar today?"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 4. Calendar - tomorrow
    r = test_tool(
        "get_calendar_events (tomorrow)",
        "What do I have tomorrow?"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 5. Calendar - week
    r = test_tool(
        "get_calendar_events (week)",
        "What's my schedule this week?"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 6. List reminders
    r = test_tool(
        "list_reminders",
        "What are my reminders?"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 7. Read messages
    r = test_tool(
        "read_imessages",
        "Show me my recent text messages"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 8. Check email
    r = test_tool(
        "check_email",
        "Any important emails in the last 24 hours?"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 9. Daily briefing
    r = test_tool(
        "daily_briefing",
        "Give me my morning briefing"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 10. Search memory
    r = test_tool(
        "search_memory",
        "What do you remember about Alex?"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # ==================== MCP TOOLS ====================
    print(f"\n{CYAN}--- MCP TOOLS (require servers) ---{RESET}")

    # 11. Music - current
    r = test_tool(
        "control_music (current)",
        "What song is playing?"
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # ==================== WRITE TESTS (SKIP BY DEFAULT) ====================
    print(f"\n{CYAN}--- WRITE TOOLS (skipped by default) ---{RESET}")

    # 13. Create calendar event
    r = test_tool(
        "create_calendar_event",
        "Create a test event tomorrow at 3pm called 'Tool Test 2'",
        skip=False
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 14. Create reminder
    r = test_tool(
        "create_reminder",
        "Remind me to clean up test data tomorrow at noon",
        skip=False
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 15. Send message - skip to avoid spam
    r = test_tool(
        "send_imessage",
        "Send a test message to Jane saying 'testing Doris'",
        skip=True  # Already tested manually
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # 16. Store memory
    r = test_tool(
        "store_memory",
        "Remember that I ran a tool test today",
        skip=True
    )
    if r is True: results["pass"] += 1
    elif r is False: results["fail"] += 1
    else: results["skip"] += 1

    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{GREEN}PASS{RESET}: {results['pass']}")
    print(f"{RED}FAIL{RESET}: {results['fail']}")
    print(f"{YELLOW}SKIP{RESET}: {results['skip']}")

    if results["fail"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
