#!/usr/bin/env python3
"""
Test script for calendar scout pickup detection and transit mode logic.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scouts.calendar_scout import CalendarScout


def test_pickup_detection():
    """Test pickup vs drop-off detection."""
    scout = CalendarScout()
    now = datetime.now()

    # Test cases
    test_cases = [
        {
            "title": "Code Academy - Alex coding class",
            "start": (now + timedelta(hours=1)).isoformat(),
            "end": (now + timedelta(hours=2)).isoformat(),
            "expected": "drop-off (kids' activity, closer to start - time to take him there)",
        },
        {
            "title": "Code Academy - Alex coding class",
            "start": (now - timedelta(minutes=30)).isoformat(),
            "end": (now + timedelta(minutes=15)).isoformat(),
            "expected": "pickup (event is running, approaching end time)",
        },
        {
            "title": "Drop off Alex at Code Academy",
            "start": (now + timedelta(hours=1)).isoformat(),
            "end": (now + timedelta(hours=2)).isoformat(),
            "expected": "drop-off (explicit keyword)",
        },
        {
            "title": "Pick up Emma from gymnastics",
            "start": (now + timedelta(hours=1)).isoformat(),
            "end": (now + timedelta(hours=2)).isoformat(),
            "expected": "pickup (explicit keyword)",
        },
        {
            "title": "Gymnastics - Emma",
            "start": (now + timedelta(minutes=5)).isoformat(),
            "end": (now + timedelta(hours=1, minutes=5)).isoformat(),
            "expected": "drop-off (very close to start time - time to take her)",
        },
        {
            "title": "Meeting with John",
            "start": (now + timedelta(hours=1)).isoformat(),
            "end": (now + timedelta(hours=2)).isoformat(),
            "expected": "drop-off (not a kids' activity)",
        },
    ]

    print("Testing pickup detection:\n")
    for i, test in enumerate(test_cases, 1):
        event = {
            "id": f"test_{i}",
            "title": test["title"],
            "start": test["start"],
            "end": test["end"],
        }

        is_pickup = scout._is_pickup_event(event, now)
        result = "pickup" if is_pickup else "drop-off"
        status = "✓" if test["expected"].startswith(result) else "✗"

        print(f"{status} Test {i}: '{test['title']}'")
        print(f"   Expected: {test['expected']}")
        print(f"   Got: {result}")
        print()


def test_transit_mode_detection():
    """Test transit mode detection."""
    scout = CalendarScout()

    test_cases = [
        {"title": "Walk to the park", "expected": "walk"},
        {"title": "Code Academy (walking distance)", "expected": "walk"},
        {"title": "Dentist appointment - take subway", "expected": "subway"},
        {"title": "Drive to Upstate", "expected": "car"},
        {"title": "Take Uber to appointment", "expected": "car"},
        {"title": "Bike to office", "expected": "bike"},
        {"title": "Generic meeting", "expected": "default"},
    ]

    print("\nTesting transit mode detection:\n")
    for i, test in enumerate(test_cases, 1):
        event = {
            "id": f"test_{i}",
            "title": test["title"],
            "notes": "",
            "location": "",
        }

        mode = scout._detect_transit_mode(event)
        status = "✓" if mode == test["expected"] else "✗"

        print(f"{status} Test {i}: '{test['title']}'")
        print(f"   Expected: {test['expected']}")
        print(f"   Got: {mode}")
        print()


def test_travel_time_estimation():
    """Test travel time estimation."""
    scout = CalendarScout()

    test_cases = [
        {"location": "Home", "mode": "walk", "expected_range": (0, 10)},
        {"location": "Code Academy", "mode": "walk", "expected_range": (15, 30)},
        {"location": "Midtown Manhattan", "mode": "subway", "expected_range": (30, 50)},
        {"location": "Upstate", "mode": "car", "expected_range": (110, 130)},
        {"location": "Central Park", "mode": "bike", "expected_range": (15, 25)},
    ]

    print("\nTesting travel time estimation:\n")
    for i, test in enumerate(test_cases, 1):
        travel_time = scout._estimate_travel_time(test["location"], test["mode"])
        in_range = test["expected_range"][0] <= travel_time <= test["expected_range"][1]
        status = "✓" if in_range else "✗"

        print(f"{status} Test {i}: {test['location']} by {test['mode']}")
        print(f"   Expected: {test['expected_range'][0]}-{test['expected_range'][1]} min")
        print(f"   Got: {travel_time} min")
        print()


def test_store_transit_preference():
    """Test storing transit preferences in knowledge graph."""
    scout = CalendarScout()

    print("\nTesting transit preference storage:\n")

    # Test storing preference
    print("1. Storing preference: Code Academy -> walk")
    success = scout.store_transit_preference("Code Academy", "walk")
    print(f"   Result: {'✓ Success' if success else '✗ Failed'}")
    print()

    # Test retrieval
    print("2. Retrieving preference for Code Academy")
    mode = scout._get_transit_mode_for_location("Code Academy")
    print(f"   Result: {mode if mode else 'None'}")
    print(f"   {'✓ Correct' if mode == 'walk' else '✗ Incorrect or not found'}")
    print()

    # Test invalid mode
    print("3. Testing invalid mode (should fail)")
    success = scout.store_transit_preference("Test Location", "airplane")
    print(f"   Result: {'✗ Accepted invalid mode' if success else '✓ Rejected correctly'}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Calendar Scout Test Suite")
    print("=" * 60)

    test_pickup_detection()
    test_transit_mode_detection()
    test_travel_time_estimation()
    test_store_transit_preference()

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
