#!/usr/bin/env python3
"""
Test script for calendar scout timing intelligence.

Tests:
1. Pickup vs drop-off detection
2. Distance-based travel time calculation
3. Transit mode detection
4. Knowledge graph integration
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scouts.calendar_scout import CalendarScout


def test_pickup_detection():
    """Test pickup vs drop-off detection."""
    print("=" * 60)
    print("TEST 1: Pickup vs Drop-off Detection")
    print("=" * 60)

    scout = CalendarScout()
    now = datetime.now()

    # Test cases
    test_events = [
        {
            'id': 'test1',
            'title': 'Code Academy coding class',
            'start': (now + timedelta(hours=1)).isoformat(),
            'end': (now + timedelta(hours=2)).isoformat(),
            'location': 'Code Academy, 123 Main St'
        },
        {
            'id': 'test2',
            'title': 'Pickup Alex from gymnastics',
            'start': now.isoformat(),
            'end': (now + timedelta(minutes=30)).isoformat(),
            'location': 'Community Center'
        },
        {
            'id': 'test3',
            'title': 'Drop off Emma at soccer',
            'start': (now + timedelta(hours=1)).isoformat(),
            'end': (now + timedelta(hours=2)).isoformat(),
            'location': 'Riverside Park'
        },
    ]

    for event in test_events:
        is_pickup = scout._is_pickup_event(event, now)
        print(f"\nEvent: {event['title']}")
        print(f"  Detected as: {'PICKUP' if is_pickup else 'DROP-OFF'}")


def test_distance_calculation():
    """Test distance-based travel time estimation."""
    print("\n" + "=" * 60)
    print("TEST 2: Distance-Based Travel Time")
    print("=" * 60)

    scout = CalendarScout()

    # Test locations with coordinates
    test_locations = [
        ("Code Academy, 123 Main St (40.7128,-74.0060)", "walk"),
        ("Central Park Zoo (40.7678,-73.9718)", "walk"),
        ("Brooklyn Bridge Park (40.7024,-73.9964)", "subway"),
        ("Home", "walk"),
    ]

    for location, mode in test_locations:
        travel_time = scout._estimate_travel_time(location, mode)
        coords = scout._parse_location_coords(location)

        print(f"\nLocation: {location}")
        print(f"  Coordinates: {coords}")
        print(f"  Transit mode: {mode}")
        print(f"  Estimated travel time: {travel_time} minutes")

        if coords:
            distance = scout._calculate_distance(scout._home_coords, coords)
            print(f"  Distance from home: {distance:.2f} miles")


def test_transit_mode_detection():
    """Test transit mode detection from event text."""
    print("\n" + "=" * 60)
    print("TEST 3: Transit Mode Detection")
    print("=" * 60)

    scout = CalendarScout()

    test_events = [
        {
            'title': 'Walk to Code Academy coding',
            'notes': '',
            'location': 'Code Academy'
        },
        {
            'title': 'Gymnastics',
            'notes': 'Take the subway',
            'location': 'Community Center'
        },
        {
            'title': 'Drive to Upstate',
            'notes': '',
            'location': 'Rhinebeck'
        },
        {
            'title': 'Regular appointment',
            'notes': '',
            'location': 'Midtown'
        },
    ]

    for event in test_events:
        mode = scout._detect_transit_mode(event)
        print(f"\nEvent: {event['title']}")
        print(f"  Location: {event['location']}")
        print(f"  Detected mode: {mode}")


def test_knowledge_graph_integration():
    """Test knowledge graph lookup for transit preferences."""
    print("\n" + "=" * 60)
    print("TEST 4: Knowledge Graph Integration")
    print("=" * 60)

    scout = CalendarScout()

    # Store some test preferences
    print("\nStoring transit preferences:")
    scout.store_transit_preference("Code Academy", "walk")
    scout.store_transit_preference("Community Center", "subway")

    # Look them up
    print("\nLooking up stored preferences:")
    locations = ["Code Academy", "Community Center", "Unknown Place"]

    for location in locations:
        mode = scout._get_transit_mode_for_location(location)
        print(f"  {location}: {mode if mode else 'not found'}")


def test_complete_timing_logic():
    """Test the complete timing logic with a realistic scenario."""
    print("\n" + "=" * 60)
    print("TEST 5: Complete Timing Logic")
    print("=" * 60)

    scout = CalendarScout()
    now = datetime.now()

    # Realistic test: Alex's Code Academy class ends in 45 minutes
    # We need to walk there (0.25 miles, ~10 min)
    event = {
        'id': 'launch_pickup',
        'title': 'Alex Code Academy coding',
        'start': (now - timedelta(hours=1)).isoformat(),  # Started 1 hour ago
        'end': (now + timedelta(minutes=45)).isoformat(),  # Ends in 45 min
        'location': 'Code Academy, 123 Main St (40.7128,-74.0060)',
        'notes': ''
    }

    print(f"\nScenario: Alex's Code Academy class")
    print(f"  Current time: {now.strftime('%I:%M %p')}")
    print(f"  Class ends: {(now + timedelta(minutes=45)).strftime('%I:%M %p')}")

    is_pickup = scout._is_pickup_event(event, now)
    print(f"\n  Detected as: {'PICKUP' if is_pickup else 'DROP-OFF'}")

    transit_mode = scout._detect_transit_mode(event)
    print(f"  Transit mode: {transit_mode}")

    travel_time = scout._estimate_travel_time(event['location'], transit_mode)
    print(f"  Estimated travel time: {travel_time} minutes")

    # Calculate when we should leave
    target_time_str = event.get('end' if is_pickup else 'start')
    target_time = datetime.fromisoformat(target_time_str)
    leave_time = target_time - timedelta(minutes=travel_time)

    print(f"\n  Should leave at: {leave_time.strftime('%I:%M %p')}")
    print(f"  Time until departure: {int((leave_time - now).total_seconds() / 60)} minutes")


if __name__ == "__main__":
    test_pickup_detection()
    test_distance_calculation()
    test_transit_mode_detection()
    test_knowledge_graph_integration()
    test_complete_timing_logic()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
