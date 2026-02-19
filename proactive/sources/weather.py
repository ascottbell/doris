"""
Weather source - monitors for weather alerts and notable conditions.

Checks for:
- Rain/snow when there are outdoor events
- Significant temperature changes
- Weather alerts
- School pickup/dropoff weather warnings
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tools.weather import get_current_weather, WEATHER_CODES
from tools.cal import list_events
from proactive.models import ProactiveEvent
from proactive.db import (
    save_event,
    is_event_processed,
    update_event_status,
    update_checkpoint,
)
from proactive.notifier import notify_action

EASTERN = ZoneInfo("America/New_York")

# Conditions that warrant a proactive alert
ALERT_CONDITIONS = {
    "heavy rain": "Rain's coming down hard",
    "heavy showers": "Heavy showers expected",
    "thunderstorm": "Thunderstorm warning",
    "thunderstorm with hail": "Thunderstorm with hail warning",
    "thunderstorm with heavy hail": "Severe thunderstorm warning",
    "heavy snow": "Heavy snow warning",
    "freezing rain": "Freezing rain - dangerous conditions",
}

# School schedule times (for weather at pickup/dropoff)
SCHOOL_DROPOFF = 8  # 8 AM
SCHOOL_PICKUP = 15  # 3 PM


def monitor():
    """
    Main weather monitor function.

    Called by scheduler hourly. Checks for:
    1. Severe weather alerts
    2. Rain before school pickup/dropoff
    3. Rain before outdoor events
    """
    print("[weather-monitor] Checking weather conditions...")

    now = datetime.now(EASTERN)

    try:
        weather = get_current_weather()
        if not weather:
            print("[weather-monitor] Could not get weather data")
            return

        # Check 1: Severe weather alerts
        _check_severe_weather(now, weather)

        # Check 2: School pickup weather (if between 2-3 PM on weekdays)
        if 14 <= now.hour < 15 and now.weekday() < 5:  # Monday=0, Friday=4
            _check_school_pickup_weather(now, weather)

        # Check 3: Rain before outdoor events
        _check_event_weather(now, weather)

        update_checkpoint("weather")
        print("[weather-monitor] Done")

    except Exception as e:
        print(f"[weather-monitor] Error: {e}")
        import traceback
        traceback.print_exc()


def _check_severe_weather(now: datetime, weather: dict):
    """Check for severe weather conditions that need immediate alert."""
    conditions = weather.get("conditions", "").lower()

    # Check against alert conditions
    for condition, message in ALERT_CONDITIONS.items():
        if condition in conditions:
            alert_id = f"weather-alert-{now.strftime('%Y%m%d%H')}"

            if is_event_processed("weather", alert_id):
                return

            proactive_event = ProactiveEvent(
                source_type="weather",
                source_id=alert_id,
                raw_data={
                    "type": "severe_weather",
                    "conditions": conditions,
                    "temp": weather.get("temperature"),
                    "alert": message,
                }
            )
            save_event(proactive_event)

            # Direct notify (skip Claude for weather alerts)
            # Format for speech with natural pauses
            from proactive.executor import _create_notification
            action = _create_notification(proactive_event, {
                "message": f"Hey, ... weather alert. ... {message}. ... Currently {weather.get('temperature')} degrees, ... with {conditions}.",
                "priority": "high"
            })

            if action:
                update_event_status(proactive_event.id, "actioned")
                notify_action(action)  # Push notification for severe weather
            return


def _check_school_pickup_weather(now: datetime, weather: dict):
    """Alert about rain/cold for school pickup."""
    conditions = weather.get("conditions", "").lower()
    temp = weather.get("temperature", 60)
    precip = weather.get("precipitation_chance", 0)

    needs_alert = False
    message_parts = []

    # Rain check
    if "rain" in conditions or "shower" in conditions or "drizzle" in conditions:
        needs_alert = True
        message_parts.append("it's raining")
    elif precip >= 60:
        needs_alert = True
        message_parts.append(f"{precip}% chance of rain")

    # Cold check
    if temp <= 35:
        needs_alert = True
        message_parts.append(f"it's only {temp} degrees")

    if not needs_alert:
        return

    alert_id = f"school-pickup-weather-{now.strftime('%Y%m%d')}"
    if is_event_processed("weather", alert_id):
        return

    proactive_event = ProactiveEvent(
        source_type="weather",
        source_id=alert_id,
        raw_data={
            "type": "school_pickup",
            "conditions": conditions,
            "temp": temp,
            "precip": precip,
        }
    )
    save_event(proactive_event)

    # Format for speech
    message = f"Hey, ... heads up for school pickup. ... {', ... and '.join(message_parts)}."
    if "rain" in conditions or precip >= 60:
        message += " ... Grab an umbrella."

    from proactive.executor import _create_notification
    action = _create_notification(proactive_event, {
        "message": message,
        "priority": "normal"
    })

    if action:
        update_event_status(proactive_event.id, "actioned")
        notify_action(action)


def _check_event_weather(now: datetime, weather: dict):
    """Check weather for upcoming outdoor events."""
    conditions = weather.get("conditions", "").lower()
    precip = weather.get("precipitation_chance", 0)

    # Only alert if rain is likely
    if precip < 50 and "rain" not in conditions:
        return

    # Look at events in the next 4 hours
    window_end = now + timedelta(hours=4)
    events = list_events(now, window_end)

    # Keywords that suggest outdoor events
    outdoor_keywords = ["park", "playground", "outdoor", "field", "game", "match", "bbq", "picnic"]

    for event in events:
        title = event.get("title", "").lower()
        location = event.get("location", "").lower()
        combined = f"{title} {location}"

        if any(keyword in combined for keyword in outdoor_keywords):
            event_id = event.get("id", title)
            alert_id = f"event-weather-{event_id}-{now.strftime('%Y%m%d')}"

            if is_event_processed("weather", alert_id):
                continue

            proactive_event = ProactiveEvent(
                source_type="weather",
                source_id=alert_id,
                raw_data={
                    "type": "event_weather",
                    "event_title": event.get("title"),
                    "conditions": conditions,
                    "precip": precip,
                }
            )
            save_event(proactive_event)

            # Format for speech
            message = f"Hey, ... rain is likely for {event.get('title', 'your event')}. ... {precip} percent chance of precipitation."

            from proactive.executor import _create_notification
            action = _create_notification(proactive_event, {
                "message": message,
                "priority": "normal"
            })

            if action:
                update_event_status(proactive_event.id, "actioned")
                notify_action(action)


def get_weather_summary() -> str:
    """Get current weather summary (for briefing)."""
    weather = get_current_weather()
    if not weather:
        return "Weather data unavailable."

    temp = weather.get("temperature", 0)
    conditions = weather.get("conditions", "unknown")
    high = weather.get("high", temp)
    low = weather.get("low", temp)
    precip = weather.get("precipitation_chance", 0)

    summary = f"Currently {temp}°F and {conditions}. "
    summary += f"High of {high}, low of {low}. "

    if precip >= 50:
        summary += f"{precip}% chance of rain."
    elif precip >= 30:
        summary += "Slight chance of rain."

    return summary
