"""
Weather-Calendar Fusion for morning briefing.

Correlates today's events with hourly weather forecast to provide
actionable insights like "bring umbrella for school pickup at 4pm".
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def get_weather_calendar_fusion() -> dict:
    """
    Get weather alerts correlated with today's calendar events.

    Returns:
        dict with 'insights' list and 'summary' string
    """
    from tools.cal import list_events
    from tools.weather import get_current_weather

    insights = []

    # Get today's events
    now = datetime.now(EASTERN)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)

    events = list_events(today_start, today_end)

    # Get weather (includes hourly forecast)
    weather = get_current_weather()

    if not weather or "error" in weather:
        return {
            "insights": [],
            "summary": "Couldn't get weather data."
        }

    # Current conditions
    current_temp = weather.get("temperature", weather.get("current_temp"))
    conditions = weather.get("conditions", "")

    # Build hourly lookup if available
    hourly = weather.get("hourly", [])
    hourly_by_hour = {}
    for h in hourly:
        hour = h.get("hour")
        if hour is not None:
            hourly_by_hour[hour] = h

    # Check morning conditions (for school/bus)
    morning_hour = 8
    if morning_hour in hourly_by_hour:
        morning = hourly_by_hour[morning_hour]
        temp = morning.get("temp", current_temp)
        cond = morning.get("conditions", conditions)

        if temp and temp < 40:
            insights.append({
                "type": "cold_morning",
                "message": f"Cold morning ({int(temp)}°) - bundle up the kids for school",
                "priority": "high"
            })

        if "rain" in cond.lower() or "shower" in cond.lower():
            insights.append({
                "type": "rain_morning",
                "message": "Rain expected this morning - grab umbrellas for school drop-off",
                "priority": "high"
            })

    # Check each event for weather alerts
    for event in events:
        title = event.get("title", "")
        start_str = event.get("start", "")
        is_all_day = event.get("isAllDay", False)

        if is_all_day or not start_str:
            continue

        # Parse event time
        try:
            from dateutil import parser as dateparser
            event_dt = dateparser.parse(start_str)
            if event_dt:
                event_dt = event_dt.astimezone(EASTERN)
                event_hour = event_dt.hour
            else:
                continue
        except:
            continue

        # Check weather at event time
        if event_hour in hourly_by_hour:
            hour_weather = hourly_by_hour[event_hour]
            temp = hour_weather.get("temp")
            cond = hour_weather.get("conditions", "")
            precip = hour_weather.get("precipitation_chance", 0)

            time_str = event_dt.strftime("%-I %p").lower()

            # Rain alert
            if precip and precip > 40 or "rain" in cond.lower():
                # Check if this is a pickup/dropoff event
                is_transport = any(w in title.lower() for w in ["pick", "drop", "bus", "launch", "school"])

                if is_transport:
                    insights.append({
                        "type": "rain_event",
                        "message": f"Rain expected at {time_str} for {title} - bring umbrella",
                        "priority": "high",
                        "event": title,
                        "time": time_str
                    })
                else:
                    insights.append({
                        "type": "rain_event",
                        "message": f"Rain likely around {time_str} ({title})",
                        "priority": "medium",
                        "event": title,
                        "time": time_str
                    })

            # Cold alert for outdoor activities
            if temp and temp < 35:
                outdoor_words = ["soccer", "park", "playground", "outside", "walk", "bike"]
                if any(w in title.lower() for w in outdoor_words):
                    insights.append({
                        "type": "cold_event",
                        "message": f"Cold ({int(temp)}°) for {title} at {time_str} - dress warm",
                        "priority": "medium",
                        "event": title
                    })

            # Hot alert
            if temp and temp > 85:
                insights.append({
                    "type": "hot_event",
                    "message": f"Hot ({int(temp)}°) at {time_str} for {title} - stay hydrated",
                    "priority": "medium",
                    "event": title
                })

    # Build summary
    if not insights:
        summary = f"Weather looks fine for today's schedule. Currently {int(current_temp) if current_temp else 'N/A'}° and {conditions.lower()}."
    else:
        high_priority = [i for i in insights if i.get("priority") == "high"]
        if high_priority:
            summary = high_priority[0]["message"]
            if len(high_priority) > 1:
                summary += f" Plus {len(high_priority) - 1} other weather alert(s)."
        else:
            summary = insights[0]["message"]

    return {
        "insights": insights,
        "summary": summary,
        "current_temp": current_temp,
        "conditions": conditions,
        "event_count": len(events)
    }


def format_summary() -> str:
    """Format weather-calendar fusion as natural language."""
    data = get_weather_calendar_fusion()

    parts = []

    # Current conditions
    temp = data.get("current_temp")
    cond = data.get("conditions", "")
    if temp:
        parts.append(f"It's currently {int(temp)} degrees and {cond.lower()}.")

    # High priority insights
    insights = data.get("insights", [])
    high = [i for i in insights if i.get("priority") == "high"]

    for insight in high[:2]:  # Max 2 alerts
        parts.append(insight["message"] + ".")

    # Medium priority if no high
    if not high:
        medium = [i for i in insights if i.get("priority") == "medium"]
        for insight in medium[:1]:
            parts.append(insight["message"] + ".")

    if not parts:
        return "Weather looks good for today's schedule."

    return " ".join(parts)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    result = get_weather_calendar_fusion()
    print("Weather-Calendar Fusion")
    print("=" * 40)
    print(f"Summary: {result['summary']}")
    print(f"\nInsights ({len(result['insights'])}):")
    for i in result["insights"]:
        print(f"  [{i['priority']}] {i['message']}")

    print(f"\nFormatted: {format_summary()}")
