"""
Morning Briefing Orchestrator.

Combines all briefing components into a cohesive morning update:
1. Weather-Calendar fusion (actionable weather alerts)
2. News digest (4 curated stories)
3. Deliveries (expected packages)
4. Kids corner (fun content for kids)
5. Health check-in (if data available)
6. Calendar summary
"""

from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def get_morning_briefing(include_news: bool = True, include_kids: bool = True) -> dict:
    """
    Get the full morning briefing.

    Args:
        include_news: Whether to fetch news (slower, costs API)
        include_kids: Whether to include kids content

    Returns:
        dict with all briefing components and formatted output
    """
    from .weather_calendar import get_weather_calendar_fusion
    from .deliveries import get_expected_deliveries

    briefing = {
        "timestamp": datetime.now(EASTERN).isoformat(),
        "components": {}
    }

    # 1. Weather-Calendar Fusion (always include - fast)
    try:
        weather_cal = get_weather_calendar_fusion()
        briefing["components"]["weather_calendar"] = weather_cal
    except Exception as e:
        briefing["components"]["weather_calendar"] = {"error": str(e)}

    # 2. News Digest (optional - slower)
    if include_news:
        try:
            from .news_digest import get_news_digest
            news = get_news_digest()
            briefing["components"]["news"] = news
        except Exception as e:
            briefing["components"]["news"] = {"error": str(e)}

    # 3. Deliveries (always include - fast)
    try:
        deliveries = get_expected_deliveries()
        briefing["components"]["deliveries"] = deliveries
    except Exception as e:
        briefing["components"]["deliveries"] = {"error": str(e)}

    # 4. Kids Corner (optional)
    if include_kids:
        try:
            from .kids_corner import get_kids_content
            kids = get_kids_content()
            briefing["components"]["kids"] = kids
        except Exception as e:
            briefing["components"]["kids"] = {"error": str(e)}

    # 5. Health check-in (if available)
    try:
        from services.health_db import get_health_summary
        health = get_health_summary(days=1)
        if health.get("available"):
            briefing["components"]["health"] = health
    except Exception as e:
        pass  # Health is optional

    # 6. Calendar summary
    try:
        from tools.cal import list_events
        today = datetime.now(EASTERN)
        today_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timedelta
        today_end = today_start + timedelta(days=1)

        events = list_events(today_start, today_end)
        briefing["components"]["calendar"] = {
            "event_count": len(events),
            "events": events[:10]  # Cap at 10
        }
    except Exception as e:
        briefing["components"]["calendar"] = {"error": str(e)}

    return briefing


def format_summary(briefing: dict = None, style: str = "full") -> str:
    """
    Format the briefing as natural language.

    Args:
        briefing: Pre-fetched briefing or None to fetch fresh
        style: "full" for complete briefing, "quick" for highlights only

    Returns:
        Natural language summary string
    """
    if briefing is None:
        briefing = get_morning_briefing(
            include_news=(style == "full"),
            include_kids=(style == "full")
        )

    parts = []
    components = briefing.get("components", {})

    # Greeting
    hour = datetime.now(EASTERN).hour
    if hour < 12:
        parts.append("Good morning.")
    else:
        parts.append("Here's your briefing.")

    # Weather-Calendar (priority - actionable)
    wc = components.get("weather_calendar", {})
    if wc and "error" not in wc:
        insights = wc.get("insights", [])
        high_priority = [i for i in insights if i.get("priority") == "high"]

        if high_priority:
            for insight in high_priority[:2]:
                parts.append(insight["message"] + ".")
        else:
            temp = wc.get("current_temp")
            cond = wc.get("conditions", "")
            if temp:
                parts.append(f"It's {int(temp)} degrees and {cond.lower()}.")

    # Calendar summary
    cal = components.get("calendar", {})
    if cal and "error" not in cal:
        count = cal.get("event_count", 0)
        if count > 0:
            parts.append(f"You have {count} event{'s' if count > 1 else ''} today.")

    # Deliveries
    deliveries = components.get("deliveries", {})
    if deliveries and "error" not in deliveries:
        today_count = deliveries.get("today_count", 0)
        if today_count > 0:
            parts.append(f"{today_count} package{'s' if today_count > 1 else ''} arriving today.")

    # Health (if available and notable)
    health = components.get("health", {})
    if health and health.get("available"):
        today_data = health.get("today", {})
        sleep = today_data.get("sleep_hours")
        if sleep and sleep < 6:
            parts.append(f"You only got {sleep:.1f} hours of sleep. Take it easy today.")

    # News (if full style)
    if style == "full":
        news = components.get("news", {})
        if news and "error" not in news:
            stories = news.get("stories", [])
            if stories:
                parts.append(f"I have {len(stories)} news stories for you.")
                # Just mention politics headline
                politics = next((s for s in stories if s["category"] == "politics"), None)
                if politics:
                    title = politics["title"][:80]
                    parts.append(f"In politics: {title}.")

    # Kids (if full style)
    if style == "full":
        kids = components.get("kids", {})
        if kids and "error" not in kids:
            parts.append("I also have a fun fact and joke for the kids when they're ready.")

    return " ".join(parts)


def get_quick_briefing() -> str:
    """Get a quick briefing - just weather, calendar, deliveries."""
    briefing = get_morning_briefing(include_news=False, include_kids=False)
    return format_summary(briefing, style="quick")


def get_full_briefing() -> str:
    """Get the full morning briefing with everything."""
    briefing = get_morning_briefing(include_news=True, include_kids=True)
    return format_summary(briefing, style="full")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Fetching morning briefing...")
    print("=" * 60)

    briefing = get_morning_briefing(include_news=True, include_kids=True)

    print("\n📋 COMPONENTS:")
    for name, data in briefing["components"].items():
        if "error" in data:
            print(f"  ❌ {name}: {data['error']}")
        else:
            print(f"  ✅ {name}")

    print("\n" + "=" * 60)
    print("BRIEFING (Full):")
    print("=" * 60)
    print(format_summary(briefing, style="full"))

    print("\n" + "=" * 60)
    print("BRIEFING (Quick):")
    print("=" * 60)
    print(format_summary(briefing, style="quick"))
