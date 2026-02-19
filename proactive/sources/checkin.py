"""
Check-in Source - Proactive outreach to the user.

Doris initiates contact with contextual, helpful messages.
Not robotic reminders - genuine, human-like check-ins.
"""

import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

EASTERN = ZoneInfo("America/New_York")

# Track last check-in to avoid being annoying
_last_checkin: Optional[datetime] = None
_daily_checkin_count = 0
_last_checkin_date: Optional[str] = None

# Limits
MAX_CHECKINS_PER_DAY = 3
MIN_HOURS_BETWEEN_CHECKINS = 4


def should_checkin() -> bool:
    """Decide if we should do a check-in right now."""
    global _daily_checkin_count, _last_checkin_date, _last_checkin

    now = datetime.now(EASTERN)
    today = now.strftime("%Y-%m-%d")

    # Reset daily counter
    if _last_checkin_date != today:
        _daily_checkin_count = 0
        _last_checkin_date = today

    # Don't exceed daily limit
    if _daily_checkin_count >= MAX_CHECKINS_PER_DAY:
        return False

    # Respect minimum interval
    if _last_checkin:
        hours_since = (now - _last_checkin).total_seconds() / 3600
        if hours_since < MIN_HOURS_BETWEEN_CHECKINS:
            return False

    # Don't check in during quiet hours (10pm - 8am)
    hour = now.hour
    if hour >= 22 or hour < 8:
        return False

    # Random chance (don't check in every time the monitor runs)
    # Higher chance mid-day, lower morning/evening
    if 10 <= hour <= 16:
        chance = 0.3  # 30% chance during prime hours
    else:
        chance = 0.15  # 15% chance other times

    return random.random() < chance


def get_checkin_context() -> dict:
    """Gather context for a meaningful check-in."""
    context = {
        "time": datetime.now(EASTERN),
        "day_of_week": datetime.now(EASTERN).strftime("%A"),
        "weather": None,
        "home_temp": None,
        "pending_followups": [],
        "recent_topics": [],
        "calendar_today": [],
    }

    # Get weather
    try:
        from tools.weather import get_current_weather
        weather = get_current_weather()
        if weather:
            context["weather"] = {
                "temp": weather.get("temperature"),
                "conditions": weather.get("conditions"),
                "description": weather.get("description"),
            }
    except Exception as e:
        print(f"[checkin] Weather fetch failed: {e}")

    # Get pending follow-ups from memory
    try:
        from memory.store import find_similar_memories
        followups = find_similar_memories("follow up get back to you pending", limit=5)
        for f in followups:
            if "pending" in f.get("content", "").lower():
                context["pending_followups"].append(f.get("content"))
    except Exception:
        pass

    # Get recent conversation topics
    try:
        from memory.store import find_similar_memories
        recent = find_similar_memories("discussed today conversation", limit=3)
        for r in recent:
            context["recent_topics"].append(r.get("content", "")[:100])
    except Exception:
        pass

    # Get today's calendar
    try:
        from tools.cal import get_todays_events
        events = get_todays_events()
        for e in events[:3]:
            context["calendar_today"].append(e.get("title", ""))
    except Exception:
        pass

    return context


def generate_checkin_message(context: dict) -> Optional[str]:
    """Use Claude to generate a natural, contextual check-in message."""
    from llm.api_client import call_claude
    from llm.providers import resolve_model

    now = context["time"]

    prompt = f"""You are Doris, a personal AI assistant. Generate a SHORT proactive check-in message.

Current time: {now.strftime("%A, %B %d at %I:%M %p")}

Context:
- Weather: {context.get('weather') or 'Unknown'}
- Home temperature: {context.get('home_temp') or 'Unknown'}°F
- Pending follow-ups: {context.get('pending_followups') or 'None'}
- Recent topics discussed: {context.get('recent_topics') or 'None'}
- Today's calendar: {context.get('calendar_today') or 'Nothing scheduled'}

Generate ONE short, natural message (1-2 sentences max). Choose from these styles:
1. Weather comment + suggestion ("Beautiful day - might be worth a walk later")
2. Day-of-week observation ("Happy Friday! Got any fun plans?")
3. Follow-up on pending item (if any exist)
4. Just checking in ("Hey, just making sure you're good")
5. Home alert if temp is extreme (below 50°F or above 85°F)
6. Interesting observation about the day/week

Rules:
- Be casual and warm, not robotic
- Don't ask yes/no questions that need a response
- If it's cold outside, mention it playfully
- If temp at home is below 50, that's urgent - mention heating
- Don't repeat the same style twice in a row
- NO emojis
- Keep it SHORT

Just output the message, nothing else."""

    try:
        response = call_claude(
            messages=[{"role": "user", "content": prompt}],
            source="proactive-checkin",
            model=resolve_model("mid"),
            max_tokens=150,
        )
        return response.text.strip()
    except Exception as e:
        print(f"[checkin] Message generation failed: {e}")
        return None


def monitor_checkin():
    """
    Main monitor function - called by scheduler.

    Decides whether to check in and generates contextual message.
    """
    global _last_checkin, _daily_checkin_count

    if not should_checkin():
        return

    print("[checkin] Initiating proactive check-in...")

    context = get_checkin_context()

    # Urgent: Home temp below 50
    if context.get("home_temp") and context["home_temp"] < 50:
        message = f"Hey - heads up, the house is at {context['home_temp']}°F. You might want to check the heat."
    else:
        message = generate_checkin_message(context)

    if not message:
        return

    # Send as proactive notification
    try:
        from proactive.models import ProactiveAction
        from proactive.notifier import notify_action
        from proactive.db import save_action

        action = ProactiveAction(
            event_id="checkin_" + datetime.now(EASTERN).strftime("%Y%m%d_%H%M%S"),
            action_type="notify",
            action_data={"message": message, "priority": "low"},
            status="completed"
        )
        save_action(action)
        notify_action(action)

        _last_checkin = datetime.now(EASTERN)
        _daily_checkin_count += 1

        print(f"[checkin] Sent: {message[:50]}...")

    except Exception as e:
        print(f"[checkin] Failed to send: {e}")


def get_monitor():
    """Return monitor config for scheduler registration."""
    return ("checkin", monitor_checkin, 60)  # Check every 60 minutes
