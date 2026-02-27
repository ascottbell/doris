"""
Behavioral pattern analysis for Doris memory.

Analyzes query logs to learn:
- What the user asks for most (to make proactive)
- Time-of-day patterns (morning vs evening preferences)
- Query category distribution
"""

import sqlite3
from datetime import datetime
from collections import defaultdict

from config import settings

DB_PATH = settings.data_dir / "doris.db"


def analyze_query_patterns() -> dict:
    """
    Analyze query logs for patterns.

    Returns dict with category stats, time patterns, and insights.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all queries
    cursor.execute("""
        SELECT message, timestamp, route
        FROM request_log
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {"error": "No query logs found"}

    # Category classification
    def categorize(msg: str) -> str:
        msg = msg.lower()
        if 'weather' in msg:
            return 'weather'
        if 'time' in msg and 'what' in msg:
            return 'time'
        if any(w in msg for w in ['calendar', 'schedule', 'tomorrow', "what's on", 'events']):
            return 'calendar'
        if 'light' in msg:
            return 'lights'
        if any(w in msg for w in ['play', 'music', 'song']):
            return 'music'
        if 'email' in msg:
            return 'email'
        if 'remind' in msg:
            return 'reminders'
        if any(w in msg for w in ['message', 'text', 'imessage']):
            return 'messaging'
        if 'brief' in msg:
            return 'briefing'
        if any(w in msg for w in ['shopping', 'grocery', 'list']):
            return 'shopping'
        return 'other'

    # Analyze categories
    category_counts = defaultdict(int)
    for msg, ts, route in rows:
        cat = categorize(msg)
        category_counts[cat] += 1

    # Analyze time of day (convert UTC to EST, -5 hours)
    time_patterns = defaultdict(lambda: defaultdict(int))
    for msg, ts, route in rows:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            # Convert to EST (rough approximation)
            hour_est = (dt.hour - 5) % 24
            cat = categorize(msg)

            if 5 <= hour_est < 12:
                period = 'morning'
            elif 12 <= hour_est < 17:
                period = 'afternoon'
            elif 17 <= hour_est < 21:
                period = 'evening'
            else:
                period = 'night'

            time_patterns[period][cat] += 1
        except:
            pass

    # Generate insights
    insights = []

    # Top categories
    sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])
    top_cats = sorted_cats[:5]
    insights.append(f"User's most common queries: {', '.join(f'{c} ({n})' for c,n in top_cats)}")

    # Time-of-day patterns
    for period, cats in time_patterns.items():
        if cats:
            top_cat = max(cats.items(), key=lambda x: x[1])
            if top_cat[1] >= 3:  # Only if significant
                insights.append(f"In the {period}, the user often asks about {top_cat[0]}")

    # Proactive opportunities (things asked > 10 times)
    proactive_candidates = [cat for cat, count in category_counts.items() if count >= 10 and cat != 'other']
    if proactive_candidates:
        insights.append(f"Could make proactive: {', '.join(proactive_candidates)}")

    return {
        'total_queries': len(rows),
        'category_counts': dict(category_counts),
        'time_patterns': {k: dict(v) for k, v in time_patterns.items()},
        'insights': insights,
        'top_categories': top_cats
    }


def store_behavioral_patterns():
    """Analyze behavior and store patterns in memory."""
    from memory.store import store_memory

    analysis = analyze_query_patterns()

    if "error" in analysis:
        print(f"[BehavioralPatterns] {analysis['error']}")
        return analysis

    stored = 0

    # Store top query pattern
    top_cats = analysis.get('top_categories', [])
    if top_cats:
        top = top_cats[0]
        content = f"User most frequently asks Doris about {top[0]} ({top[1]} times)"
        store_memory(
            content=content,
            category="preference",
            subject="User",
            source="doris:behavioral_analysis",
            confidence=0.9
        )
        stored += 1
        print(f"[BehavioralPatterns] Stored: {content}")

    # Store time-of-day patterns
    for insight in analysis.get('insights', []):
        if 'In the ' in insight:  # Time-based insight
            store_memory(
                content=insight,
                category="preference",
                subject="User",
                source="doris:behavioral_analysis",
                confidence=0.85
            )
            stored += 1
            print(f"[BehavioralPatterns] Stored: {insight}")

    # Store proactive candidates
    for insight in analysis.get('insights', []):
        if 'Could make proactive' in insight:
            store_memory(
                content=insight,
                category="preference",
                subject="Doris",
                source="doris:behavioral_analysis",
                confidence=0.8
            )
            stored += 1
            print(f"[BehavioralPatterns] Stored: {insight}")

    print(f"[BehavioralPatterns] Analyzed {analysis['total_queries']} queries, stored {stored} patterns")
    return analysis


if __name__ == "__main__":
    result = store_behavioral_patterns()

    print(f"\n=== Behavioral Analysis Summary ===")
    print(f"Total queries: {result.get('total_queries', 0)}")

    print("\nCategory distribution:")
    for cat, count in sorted(result.get('category_counts', {}).items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print("\nTime-of-day patterns:")
    for period, cats in result.get('time_patterns', {}).items():
        print(f"  {period}: {cats}")

    print("\nInsights:")
    for insight in result.get('insights', []):
        print(f"  - {insight}")
