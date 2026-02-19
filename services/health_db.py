"""
Health data storage and insights.

Stores Apple Health data in SQLite and generates insights for Doris memory.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "doris.db"


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_health_db():
    """Create health tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_daily (
            date TEXT PRIMARY KEY,
            steps INTEGER,
            active_calories INTEGER,
            stand_hours INTEGER,
            resting_hr INTEGER,
            hrv INTEGER,
            vo2_max REAL,
            sleep_hours REAL,
            sleep_stages TEXT,
            workouts TEXT,
            synced_at TEXT
        )
    """)

    # Add new columns if they don't exist (migration for existing DBs)
    try:
        cursor.execute("ALTER TABLE health_daily ADD COLUMN hrv INTEGER")
    except:
        pass
    try:
        cursor.execute("ALTER TABLE health_daily ADD COLUMN vo2_max REAL")
    except:
        pass
    try:
        cursor.execute("ALTER TABLE health_daily ADD COLUMN sleep_stages TEXT")
    except:
        pass

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_health_date
        ON health_daily(date DESC)
    """)

    conn.commit()
    conn.close()


def store_health_data(data: dict):
    """Store a day's health data and generate insights."""
    init_health_db()

    conn = get_connection()
    cursor = conn.cursor()

    workouts_json = json.dumps(data.get("workouts") or [])
    sleep_stages_json = json.dumps(data.get("sleep_stages")) if data.get("sleep_stages") else None

    cursor.execute("""
        INSERT OR REPLACE INTO health_daily
        (date, steps, active_calories, stand_hours, resting_hr, hrv, vo2_max, sleep_hours, sleep_stages, workouts, synced_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("date"),
        data.get("steps"),
        data.get("active_calories"),
        data.get("stand_hours"),
        data.get("resting_hr"),
        data.get("hrv"),
        data.get("vo2_max"),
        data.get("sleep_hours"),
        sleep_stages_json,
        workouts_json,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

    # Generate insights and store in memory
    _generate_insights(data)


def get_health_data(date: str = None, days: int = 7) -> dict | list:
    """Get health data for a specific date or last N days."""
    init_health_db()

    conn = get_connection()
    cursor = conn.cursor()

    if date:
        cursor.execute("""
            SELECT date, steps, active_calories, stand_hours, resting_hr, hrv, vo2_max, sleep_hours, sleep_stages, workouts
            FROM health_daily
            WHERE date = ?
        """, (date,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return _row_to_dict(row)
        return {}

    else:
        cursor.execute("""
            SELECT date, steps, active_calories, stand_hours, resting_hr, hrv, vo2_max, sleep_hours, sleep_stages, workouts
            FROM health_daily
            ORDER BY date DESC
            LIMIT ?
        """, (days,))
        rows = cursor.fetchall()
        conn.close()

        return [_row_to_dict(row) for row in rows]


def get_health_summary(days: int = 7) -> dict:
    """Get summary stats for the last N days (for briefings)."""
    data = get_health_data(days=days)

    if not data:
        return {"available": False}

    # Calculate averages
    steps = [d["steps"] for d in data if d.get("steps")]
    sleep = [d["sleep_hours"] for d in data if d.get("sleep_hours")]
    calories = [d["active_calories"] for d in data if d.get("active_calories")]

    # Count workouts
    total_workouts = sum(len(d.get("workouts") or []) for d in data)

    # Get today's data
    today = data[0] if data else {}

    return {
        "available": True,
        "today": today,
        "averages": {
            "steps": int(sum(steps) / len(steps)) if steps else None,
            "sleep_hours": round(sum(sleep) / len(sleep), 1) if sleep else None,
            "active_calories": int(sum(calories) / len(calories)) if calories else None,
        },
        "trends": {
            "workouts_this_week": total_workouts,
            "days_with_data": len(data),
        }
    }


def _row_to_dict(row) -> dict:
    """Convert a database row to a dictionary."""
    return {
        "date": row[0],
        "steps": row[1],
        "active_calories": row[2],
        "stand_hours": row[3],
        "resting_hr": row[4],
        "hrv": row[5],
        "vo2_max": row[6],
        "sleep_hours": row[7],
        "sleep_stages": json.loads(row[8]) if row[8] else None,
        "workouts": json.loads(row[9]) if row[9] else []
    }


def _generate_insights(data: dict):
    """Generate insights from health data and store in memory."""
    from memory.store import store_memory

    date = data.get("date")
    steps = data.get("steps")
    sleep = data.get("sleep_hours")
    sleep_stages = data.get("sleep_stages")
    workouts = data.get("workouts") or []
    resting_hr = data.get("resting_hr")
    hrv = data.get("hrv")
    vo2_max = data.get("vo2_max")

    insights = []

    # Sleep insight
    if sleep is not None:
        if sleep < 6:
            insights.append(f"User got only {sleep:.1f} hours of sleep on {date} - below recommended")
        elif sleep >= 8:
            insights.append(f"User got a solid {sleep:.1f} hours of sleep on {date}")

    # Sleep stages insight
    if sleep_stages:
        deep = sleep_stages.get("deep_hours", 0)
        rem = sleep_stages.get("rem_hours", 0)
        if deep < 0.5:
            insights.append(f"User got very little deep sleep on {date} ({deep:.1f}h) - may feel tired")
        if rem >= 2:
            insights.append(f"User got good REM sleep on {date} ({rem:.1f}h)")

    # HRV insight (stress/recovery indicator)
    if hrv is not None:
        if hrv < 30:
            insights.append(f"User's HRV was low ({hrv}ms) on {date} - may indicate stress or poor recovery")
        elif hrv > 60:
            insights.append(f"User's HRV was excellent ({hrv}ms) on {date} - well recovered")

    # Activity insight
    if steps is not None:
        if steps >= 10000:
            insights.append(f"User hit {steps:,} steps on {date} - great activity day")
        elif steps < 3000:
            insights.append(f"User only got {steps:,} steps on {date} - low activity day")

    # Workout insight
    if workouts:
        workout_types = [w.get("type", "workout") for w in workouts]
        insights.append(f"User did {len(workouts)} workout(s) on {date}: {', '.join(workout_types)}")

    # Store notable insights in memory (not every day's data)
    for insight in insights:
        store_memory(
            content=insight,
            category="health",
            subject="User",
            source="health:apple_health",
            confidence=1.0
        )

    if insights:
        print(f"[Health] Generated {len(insights)} insights for {date}")
