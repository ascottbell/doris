"""Database operations for the proactive system."""

import sqlite3
from datetime import datetime
from typing import Optional
from pathlib import Path

from .models import ProactiveEvent, ProactiveAction

DB_PATH = Path(__file__).parent.parent / "data" / "doris.db"


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_proactive_db():
    """Create proactive tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS proactive_events (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_id TEXT,
            raw_data TEXT NOT NULL,
            detected_at TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            evaluation TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS proactive_actions (
            id TEXT PRIMARY KEY,
            event_id TEXT REFERENCES proactive_events(id),
            action_type TEXT NOT NULL,
            action_data TEXT NOT NULL,
            result_id TEXT,
            executed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'completed',
            notification_sent INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS proactive_checkpoints (
            source_type TEXT PRIMARY KEY,
            last_check TEXT,
            last_id TEXT
        )
    """)

    # Indexes for common queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_status
        ON proactive_events(status)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_source
        ON proactive_events(source_type, detected_at)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_actions_event
        ON proactive_actions(event_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_actions_status
        ON proactive_actions(status, executed_at)
    """)

    conn.commit()
    conn.close()


def save_event(event: ProactiveEvent) -> str:
    """Save an event to the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO proactive_events
        (id, source_type, source_id, raw_data, detected_at, status, evaluation)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, event.to_db_row())

    conn.commit()
    conn.close()
    return event.id


def save_action(action: ProactiveAction) -> str:
    """Save an action to the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO proactive_actions
        (id, event_id, action_type, action_data, result_id, executed_at, status, notification_sent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, action.to_db_row())

    conn.commit()
    conn.close()
    return action.id


def get_pending_events(limit: int = 50) -> list[ProactiveEvent]:
    """Get events that haven't been evaluated yet."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, source_type, source_id, raw_data, detected_at, status, evaluation
        FROM proactive_events
        WHERE status = 'pending'
        ORDER BY detected_at ASC
        LIMIT ?
    """, (limit,))

    events = [ProactiveEvent.from_db_row(row) for row in cursor.fetchall()]
    conn.close()
    return events


def get_event(event_id: str) -> Optional[ProactiveEvent]:
    """Get a specific event by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, source_type, source_id, raw_data, detected_at, status, evaluation
        FROM proactive_events
        WHERE id = ?
    """, (event_id,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return ProactiveEvent.from_db_row(row)
    return None


def update_event_status(event_id: str, status: str, evaluation: dict = None):
    """Update an event's status and optionally its evaluation."""
    conn = get_connection()
    cursor = conn.cursor()

    import json
    if evaluation:
        cursor.execute("""
            UPDATE proactive_events
            SET status = ?, evaluation = ?
            WHERE id = ?
        """, (status, json.dumps(evaluation), event_id))
    else:
        cursor.execute("""
            UPDATE proactive_events
            SET status = ?
            WHERE id = ?
        """, (status, event_id))

    conn.commit()
    conn.close()


def get_recent_actions(limit: int = 10) -> list[ProactiveAction]:
    """Get recent actions for correction handling."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, event_id, action_type, action_data, result_id, executed_at, status, notification_sent
        FROM proactive_actions
        WHERE status = 'completed'
        ORDER BY executed_at DESC
        LIMIT ?
    """, (limit,))

    actions = [ProactiveAction.from_db_row(row) for row in cursor.fetchall()]
    conn.close()
    return actions


def update_action_status(action_id: str, status: str):
    """Update an action's status (e.g., marking as undone)."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE proactive_actions
        SET status = ?
        WHERE id = ?
    """, (status, action_id))

    conn.commit()
    conn.close()


def get_checkpoint(source_type: str) -> tuple[Optional[str], Optional[str]]:
    """Get the last check time and ID for a source."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT last_check, last_id
        FROM proactive_checkpoints
        WHERE source_type = ?
    """, (source_type,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return row[0], row[1]
    return None, None


def update_checkpoint(source_type: str, last_check: str = None, last_id: str = None):
    """Update the checkpoint for a source."""
    conn = get_connection()
    cursor = conn.cursor()

    if last_check is None:
        last_check = datetime.now().isoformat()

    cursor.execute("""
        INSERT OR REPLACE INTO proactive_checkpoints (source_type, last_check, last_id)
        VALUES (?, ?, ?)
    """, (source_type, last_check, last_id))

    conn.commit()
    conn.close()


def is_event_processed(source_type: str, source_id: str) -> bool:
    """Check if we've already processed this source event."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 1 FROM proactive_events
        WHERE source_type = ? AND source_id = ?
    """, (source_type, source_id))

    exists = cursor.fetchone() is not None
    conn.close()
    return exists
