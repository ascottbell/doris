"""
Doris Conversation Sync API

Provides server-side chat history storage and sync for iOS/macOS clients.
Uses local SQLite for persistence — zero network dependency, sub-ms reads.
"""

import json
import sqlite3
import uuid
import logging
from datetime import datetime, timedelta
from typing import Literal, Optional
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)

# Database path — same file used by memory/store.py
DB_PATH = settings.data_dir / "doris.db"


def _get_db() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode and row_factory."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_conversations_db():
    """
    Create conversations and chat_messages tables if they don't exist.

    Call once at server startup. Safe to call multiple times.
    """
    db = _get_db()
    try:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_message_at TEXT DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_device_id
                ON conversations(device_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_last_message
                ON conversations(last_message_at);

            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                content TEXT NOT NULL,
                role TEXT NOT NULL,
                device_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation
                ON chat_messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at
                ON chat_messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_device_id
                ON chat_messages(device_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chat_messages_fts USING fts5(
                content,
                content=chat_messages,
                content_rowid=rowid
            );

            -- Trigger: keep FTS in sync on INSERT
            CREATE TRIGGER IF NOT EXISTS chat_messages_ai
                AFTER INSERT ON chat_messages
            BEGIN
                INSERT INTO chat_messages_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;

            -- Trigger: keep FTS in sync on DELETE
            CREATE TRIGGER IF NOT EXISTS chat_messages_ad
                AFTER DELETE ON chat_messages
            BEGIN
                INSERT INTO chat_messages_fts(chat_messages_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;

            -- Trigger: keep FTS in sync on UPDATE
            CREATE TRIGGER IF NOT EXISTS chat_messages_au
                AFTER UPDATE ON chat_messages
            BEGIN
                INSERT INTO chat_messages_fts(chat_messages_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO chat_messages_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;
        """)
        db.commit()
        logger.info("Conversations DB initialized (SQLite, WAL mode)")
    finally:
        db.close()


# =============================================================================
# Pydantic Models
# =============================================================================

class MessageCreate(BaseModel):
    """Request to save a new message."""
    id: str  # UUID from client
    content: str
    role: Literal["user", "assistant"]
    device_id: str
    created_at: Optional[str] = None  # ISO timestamp
    conversation_id: Optional[str] = None
    metadata: Optional[dict] = None


class MessageResponse(BaseModel):
    """Message returned from API."""
    id: str
    content: str
    role: str
    device_id: Optional[str]
    created_at: str
    conversation_id: Optional[str]
    metadata: Optional[dict]


class MessagesResponse(BaseModel):
    """Response containing multiple messages."""
    messages: list[MessageResponse]
    count: int
    has_more: bool


class SearchResult(BaseModel):
    """Search result with relevance info."""
    id: str
    content: str
    role: str
    created_at: str
    rank: Optional[float] = None


# =============================================================================
# Internal helpers
# =============================================================================

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict, parsing JSON metadata."""
    d = dict(row)
    if "metadata" in d and isinstance(d["metadata"], str):
        try:
            d["metadata"] = json.loads(d["metadata"])
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
    return d


# =============================================================================
# Conversation Management
# =============================================================================

def _get_or_create_conversation_in_tx(
    db: sqlite3.Connection, device_id: str, metadata: Optional[dict] = None
) -> str:
    """
    Get the active conversation for a device, or create a new one.
    Operates within an existing connection/transaction — caller manages commit.

    A new conversation is created if:
    - No existing conversation for this device
    - Last message was more than 1 hour ago
    """
    one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()

    row = db.execute(
        """SELECT id FROM conversations
           WHERE device_id = ? AND last_message_at >= ?
           ORDER BY last_message_at DESC LIMIT 1""",
        (device_id, one_hour_ago),
    ).fetchone()

    if row:
        return row["id"]

    # Create new conversation
    conv_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    meta_json = json.dumps(metadata) if metadata else None

    db.execute(
        """INSERT INTO conversations (id, device_id, started_at, last_message_at, message_count, metadata)
           VALUES (?, ?, ?, ?, 0, ?)""",
        (conv_id, device_id, now, now, meta_json),
    )
    return conv_id


def get_or_create_conversation(device_id: str, metadata: Optional[dict] = None) -> str:
    """
    Get the active conversation for a device, or create a new one.
    Public wrapper that manages its own connection. Prefer _get_or_create_conversation_in_tx
    when you already hold a connection (e.g., inside save_message).
    """
    db = _get_db()
    try:
        conv_id = _get_or_create_conversation_in_tx(db, device_id, metadata)
        db.commit()
        return conv_id
    finally:
        db.close()


# =============================================================================
# Message Operations
# =============================================================================

def save_message(message: MessageCreate) -> dict:
    """
    Save a new message to the database.

    All operations (get/create conversation, insert message, update counter)
    run in a single connection and transaction for atomicity.

    Returns the saved message with server-assigned conversation_id if needed.
    """
    db = _get_db()
    try:
        # Get or create conversation within the same connection
        conv_id = message.conversation_id
        if not conv_id:
            conv_id = _get_or_create_conversation_in_tx(
                db, message.device_id, message.metadata
            )

        created_at = message.created_at or datetime.utcnow().isoformat()
        meta_json = json.dumps(message.metadata) if message.metadata else None

        db.execute(
            """INSERT OR REPLACE INTO chat_messages
               (id, conversation_id, content, role, device_id, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (message.id, conv_id, message.content, message.role,
             message.device_id, created_at, meta_json),
        )

        # Update conversation stats
        db.execute(
            """UPDATE conversations
               SET last_message_at = ?, message_count = message_count + 1
               WHERE id = ?""",
            (created_at, conv_id),
        )

        # Single commit for all three operations
        db.commit()

        return {
            "id": message.id,
            "conversation_id": conv_id,
            "content": message.content,
            "role": message.role,
            "device_id": message.device_id,
            "created_at": created_at,
            "metadata": message.metadata,
        }
    finally:
        db.close()


def get_messages(
    since: Optional[str] = None,
    limit: int = 100,
    device_id: Optional[str] = None,
) -> tuple[list[dict], bool]:
    """
    Get messages, optionally filtered by timestamp and device.

    Args:
        since: ISO timestamp to fetch messages after
        limit: Maximum number of messages to return
        device_id: Optional filter for specific device

    Returns:
        Tuple of (messages, has_more)
    """
    db = _get_db()
    try:
        clauses = []
        params: list = []

        if since:
            clauses.append("created_at > ?")
            params.append(since)

        if device_id:
            clauses.append("device_id = ?")
            params.append(device_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit + 1)  # fetch one extra for has_more

        rows = db.execute(
            f"""SELECT id, content, role, device_id, created_at, conversation_id, metadata
                FROM chat_messages
                {where}
                ORDER BY created_at ASC
                LIMIT ?""",
            params,
        ).fetchall()

        messages = [_row_to_dict(r) for r in rows]
        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]

        return messages, has_more
    finally:
        db.close()


def get_recent_messages(limit: int = 50) -> list[dict]:
    """Get the most recent messages across all devices."""
    db = _get_db()
    try:
        rows = db.execute(
            """SELECT id, content, role, device_id, created_at, conversation_id, metadata
               FROM chat_messages
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

        # Return in chronological order (oldest first)
        return [_row_to_dict(r) for r in reversed(rows)]
    finally:
        db.close()


def _quote_fts5_query(query: str) -> str:
    """
    Quote a user-provided search string for safe use in FTS5 MATCH.

    FTS5 has its own query syntax (AND, OR, NOT, NEAR, *, column filters).
    Wrapping each token in double quotes treats it as a literal phrase and
    prevents syntax injection.
    """
    # Escape any double quotes inside the query, then wrap in quotes
    escaped = query.replace('"', '""')
    return f'"{escaped}"'


def _escape_like(query: str) -> str:
    """
    Escape LIKE wildcards in user input.

    Escapes %, _, and the escape character itself so user input
    is treated as literal text in a LIKE clause.
    """
    return query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_messages(query: str, limit: int = 20) -> list[dict]:
    """
    Search messages using FTS5 full-text search.

    Returns messages matching the query, ordered by relevance.
    """
    db = _get_db()
    try:
        fts_query = _quote_fts5_query(query)
        rows = db.execute(
            """SELECT m.id, m.content, m.role, m.created_at,
                      rank AS rank
               FROM chat_messages_fts fts
               JOIN chat_messages m ON m.rowid = fts.rowid
               WHERE chat_messages_fts MATCH ?
               ORDER BY fts.rank
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()

        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        logger.error(f"FTS search failed for query '{query}': {e}")
        # Fallback: LIKE search with escaped wildcards
        escaped = _escape_like(query)
        rows = db.execute(
            """SELECT id, content, role, created_at, NULL as rank
               FROM chat_messages
               WHERE content LIKE ? ESCAPE '\\'
               ORDER BY created_at DESC
               LIMIT ?""",
            (f"%{escaped}%", limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        db.close()


def delete_all_messages() -> int:
    """
    Delete all messages and conversations.

    Returns count of deleted messages.
    """
    db = _get_db()
    try:
        count = db.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0]

        db.execute("DELETE FROM chat_messages")
        db.execute("DELETE FROM conversations")
        db.commit()

        return count
    finally:
        db.close()


def get_conversation_for_api(limit: int = 20) -> list[dict]:
    """
    Get recent conversation history formatted for the Claude API.

    Returns messages in the format expected by the chat API:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    messages = get_recent_messages(limit=limit)
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]
