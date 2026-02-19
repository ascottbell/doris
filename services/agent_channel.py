"""
Agent-to-Agent Communication Channel

Enables structured communication between Claude Code and Doris.
Messages are persisted so either agent can catch up on conversations.

Message types:
- notification: One-way alert, no response expected
- question: Asking for input, expects response
- handoff: Passing context/task to the other agent
- update: Status update on ongoing work
- chat: General conversation
"""

import hashlib
import hmac
import json
import logging
import sqlite3
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Agent(str, Enum):
    CLAUDE_CODE = "claude_code"
    DORIS = "doris"


class MessageType(str, Enum):
    NOTIFICATION = "notification"
    QUESTION = "question"
    HANDOFF = "handoff"
    UPDATE = "update"
    CHAT = "chat"


class AgentMessage(BaseModel):
    from_agent: Agent
    to_agent: Agent
    message_type: MessageType
    content: str
    context: Optional[dict] = None  # Additional structured data
    priority: str = "normal"  # low, normal, high, urgent
    expects_response: bool = False
    related_to: Optional[int] = None  # ID of message this responds to


class StoredMessage(BaseModel):
    id: int
    from_agent: str
    to_agent: str
    message_type: str
    content: str
    context: Optional[dict]
    priority: str
    expects_response: bool
    related_to: Optional[int]
    created_at: str
    responded: bool
    response_id: Optional[int]


# Database setup
DB_PATH = Path(__file__).parent.parent / "data" / "agent_channel.db"


def _get_hmac_key() -> Optional[bytes]:
    """Get HMAC signing key from DORIS_API_TOKEN (first token if comma-separated)."""
    from config import settings
    token = settings.doris_api_token
    if not token:
        return None
    # Use only the primary token for HMAC (first if comma-separated for rotation)
    primary = token.split(",")[0].strip()
    if not primary:
        return None
    return primary.encode("utf-8")


def _compute_hmac(from_agent: str, to_agent: str, message_type: str,
                  content: str, created_at: str) -> str:
    """Compute HMAC-SHA256 over the canonical message fields."""
    key = _get_hmac_key()
    if key is None:
        return ""
    # Canonical form: fields joined by null byte (unambiguous separator)
    payload = "\0".join([from_agent, to_agent, message_type, content, created_at])
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _verify_hmac(row: sqlite3.Row) -> bool:
    """Verify HMAC signature on a stored message. Returns True if valid or unsigned."""
    sig = row["hmac_signature"] if "hmac_signature" in row.keys() else None
    if not sig:
        # Unsigned legacy message — log but allow (migration period)
        return True
    key = _get_hmac_key()
    if key is None:
        # No key configured — can't verify, allow but warn
        logger.warning("Agent channel: HMAC signature present but no signing key configured")
        return True
    expected = _compute_hmac(
        row["from_agent"], row["to_agent"], row["message_type"],
        row["content"], row["created_at"],
    )
    if hmac.compare_digest(sig, expected):
        return True
    logger.warning(
        f"Agent channel: HMAC verification FAILED for message {row['id']} "
        f"from {row['from_agent']} — possible tampering"
    )
    return False


def _get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Initialize the database schema."""
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_agent TEXT NOT NULL,
            to_agent TEXT NOT NULL,
            message_type TEXT NOT NULL,
            content TEXT NOT NULL,
            context TEXT,
            priority TEXT DEFAULT 'normal',
            expects_response BOOLEAN DEFAULT FALSE,
            related_to INTEGER,
            created_at TEXT NOT NULL,
            responded BOOLEAN DEFAULT FALSE,
            response_id INTEGER,
            hmac_signature TEXT DEFAULT '',
            FOREIGN KEY (related_to) REFERENCES messages(id),
            FOREIGN KEY (response_id) REFERENCES messages(id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_to_agent
        ON messages(to_agent, responded, created_at)
    """)
    # Migrate: add hmac_signature column if missing (existing DBs)
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN hmac_signature TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()


# Initialize on import
_init_db()


def send_message(message: AgentMessage) -> int:
    """
    Send a message from one agent to another.
    Returns the message ID.
    """
    created_at = datetime.now().isoformat()
    signature = _compute_hmac(
        message.from_agent.value,
        message.to_agent.value,
        message.message_type.value,
        message.content,
        created_at,
    )

    conn = _get_db()
    cursor = conn.execute("""
        INSERT INTO messages
        (from_agent, to_agent, message_type, content, context, priority,
         expects_response, related_to, created_at, responded, hmac_signature)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE, ?)
    """, (
        message.from_agent.value,
        message.to_agent.value,
        message.message_type.value,
        message.content,
        json.dumps(message.context) if message.context else None,
        message.priority,
        message.expects_response,
        message.related_to,
        created_at,
        signature,
    ))
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"[AgentChannel] {message.from_agent.value} → {message.to_agent.value}: {message.content[:50]}...")
    return message_id


def get_pending_messages(for_agent: Agent, limit: int = 20) -> list[StoredMessage]:
    """Get messages waiting for this agent that haven't been responded to."""
    conn = _get_db()
    rows = conn.execute("""
        SELECT * FROM messages
        WHERE to_agent = ? AND expects_response = TRUE AND responded = FALSE
        ORDER BY
            CASE priority
                WHEN 'urgent' THEN 0
                WHEN 'high' THEN 1
                WHEN 'normal' THEN 2
                WHEN 'low' THEN 3
            END,
            created_at ASC
        LIMIT ?
    """, (for_agent.value, limit)).fetchall()
    conn.close()

    return [msg for row in rows if (msg := _row_to_message(row)) is not None]


def get_agent_messages(
    agent: Optional[Agent] = None,
    limit: int = 50,
    include_all: bool = False
) -> list[StoredMessage]:
    """
    Get recent messages.
    If agent is specified, gets messages to/from that agent.
    """
    conn = _get_db()

    if agent and not include_all:
        rows = conn.execute("""
            SELECT * FROM messages
            WHERE from_agent = ? OR to_agent = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (agent.value, agent.value, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM messages
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

    conn.close()
    return [msg for row in rows if (msg := _row_to_message(row)) is not None]


def get_conversation_thread(message_id: int) -> list[StoredMessage]:
    """Get a conversation thread starting from a message."""
    conn = _get_db()

    # Get the root message
    messages = []
    current_id = message_id

    # Walk up to find the root
    while current_id:
        row = conn.execute(
            "SELECT * FROM messages WHERE id = ?",
            (current_id,)
        ).fetchone()
        if row:
            msg = _row_to_message(row)
            if msg is not None:
                messages.insert(0, msg)
            current_id = row['related_to']
        else:
            break

    # Now get all responses
    root_id = messages[0].id if messages else message_id

    def get_responses(parent_id):
        rows = conn.execute(
            "SELECT * FROM messages WHERE related_to = ? ORDER BY created_at",
            (parent_id,)
        ).fetchall()
        for row in rows:
            msg = _row_to_message(row)
            if msg is not None:
                messages.append(msg)
                get_responses(msg.id)

    get_responses(root_id)
    conn.close()

    return messages


def mark_responded(message_id: int, response_id: int):
    """Mark a message as responded to."""
    conn = _get_db()
    conn.execute("""
        UPDATE messages SET responded = TRUE, response_id = ?
        WHERE id = ?
    """, (response_id, message_id))
    conn.commit()
    conn.close()


def respond_to_message(original_id: int, response: AgentMessage) -> int:
    """
    Send a response to a message.
    Automatically sets related_to and marks original as responded.
    """
    response.related_to = original_id
    response_id = send_message(response)
    mark_responded(original_id, response_id)
    return response_id


def _row_to_message(row: sqlite3.Row) -> Optional[StoredMessage]:
    """Convert a database row to a StoredMessage.

    Returns None if HMAC verification fails (message tampered with).
    """
    if not _verify_hmac(row):
        # Tampered message — skip it entirely
        return None

    return StoredMessage(
        id=row['id'],
        from_agent=row['from_agent'],
        to_agent=row['to_agent'],
        message_type=row['message_type'],
        content=row['content'],
        context=json.loads(row['context']) if row['context'] else None,
        priority=row['priority'],
        expects_response=bool(row['expects_response']),
        related_to=row['related_to'],
        created_at=row['created_at'],
        responded=bool(row['responded']),
        response_id=row['response_id'],
    )


# Convenience functions for common operations

def notify_user_via_doris(message: str, context: Optional[dict] = None) -> int:
    """
    Claude Code asking Doris to notify the user.
    This is the simple "I need attention" flow.
    """
    return send_message(AgentMessage(
        from_agent=Agent.CLAUDE_CODE,
        to_agent=Agent.DORIS,
        message_type=MessageType.NOTIFICATION,
        content=message,
        context=context,
        priority="high",
        expects_response=False,
    ))


def ask_doris(question: str, context: Optional[dict] = None) -> int:
    """
    Claude Code asking Doris a question.
    Expects a response.
    """
    return send_message(AgentMessage(
        from_agent=Agent.CLAUDE_CODE,
        to_agent=Agent.DORIS,
        message_type=MessageType.QUESTION,
        content=question,
        context=context,
        expects_response=True,
    ))


def handoff_to_doris(task: str, context: dict) -> int:
    """
    Claude Code handing off a task to Doris.
    E.g., "Remind the user about X in 2 hours"
    """
    return send_message(AgentMessage(
        from_agent=Agent.CLAUDE_CODE,
        to_agent=Agent.DORIS,
        message_type=MessageType.HANDOFF,
        content=task,
        context=context,
        expects_response=False,
    ))


def get_cc_completion_updates(limit: int = 10) -> list[StoredMessage]:
    """
    Get Claude Code completion updates that haven't been acknowledged.

    These are UPDATE messages from CC to Doris about task completions.
    Used by the daemon to poll for completed tasks and notify the user.
    """
    conn = _get_db()
    rows = conn.execute("""
        SELECT * FROM messages
        WHERE from_agent = ?
          AND to_agent = ?
          AND message_type = ?
          AND responded = FALSE
        ORDER BY created_at DESC
        LIMIT ?
    """, (Agent.CLAUDE_CODE.value, Agent.DORIS.value, MessageType.UPDATE.value, limit)).fetchall()
    conn.close()

    return [msg for row in rows if (msg := _row_to_message(row)) is not None]


def acknowledge_message(message_id: int):
    """
    Acknowledge a message without sending a response.
    Marks the message as responded so it won't appear in pending lists.
    """
    conn = _get_db()
    conn.execute("""
        UPDATE messages SET responded = TRUE
        WHERE id = ?
    """, (message_id,))
    conn.commit()
    conn.close()
