"""
Session Management for Doris Daemon

Manages persistent Claude sessions for the "Queen Doris" brain.
Tracks session state, estimates token usage, and handles session
resumption across wake-ups.

Note: Full Claude Agent SDK integration pending SDK availability.
This module provides the interface and state management that will
integrate with the SDK when ready.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from config import settings

logger = logging.getLogger("doris.session")

# Session state file
SESSION_FILE = settings.data_dir / "session_state.json"

# Token thresholds
MAX_CONTEXT_TOKENS = 100_000  # Rough limit before compaction
WARNING_THRESHOLD = 0.8  # Warn at 80% of max


@dataclass
class Message:
    """A message in the session history."""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tokens_estimate: int = 0

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tokens_estimate": self.tokens_estimate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tokens_estimate=data.get("tokens_estimate", 0),
        )


@dataclass
class SessionState:
    """
    Persistent session state for Doris.

    Tracks the ongoing conversation context, token estimates,
    and provides hooks for memory extraction before compaction.
    """

    session_id: Optional[str] = None
    messages: list[Message] = field(default_factory=list)
    total_tokens: int = 0
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    wake_count: int = 0  # How many times Doris has woken up

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()

    @property
    def token_usage_ratio(self) -> float:
        """What percentage of max tokens have been used."""
        return self.total_tokens / MAX_CONTEXT_TOKENS

    @property
    def approaching_compaction(self) -> bool:
        """Check if we're approaching the compaction threshold."""
        return self.token_usage_ratio >= WARNING_THRESHOLD

    @property
    def needs_compaction(self) -> bool:
        """Check if we've exceeded the max and need compaction."""
        return self.total_tokens >= MAX_CONTEXT_TOKENS

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session."""
        tokens = estimate_tokens(content)
        msg = Message(role=role, content=content, tokens_estimate=tokens)
        self.messages.append(msg)
        self.total_tokens += tokens
        self.last_active = datetime.now()

        if self.approaching_compaction:
            logger.warning(
                f"Session approaching compaction threshold: "
                f"{self.total_tokens}/{MAX_CONTEXT_TOKENS} tokens "
                f"({self.token_usage_ratio:.1%})"
            )

    def get_recent_messages(self, limit: int = 10) -> list[Message]:
        """Get the most recent messages."""
        return self.messages[-limit:]

    def get_context_for_sdk(self) -> list[dict]:
        """Format messages for Claude SDK."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "total_tokens": self.total_tokens,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "wake_count": self.wake_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        return cls(
            session_id=data.get("session_id"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            total_tokens=data.get("total_tokens", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            last_active=datetime.fromisoformat(data["last_active"]) if data.get("last_active") else None,
            wake_count=data.get("wake_count", 0),
        )


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Rough estimate: ~4 characters per token on average.
    More accurate counting would use tiktoken or similar.
    """
    return len(text) // 4


class SessionManager:
    """
    Manages Doris's persistent session.

    Provides:
    - Session state persistence
    - Token tracking
    - Compaction detection
    - Memory extraction triggers
    """

    def __init__(self):
        self.state: Optional[SessionState] = None
        self._on_compaction_needed: Optional[callable] = None

    def set_compaction_callback(self, callback: callable) -> None:
        """Set callback for when compaction is needed."""
        self._on_compaction_needed = callback

    def load(self) -> SessionState:
        """Load session state from disk."""
        if SESSION_FILE.exists():
            try:
                data = json.loads(SESSION_FILE.read_text())
                self.state = SessionState.from_dict(data)
                logger.info(
                    f"Loaded session: {len(self.state.messages)} messages, "
                    f"{self.state.total_tokens} tokens"
                )
            except Exception as e:
                logger.error(f"Error loading session: {e}")
                self.state = SessionState()
        else:
            self.state = SessionState()

        return self.state

    def save(self) -> None:
        """Save session state to disk."""
        if self.state is None:
            return

        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps(self.state.to_dict(), indent=2))
        logger.debug("Session saved")

    def add_message(self, role: str, content: str) -> None:
        """Add a message and check for compaction needs."""
        if self.state is None:
            self.load()

        self.state.add_message(role, content)
        self.save()

        # Check if compaction is needed
        if self.state.needs_compaction and self._on_compaction_needed:
            self._on_compaction_needed(self.state)

    def start_new_session(self) -> SessionState:
        """Start a fresh session."""
        self.state = SessionState()
        self.save()
        logger.info("Started new session")
        return self.state

    def record_wake(self) -> None:
        """Record a wake-up event."""
        if self.state is None:
            self.load()

        self.state.wake_count += 1
        self.state.last_active = datetime.now()
        self.save()
        logger.info(f"Wake #{self.state.wake_count} recorded")

    async def extract_memories_before_compaction(self) -> dict:
        """
        Extract memories from session before compaction.

        Called when approaching token limit. Summarizes the session
        and stores rich conversation memories.

        Returns dict with extraction results.
        """
        if self.state is None or not self.state.messages:
            return {"status": "empty", "extracted": 0}

        # Build conversation text for extraction
        conversation_text = "\n".join([
            f"{m.role}: {m.content}"
            for m in self.state.messages
        ])

        # TODO: Call Claude to extract memories
        # For now, return placeholder
        logger.info(f"Memory extraction triggered: {len(self.state.messages)} messages")

        return {
            "status": "pending_implementation",
            "message_count": len(self.state.messages),
            "token_count": self.state.total_tokens,
        }


# Singleton instance
_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
        _manager.load()
    return _manager


# For testing
if __name__ == "__main__":
    manager = SessionManager()
    state = manager.load()

    print("Session Manager Test")
    print("=" * 50)
    print(f"Session ID: {state.session_id or 'None (new session)'}")
    print(f"Messages: {len(state.messages)}")
    print(f"Tokens: {state.total_tokens}/{MAX_CONTEXT_TOKENS}")
    print(f"Usage: {state.token_usage_ratio:.1%}")
    print(f"Approaching compaction: {state.approaching_compaction}")
    print(f"Wake count: {state.wake_count}")

    # Test adding a message
    manager.add_message("user", "Test message from session.py")
    manager.record_wake()

    print(f"\nAfter test:")
    print(f"Messages: {len(state.messages)}")
    print(f"Tokens: {state.total_tokens}")
