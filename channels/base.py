"""
Channel adapter protocol for Doris.

Defines the base class and message types that all channel adapters implement.
LLM providers use Protocol (structural typing) because they're stateless
converters. Channel adapters use ABC because they carry state (connections,
tokens, webhook servers) and benefit from shared behavior.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


class IncomingMessage(BaseModel):
    """Message received from an external channel."""

    text: str = Field(max_length=50_000)
    """Message content (capped at 50K chars, matching API endpoint limit)."""

    sender_id: str
    """Platform-specific user ID."""

    conversation_id: str
    """Platform-specific chat/thread ID."""

    channel: str
    """Channel identifier: 'telegram', 'bluebubbles', 'cli', etc."""

    sender_name: str | None = None
    """Display name if available."""

    timestamp: datetime | None = None
    """When the message was sent (platform time, not receive time)."""

    metadata: dict = Field(default_factory=dict)
    """Platform-specific extras (e.g., telegram message_id, attachments)."""


class OutgoingMessage(BaseModel):
    """Message to send back through a channel."""

    text: str
    """Response content."""

    conversation_id: str
    """Where to send it."""

    metadata: dict = Field(default_factory=dict)
    """Platform-specific extras (e.g., reply_to_message_id)."""


# ---------------------------------------------------------------------------
# Handler type
# ---------------------------------------------------------------------------

MessageHandler = Callable[[IncomingMessage], AsyncGenerator[str, None]]
"""
Async generator that yields text chunks as Doris generates a response.

Each adapter decides how to consume the stream:
  - CLI: print chunks as they arrive (natural terminal streaming)
  - Telegram: send initial message, edit every ~500ms with accumulated text
  - BlueBubbles: accumulate all chunks, send once (iMessage is atomic)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def collect_response(stream: AsyncGenerator[str, None]) -> str:
    """Accumulate all chunks from a streaming handler into a single string.

    Use this in adapters that don't support incremental delivery
    (e.g., BlueBubbles/iMessage where messages are atomic).
    """
    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ChannelAdapter(ABC):
    """Base class for channel adapters.

    Subclasses must implement:
      - name (property): channel identifier string
      - start(handler): begin listening for messages
      - stop(): gracefully shut down
      - send_message(conversation_id, text): send a proactive message

    Shared behavior provided:
      - _running state tracking
      - _safe_handle() wrapper for error handling
    """

    def __init__(self) -> None:
        self._running: bool = False

    # -- Abstract interface --------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel identifier (e.g., 'telegram', 'bluebubbles', 'cli')."""

    @abstractmethod
    async def start(self, handler: MessageHandler) -> None:
        """Start listening for messages. Call handler for each incoming message.

        Must set self._running = True on success. If the adapter cannot start
        (bad token, network error, etc.), it must raise — not silently fail.
        The runner decides how to handle startup failures.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the adapter.

        Contract:
          - Stop accepting new messages immediately
          - Finish any in-flight message handling before returning
          - Set self._running = False when done
        """

    @abstractmethod
    async def send_message(self, conversation_id: str, text: str) -> None:
        """Send a proactive message (Doris-initiated, e.g., notifications).

        Raises RuntimeError if the adapter is not running.
        """

    # -- Shared behavior -----------------------------------------------------

    async def _safe_handle(
        self,
        handler: MessageHandler,
        message: IncomingMessage,
    ) -> AsyncGenerator[str, None]:
        """Wrap a handler call with error handling.

        Catches exceptions from the handler, logs them, and yields an
        error message so the user always gets feedback — never a silent
        failure or a dead end.

        Adapters should call this instead of invoking the handler directly.
        """
        try:
            async for chunk in handler(message):
                yield chunk
        except Exception:
            logger.exception(
                "Handler error for message from %s on %s (conversation %s)",
                message.sender_id,
                self.name,
                message.conversation_id,
            )
            yield "Sorry, something went wrong processing your message. Please try again."

    def _check_running(self) -> None:
        """Raise RuntimeError if the adapter hasn't been started.

        Call this at the top of send_message() implementations.
        """
        if not self._running:
            raise RuntimeError(
                f"{self.name} adapter is not running. Call start() first."
            )
