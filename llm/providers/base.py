"""
Provider protocols for Doris's LLM layer.

Any LLM backend must implement LLMProvider.
Any embedding backend must implement EmbedProvider.
"""

from __future__ import annotations

from typing import Any, Generator, AsyncGenerator, Protocol

from llm.types import LLMResponse, StreamEvent, ToolDef, ToolResult


class LLMProvider(Protocol):
    """
    Protocol for LLM providers.

    Each provider converts canonical ToolDefs and messages to its wire format
    internally. Doris never touches provider-specific types.
    """

    def complete(
        self,
        messages: list[dict],
        *,
        system: str | list | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 1024,
        model: str | None = None,
        source: str = "",
    ) -> LLMResponse:
        """Synchronous, non-streaming completion."""
        ...

    def stream(
        self,
        messages: list[dict],
        *,
        system: str | list | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 1024,
        model: str | None = None,
        source: str = "",
    ) -> Generator[StreamEvent, None, None]:
        """Synchronous streaming — yields StreamEvents."""
        ...

    async def astream(
        self,
        messages: list[dict],
        *,
        system: str | list | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 1024,
        model: str | None = None,
        source: str = "",
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming — yields StreamEvents."""
        ...

    def build_tool_result_messages(
        self,
        assistant_content: Any,
        tool_results: list[ToolResult],
    ) -> list[dict]:
        """
        Build provider-specific messages to thread tool results back.

        Claude: assistant message with raw content + user message with tool_result blocks.
        OpenAI: assistant message + separate tool-role messages per result.
        """
        ...


class EmbedProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Embed a document/memory (no instruction prefix)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (with instruction prefix if model supports it)."""
        ...
