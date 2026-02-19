"""
Provider-agnostic types for Doris's LLM layer.

These types decouple Doris logic from any specific LLM SDK.
All providers convert to/from these types internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StopReason(Enum):
    """Why the LLM stopped generating."""
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"


@dataclass
class TokenUsage:
    """Token counts for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )


@dataclass
class ToolDef:
    """Canonical tool definition — provider converts to its wire format."""
    name: str
    description: str
    input_schema: dict  # Raw JSON Schema


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """Result of executing a tool, sent back to the LLM."""
    tool_call_id: str
    content: str


@dataclass
class LLMResponse:
    """
    Provider-agnostic response from an LLM call.

    `raw_content` preserves the provider's native content representation
    for message threading (e.g., Anthropic content blocks for tool loops).
    """
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw_content: Any = None  # Provider-native content for message threading

    def __str__(self) -> str:
        return self.text


@dataclass
class StreamEvent:
    """
    A single event from a streaming LLM response.

    Flat union — check `type` to know which fields are populated.
    """
    type: str  # "text", "tool_start", "tool_input", "tool_stop", "done", "error"
    text: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    tool_input_json: str = ""  # Partial JSON accumulated during streaming
    usage: TokenUsage | None = None
    stop_reason: StopReason | None = None
    raw_content: Any = None  # Provider-native final message content
