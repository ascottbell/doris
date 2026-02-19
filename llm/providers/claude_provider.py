"""
Anthropic Claude provider — wraps the Anthropic SDK.

Handles:
- Tool format conversion (ToolDef → Anthropic schema with cache_control)
- Response parsing (Anthropic Message → LLMResponse)
- Streaming event mapping (Anthropic events → StreamEvent)
- System prompt pass-through (string or list-of-blocks with cache_control)
"""

from __future__ import annotations

import logging
from typing import Any, Generator, AsyncGenerator

from llm.types import (
    LLMResponse,
    StreamEvent,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolDef,
    ToolResult,
)

logger = logging.getLogger("doris.providers.claude")

# Map Anthropic stop reasons to our enum
_STOP_REASON_MAP = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
}


def _parse_usage(usage: Any) -> TokenUsage:
    """Extract token counts from an Anthropic usage object."""
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
    )


def _parse_response(response: Any) -> LLMResponse:
    """Convert an Anthropic Message to an LLMResponse."""
    # Extract text
    text_parts = []
    tool_calls = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input,
            ))

    return LLMResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=_STOP_REASON_MAP.get(response.stop_reason, StopReason.END_TURN),
        usage=_parse_usage(response.usage),
        raw_content=response.content,
    )


def _convert_tools(tools: list[ToolDef], add_cache_control: bool = True) -> list[dict]:
    """
    Convert canonical ToolDefs to Anthropic tool format.

    Adds cache_control to the last tool so the entire array is cached
    via Anthropic's ephemeral prompt caching (~90% cost reduction).
    """
    result = []
    for i, tool in enumerate(tools):
        entry = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        # Cache the entire tools array by marking the last one
        if add_cache_control and i == len(tools) - 1:
            entry["cache_control"] = {"type": "ephemeral"}
        result.append(entry)
    return result


def _normalize_system(system: str | list | None) -> str | list | None:
    """
    Pass-through for Claude's system format.

    Claude accepts either a plain string or a list of content blocks
    (with optional cache_control). We preserve whichever format the
    caller provides.
    """
    return system


class ClaudeLLMProvider:
    """Anthropic Claude provider."""

    def __init__(self):
        self._client = None
        self._async_client = None

    def _get_client(self):
        """Lazy-init the sync Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "Anthropic SDK not installed. Run: pip install anthropic"
                )
            from config import settings
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def _get_async_client(self):
        """Lazy-init the async Anthropic client."""
        if self._async_client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "Anthropic SDK not installed. Run: pip install anthropic"
                )
            from config import settings
            self._async_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._async_client

    def _resolve_model(self, model: str | None) -> str:
        """Resolve model string, falling back to config default."""
        if model:
            return model
        from config import settings
        return settings.claude_model

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
        """Synchronous completion via Anthropic SDK."""
        client = self._get_client()
        model = self._resolve_model(model)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        system = _normalize_system(system)
        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = _convert_tools(tools)

        response = client.messages.create(**kwargs)
        result = _parse_response(response)

        logger.debug(
            f"complete [{source}] model={model} "
            f"in={result.usage.input_tokens} out={result.usage.output_tokens} "
            f"stop={result.stop_reason.value}"
        )

        return result

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
        """Synchronous streaming via Anthropic SDK."""
        client = self._get_client()
        model = self._resolve_model(model)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        system = _normalize_system(system)
        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = _convert_tools(tools)

        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                if not hasattr(event, "type"):
                    continue

                if event.type == "content_block_start":
                    if hasattr(event, "content_block") and event.content_block.type == "tool_use":
                        yield StreamEvent(
                            type="tool_start",
                            tool_name=event.content_block.name,
                            tool_call_id=event.content_block.id,
                        )

                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield StreamEvent(type="text", text=event.delta.text)
                    elif hasattr(event.delta, "partial_json"):
                        yield StreamEvent(
                            type="tool_input",
                            tool_input_json=event.delta.partial_json,
                        )

                elif event.type == "content_block_stop":
                    # Tool stop events are synthesized by the caller when
                    # they accumulate tool_input_json into a complete ToolCall
                    pass

            # Final message with usage and stop reason
            final = stream.get_final_message()
            parsed = _parse_response(final)

            yield StreamEvent(
                type="done",
                usage=parsed.usage,
                stop_reason=parsed.stop_reason,
                raw_content=parsed.raw_content,
            )

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
        """Async streaming via Anthropic SDK."""
        client = self._get_async_client()
        model = self._resolve_model(model)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        system = _normalize_system(system)
        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = _convert_tools(tools)

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if not hasattr(event, "type"):
                    continue

                if event.type == "content_block_start":
                    if hasattr(event, "content_block") and event.content_block.type == "tool_use":
                        yield StreamEvent(
                            type="tool_start",
                            tool_name=event.content_block.name,
                            tool_call_id=event.content_block.id,
                        )

                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield StreamEvent(type="text", text=event.delta.text)
                    elif hasattr(event.delta, "partial_json"):
                        yield StreamEvent(
                            type="tool_input",
                            tool_input_json=event.delta.partial_json,
                        )

                elif event.type == "content_block_stop":
                    pass

            # Final message
            final = await stream.get_final_message()
            parsed = _parse_response(final)

            yield StreamEvent(
                type="done",
                usage=parsed.usage,
                stop_reason=parsed.stop_reason,
                raw_content=parsed.raw_content,
            )

    def build_tool_result_messages(
        self,
        assistant_content: Any,
        tool_results: list[ToolResult],
    ) -> list[dict]:
        """
        Build Claude-format messages for threading tool results.

        Claude expects:
        1. {"role": "assistant", "content": <raw content blocks>}
        2. {"role": "user", "content": [{"type": "tool_result", ...}, ...]}
        """
        return [
            {"role": "assistant", "content": assistant_content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": tr.content,
                    }
                    for tr in tool_results
                ],
            },
        ]
