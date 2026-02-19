"""
OpenAI provider — wraps the OpenAI SDK.

Handles:
- Tool format conversion (ToolDef → OpenAI function-calling schema)
- Response parsing (ChatCompletion → LLMResponse)
- Streaming event mapping (OpenAI chunks → StreamEvent)
- System prompt injection (as first message with role "system")
- Tool result threading (assistant + tool-role messages)
"""

from __future__ import annotations

import json
import logging
import uuid
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

logger = logging.getLogger("doris.providers.openai")

# Map OpenAI finish reasons to our enum
_STOP_REASON_MAP = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
}


def _parse_usage(usage: Any) -> TokenUsage:
    """Extract token counts from an OpenAI usage object."""
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def _parse_response(response: Any) -> LLMResponse:
    """Convert an OpenAI ChatCompletion to an LLMResponse."""
    choice = response.choices[0]
    message = choice.message

    text = message.content or ""
    tool_calls = []

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=arguments,
            ))

    stop_reason = _STOP_REASON_MAP.get(
        choice.finish_reason or "stop", StopReason.END_TURN
    )

    # Preserve raw message for tool result threading
    raw_content = message

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=_parse_usage(response.usage),
        raw_content=raw_content,
    )


def _convert_tools(tools: list[ToolDef]) -> list[dict]:
    """Convert canonical ToolDefs to OpenAI function-calling format."""
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        })
    return result


def _build_messages(
    messages: list[dict],
    system: str | list | None,
) -> list[dict]:
    """
    Prepend system prompt as a system message.

    OpenAI uses role="system" messages instead of a separate system parameter.
    If system is a list of content blocks (Claude format), concatenate text parts.
    """
    result = []
    if system:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            # Extract text from Claude-style content blocks
            parts = []
            for block in system:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            system_text = "\n\n".join(parts)
        else:
            system_text = str(system)

        result.append({"role": "system", "content": system_text})

    result.extend(messages)
    return result


class OpenAILLMProvider:
    """OpenAI provider."""

    def __init__(self):
        self._client = None
        self._async_client = None

    def _get_client(self):
        """Lazy-init the sync OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "OpenAI SDK not installed. Run: pip install openai"
                )
            from config import settings
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Required when LLM_PROVIDER=openai."
                )
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def _get_async_client(self):
        """Lazy-init the async OpenAI client."""
        if self._async_client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "OpenAI SDK not installed. Run: pip install openai"
                )
            from config import settings
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Required when LLM_PROVIDER=openai."
                )
            self._async_client = openai.AsyncOpenAI(api_key=api_key)
        return self._async_client

    def _resolve_model(self, model: str | None) -> str:
        """Resolve model string, falling back to config default."""
        if model:
            return model
        from config import settings
        return settings.default_model or "gpt-5.2"

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
        """Synchronous completion via OpenAI SDK."""
        client = self._get_client()
        model = self._resolve_model(model)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _build_messages(messages, system),
        }

        if tools:
            kwargs["tools"] = _convert_tools(tools)

        response = client.chat.completions.create(**kwargs)
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
        """Synchronous streaming via OpenAI SDK."""
        client = self._get_client()
        model = self._resolve_model(model)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _build_messages(messages, system),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            kwargs["tools"] = _convert_tools(tools)

        # Track tool calls being built up across chunks
        active_tool_calls: dict[int, dict] = {}
        finish_reason = None
        usage = TokenUsage()

        response = client.chat.completions.create(**kwargs)

        for chunk in response:
            # Usage comes in the final chunk
            if chunk.usage:
                usage = _parse_usage(chunk.usage)

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            delta = choice.delta
            if delta is None:
                continue

            # Text content
            if delta.content:
                yield StreamEvent(type="text", text=delta.content)

            # Tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    if idx not in active_tool_calls:
                        # New tool call starting
                        active_tool_calls[idx] = {
                            "id": tc_delta.id or f"call_{uuid.uuid4().hex[:8]}",
                            "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                            "arguments": "",
                        }
                        if active_tool_calls[idx]["name"]:
                            yield StreamEvent(
                                type="tool_start",
                                tool_name=active_tool_calls[idx]["name"],
                                tool_call_id=active_tool_calls[idx]["id"],
                            )

                    # Accumulate argument fragments
                    if tc_delta.function and tc_delta.function.arguments:
                        active_tool_calls[idx]["arguments"] += tc_delta.function.arguments
                        yield StreamEvent(
                            type="tool_input",
                            tool_input_json=tc_delta.function.arguments,
                        )

        # Build final raw_content for tool result threading
        raw_content = self._build_raw_assistant_message(
            finish_reason, active_tool_calls
        )

        stop = _STOP_REASON_MAP.get(finish_reason or "stop", StopReason.END_TURN)

        yield StreamEvent(
            type="done",
            usage=usage,
            stop_reason=stop,
            raw_content=raw_content,
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
        """Async streaming via OpenAI SDK."""
        client = self._get_async_client()
        model = self._resolve_model(model)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _build_messages(messages, system),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            kwargs["tools"] = _convert_tools(tools)

        active_tool_calls: dict[int, dict] = {}
        finish_reason = None
        usage = TokenUsage()
        accumulated_text = ""

        response = await client.chat.completions.create(**kwargs)

        async for chunk in response:
            if chunk.usage:
                usage = _parse_usage(chunk.usage)

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            delta = choice.delta
            if delta is None:
                continue

            if delta.content:
                accumulated_text += delta.content
                yield StreamEvent(type="text", text=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    if idx not in active_tool_calls:
                        active_tool_calls[idx] = {
                            "id": tc_delta.id or f"call_{uuid.uuid4().hex[:8]}",
                            "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                            "arguments": "",
                        }
                        if active_tool_calls[idx]["name"]:
                            yield StreamEvent(
                                type="tool_start",
                                tool_name=active_tool_calls[idx]["name"],
                                tool_call_id=active_tool_calls[idx]["id"],
                            )

                    if tc_delta.function and tc_delta.function.arguments:
                        active_tool_calls[idx]["arguments"] += tc_delta.function.arguments
                        yield StreamEvent(
                            type="tool_input",
                            tool_input_json=tc_delta.function.arguments,
                        )

        raw_content = self._build_raw_assistant_message(
            finish_reason, active_tool_calls, accumulated_text
        )

        stop = _STOP_REASON_MAP.get(finish_reason or "stop", StopReason.END_TURN)

        yield StreamEvent(
            type="done",
            usage=usage,
            stop_reason=stop,
            raw_content=raw_content,
        )

    def _build_raw_assistant_message(
        self,
        finish_reason: str | None,
        active_tool_calls: dict[int, dict],
        text: str = "",
    ) -> dict:
        """
        Build a raw assistant message dict for tool result threading.

        This is stored as raw_content on the StreamEvent/LLMResponse so the
        caller can thread it back as the assistant message in the next turn.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": text or None}

        if active_tool_calls:
            msg["tool_calls"] = []
            for idx in sorted(active_tool_calls.keys()):
                tc = active_tool_calls[idx]
                msg["tool_calls"].append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                })

        return msg

    def build_tool_result_messages(
        self,
        assistant_content: Any,
        tool_results: list[ToolResult],
    ) -> list[dict]:
        """
        Build OpenAI-format messages for threading tool results.

        OpenAI expects:
        1. {"role": "assistant", ...} with tool_calls
        2. One {"role": "tool", "tool_call_id": "...", "content": "..."} per result
        """
        messages = []

        # Assistant message — use raw_content if it's already an OpenAI message dict
        if isinstance(assistant_content, dict) and "role" in assistant_content:
            messages.append(assistant_content)
        else:
            # Fallback: wrap as assistant message
            messages.append({"role": "assistant", "content": str(assistant_content) if assistant_content else None})

        # Tool result messages
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr.tool_call_id,
                "content": tr.content,
            })

        return messages
