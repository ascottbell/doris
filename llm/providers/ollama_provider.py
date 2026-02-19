"""
Ollama provider — wraps Ollama's native HTTP API via httpx.

Handles:
- Tool format conversion (ToolDef → Ollama function-calling schema)
- Response parsing (Ollama chat response → LLMResponse)
- Streaming via newline-delimited JSON
- System prompt injection (as first message with role "system")
- Tool result threading (assistant + tool-role messages)

Requires Ollama running locally (default: http://localhost:11434).
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

logger = logging.getLogger("doris.providers.ollama")

OLLAMA_BASE_URL = "http://localhost:11434"


def _parse_usage(response_data: dict) -> TokenUsage:
    """Extract token counts from an Ollama response."""
    return TokenUsage(
        input_tokens=response_data.get("prompt_eval_count", 0) or 0,
        output_tokens=response_data.get("eval_count", 0) or 0,
    )


def _parse_tool_calls(message: dict) -> list[ToolCall]:
    """Extract tool calls from an Ollama response message."""
    tool_calls = []
    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        tool_calls.append(ToolCall(
            id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            name=func.get("name", ""),
            arguments=func.get("arguments", {}),
        ))
    return tool_calls


def _parse_response(data: dict) -> LLMResponse:
    """Convert an Ollama chat response to an LLMResponse."""
    message = data.get("message", {})
    text = message.get("content", "")
    tool_calls = _parse_tool_calls(message)

    # Determine stop reason
    done = data.get("done", True)
    if tool_calls:
        stop_reason = StopReason.TOOL_USE
    elif not done:
        stop_reason = StopReason.MAX_TOKENS
    else:
        stop_reason = StopReason.END_TURN

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=_parse_usage(data),
        raw_content=message,
    )


def _convert_tools(tools: list[ToolDef]) -> list[dict]:
    """Convert canonical ToolDefs to Ollama function-calling format."""
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

    Ollama uses role="system" messages like OpenAI.
    If system is a list of content blocks (Claude format), concatenate text parts.
    """
    result = []
    if system:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
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


class OllamaLLMProvider:
    """Ollama provider using native HTTP API."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self._base_url = base_url
        self._client = None
        self._async_client = None

    def _get_client(self):
        """Lazy-init the sync httpx client."""
        if self._client is None:
            import httpx
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=120.0,
            )
        return self._client

    def _get_async_client(self):
        """Lazy-init the async httpx client."""
        if self._async_client is None:
            import httpx
            self._async_client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=120.0,
            )
        return self._async_client

    def _resolve_model(self, model: str | None) -> str:
        """Resolve model string, falling back to config default."""
        if model:
            return model
        from config import settings
        return settings.ollama_model

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
        """Synchronous completion via Ollama HTTP API."""
        client = self._get_client()
        model = self._resolve_model(model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": _build_messages(messages, system),
            "stream": False,
            "options": {
                "num_predict": max_tokens,
            },
        }

        if tools:
            payload["tools"] = _convert_tools(tools)

        response = client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        result = _parse_response(data)

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
        """Synchronous streaming via Ollama HTTP API (NDJSON)."""
        client = self._get_client()
        model = self._resolve_model(model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": _build_messages(messages, system),
            "stream": True,
            "options": {
                "num_predict": max_tokens,
            },
        }

        if tools:
            payload["tools"] = _convert_tools(tools)

        accumulated_text = ""
        usage = TokenUsage()
        tool_calls: list[ToolCall] = []
        done = False

        with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = data.get("message", {})
                content = message.get("content", "")

                if content:
                    accumulated_text += content
                    yield StreamEvent(type="text", text=content)

                # Tool calls come in the final message (Ollama doesn't stream them)
                if message.get("tool_calls"):
                    for tc in message["tool_calls"]:
                        func = tc.get("function", {})
                        call_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                        name = func.get("name", "")
                        args = json.dumps(func.get("arguments", {}))

                        tool_calls.append(ToolCall(
                            id=call_id,
                            name=name,
                            arguments=func.get("arguments", {}),
                        ))

                        yield StreamEvent(
                            type="tool_start",
                            tool_name=name,
                            tool_call_id=call_id,
                        )
                        yield StreamEvent(
                            type="tool_input",
                            tool_input_json=args,
                        )

                if data.get("done", False):
                    done = True
                    usage = _parse_usage(data)

        # Determine stop reason
        if tool_calls:
            stop_reason = StopReason.TOOL_USE
        elif not done:
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.END_TURN

        # Build raw content for threading
        raw_content = {"role": "assistant", "content": accumulated_text}
        if tool_calls:
            raw_content["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                    "id": tc.id,
                }
                for tc in tool_calls
            ]

        yield StreamEvent(
            type="done",
            usage=usage,
            stop_reason=stop_reason,
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
        """Async streaming via Ollama HTTP API (NDJSON)."""
        client = self._get_async_client()
        model = self._resolve_model(model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": _build_messages(messages, system),
            "stream": True,
            "options": {
                "num_predict": max_tokens,
            },
        }

        if tools:
            payload["tools"] = _convert_tools(tools)

        accumulated_text = ""
        usage = TokenUsage()
        tool_calls: list[ToolCall] = []
        done = False

        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = data.get("message", {})
                content = message.get("content", "")

                if content:
                    accumulated_text += content
                    yield StreamEvent(type="text", text=content)

                if message.get("tool_calls"):
                    for tc in message["tool_calls"]:
                        func = tc.get("function", {})
                        call_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                        name = func.get("name", "")
                        args = json.dumps(func.get("arguments", {}))

                        tool_calls.append(ToolCall(
                            id=call_id,
                            name=name,
                            arguments=func.get("arguments", {}),
                        ))

                        yield StreamEvent(
                            type="tool_start",
                            tool_name=name,
                            tool_call_id=call_id,
                        )
                        yield StreamEvent(
                            type="tool_input",
                            tool_input_json=args,
                        )

                if data.get("done", False):
                    done = True
                    usage = _parse_usage(data)

        if tool_calls:
            stop_reason = StopReason.TOOL_USE
        elif not done:
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.END_TURN

        raw_content = {"role": "assistant", "content": accumulated_text}
        if tool_calls:
            raw_content["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                    "id": tc.id,
                }
                for tc in tool_calls
            ]

        yield StreamEvent(
            type="done",
            usage=usage,
            stop_reason=stop_reason,
            raw_content=raw_content,
        )

    def build_tool_result_messages(
        self,
        assistant_content: Any,
        tool_results: list[ToolResult],
    ) -> list[dict]:
        """
        Build Ollama-format messages for threading tool results.

        Ollama uses OpenAI-compatible format:
        1. {"role": "assistant", ...} with tool_calls
        2. One {"role": "tool", ...} per result
        """
        messages = []

        # Assistant message
        if isinstance(assistant_content, dict) and "role" in assistant_content:
            messages.append(assistant_content)
        else:
            messages.append({
                "role": "assistant",
                "content": str(assistant_content) if assistant_content else "",
            })

        # Tool result messages
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "content": tr.content,
            })

        return messages
