"""
Provider test suite — tool conversion, response parsing, tool threading, resolve_model.

Tests are pure unit tests (no SDK calls). Each provider's conversion and parsing
logic is tested against known inputs/outputs.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from llm.types import (
    LLMResponse,
    StreamEvent,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolDef,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TOOLS = [
    ToolDef(
        name="get_weather",
        description="Get weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    ),
    ToolDef(
        name="get_time",
        description="Get current time",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]

SAMPLE_TOOL_RESULTS = [
    ToolResult(tool_call_id="call_abc123", content='{"temp": 72, "condition": "sunny"}'),
    ToolResult(tool_call_id="call_def456", content='{"time": "2:30 PM"}'),
]


# ===========================================================================
# Claude Provider Tests
# ===========================================================================


class TestClaudeToolConversion:
    """Test ToolDef → Anthropic wire format."""

    def test_basic_conversion(self):
        from llm.providers.claude_provider import _convert_tools

        result = _convert_tools(SAMPLE_TOOLS)
        assert len(result) == 2
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a location"
        assert result[0]["input_schema"]["type"] == "object"
        assert "location" in result[0]["input_schema"]["properties"]

    def test_cache_control_on_last_tool(self):
        from llm.providers.claude_provider import _convert_tools

        result = _convert_tools(SAMPLE_TOOLS, add_cache_control=True)
        # Only the last tool should have cache_control
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_disabled(self):
        from llm.providers.claude_provider import _convert_tools

        result = _convert_tools(SAMPLE_TOOLS, add_cache_control=False)
        for tool in result:
            assert "cache_control" not in tool

    def test_single_tool_gets_cache_control(self):
        from llm.providers.claude_provider import _convert_tools

        result = _convert_tools([SAMPLE_TOOLS[0]])
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_empty_tools(self):
        from llm.providers.claude_provider import _convert_tools

        result = _convert_tools([])
        assert result == []


class TestClaudeResponseParsing:
    """Test Anthropic Message → LLMResponse."""

    def test_text_response(self):
        from llm.providers.claude_provider import _parse_response

        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello, world!"
        mock_response.content = [text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

        result = _parse_response(mock_response)
        assert result.text == "Hello, world!"
        assert result.tool_calls == []
        assert result.stop_reason == StopReason.END_TURN
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50

    def test_tool_use_response(self):
        from llm.providers.claude_provider import _parse_response

        mock_response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_123"
        tool_block.name = "get_weather"
        tool_block.input = {"location": "NYC"}
        # hasattr(tool_block, "text") should be False
        del tool_block.text
        mock_response.content = [tool_block]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = MagicMock(
            input_tokens=200,
            output_tokens=80,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=50,
        )

        result = _parse_response(mock_response)
        assert result.text == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_123"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "NYC"}
        assert result.stop_reason == StopReason.TOOL_USE
        assert result.usage.cache_creation_tokens == 10
        assert result.usage.cache_read_tokens == 50

    def test_mixed_text_and_tool_response(self):
        from llm.providers.claude_provider import _parse_response

        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me check the weather."
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_456"
        tool_block.name = "get_weather"
        tool_block.input = {"location": "NYC"}
        del tool_block.text
        mock_response.content = [text_block, tool_block]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = MagicMock(
            input_tokens=150,
            output_tokens=60,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

        result = _parse_response(mock_response)
        assert result.text == "Let me check the weather."
        assert len(result.tool_calls) == 1
        assert result.raw_content == [text_block, tool_block]


class TestClaudeToolResultThreading:
    """Test Claude tool result message building."""

    def test_build_tool_result_messages(self):
        from llm.providers.claude_provider import ClaudeLLMProvider

        provider = ClaudeLLMProvider()
        raw_content = [{"type": "tool_use", "id": "call_abc123", "name": "get_weather"}]
        results = [ToolResult(tool_call_id="call_abc123", content="sunny")]

        messages = provider.build_tool_result_messages(raw_content, results)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == raw_content
        assert messages[1]["role"] == "user"
        assert messages[1]["content"][0]["type"] == "tool_result"
        assert messages[1]["content"][0]["tool_use_id"] == "call_abc123"
        assert messages[1]["content"][0]["content"] == "sunny"

    def test_multiple_tool_results(self):
        from llm.providers.claude_provider import ClaudeLLMProvider

        provider = ClaudeLLMProvider()
        raw_content = "mock_content"
        messages = provider.build_tool_result_messages(raw_content, SAMPLE_TOOL_RESULTS)

        assert len(messages) == 2
        user_msg = messages[1]
        assert len(user_msg["content"]) == 2
        assert user_msg["content"][0]["tool_use_id"] == "call_abc123"
        assert user_msg["content"][1]["tool_use_id"] == "call_def456"


# ===========================================================================
# OpenAI Provider Tests
# ===========================================================================


class TestOpenAIToolConversion:
    """Test ToolDef → OpenAI function-calling format."""

    def test_basic_conversion(self):
        from llm.providers.openai_provider import _convert_tools

        result = _convert_tools(SAMPLE_TOOLS)
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather for a location"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_no_cache_control(self):
        """OpenAI doesn't use cache_control — ensure it's absent."""
        from llm.providers.openai_provider import _convert_tools

        result = _convert_tools(SAMPLE_TOOLS)
        for tool in result:
            assert "cache_control" not in tool
            assert "cache_control" not in tool["function"]


class TestOpenAISystemPrompt:
    """Test system prompt injection as first message."""

    def test_string_system(self):
        from llm.providers.openai_provider import _build_messages

        messages = [{"role": "user", "content": "Hello"}]
        result = _build_messages(messages, "You are helpful.")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"

    def test_list_system_with_text_blocks(self):
        """Claude-style list of content blocks → concatenated text."""
        from llm.providers.openai_provider import _build_messages

        system = [
            {"type": "text", "text": "You are Doris."},
            {"type": "text", "text": "Be helpful."},
        ]
        result = _build_messages([{"role": "user", "content": "Hi"}], system)

        assert result[0]["role"] == "system"
        assert "You are Doris." in result[0]["content"]
        assert "Be helpful." in result[0]["content"]

    def test_no_system(self):
        from llm.providers.openai_provider import _build_messages

        messages = [{"role": "user", "content": "Hello"}]
        result = _build_messages(messages, None)
        assert len(result) == 1
        assert result[0]["role"] == "user"


class TestOpenAIResponseParsing:
    """Test OpenAI ChatCompletion → LLMResponse."""

    def test_text_response(self):
        from llm.providers.openai_provider import _parse_response

        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Hello from GPT!"
        mock_message.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)

        result = _parse_response(mock_response)
        assert result.text == "Hello from GPT!"
        assert result.tool_calls == []
        assert result.stop_reason == StopReason.END_TURN
        assert result.usage.input_tokens == 50
        assert result.usage.output_tokens == 20

    def test_tool_call_response(self):
        from llm.providers.openai_provider import _parse_response

        mock_tc = MagicMock()
        mock_tc.id = "call_xyz"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"location": "NYC"}'

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tc]
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=30)

        result = _parse_response(mock_response)
        assert result.text == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "NYC"}
        assert result.stop_reason == StopReason.TOOL_USE

    def test_malformed_tool_arguments(self):
        """Invalid JSON in tool arguments → empty dict."""
        from llm.providers.openai_provider import _parse_response

        mock_tc = MagicMock()
        mock_tc.id = "call_bad"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = "not json"

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tc]
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        result = _parse_response(mock_response)
        assert result.tool_calls[0].arguments == {}
        assert result.usage.input_tokens == 0  # None usage → defaults


class TestOpenAIToolResultThreading:
    """Test OpenAI tool result message building."""

    def test_build_with_raw_dict(self):
        from llm.providers.openai_provider import OpenAILLMProvider

        provider = OpenAILLMProvider()
        raw_content = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
        }
        results = [ToolResult(tool_call_id="call_abc123", content="sunny")]

        messages = provider.build_tool_result_messages(raw_content, results)
        assert len(messages) == 2
        assert messages[0] == raw_content  # Passed through as-is
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_abc123"
        assert messages[1]["content"] == "sunny"

    def test_build_with_non_dict_fallback(self):
        from llm.providers.openai_provider import OpenAILLMProvider

        provider = OpenAILLMProvider()
        # If raw_content isn't a dict, it wraps it
        messages = provider.build_tool_result_messages("some text", [SAMPLE_TOOL_RESULTS[0]])

        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "some text"


# ===========================================================================
# Ollama Provider Tests
# ===========================================================================


class TestOllamaToolConversion:
    """Test ToolDef → Ollama function-calling format."""

    def test_basic_conversion(self):
        from llm.providers.ollama_provider import _convert_tools

        result = _convert_tools(SAMPLE_TOOLS)
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"

    def test_matches_openai_format(self):
        """Ollama uses the same format as OpenAI for tools."""
        from llm.providers.ollama_provider import _convert_tools as ollama_convert
        from llm.providers.openai_provider import _convert_tools as openai_convert

        ollama_result = ollama_convert(SAMPLE_TOOLS)
        openai_result = openai_convert(SAMPLE_TOOLS)
        assert ollama_result == openai_result


class TestOllamaResponseParsing:
    """Test Ollama JSON response → LLMResponse."""

    def test_text_response(self):
        from llm.providers.ollama_provider import _parse_response

        data = {
            "message": {"role": "assistant", "content": "Hello from Ollama!"},
            "done": True,
            "prompt_eval_count": 40,
            "eval_count": 15,
        }

        result = _parse_response(data)
        assert result.text == "Hello from Ollama!"
        assert result.tool_calls == []
        assert result.stop_reason == StopReason.END_TURN
        assert result.usage.input_tokens == 40
        assert result.usage.output_tokens == 15

    def test_tool_call_response(self):
        from llm.providers.ollama_provider import _parse_response

        data = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "NYC"},
                        },
                    }
                ],
            },
            "done": True,
            "prompt_eval_count": 60,
            "eval_count": 25,
        }

        result = _parse_response(data)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "NYC"}
        assert result.stop_reason == StopReason.TOOL_USE

    def test_incomplete_response(self):
        """done=False with no tool calls → MAX_TOKENS."""
        from llm.providers.ollama_provider import _parse_response

        data = {
            "message": {"role": "assistant", "content": "Partial..."},
            "done": False,
        }

        result = _parse_response(data)
        assert result.stop_reason == StopReason.MAX_TOKENS

    def test_tool_call_without_id(self):
        """Ollama may not always provide tool call IDs — we generate one."""
        from llm.providers.ollama_provider import _parse_response

        data = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {"name": "get_time", "arguments": {}},
                    }
                ],
            },
            "done": True,
        }

        result = _parse_response(data)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id.startswith("call_")  # Generated
        assert result.tool_calls[0].name == "get_time"


class TestOllamaSystemPrompt:
    """Test Ollama system prompt handling (same as OpenAI)."""

    def test_string_system(self):
        from llm.providers.ollama_provider import _build_messages

        result = _build_messages([{"role": "user", "content": "Hi"}], "Be helpful.")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful."

    def test_claude_format_list(self):
        from llm.providers.ollama_provider import _build_messages

        system = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]
        result = _build_messages([], system)
        assert result[0]["role"] == "system"
        assert "Part 1" in result[0]["content"]
        assert "Part 2" in result[0]["content"]


class TestOllamaToolResultThreading:
    """Test Ollama tool result message building."""

    def test_build_with_raw_dict(self):
        from llm.providers.ollama_provider import OllamaLLMProvider

        provider = OllamaLLMProvider()
        raw = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_time"}}]}
        results = [ToolResult(tool_call_id="call_1", content="2:30 PM")]

        messages = provider.build_tool_result_messages(raw, results)
        assert messages[0] == raw
        assert messages[1]["role"] == "tool"
        assert messages[1]["content"] == "2:30 PM"


# ===========================================================================
# Provider Registry Tests
# ===========================================================================


class TestResolveModel:
    """Test resolve_model() tier resolution."""

    def setup_method(self):
        """Reset provider cache before each test."""
        from llm.providers import reset_providers
        reset_providers()

    @patch("llm.providers.resolve_model.__module__", "llm.providers")
    def test_default_tier(self):
        with patch("config.settings") as mock_settings:
            mock_settings.default_model = "my-custom-model"
            mock_settings.claude_model = "claude-opus-4-6"
            from llm.providers import resolve_model
            result = resolve_model("default")
            # default_model takes precedence when set
            assert result == "my-custom-model"

    def test_default_tier_fallback_to_claude_model(self):
        with patch("config.settings") as mock_settings:
            mock_settings.default_model = ""
            mock_settings.claude_model = "claude-opus-4-6"
            from llm.providers import resolve_model
            result = resolve_model("default")
            assert result == "claude-opus-4-6"

    def test_utility_tier_explicit(self):
        with patch("config.settings") as mock_settings:
            mock_settings.utility_model = "my-haiku"
            from llm.providers import resolve_model
            result = resolve_model("utility")
            assert result == "my-haiku"

    def test_utility_tier_provider_default_claude(self):
        with patch("config.settings") as mock_settings:
            mock_settings.utility_model = ""
            mock_settings.llm_provider = "claude"
            mock_settings.claude_model = "claude-opus-4-6"
            from llm.providers import resolve_model
            result = resolve_model("utility")
            assert result == "claude-haiku-4-5-20251001"

    def test_utility_tier_provider_default_openai(self):
        with patch("config.settings") as mock_settings:
            mock_settings.utility_model = ""
            mock_settings.llm_provider = "openai"
            mock_settings.claude_model = "claude-opus-4-6"
            from llm.providers import resolve_model
            result = resolve_model("utility")
            assert result == "gpt-5-mini"

    def test_mid_tier_explicit(self):
        with patch("config.settings") as mock_settings:
            mock_settings.mid_model = "my-sonnet"
            from llm.providers import resolve_model
            result = resolve_model("mid")
            assert result == "my-sonnet"

    def test_mid_tier_provider_default_claude(self):
        with patch("config.settings") as mock_settings:
            mock_settings.mid_model = ""
            mock_settings.llm_provider = "claude"
            mock_settings.claude_model = "claude-opus-4-6"
            from llm.providers import resolve_model
            result = resolve_model("mid")
            assert result == "claude-sonnet-4-6"

    def test_unknown_tier_raises(self):
        from llm.providers import resolve_model
        with pytest.raises(ValueError, match="Unknown model tier"):
            resolve_model("turbo")


class TestProviderFactory:
    """Test get_llm_provider() factory."""

    def setup_method(self):
        from llm.providers import reset_providers
        reset_providers()

    @patch("config.settings")
    def test_claude_provider(self, mock_settings):
        mock_settings.llm_provider = "claude"
        from llm.providers import get_llm_provider
        provider = get_llm_provider()
        from llm.providers.claude_provider import ClaudeLLMProvider
        assert isinstance(provider, ClaudeLLMProvider)

    @patch("config.settings")
    def test_openai_provider(self, mock_settings):
        mock_settings.llm_provider = "openai"
        from llm.providers import get_llm_provider
        provider = get_llm_provider()
        from llm.providers.openai_provider import OpenAILLMProvider
        assert isinstance(provider, OpenAILLMProvider)

    @patch("config.settings")
    def test_ollama_provider(self, mock_settings):
        mock_settings.llm_provider = "ollama"
        from llm.providers import get_llm_provider
        provider = get_llm_provider()
        from llm.providers.ollama_provider import OllamaLLMProvider
        assert isinstance(provider, OllamaLLMProvider)

    @patch("config.settings")
    def test_unknown_provider_raises(self, mock_settings):
        mock_settings.llm_provider = "gemini"
        from llm.providers import get_llm_provider
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider()

    def test_singleton_behavior(self):
        """Same provider instance returned on subsequent calls."""
        with patch("config.settings") as mock_settings:
            mock_settings.llm_provider = "claude"
            from llm.providers import get_llm_provider
            p1 = get_llm_provider()
            p2 = get_llm_provider()
            assert p1 is p2

    def test_reset_clears_cache(self):
        """reset_providers() forces re-creation."""
        with patch("config.settings") as mock_settings:
            mock_settings.llm_provider = "claude"
            from llm.providers import get_llm_provider, reset_providers
            p1 = get_llm_provider()
            reset_providers()
            p2 = get_llm_provider()
            assert p1 is not p2


# ===========================================================================
# Cross-Provider: ToolDef canonical format consistency
# ===========================================================================


class TestToolDefConsistency:
    """Verify all tools in llm/tools.py produce valid ToolDefs."""

    def test_all_tools_are_tool_defs(self):
        from llm.tools import TOOLS
        assert len(TOOLS) > 0
        for tool in TOOLS:
            assert isinstance(tool, ToolDef)
            assert tool.name
            assert tool.description
            assert isinstance(tool.input_schema, dict)
            assert "type" in tool.input_schema

    def test_all_tools_convert_to_claude_format(self):
        from llm.tools import TOOLS
        from llm.providers.claude_provider import _convert_tools

        result = _convert_tools(TOOLS)
        assert len(result) == len(TOOLS)
        # Last tool should have cache_control
        assert result[-1].get("cache_control") == {"type": "ephemeral"}

    def test_all_tools_convert_to_openai_format(self):
        from llm.tools import TOOLS
        from llm.providers.openai_provider import _convert_tools

        result = _convert_tools(TOOLS)
        assert len(result) == len(TOOLS)
        for tool in result:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]

    def test_all_tools_convert_to_ollama_format(self):
        from llm.tools import TOOLS
        from llm.providers.ollama_provider import _convert_tools

        result = _convert_tools(TOOLS)
        assert len(result) == len(TOOLS)
        for tool in result:
            assert tool["type"] == "function"


# ===========================================================================
# Types Tests
# ===========================================================================


class TestTokenUsage:
    """Test TokenUsage arithmetic."""

    def test_addition(self):
        a = TokenUsage(input_tokens=100, output_tokens=50, cache_creation_tokens=10, cache_read_tokens=20)
        b = TokenUsage(input_tokens=200, output_tokens=30, cache_creation_tokens=5, cache_read_tokens=10)
        c = a + b
        assert c.input_tokens == 300
        assert c.output_tokens == 80
        assert c.cache_creation_tokens == 15
        assert c.cache_read_tokens == 30

    def test_defaults(self):
        t = TokenUsage()
        assert t.input_tokens == 0
        assert t.output_tokens == 0
        assert t.cache_creation_tokens == 0
        assert t.cache_read_tokens == 0


class TestLLMResponse:
    """Test LLMResponse behavior."""

    def test_str_returns_text(self):
        r = LLMResponse(text="Hello")
        assert str(r) == "Hello"

    def test_defaults(self):
        r = LLMResponse(text="")
        assert r.tool_calls == []
        assert r.stop_reason == StopReason.END_TURN
        assert r.usage.input_tokens == 0
        assert r.raw_content is None


class TestStreamEvent:
    """Test StreamEvent structure."""

    def test_text_event(self):
        e = StreamEvent(type="text", text="chunk")
        assert e.type == "text"
        assert e.text == "chunk"

    def test_tool_start_event(self):
        e = StreamEvent(type="tool_start", tool_name="get_weather", tool_call_id="call_1")
        assert e.tool_name == "get_weather"

    def test_done_event(self):
        e = StreamEvent(
            type="done",
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            stop_reason=StopReason.END_TURN,
        )
        assert e.usage.input_tokens == 100
        assert e.stop_reason == StopReason.END_TURN
