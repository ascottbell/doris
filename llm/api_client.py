"""
Shared LLM client with automatic token logging.

All API call sites outside the main chat loop should use call_llm()
instead of creating their own SDK instances. This ensures:
1. Every call is logged to token_usage.jsonl with a source tag
2. Provider/model updates happen in one place
3. Cost-by-feature visibility via source tags

call_claude() is a backward-compatible alias for call_llm().
"""

import logging
from typing import Optional

from config import settings
from llm.providers import resolve_model
from llm.types import LLMResponse

logger = logging.getLogger("doris.api_client")


def call_llm(
    messages: list[dict],
    source: str,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    system: Optional[str] = None,
) -> LLMResponse:
    """
    Call the configured LLM provider with automatic token logging.

    Args:
        messages: List of message dicts (role + content)
        source: Tag for cost tracking (e.g. "proactive-eval", "briefing-news")
        model: Model override (defaults to provider's configured model)
        max_tokens: Max output tokens
        system: Optional system prompt

    Returns:
        LLMResponse with .text, .tool_calls, .usage, etc.
    """
    from llm.providers import get_llm_provider

    provider = get_llm_provider()
    response = provider.complete(
        messages,
        system=system,
        max_tokens=max_tokens,
        model=model,
        source=source,
    )

    # Log token usage
    try:
        from llm.brain import log_token_usage

        log_token_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model or resolve_model("default"),
            source=source,
            cache_creation_tokens=response.usage.cache_creation_tokens,
            cache_read_tokens=response.usage.cache_read_tokens,
        )
    except Exception as e:
        logger.warning(f"Token logging failed for {source}: {e}")

    return response


# Backward-compatible alias — existing code imports call_claude
call_claude = call_llm
