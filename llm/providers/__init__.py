"""
Provider registry — resolves configured providers and model tiers.

Usage:
    from llm.providers import get_llm_provider, get_embed_provider, resolve_model

    provider = get_llm_provider()
    response = provider.complete(messages, system=prompt, source="briefing")

    model = resolve_model("utility")  # e.g. "claude-haiku-4-5-20251001"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.providers.base import LLMProvider, EmbedProvider

logger = logging.getLogger("doris.providers")

# Cached provider instances
_llm_provider: LLMProvider | None = None
_embed_provider: EmbedProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Get or create the configured LLM provider (singleton)."""
    global _llm_provider
    if _llm_provider is None:
        from config import settings
        provider_name = getattr(settings, "llm_provider", "claude")

        if provider_name == "claude":
            from llm.providers.claude_provider import ClaudeLLMProvider
            _llm_provider = ClaudeLLMProvider()
        elif provider_name == "openai":
            from llm.providers.openai_provider import OpenAILLMProvider
            _llm_provider = OpenAILLMProvider()
        elif provider_name == "ollama":
            from llm.providers.ollama_provider import OllamaLLMProvider
            _llm_provider = OllamaLLMProvider()
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider_name!r}. "
                f"Set LLM_PROVIDER to one of: claude, openai, ollama"
            )

        logger.info(f"LLM provider: {provider_name}")

    return _llm_provider


def get_embed_provider() -> EmbedProvider:
    """Get or create the configured embedding provider (singleton)."""
    global _embed_provider
    if _embed_provider is None:
        from config import settings
        provider_name = getattr(settings, "embed_provider", "ollama")

        if provider_name == "ollama":
            from llm.providers.ollama_embed import OllamaEmbedProvider
            _embed_provider = OllamaEmbedProvider()
        else:
            raise ValueError(
                f"Unknown embed provider: {provider_name!r}. "
                f"Set EMBED_PROVIDER to one of: ollama"
            )

        logger.info(f"Embed provider: {provider_name}")

    return _embed_provider


def resolve_model(tier: str = "default") -> str:
    """
    Resolve a model tier to a concrete model string.

    Tiers:
        "default" — main brain (chat, briefings, check-ins)
        "utility" — cheap/fast tasks (extraction, evaluation, classification)

    Falls back to the provider's sensible defaults if not configured.
    """
    from config import settings

    if tier == "default":
        # Check explicit setting first, then backward-compat alias
        model = getattr(settings, "default_model", "") or settings.claude_model
        return model

    if tier == "mid":
        model = getattr(settings, "mid_model", "")
        if model:
            return model
        # Provider-specific defaults for mid tier
        provider_name = getattr(settings, "llm_provider", "claude")
        mid_defaults = {
            "claude": "claude-sonnet-4-6",
            "openai": "gpt-5.2",
            "ollama": settings.ollama_model,
        }
        return mid_defaults.get(provider_name, settings.claude_model)

    if tier == "utility":
        model = getattr(settings, "utility_model", "")
        if model:
            return model
        # Provider-specific defaults for utility tier
        provider_name = getattr(settings, "llm_provider", "claude")
        utility_defaults = {
            "claude": "claude-haiku-4-5-20251001",
            "openai": "gpt-5-mini",
            "ollama": settings.ollama_model,
        }
        return utility_defaults.get(provider_name, settings.claude_model)

    raise ValueError(f"Unknown model tier: {tier!r}. Use 'default', 'mid', or 'utility'.")


def reset_providers():
    """Reset cached providers (for testing)."""
    global _llm_provider, _embed_provider
    _llm_provider = None
    _embed_provider = None
