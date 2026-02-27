"""
Bridge between Doris and the maasv cognition layer.

Creates Doris-specific LLM and Embed providers that satisfy the
maasv protocol, then initializes maasv with Doris's config.

Call init_maasv() once at startup, before any memory/sleep imports.
"""

import logging
from pathlib import Path

import maasv
from llm.providers import resolve_model
from maasv.config import MaasvConfig
from maasv.protocols import LLMProvider, EmbedProvider

logger = logging.getLogger("doris.maasv_bridge")


class DorisLLMProvider:
    """Routes maasv LLM calls through Doris's provider-agnostic LLM layer."""

    def call(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 1024,
        source: str = "",
    ) -> str:
        from llm.api_client import call_claude

        response = call_claude(
            messages=messages,
            source=source or "maasv",
            model=model,
            max_tokens=max_tokens,
        )
        return response.text


class DorisEmbedProvider:
    """Routes maasv embedding calls through Doris's configured backend."""

    QUERY_INSTRUCTION = (
        "Instruct: Given a user query about a person's life, "
        "retrieve relevant personal memories and facts.\nQuery:"
    )

    def __init__(self):
        import os
        self._model = os.environ.get("DORIS_EMBED_MODEL", "qwen3-embedding:8b")
        self._truncate_dims = int(os.environ.get("DORIS_EMBED_TRUNCATE", "1024"))

    def embed(self, text: str) -> list[float]:
        """Embed a document/memory (no instruction prefix)."""
        return self._embed_ollama(text)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (with instruction prefix for Qwen3)."""
        prefixed = f"{self.QUERY_INSTRUCTION}{text}"
        return self._embed_ollama(prefixed)

    def _embed_ollama(self, text: str) -> list[float]:
        import ollama
        import numpy as np
        response = ollama.embed(model=self._model, input=text)
        vec = response["embeddings"][0]
        # MRL truncation + L2 normalize
        vec = vec[:self._truncate_dims]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = (np.array(vec) / norm).tolist()
        return vec


def init_maasv():
    """Initialize maasv with Doris's configuration and providers."""
    import os
    from config import settings

    embed_dims = int(os.environ.get("DORIS_EMBED_DIMS", "1024"))

    config = MaasvConfig(
        db_path=settings.db_path,
        embed_dims=embed_dims,
        # Models for sleep-time jobs
        extraction_model=resolve_model("utility"),
        inference_model=resolve_model("utility"),
        review_model=resolve_model("utility"),
        # Hygiene
        backup_dir=settings.data_dir / "backups",
        max_hygiene_backups=int(os.environ.get("DORIS_MAX_HYGIENE_BACKUPS", "3")),
        protected_categories={"identity", "family"},
        protected_subjects=set(),  # Configure with your protected family member names
        # Known entities for extraction prompts
        # Add people, projects, and technologies relevant to your use case
        known_entities={
            # People — add your own
            # "Alice": "person",
            # Projects
            "Doris": "project",
            "maasv": "project",
            # Technologies
            "FastAPI": "technology",
            "Python": "technology",
            "SQLite": "technology",
            "Ollama": "technology",
            "Claude": "technology",
            "Next.js": "technology",
            "Railway": "technology",
            "PostgreSQL": "technology",
        },
        # Action families for wisdom "similar enough" matching
        action_families={
            "calendar": ["create_calendar_event", "move_calendar_event", "delete_calendar_event"],
            "reminders": ["create_reminder", "complete_reminder"],
            "messaging": ["send_imessage", "send_email"],
            "home": ["control_music"],
            "memory": ["store_memory"],
            "notifications": ["notify_user"],
            "creative": ["create_note"],
            "escalation": ["email_escalation_miss", "email_escalation_correct", "calendar_escalation_miss"],
            "development": [
                "architecture_decision",
                "debugging_resolution",
                "dependency_choice",
                "config_change",
                "gotcha",
                "user_preference",
                "approach_validated",
                "approach_rejected",
            ],
        },
        # Cross-encoder reranking (Doris has torch installed)
        cross_encoder_enabled=True,
        # Hygiene log
        hygiene_log_path=settings.data_dir / "memory_hygiene_log.json",
    )

    maasv.init(
        config=config,
        llm=DorisLLMProvider(),
        embed=DorisEmbedProvider(),
    )
    logger.info("[maasv] Initialized cognition layer")
