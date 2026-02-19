"""
Ollama embedding provider — wraps Ollama's native HTTP API via httpx.

Provides document and query embeddings using Ollama-served models.
Supports MRL truncation and L2 normalization for consistent vector dimensions.

Requires Ollama running locally (default: http://localhost:11434).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("doris.providers.ollama_embed")

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaEmbedProvider:
    """Ollama embedding provider using native HTTP API."""

    QUERY_INSTRUCTION = (
        "Instruct: Given a user query about a person's life, "
        "retrieve relevant personal memories and facts.\nQuery:"
    )

    def __init__(self):
        self._client = None
        self._model = os.environ.get("DORIS_EMBED_MODEL", "qwen3-embedding:8b")
        self._truncate_dims = int(os.environ.get("DORIS_EMBED_TRUNCATE", "1024"))

    def _get_client(self):
        """Lazy-init the httpx client."""
        if self._client is None:
            import httpx
            self._client = httpx.Client(
                base_url=OLLAMA_BASE_URL,
                timeout=30.0,
            )
        return self._client

    def _embed_raw(self, text: str) -> list[float]:
        """Call Ollama's embedding endpoint and return normalized vector."""
        client = self._get_client()

        response = client.post("/api/embed", json={
            "model": self._model,
            "input": text,
        })
        response.raise_for_status()
        data = response.json()

        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError(f"Ollama returned no embeddings for model {self._model}")

        vec = embeddings[0]

        # MRL truncation
        vec = vec[:self._truncate_dims]

        # L2 normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec

    def embed(self, text: str) -> list[float]:
        """Embed a document/memory (no instruction prefix)."""
        return self._embed_raw(text)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (with instruction prefix for retrieval models)."""
        prefixed = f"{self.QUERY_INSTRUCTION}{text}"
        return self._embed_raw(prefixed)
