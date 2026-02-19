"""
Session module for Doris.

Provides persistent conversation context across all entry points.
"""

from .persistent import (
    PersistentSession,
    get_session,
    estimate_tokens,
    Message,
    MAX_CONTEXT_TOKENS,
    COMPACTION_THRESHOLD,
    VERBATIM_TURNS,
)
from .compaction import compact_session

__all__ = [
    "PersistentSession",
    "get_session",
    "estimate_tokens",
    "compact_session",
    "Message",
    "MAX_CONTEXT_TOKENS",
    "COMPACTION_THRESHOLD",
    "VERBATIM_TURNS",
]
