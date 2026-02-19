"""
Memory Store — thin wrapper over maasv.core.store.

All memory operations now live in the maasv package.
This module re-exports everything for backward compatibility.

Security: store_memory() is wrapped with injection scanning so all callers
get automatic protection against prompt injection in stored content.
"""

import json
import logging
from typing import Optional

from security.injection_scanner import scan_for_injection, strip_invisible_chars

logger = logging.getLogger(__name__)

# After maasv Session 14 decomposition, imports are spread across submodules:
# db.py: get_db, _db, init_db, get_embedding, serialize_embedding
# store.py: store_memory, supersede_memory, get_all_active, get_recent_memories, delete_memory, update_memory_metadata
# retrieval.py: find_similar_memories, find_by_subject, search_fts, get_core_memories, get_tiered_memory_context
# graph.py: all entity/relationship functions

from maasv.core.db import (  # noqa: F401
    get_db,
    _db,
    init_db,
    get_embedding,
    serialize_embedding,
)

from maasv.core.store import (  # noqa: F401
    store_memory as _raw_store_memory,
    supersede_memory,
    get_all_active,
    get_recent_memories,
    delete_memory,
    update_memory_metadata,
)


def store_memory(
    content: str,
    category: str,
    subject: Optional[str] = None,
    source: str = "manual",
    confidence: float = 1.0,
    metadata: Optional[dict] = None,
) -> str:
    """
    Store a memory with injection scanning and invisible character stripping.

    Wraps the maasv store_memory() to provide security at the storage layer.
    All callers automatically get:
    - Invisible Unicode character removal
    - Injection pattern scanning with logging
    - Suspicious content flagged in metadata for audit

    Args:
        content: Memory content to store
        category: Memory category
        subject: Who/what this memory is about
        source: Where this memory came from
        confidence: Confidence score 0-1
        metadata: Additional metadata dict

    Returns:
        Memory ID string
    """
    # Strip invisible characters that could hide injection payloads
    content = strip_invisible_chars(content)
    if subject:
        subject = strip_invisible_chars(subject)

    # Scan for injection patterns
    scan_result = scan_for_injection(content, source=f"memory_write:{source}")

    if scan_result.is_suspicious:
        logger.warning(
            f"Suspicious content in memory write from {source}: "
            f"risk={scan_result.risk_level}, patterns={scan_result.matched_patterns}"
        )
        # Tag the metadata so the content can be audited later
        metadata = metadata or {}
        metadata["injection_scan"] = {
            "risk_level": scan_result.risk_level,
            "patterns": scan_result.matched_patterns,
            "flagged": True,
        }

    return _raw_store_memory(
        content=content,
        category=category,
        subject=subject,
        source=source,
        confidence=confidence,
        metadata=metadata,
    )

from maasv.core.retrieval import (  # noqa: F401
    find_similar_memories,
    find_by_subject,
    search_fts,
    get_core_memories,
    get_tiered_memory_context,
)

from maasv.core.graph import (  # noqa: F401
    create_entity,
    get_entity,
    find_entity_by_name,
    find_or_create_entity,
    search_entities,
    get_entities_by_type,
    add_relationship,
    expire_relationship,
    get_entity_relationships,
    get_causal_chain,
    update_relationship_value,
    graph_query,
    get_entity_profile,
)

# Doris-specific: backward-compat DB_PATH for modules that reference it directly
from config import settings as _settings

DB_PATH = _settings.db_path

# Moved from maasv to config-driven in maasv, kept here for Doris backward compat
CATEGORY_PRIORITY = {
    'family': 1,
    'identity': 2,
    'preference': 3,
    'project': 4,
    'decision': 5,
    'person': 6,
    'learning': 7,
    'history': 8,
    'home': 9,
    'conversation': 10,
}


def check_conflicts(content: str, subject: Optional[str] = None) -> list[dict]:
    """Check if new memory conflicts with existing ones."""
    conflicts = []

    similar = find_similar_memories(content, limit=5)
    conflicts.extend(similar)

    if subject:
        by_subject = find_by_subject(subject)
        for mem in by_subject:
            if not any(c['id'] == mem['id'] for c in conflicts):
                conflicts.append(mem)

    return conflicts


def get_unprocessed_thoughts() -> list[dict]:
    """Get all unprocessed brain dump thoughts."""
    with _db() as db:
        rows = db.execute("""
            SELECT id, content, category, subject, source, confidence, created_at, metadata
            FROM memories
            WHERE category = 'thought'
            AND superseded_by IS NULL
            ORDER BY created_at ASC
        """).fetchall()

    unprocessed = []
    for row in rows:
        row_dict = dict(row)
        metadata = json.loads(row_dict['metadata']) if row_dict['metadata'] else {}
        if not metadata.get('processed', False):
            row_dict['_metadata'] = metadata
            unprocessed.append(row_dict)

    return unprocessed


