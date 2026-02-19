"""
Doris Memory MCP Server

Exposes Doris memory system to Claude Desktop/Code via Model Context Protocol.
Uses FastMCP for a clean decorator-based API.

Tools:
- doris_memory_bootstrap: Session start context
- doris_memory_query: Semantic search across memories
- doris_memory_log: Log conversation for fact extraction
- doris_memory_facts: Get facts by category (fast, no embedding)
- doris_memory_forget: Delete memories

Run modes:
- STDIO (local):  python -m mcp_server.server
- HTTP (remote):  DORIS_MCP_TRANSPORT=http python -m mcp_server.server

Environment variables:
- DORIS_MCP_TRANSPORT: "stdio" (default) or "http"
- DORIS_MCP_PORT: Port for HTTP transport (default 8000)
- DORIS_MCP_HOST: Host for HTTP transport (default 127.0.0.1)
- DORIS_MCP_AUTH_TOKEN: API token for HTTP auth (required for HTTP)
"""

import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP
from config import settings

# Initialize maasv cognition layer (MCP server is a separate process)
from maasv_bridge import init_maasv
init_maasv()

from memory.store import (
    store_memory,
    find_similar_memories,
    get_all_active,
    delete_memory,
    search_fts,
    # Graph functions
    create_entity,
    get_entity,
    find_entity_by_name,
    find_or_create_entity,
    search_entities,
    get_entities_by_type,
    add_relationship,
    expire_relationship,
    get_entity_relationships,
    update_relationship_value,
    graph_query,
    get_entity_profile,
)

from mcp_server.bootstrap import (
    build_bootstrap_context,
    format_bootstrap_for_prompt,
    get_category_memories,
)
from mcp_server.background import get_worker


# ── Input validation constants ──────────────────────────────────────────────
VALID_MEMORY_CATEGORIES = frozenset({
    "identity", "family", "preference", "project", "decision",
    "person", "event", "learning", "conversation", "context",
    "health", "financial",
})
VALID_ENTITY_TYPES = frozenset({
    "person", "place", "project", "event", "concept", "unknown",
})
MAX_CONTENT_LENGTH = 10_000   # characters
MAX_LIMIT = 100
MAX_QUERY_LENGTH = 2_000
MAX_FACTS_PER_CALL = 50
MAX_CONVERSATION_LENGTH = 50_000  # characters for graph extraction input


def _validate_category(category: str, label: str = "category") -> str:
    """Validate a memory category. Returns the validated value or raises ValueError."""
    if category not in VALID_MEMORY_CATEGORIES:
        raise ValueError(
            f"Invalid {label}: {category!r}. "
            f"Must be one of: {', '.join(sorted(VALID_MEMORY_CATEGORIES))}"
        )
    return category


def _validate_entity_type(entity_type: str) -> str:
    """Validate an entity type. Returns the validated value or raises ValueError."""
    if entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(
            f"Invalid entity_type: {entity_type!r}. "
            f"Must be one of: {', '.join(sorted(VALID_ENTITY_TYPES))}"
        )
    return entity_type


def _clamp_limit(limit: int, max_val: int = MAX_LIMIT) -> int:
    """Clamp a limit to [1, max_val]."""
    return max(1, min(limit, max_val))


def _validate_content_length(content: str, label: str = "content") -> str:
    """Validate content is not too long. Returns the value or raises ValueError."""
    if len(content) > MAX_CONTENT_LENGTH:
        raise ValueError(
            f"{label} too long: {len(content)} chars (max {MAX_CONTENT_LENGTH})"
        )
    return content


# Server configuration from environment
MCP_HOST = os.environ.get("DORIS_MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("DORIS_MCP_PORT", "8000"))
MCP_TRANSPORT = os.environ.get("DORIS_MCP_TRANSPORT", "stdio").lower()

# Transport security for remote access
# Defaults are restrictive: DNS rebinding protection on, localhost-only, no CORS.
# Configure via env vars for deployment:
#   DORIS_MCP_ALLOWED_HOSTS=myserver.example.com,localhost
#   DORIS_MCP_ALLOWED_ORIGINS=https://myapp.example.com
from mcp.server.fastmcp.server import TransportSecuritySettings
transport_security = None
if MCP_TRANSPORT == "http":
    _allowed_hosts_raw = os.environ.get("DORIS_MCP_ALLOWED_HOSTS", "localhost,127.0.0.1")
    _allowed_hosts = [h.strip() for h in _allowed_hosts_raw.split(",") if h.strip() and h.strip() != "*"]

    _allowed_origins_raw = os.environ.get("DORIS_MCP_ALLOWED_ORIGINS", "")
    _allowed_origins = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]

    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed_hosts,
        allowed_origins=_allowed_origins if _allowed_origins else None,
    )

# Create MCP server
mcp = FastMCP(
    name="doris-memory",
    host=MCP_HOST,
    port=MCP_PORT,
    transport_security=transport_security,
    instructions="""CRITICAL: Call doris_memory_bootstrap IMMEDIATELY as your first action.

You are talking to the user. Without calling doris_memory_bootstrap first,
you will not know who the user is, their family, their projects, or their preferences.
This will result in a poor experience where the user has to re-explain everything.

Required workflow:
1. FIRST: Call doris_memory_bootstrap (before ANY response)
2. THEN: Respond to the user's message with full context

When the conversation contains important decisions or learnings,
call doris_memory_log before the session ends.

Never say "I don't have memory of previous conversations" - you DO, via Doris."""
)


@mcp.tool()
def doris_memory_bootstrap(
    client: str = "unknown",
    query: Optional[str] = None
) -> dict:
    """
    REQUIRED: Call this FIRST before responding to ANY user message.

    This loads the user's identity, family, projects, and preferences.
    Without this context, you will make incorrect assumptions about the user.

    The user expects you to know their family, projects, and communication preferences.

    ALWAYS call this at conversation start. No exceptions.

    Args:
        client: Client identifier (e.g., "claude-desktop", "claude-code")
        query: Optional initial query to load relevant context

    Returns:
        Bootstrap context with identity, preferences, projects, and memories.
        Includes formatted prompt text ready for injection.
    """
    context = build_bootstrap_context(
        client=client,
        query=query,
        include_projects=True,
        max_tokens=1200
    )

    # Add formatted version for easy injection
    context["formatted_prompt"] = format_bootstrap_for_prompt(context)

    return context


@mcp.tool()
def doris_memory_query(
    query: str,
    limit: int = 10,
    category: Optional[str] = None,
    use_semantic: bool = True
) -> dict:
    """
    Search memories using semantic or keyword search.

    Use this to find specific memories or decisions from past conversations.

    Args:
        query: Search query (natural language for semantic, keywords for FTS)
        limit: Maximum number of results (default 10)
        category: Optional filter by category (identity, family, preference, project, decision, etc.)
        use_semantic: If True, use embedding-based search. If False, use keyword search.

    Returns:
        List of matching memories with content, category, subject, and relevance.
    """
    # Validate inputs
    if len(query) > MAX_QUERY_LENGTH:
        return {"error": f"Query too long: {len(query)} chars (max {MAX_QUERY_LENGTH})"}
    limit = _clamp_limit(limit)
    if category is not None:
        try:
            _validate_category(category)
        except ValueError as e:
            return {"error": str(e)}

    if use_semantic:
        results = find_similar_memories(
            query=query,
            limit=limit,
            category=category
        )
    else:
        # Keyword search via FTS (category filter is applied in the SQL query)
        results = search_fts(query, limit=limit, category=category)

    return {
        "query": query,
        "count": len(results),
        "results": results
    }


@mcp.tool()
def doris_memory_log(
    client: str,
    summary: str,
    key_facts: Optional[list[dict]] = None,
    decisions: Optional[list[dict]] = None,
    auto_extract: bool = False,  # DEPRECATED - ignored, see docstring
    conversation: Optional[str] = None
) -> dict:
    """
    Log conversation facts for memory storage.

    Call this before ending a session that contains important information.

    **IMPORTANT: Agent-driven extraction (AgeMem pattern)**
    The agent (Claude) should extract facts and pass them via `key_facts`.
    Do NOT rely on auto_extract - it is deprecated and ignored.

    If `conversation` text is provided, graph entity extraction will run
    in the background (using Haiku, ~1s) to update the knowledge graph
    with people, places, projects, and relationships mentioned.

    Args:
        client: Client identifier
        summary: Brief summary of the conversation
        key_facts: List of facts to store, each with:
            - content: The fact text
            - category: Category (decision, learning, project, person, preference, event)
            - subject: Optional subject (who/what this is about)
            - confidence: Optional confidence score (0.0-1.0)
        decisions: List of decisions made (shorthand for decision category)
        auto_extract: DEPRECATED - ignored. Agent should extract facts via key_facts.
        conversation: Conversation text for graph entity extraction (optional)

    Returns:
        List of stored memory IDs and status.
    """
    stored = []
    errors = []

    # Validate inputs
    if key_facts and len(key_facts) > MAX_FACTS_PER_CALL:
        return {"status": "error", "message": f"Too many facts: {len(key_facts)} (max {MAX_FACTS_PER_CALL})"}
    if conversation and len(conversation) > MAX_CONVERSATION_LENGTH:
        return {"status": "error", "message": f"Conversation too long: {len(conversation)} chars (max {MAX_CONVERSATION_LENGTH})"}

    # Warn if auto_extract is used
    if auto_extract:
        print("[MCP] WARNING: auto_extract is deprecated. Agent should extract facts via key_facts param.")

    # Store provided facts (agent-driven extraction - the right way)
    if key_facts:
        for fact in key_facts:
            try:
                content = fact.get("content", "")
                if not content or not isinstance(content, str):
                    errors.append({"fact": str(content)[:100], "error": "content must be a non-empty string"})
                    continue
                if len(content) > MAX_CONTENT_LENGTH:
                    errors.append({"fact": content[:100], "error": f"content too long: {len(content)} chars (max {MAX_CONTENT_LENGTH})"})
                    continue
                cat = fact.get("category", "learning")
                if cat not in VALID_MEMORY_CATEGORIES:
                    errors.append({"fact": content[:100], "error": f"invalid category: {cat!r}"})
                    continue

                mem_id = store_memory(
                    content=content,
                    category=cat,
                    subject=fact.get("subject"),
                    source=f"conversation:{client}",
                    confidence=fact.get("confidence", 0.9)
                )
                stored.append({"id": mem_id, "content": content})
            except Exception as e:
                errors.append({"fact": fact.get("content", "")[:100], "error": str(e)})

    # Store decisions separately
    if decisions:
        for dec in decisions:
            try:
                content = dec if isinstance(dec, str) else dec.get("content", str(dec))
                subject = dec.get("subject") if isinstance(dec, dict) else None
                mem_id = store_memory(
                    content=content,
                    category="decision",
                    subject=subject,
                    source=f"conversation:{client}",
                    confidence=0.95
                )
                stored.append({"id": mem_id, "content": content, "category": "decision"})
            except Exception as e:
                errors.append({"decision": content, "error": str(e)})

    # Graph extraction (Slow Path) - runs in background with Haiku
    # Extracts entities and relationships for knowledge graph
    graph_result = None
    if conversation:
        worker = get_worker()
        if worker.queue_graph_extraction(conversation, source=f"conversation:{client}"):
            graph_result = {"status": "queued", "message": "Graph extraction running in background (Haiku)"}
        else:
            graph_result = {"status": "queue_full", "message": "Background queue full, extraction skipped"}

    return {
        "status": "success" if not errors else "partial",
        "summary": summary,
        "stored_count": len(stored),
        "stored": stored,
        "graph": graph_result,
        "errors": errors if errors else None,
        "timestamp": datetime.now().isoformat()
    }


@mcp.tool()
def doris_memory_facts(
    category: str,
    limit: int = 20,
    subject: Optional[str] = None
) -> dict:
    """
    Get facts by category (fast, no embedding lookup).

    Use this for quick retrieval of known fact categories.

    Args:
        category: Category to retrieve. Options:
            - identity: Core facts about the user
            - family: Family members and context
            - preference: How the user likes to work
            - project: Active and past projects
            - decision: Decisions made in conversations
            - person: People the user works with
            - event: Calendar/time-based memories
            - learning: Things discovered/learned
            - conversation: Rich session summaries with reasoning and open questions
        limit: Maximum results (default 20)
        subject: Optional filter by subject (e.g., "Alice", "MyProject")

    Returns:
        List of facts in the category.
    """
    # Validate inputs
    try:
        _validate_category(category)
    except ValueError as e:
        return {"error": str(e)}
    limit = _clamp_limit(limit)

    memories = get_all_active(category=category)

    if subject:
        memories = [m for m in memories if subject.lower() in (m.get("subject") or "").lower()]

    memories = memories[:limit]

    return {
        "category": category,
        "subject_filter": subject,
        "count": len(memories),
        "facts": [
            {
                "id": m["id"],
                "content": m["content"],
                "subject": m.get("subject"),
                "confidence": m.get("confidence", 1.0),
                "created_at": m.get("created_at")
            }
            for m in memories
        ]
    }


@mcp.tool()
def doris_memory_forget(
    memory_id: Optional[str] = None,
    memory_ids: Optional[list[str]] = None,
    query: Optional[str] = None,
    confirm: bool = False
) -> dict:
    """
    Delete memories (for corrections or privacy).

    Two-step process for safety:
    1. Use `query` to SEARCH for memories (returns preview, never deletes)
    2. Use `memory_id` or `memory_ids` with `confirm=True` to delete specific IDs

    Query-based deletion is NOT supported — you must specify exact IDs.
    This prevents vague queries from accidentally deleting unrelated memories.

    Args:
        memory_id: Specific memory ID to delete (single)
        memory_ids: List of specific memory IDs to delete (batch)
        query: Search query to find memories (preview only, never deletes)
        confirm: Must be True to actually delete. If False, shows what would be deleted.

    Returns:
        Status and list of deleted/affected memories.
    """
    # Validate inputs
    if query and len(query) > MAX_QUERY_LENGTH:
        return {"error": f"Query too long: {len(query)} chars (max {MAX_QUERY_LENGTH})"}
    if memory_ids and len(memory_ids) > MAX_LIMIT:
        return {"error": f"Too many memory_ids: {len(memory_ids)} (max {MAX_LIMIT})"}

    # Query mode: preview only, never delete
    if query:
        results = find_similar_memories(query, limit=10)
        preview = [
            {
                "id": r["id"],
                "content": r["content"],
                "category": r["category"]
            }
            for r in results
        ]
        return {
            "status": "preview",
            "message": "Use memory_id or memory_ids with confirm=True to delete specific entries",
            "matches": preview
        }

    # Collect target IDs
    target_ids = []
    if memory_id:
        target_ids.append(memory_id)
    if memory_ids:
        target_ids.extend(memory_ids)

    if not target_ids:
        return {
            "status": "error",
            "message": "Must provide memory_id, memory_ids, or query"
        }

    # Resolve IDs to actual memories
    all_memories = get_all_active()
    memory_map = {m["id"]: m for m in all_memories}

    affected = []
    not_found = []
    for mid in target_ids:
        if mid in memory_map:
            m = memory_map[mid]
            affected.append({
                "id": m["id"],
                "content": m["content"],
                "category": m["category"]
            })
        else:
            not_found.append(mid)

    if not affected:
        return {
            "status": "not_found",
            "message": "No matching memories found",
            "not_found_ids": not_found
        }

    if not confirm:
        result = {
            "status": "preview",
            "message": "Set confirm=True to delete these memories",
            "would_delete": affected
        }
        if not_found:
            result["not_found_ids"] = not_found
        return result

    # Actually delete
    deleted = []
    for mem in affected:
        if delete_memory(mem["id"]):
            deleted.append(mem)

    result = {
        "status": "deleted",
        "count": len(deleted),
        "deleted": deleted
    }
    if not_found:
        result["not_found_ids"] = not_found
    return result


# ============================================================================
# GRAPH MEMORY TOOLS (Jan 2026)
# ============================================================================

@mcp.tool()
def doris_graph_entity_create(
    name: str,
    entity_type: str,
    metadata: Optional[dict] = None
) -> dict:
    """
    Create a new entity in the knowledge graph.

    Use this to add people, places, projects, events, or concepts that Doris
    should remember. Entities are nodes in the graph that can have relationships.

    Args:
        name: Display name (e.g., "Alice", "MyProject", "Favorite Restaurant")
        entity_type: Type of entity. Options:
            - person: People (family, colleagues, contacts)
            - place: Locations (restaurants, offices, addresses)
            - project: Work projects or personal projects
            - event: Scheduled events or occasions
            - concept: Abstract concepts or topics
        metadata: Optional additional data (e.g., {"email": "...", "role": "..."})

    Returns:
        Created entity with ID
    """
    try:
        _validate_entity_type(entity_type)
        _validate_content_length(name, "name")
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    try:
        entity_id = create_entity(name, entity_type, metadata=metadata)
        return {
            "status": "created",
            "entity": {
                "id": entity_id,
                "name": name,
                "entity_type": entity_type,
                "metadata": metadata
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def doris_graph_entity_get(entity_id: str) -> dict:
    """
    Get an entity by ID.

    Returns the full entity details including metadata.

    Args:
        entity_id: The entity ID (starts with "ent_")

    Returns:
        Entity details or not_found status
    """
    entity = get_entity(entity_id)
    if entity:
        return {"status": "found", "entity": entity}
    return {"status": "not_found", "entity_id": entity_id}


@mcp.tool()
def doris_graph_entity_search(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 10
) -> dict:
    """
    Search for entities by name.

    Uses full-text search to find matching entities.

    Args:
        query: Search query (name or partial name)
        entity_type: Optional filter (person, place, project, event, concept)
        limit: Maximum results (default 10)

    Returns:
        List of matching entities
    """
    limit = _clamp_limit(limit)
    if entity_type is not None:
        try:
            _validate_entity_type(entity_type)
        except ValueError as e:
            return {"error": str(e)}
    if len(query) > MAX_QUERY_LENGTH:
        return {"error": f"Query too long: {len(query)} chars (max {MAX_QUERY_LENGTH})"}

    results = search_entities(query, entity_type, limit)
    return {
        "query": query,
        "type_filter": entity_type,
        "count": len(results),
        "entities": results
    }


@mcp.tool()
def doris_graph_entity_find_or_create(
    name: str,
    entity_type: str,
    metadata: Optional[dict] = None
) -> dict:
    """
    Find an existing entity by name, or create it if it doesn't exist.

    Uses canonical name matching (case-insensitive, spaces normalized).
    Useful when you're not sure if an entity already exists.

    Args:
        name: Entity name
        entity_type: Type (person, place, project, event, concept)
        metadata: Optional metadata (only used if creating new)

    Returns:
        Entity ID and whether it was newly created
    """
    try:
        _validate_entity_type(entity_type)
        _validate_content_length(name, "name")
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    existing = find_entity_by_name(name, entity_type)
    if existing:
        return {
            "status": "found",
            "created": False,
            "entity": existing
        }

    entity_id = create_entity(name, entity_type, metadata=metadata)
    return {
        "status": "created",
        "created": True,
        "entity": {
            "id": entity_id,
            "name": name,
            "entity_type": entity_type,
            "metadata": metadata
        }
    }


@mcp.tool()
def doris_graph_relationship_add(
    subject_id: str,
    predicate: str,
    object_id: Optional[str] = None,
    object_value: Optional[str] = None,
    source: Optional[str] = None
) -> dict:
    """
    Add a relationship between entities (or entity to value).

    Relationships are temporal - they have a valid_from timestamp and can be
    expired later when information changes. This preserves history.

    Args:
        subject_id: Source entity ID
        predicate: Relationship type. Common predicates:
            - married_to, parent_of, child_of (family)
            - works_on, owns (projects)
            - located_at, lives_in (places)
            - has_email, has_phone, has_age (attributes)
            - scheduled_for, attending (events)
        object_id: Target entity ID (for entity-to-entity relationships)
        object_value: Literal value (for entity-to-value, e.g., email address)
        source: Where this came from (e.g., "conversation:claude-code")

    Returns:
        Created relationship with ID
    """
    if object_id is None and object_value is None:
        return {"status": "error", "message": "Must provide either object_id or object_value"}

    if object_value is not None and len(object_value) > MAX_CONTENT_LENGTH:
        return {"status": "error", "message": f"object_value too long: {len(object_value)} chars (max {MAX_CONTENT_LENGTH})"}

    try:
        rel_id = add_relationship(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            object_value=object_value,
            source=source
        )
        return {
            "status": "created",
            "relationship": {
                "id": rel_id,
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                "object_value": object_value
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def doris_graph_relationship_expire(
    relationship_id: str
) -> dict:
    """
    Mark a relationship as expired (no longer current).

    The relationship is NOT deleted - it becomes historical.
    Use this when information changes (e.g., someone's email changed).
    The old value is still queryable with include_expired=True.

    Args:
        relationship_id: ID of relationship to expire (starts with "rel_")

    Returns:
        Status of the expiration
    """
    success = expire_relationship(relationship_id)
    if success:
        return {"status": "expired", "relationship_id": relationship_id}
    return {"status": "not_found", "relationship_id": relationship_id}


@mcp.tool()
def doris_graph_entity_edges(
    entity_id: str,
    include_expired: bool = False,
    predicate: Optional[str] = None,
    direction: str = "both"
) -> dict:
    """
    Get all relationships for an entity.

    Returns both outgoing (entity is subject) and incoming (entity is object)
    relationships by default.

    Args:
        entity_id: Entity to get relationships for
        include_expired: If True, include historical (expired) relationships
        predicate: Optional filter by relationship type
        direction: "outgoing", "incoming", or "both" (default)

    Returns:
        List of relationships with resolved entity names
    """
    relationships = get_entity_relationships(
        entity_id,
        include_expired=include_expired,
        predicate=predicate,
        direction=direction
    )
    return {
        "entity_id": entity_id,
        "include_expired": include_expired,
        "count": len(relationships),
        "relationships": relationships
    }


@mcp.tool()
def doris_graph_query(
    subject_type: Optional[str] = None,
    predicate: Optional[str] = None,
    object_type: Optional[str] = None,
    include_expired: bool = False,
    limit: int = 50
) -> dict:
    """
    Query the knowledge graph with pattern matching.

    Find relationships matching a pattern. All parameters are optional filters.

    Examples:
        - All family relationships: predicate="parent_of" OR "married_to"
        - All project work: subject_type="person", predicate="works_on", object_type="project"
        - All scheduled events: predicate="scheduled_for"

    Args:
        subject_type: Filter by subject entity type (person, place, project, etc.)
        predicate: Filter by relationship type
        object_type: Filter by object entity type
        include_expired: Include historical relationships
        limit: Maximum results (default 50)

    Returns:
        List of matching relationships with entity details
    """
    limit = _clamp_limit(limit)
    for label, val in [("subject_type", subject_type), ("object_type", object_type)]:
        if val is not None:
            try:
                _validate_entity_type(val)
            except ValueError as e:
                return {"error": str(e)}

    results = graph_query(
        subject_type=subject_type,
        predicate=predicate,
        object_type=object_type,
        include_expired=include_expired,
        limit=limit
    )
    return {
        "pattern": {
            "subject_type": subject_type,
            "predicate": predicate,
            "object_type": object_type
        },
        "include_expired": include_expired,
        "count": len(results),
        "results": results
    }


@mcp.tool()
def doris_graph_entity_profile(entity_id: str) -> dict:
    """
    Get a complete profile for an entity.

    Returns the entity with all current relationships organized by predicate.
    This is the main function for building context about a person/thing.

    Args:
        entity_id: Entity ID to get profile for

    Returns:
        Entity with nested relationships and related entities
    """
    profile = get_entity_profile(entity_id)
    if not profile:
        return {"status": "not_found", "entity_id": entity_id}
    return {"status": "found", "profile": profile}


@mcp.tool()
def doris_graph_relationship_update(
    subject_id: str,
    predicate: str,
    new_value: str,
    source: Optional[str] = None
) -> dict:
    """
    Update a relationship value by expiring the old one and creating a new one.

    This preserves history (Zep pattern). The old value is still queryable.
    Use this when a value changes (e.g., email address, phone number, status).

    Args:
        subject_id: Entity ID
        predicate: Relationship type (e.g., 'has_email', 'has_status')
        new_value: The new value
        source: Source of the update

    Returns:
        Old and new relationship IDs
    """
    if len(new_value) > MAX_CONTENT_LENGTH:
        return {"status": "error", "message": f"new_value too long: {len(new_value)} chars (max {MAX_CONTENT_LENGTH})"}

    old_id, new_id = update_relationship_value(
        subject_id=subject_id,
        predicate=predicate,
        new_value=new_value,
        source=source
    )
    return {
        "status": "updated",
        "expired_relationship_id": old_id,
        "new_relationship_id": new_id,
        "predicate": predicate,
        "new_value": new_value
    }


# ============================================================================
# FACT EXTRACTION
# ============================================================================

def extract_facts_from_conversation(conversation: str) -> list[dict]:
    """
    DEPRECATED: This function is no longer used.

    Agent-driven extraction via key_facts param is the canonical approach.
    See AgeMem pattern in persistent-memory-architecture.md.

    Keeping for backwards compatibility but will be removed in future.
    """
    print("[MCP] WARNING: extract_facts_from_conversation is deprecated. Use agent-driven extraction.")
    try:
        import ollama

        prompt = f"""Review this conversation and extract facts worth remembering.

Categories:
- decision: Choices made ("decided to use X for Y")
- learning: New information discovered ("found out that X")
- project: Project status updates ("MyProject now has X working")
- person: Information about people ("Nathan said X")
- preference: Stated preferences ("I prefer X over Y")
- event: Time-based facts ("Meeting scheduled for X")

Return JSON array:
[
  {{"category": "decision", "content": "...", "subject": "optional", "confidence": 0.9}},
  ...
]

Only extract facts useful for future conversations.
Skip: pleasantries, routine confirmations, obvious context.

Conversation:
{conversation[:4000]}"""  # Limit to ~4k chars

        response = ollama.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )

        import json
        result = json.loads(response["message"]["content"])
        if isinstance(result, list):
            return result
        return result.get("facts", [])

    except Exception as e:
        print(f"[MCP] Fact extraction failed: {e}")
        return []


def extract_graph_from_conversation(conversation: str, source: str = "extracted") -> dict:
    """
    Extract entities and relationships from conversation for the knowledge graph.

    Uses Claude Haiku for fast, high-quality extraction (~1s vs 5-10s with local Ollama).

    Identifies:
    - Entities: people, places, projects, events, concepts mentioned
    - Relationships: how entities relate to each other with temporal context

    Returns dict with entities and relationships to create.
    """
    try:
        import json

        from llm.api_client import call_claude
        from llm.providers import resolve_model
        from security.prompt_safety import wrap_with_scan

        wrapped_conversation = wrap_with_scan(
            conversation[:4000], "graph-extraction-conversation"
        )

        prompt = f"""Analyze this conversation and extract knowledge graph data.

Extract:
1. ENTITIES - People, places, projects, events, or concepts mentioned
2. RELATIONSHIPS - How entities relate to each other

Entity types: person, place, project, event, concept

Common relationship predicates:
- Family: married_to, parent_of, child_of, sibling_of
- Work: works_on, owns, manages, created
- Location: lives_in, located_at, visited
- Events: scheduled_for, attending, happened_on
- Attributes: has_email, has_phone, has_age, has_status

Return ONLY valid JSON (no markdown, no explanation):
{{
  "entities": [
    {{"name": "...", "type": "person|place|project|event|concept", "metadata": {{}}}},
    ...
  ],
  "relationships": [
    {{
      "subject": "entity name",
      "predicate": "relationship type",
      "object": "entity name or literal value",
      "is_entity": true/false,
      "temporal": "optional date/time context"
    }},
    ...
  ]
}}

Focus on:
- New information not likely already known
- Specific facts with clear subjects and objects
- Temporal context when dates/times are mentioned

Skip:
- Generic statements without specific entities
- Routine pleasantries
- Things that would already be in the knowledge graph

{wrapped_conversation}"""

        response = call_claude(
            messages=[{"role": "user", "content": prompt}],
            source="graph-extract",
            model=resolve_model("utility"),
            max_tokens=2048,
        )

        # Extract text from response
        result_text = response.text

        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            # Strip markdown code fences
            lines = result_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            result_text = "\n".join(lines)

        result = json.loads(result_text)

        # Validate structure
        if not isinstance(result, dict):
            return {"entities": [], "relationships": []}

        return {
            "entities": result.get("entities", []),
            "relationships": result.get("relationships", [])
        }

    except Exception as e:
        print(f"[MCP] Graph extraction failed: {e}")
        return {"entities": [], "relationships": []}


def process_graph_extraction(extraction: dict, source: str = "extracted") -> dict:
    """
    Process extracted graph data and create entities/relationships in the database.

    Returns summary of what was created.
    """
    created_entities = []
    created_relationships = []
    errors = []

    # Create entities
    for ent in extraction.get("entities", []):
        try:
            name = ent.get("name")
            ent_type = ent.get("type", "concept")
            metadata = ent.get("metadata", {})

            if not name:
                continue

            entity_id = find_or_create_entity(name, ent_type, metadata)
            created_entities.append({
                "id": entity_id,
                "name": name,
                "type": ent_type
            })
        except Exception as e:
            errors.append({"entity": ent.get("name"), "error": str(e)})

    # Create relationships
    for rel in extraction.get("relationships", []):
        try:
            subject_name = rel.get("subject")
            predicate = rel.get("predicate")
            object_val = rel.get("object")
            is_entity = rel.get("is_entity", True)

            if not subject_name or not predicate or not object_val:
                continue

            # Find or create subject entity (type from extraction data or "unknown")
            subject_id = find_or_create_entity(subject_name, "unknown")

            if is_entity:
                # Object is another entity
                object_id = find_or_create_entity(object_val, "unknown")
                rel_id = add_relationship(
                    subject_id=subject_id,
                    predicate=predicate,
                    object_id=object_id,
                    source=source
                )
            else:
                # Object is a literal value
                rel_id = add_relationship(
                    subject_id=subject_id,
                    predicate=predicate,
                    object_value=object_val,
                    source=source
                )

            created_relationships.append({
                "id": rel_id,
                "subject": subject_name,
                "predicate": predicate,
                "object": object_val
            })
        except Exception as e:
            errors.append({"relationship": f"{rel.get('subject')} -> {rel.get('object')}", "error": str(e)})

    return {
        "entities_created": len(created_entities),
        "relationships_created": len(created_relationships),
        "entities": created_entities,
        "relationships": created_relationships,
        "errors": errors if errors else None
    }


# Pure ASGI auth middleware for HTTP transport.
# Wraps the Starlette app returned by FastMCP.streamable_http_app().
def _create_auth_asgi_middleware(app, api_key: str):
    """Wrap an ASGI app with X-API-Key header authentication.

    Allows /health without auth. All other paths require a valid
    X-API-Key header, compared with constant-time comparison to prevent
    timing attacks. Supports comma-separated tokens for key rotation.

    Returns a new ASGI callable that wraps the original app.
    """
    import json
    from security.crypto import token_matches_any

    async def auth_middleware(scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await app(scope, receive, send)

        # Allow health checks without auth
        path = scope.get("path", "")
        if path == "/health":
            return await app(scope, receive, send)

        # Extract X-API-Key from headers
        headers = dict(scope.get("headers", []))
        provided_key = headers.get(b"x-api-key", b"").decode("utf-8", errors="ignore")

        if not token_matches_any(provided_key, api_key):
            # Audit log the failure
            from security.audit import audit as _audit
            # Extract client IP from ASGI scope
            client_info = scope.get("client")
            client_ip = client_info[0] if client_info else "unknown"
            reason = "missing_key" if not provided_key else "invalid_key"
            _audit.auth_failure(ip=client_ip, reason=reason, endpoint=path, component="mcp")

            # Reject with 401
            body = json.dumps({"error": "Invalid or missing API key"}).encode()
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
            })
            return

        return await app(scope, receive, send)

    return auth_middleware


# Entry point for running as MCP server
if __name__ == "__main__":
    print(f"[Doris MCP] Starting server...")
    print(f"[Doris MCP] Transport: {MCP_TRANSPORT}")

    if MCP_TRANSPORT == "http":
        print(f"[Doris MCP] Host: {MCP_HOST}:{MCP_PORT}")

        # Read auth token from environment
        _mcp_auth_token = os.environ.get("DORIS_MCP_AUTH_TOKEN", "").strip()
        _dev_mode = os.environ.get("DORIS_DEV_MODE", "").strip().lower() in ("true", "1", "yes")

        if not _mcp_auth_token and not _dev_mode:
            print("\n" + "=" * 70)
            print("FATAL: DORIS_MCP_AUTH_TOKEN is not set!")
            print("=" * 70)
            print("The MCP HTTP server refuses to start without authentication.")
            print("Anyone with network access could read/write all memories.")
            print()
            print("Either:")
            print("  1. Set DORIS_MCP_AUTH_TOKEN in your .env file")
            print("  2. Set DORIS_DEV_MODE=true for local development (NOT for production)")
            print("=" * 70 + "\n")
            sys.exit(1)

        # Get the Starlette ASGI app from FastMCP
        asgi_app = mcp.streamable_http_app()

        if _mcp_auth_token:
            # Wrap with auth middleware
            asgi_app = _create_auth_asgi_middleware(asgi_app, _mcp_auth_token)
            print("[Doris MCP] Auth enabled — X-API-Key header required on all requests (except /health)")
        else:
            print("[Doris MCP] WARNING: Running without auth (DORIS_DEV_MODE=true)")

        # Run via uvicorn directly (gives us control over the ASGI app)
        import uvicorn
        uvicorn.run(asgi_app, host=MCP_HOST, port=MCP_PORT, log_level="info")
    else:
        # Default STDIO transport for local Claude Desktop/Code
        mcp.run()
