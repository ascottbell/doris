"""
Bootstrap Context Generation

Builds tiered context payload for Claude sessions.
Target: ~800-1200 tokens total.

Tier 1: Always included (~400 tokens) - Identity, active projects, preferences
Tier 2: Time-based (~200 tokens) - Morning/evening/weekend context
Tier 3: Query-based (~200-400 tokens) - Based on initial query keywords

Jan 2026 Update: Now queries the knowledge graph for identity/relationships.
Falls back to hardcoded defaults if graph is empty.
"""

from datetime import datetime
from typing import Optional
import json

from memory.store import get_all_active, find_similar_memories, search_fts
from security.prompt_safety import wrap_with_scan

# Graph functions (always local for now)
from memory.store import (
    find_entity_by_name,
    get_entity_profile,
    get_entities_by_type,
    graph_query,
)


# Core identity facts - always included
# NOTE: Override these via the knowledge graph (memory/seed.py) for your own setup.
# These defaults are intentionally generic.
IDENTITY = {
    "name": "User",
    "family": [],
    "locations": {
        "primary": "",
        "secondary": ""
    }
}

# Preferences - always included
PREFERENCES = [
    "Direct communication, no cheerleading or excessive praise",
    "Truth over comfort - push back when wrong",
    "Concise responses preferred",
    "Quality over speed for personal projects"
]

# Active projects - dynamically loaded from memory
# NOTE: Override via the knowledge graph for your own projects.
ACTIVE_PROJECTS = [
    {
        "name": "Doris",
        "description": "Personal AI assistant framework",
        "status": "Active development",
        "key_tech": ["Python", "FastAPI", "Claude"]
    },
]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token average)."""
    return len(text) // 4


def get_time_context() -> dict:
    """
    Generate time-based context.

    Morning (5am-12pm): Focus on today's schedule
    Afternoon (12pm-6pm): Work mode
    Evening (6pm-10pm): Wind-down context
    Weekend: Different focus
    """
    now = datetime.now()
    hour = now.hour
    is_weekend = now.weekday() >= 5

    context = {
        "current_time": now.strftime("%I:%M %p"),
        "day_of_week": now.strftime("%A"),
        "date": now.strftime("%B %d, %Y"),
    }

    if is_weekend:
        context["mode"] = "weekend"
        context["notes"] = "Weekend - likely family time or personal projects"
    elif hour < 12:
        context["mode"] = "morning"
        context["notes"] = "Morning - check calendar and priorities"
    elif hour < 18:
        context["mode"] = "afternoon"
        context["notes"] = "Work hours - focus mode"
    else:
        context["mode"] = "evening"
        context["notes"] = "Evening - winding down"

    return context


def get_identity_from_graph() -> Optional[dict]:
    """
    Build identity context from the knowledge graph.

    Queries for the user entity and builds identity dict from relationships.
    Returns None if graph has no user entity (fallback to defaults).
    """
    try:
        owner_name = IDENTITY.get("name", "User")
        owner = find_entity_by_name(owner_name, "person")
        if not owner:
            return None

        profile = get_entity_profile(owner['id'])
        if not profile or not profile.get('relationships'):
            return None

        # Build identity dict from graph
        identity = {
            "name": owner_name,
            "family": [],
            "locations": {}
        }

        rels = profile['relationships']

        # Extract family members
        if 'married_to' in rels:
            for r in rels['married_to']:
                if r.get('entity_name'):
                    # Get the entity for more details
                    identity['family'].append(f"{r['entity_name']} (wife)")

        if 'parent_of' in rels:
            for r in rels['parent_of']:
                if r.get('entity_name'):
                    # Try to get age/role from related entity
                    for related in profile.get('related_entities', []):
                        if related['id'] == r.get('entity_id'):
                            meta = related.get('metadata', {})
                            if isinstance(meta, str):
                                meta = json.loads(meta) if meta else {}
                            role = meta.get('role', 'child')
                            age = meta.get('age')
                            bday = meta.get('birthday')
                            desc = f"{r['entity_name']} ({role}"
                            if age:
                                desc += f", {age}"
                            if bday:
                                desc += f", birthday {bday}"
                            desc += ")"
                            identity['family'].append(desc)
                            break
                    else:
                        identity['family'].append(r['entity_name'])

        if 'owns_pet' in rels:
            for r in rels['owns_pet']:
                if r.get('entity_name'):
                    identity['family'].append(f"{r['entity_name']} (dog)")

        # Extract locations
        if 'lives_in' in rels:
            for r in rels['lives_in']:
                if r.get('entity_name'):
                    identity['locations']['primary'] = r['entity_name']

        if 'has_property_in' in rels:
            for r in rels['has_property_in']:
                if r.get('entity_name'):
                    identity['locations']['secondary'] = f"{r['entity_name']} (second home)"

        return identity if identity['family'] else None

    except Exception as e:
        print(f"[Bootstrap] Graph identity error: {e}")
        return None


def get_projects_from_graph() -> list[dict]:
    """
    Get active projects from the knowledge graph.

    Returns list of project dicts with name, description, status, tech.
    """
    try:
        # Query for person-works_on-project relationships
        work_rels = graph_query(
            subject_type="person",
            predicate="works_on",
            object_type="project"
        )

        projects = []
        seen_ids = set()

        for rel in work_rels:
            proj_id = rel.get('object_id')
            if proj_id and proj_id not in seen_ids:
                seen_ids.add(proj_id)
                # Get project details from related entity
                projects.append({
                    "name": rel.get('object_name', 'Unknown'),
                    "description": "",  # Would need to fetch entity metadata
                    "status": "Active",
                    "key_tech": []
                })

        # Enrich with entity metadata
        project_entities = get_entities_by_type("project")
        for proj in projects:
            for ent in project_entities:
                if ent['name'] == proj['name']:
                    meta = ent.get('metadata', {})
                    if isinstance(meta, str):
                        meta = json.loads(meta) if meta else {}
                    proj['description'] = meta.get('description', '')
                    proj['status'] = meta.get('status', 'Active')
                    proj['key_tech'] = meta.get('tech', [])
                    break

        return projects if projects else None

    except Exception as e:
        print(f"[Bootstrap] Graph projects error: {e}")
        return None


def get_relevant_memories_for_query(query: Optional[str], limit: int = 5, use_semantic: bool = False) -> list[dict]:
    """
    Find memories relevant to an initial query.

    Args:
        query: Search query
        limit: Max results
        use_semantic: If True, use embedding search (~400ms). If False, use FTS (~2ms).

    For bootstrap, we use FTS to keep latency low. Semantic search is available
    via doris_memory_query for explicit searches.
    """
    if not query:
        return []

    if use_semantic:
        # Semantic search - slow (~400ms) but higher quality
        similar = find_similar_memories(query, limit=limit)
        return [
            {
                "content": m["content"],
                "category": m["category"],
                "subject": m.get("subject")
            }
            for m in similar
        ]

    # FTS keyword search - fast (~2ms)
    try:
        fts_results = search_fts(query, limit=limit)
        return [
            {
                "content": m["content"],
                "category": m["category"],
                "subject": m.get("subject")
            }
            for m in fts_results
        ]
    except Exception:
        return []  # FTS can fail on special characters


def get_recent_decisions(limit: int = 5) -> list[dict]:
    """Get recent decisions from memory."""
    decisions = get_all_active(category="decision")
    return [
        {
            "content": d["content"],
            "subject": d.get("subject"),
            "created": d.get("created_at")
        }
        for d in decisions[:limit]
    ]


def get_recent_conversations(limit: int = 3) -> list[dict]:
    """
    Get recent conversation memories.

    Conversation memories are rich session summaries with:
    - topic: What the session was about
    - summary: High-level description
    - key_decisions: Decisions made during session
    - reasoning: Why decisions were made
    - open_questions: Unresolved issues
    - emotional_context: Mood/tone of conversation
    - participants: Who was involved

    These are stored in the metadata field.
    """
    conversations = get_all_active(category="conversation")
    results = []
    for c in conversations[:limit]:
        metadata = c.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                import json
                metadata = json.loads(metadata)
            except:
                metadata = {}
        results.append({
            "content": c["content"],
            "subject": c.get("subject"),
            "created": c.get("created_at"),
            "topic": metadata.get("topic"),
            "key_decisions": metadata.get("key_decisions", []),
            "open_questions": metadata.get("open_questions", []),
            "reasoning": metadata.get("reasoning"),
        })
    return results


def get_category_memories(category: str, limit: int = 10) -> list[dict]:
    """Get memories by category."""
    memories = get_all_active(category=category)
    return [
        {
            "id": m["id"],
            "content": m["content"],
            "subject": m.get("subject"),
            "created_at": m.get("created_at")
        }
        for m in memories[:limit]
    ]


def build_bootstrap_context(
    client: str = "unknown",
    query: Optional[str] = None,
    include_projects: bool = True,
    max_tokens: int = 1200
) -> dict:
    """
    Build the full bootstrap context payload.

    Jan 2026: Now queries knowledge graph first, falls back to hardcoded defaults.

    Args:
        client: Client identifier (claude-desktop, claude-code, etc.)
        query: Optional initial query to inform context
        include_projects: Whether to include project summaries
        max_tokens: Target max tokens for payload

    Returns:
        Bootstrap context dict with token count
    """
    # Try to get identity from graph first
    graph_identity = get_identity_from_graph()
    identity = graph_identity if graph_identity else IDENTITY

    context = {
        "identity": identity,
        "preferences": PREFERENCES,
        "time_context": get_time_context(),
    }

    # Tier 1: Always included
    if include_projects:
        # Try graph first, fall back to hardcoded
        graph_projects = get_projects_from_graph()
        context["active_projects"] = graph_projects if graph_projects else ACTIVE_PROJECTS

    # Tier 2: Recent decisions (wrap DB content as untrusted — could be poisoned)
    decisions = get_recent_decisions(limit=3)
    if decisions:
        for d in decisions:
            d["content"] = wrap_with_scan(d["content"], "memory")
        context["recent_decisions"] = decisions

    # Tier 3: Query-relevant memories (wrap DB content as untrusted)
    if query:
        relevant = get_relevant_memories_for_query(query, limit=5)
        if relevant:
            for m in relevant:
                m["content"] = wrap_with_scan(m["content"], "memory")
            context["relevant_memories"] = relevant

    # Load family memories from DB (wrap DB content as untrusted)
    family_memories = get_category_memories("family", limit=10)
    if family_memories:
        context["family_context"] = [
            wrap_with_scan(m["content"], "memory") for m in family_memories
        ]

    # Calculate token count
    json_str = json.dumps(context, indent=2)
    token_count = estimate_tokens(json_str)
    context["_meta"] = {
        "token_count": token_count,
        "client": client,
        "generated_at": datetime.now().isoformat()
    }

    return context


def format_bootstrap_for_prompt(context: dict) -> str:
    """
    Format bootstrap context as a system prompt section.
    This is what gets injected into Claude's context.
    """
    lines = [
        "=== DORIS CONTEXT ===",
        "",
        f"User: {context['identity']['name']}",
        f"Family: {', '.join(context['identity']['family'])}",
        f"Location: {context['identity']['locations']['primary']}",
        "",
        "Preferences:",
    ]

    for pref in context.get("preferences", []):
        lines.append(f"- {pref}")

    lines.append("")
    lines.append(f"Current: {context['time_context']['day_of_week']}, {context['time_context']['current_time']}")

    if context.get("active_projects"):
        lines.append("")
        lines.append("Active Projects:")
        for proj in context["active_projects"]:
            lines.append(f"- {proj['name']}: {proj['description']} ({proj['status']})")

    if context.get("recent_decisions"):
        lines.append("")
        lines.append("Recent Decisions:")
        for dec in context["recent_decisions"]:
            lines.append(f"- {dec['content']}")

    if context.get("relevant_memories"):
        lines.append("")
        lines.append("Relevant Context:")
        for mem in context["relevant_memories"]:
            lines.append(f"- {mem['content']}")

    if context.get("family_context"):
        lines.append("")
        lines.append("Family Notes:")
        for note in context["family_context"][:5]:  # Limit to 5
            lines.append(f"- {note}")

    lines.append("")
    lines.append("=== END DORIS CONTEXT ===")

    return "\n".join(lines)
