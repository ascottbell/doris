"""
Wisdom Compilation System

Background job that distills raw wisdom entries into a living summary document.
Two sections:
  1. Narrative Understanding — prose about the user as a builder/decision-maker
  2. Operational Patterns — specific do's, don'ts, gotchas

Triggered when 5+ new entries exist since last compilation.
Uses Sonnet-class model (mid tier) for quality.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger("doris.wisdom_compiler")

# Paths
DATA_DIR = settings.data_dir
SUMMARY_PATH = DATA_DIR / "wisdom_summary.md"
STATE_PATH = DATA_DIR / "wisdom_compiler_state.json"

# Compilation threshold
MIN_NEW_ENTRIES = 5

COMPILATION_SYSTEM_PROMPT = """\
You are a reflective analyst studying one person's decision-making patterns over time.

You will receive a collection of wisdom entries — each one captures a decision, \
its reasoning, the outcome, and sometimes feedback. Some entries are polished; \
many are rough notes from automated systems. Your job is to distill ALL of them \
into a living summary that helps future AI assistants work effectively with this person.

Write TWO sections:

## Narrative Understanding

1-2 paragraphs of prose. Who is this person as a builder and decision-maker? \
What do they value? How do they think about quality, speed, risk, and tradeoffs? \
Write it like you're briefing a new collaborator who needs to understand this person \
deeply — not a resume, not flattery, just honest insight.

## Operational Patterns

A bullet list of specific, actionable patterns. Each should be a concrete \
do/don't/gotcha that would save a collaborator from making a mistake. \
Format: `- pattern (source context)`. Examples:
- Verify enforcement, not existence — checking that security code exists without \
testing it runs is the #1 way we've shipped broken security
- FTS5 default tokenizer doesn't split camelCase — use trigram for substring search
- Scan-and-tag, not scan-and-block

Keep it tight. No fluff. If two patterns overlap, merge them. \
If a pattern has strong feedback (positive or negative), weight it more heavily.

If you receive a PREVIOUS SUMMARY, evolve it — don't start from scratch. \
Add new patterns, refine the narrative, remove anything contradicted by newer evidence. \
The summary should get better over time, not just longer.
"""


def _load_state() -> dict:
    """Load compiler state (last compilation timestamp, entry count)."""
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt compiler state, resetting")
    return {"last_compiled_at": None, "last_entry_count": 0}


def _save_state(state: dict) -> None:
    """Save compiler state."""
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _load_previous_summary() -> Optional[str]:
    """Load the previous compiled summary, if it exists."""
    if SUMMARY_PATH.exists():
        try:
            return SUMMARY_PATH.read_text()
        except OSError:
            return None
    return None


def _get_all_wisdom_entries() -> list[dict]:
    """Fetch all wisdom entries from maasv."""
    from memory.wisdom import get_recent_wisdom
    # get_recent_wisdom returns sorted by timestamp DESC, get all
    return get_recent_wisdom(limit=10000)


def _count_entries_since(timestamp: Optional[str]) -> int:
    """Count wisdom entries created after the given timestamp."""
    if timestamp is None:
        return _get_total_count()

    from maasv.core.db import _plain_db as _db
    with _db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM wisdom WHERE timestamp > ?",
            (timestamp,)
        ).fetchone()
        return row[0] if row else 0


def _get_total_count() -> int:
    """Get total wisdom entry count."""
    from maasv.core.db import _plain_db as _db
    with _db() as conn:
        row = conn.execute("SELECT COUNT(*) FROM wisdom").fetchone()
        return row[0] if row else 0


def should_compile() -> tuple[bool, int]:
    """Check if compilation is needed. Returns (should_compile, new_entry_count)."""
    state = _load_state()
    new_count = _count_entries_since(state["last_compiled_at"])
    return new_count >= MIN_NEW_ENTRIES, new_count


def _format_entries_for_prompt(entries: list[dict]) -> str:
    """Format wisdom entries for the compilation prompt."""
    lines = []
    for entry in entries:
        parts = [f"[{entry.get('action_type', 'unknown')}]"]

        if entry.get('reasoning'):
            # Truncate very long reasoning (some entries have full prompts)
            reasoning = entry['reasoning']
            if len(reasoning) > 500:
                reasoning = reasoning[:500] + "..."
            parts.append(f"Reasoning: {reasoning}")

        if entry.get('context'):
            context = entry['context']
            if len(context) > 200:
                context = context[:200] + "..."
            parts.append(f"Context: {context}")

        if entry.get('outcome') and entry['outcome'] != 'pending':
            parts.append(f"Outcome: {entry['outcome']}")

        if entry.get('outcome_details'):
            details = entry['outcome_details']
            if len(details) > 300:
                details = details[:300] + "..."
            parts.append(f"Details: {details}")

        if entry.get('feedback_score') is not None:
            score = entry['feedback_score']
            notes = entry.get('feedback_notes', '')
            parts.append(f"Feedback: {score}/5" + (f" — {notes}" if notes else ""))

        if entry.get('tags'):
            tags = entry['tags']
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except (json.JSONDecodeError, TypeError):
                    tags = [tags]
            if tags:
                parts.append(f"Tags: {', '.join(tags)}")

        lines.append("\n".join(parts))

    return "\n---\n".join(lines)


def compile_wisdom(force: bool = False) -> Optional[str]:
    """
    Run the wisdom compilation.

    Args:
        force: If True, compile regardless of threshold

    Returns:
        The compiled summary text, or None if compilation was skipped.
    """
    from llm.api_client import call_llm
    from llm.providers import resolve_model

    state = _load_state()
    new_count = _count_entries_since(state["last_compiled_at"])

    if not force and new_count < MIN_NEW_ENTRIES:
        logger.debug(f"Wisdom compilation skipped: {new_count} new entries (need {MIN_NEW_ENTRIES})")
        return None

    logger.info(f"Starting wisdom compilation ({new_count} new entries)")

    # Get all entries
    entries = _get_all_wisdom_entries()
    if not entries:
        logger.info("No wisdom entries to compile")
        return None

    total_count = len(entries)
    logger.info(f"Compiling {total_count} total wisdom entries")

    # Build the user message
    previous_summary = _load_previous_summary()
    formatted_entries = _format_entries_for_prompt(entries)

    user_content = f"Here are {total_count} wisdom entries to compile:\n\n{formatted_entries}"
    if previous_summary:
        user_content = f"PREVIOUS SUMMARY (evolve this, don't start from scratch):\n\n{previous_summary}\n\n---\n\n{user_content}"

    # Call mid-tier model (Sonnet-class)
    model = resolve_model("mid")
    response = call_llm(
        messages=[{"role": "user", "content": user_content}],
        source="wisdom_compiler",
        model=model,
        max_tokens=4096,
        system=COMPILATION_SYSTEM_PROMPT,
    )

    summary = response.text
    if not summary:
        logger.error("Wisdom compilation returned empty response")
        return None

    # Add metadata header
    now = datetime.now(timezone.utc).isoformat()
    header = (
        f"<!-- Compiled: {now} | Entries: {total_count} | "
        f"New since last: {new_count} | Model: {model} -->\n\n"
    )
    full_summary = header + summary

    # Write to data/wisdom_summary.md
    SUMMARY_PATH.write_text(full_summary)
    logger.info(f"Wisdom summary written to {SUMMARY_PATH}")

    # Update state
    _save_state({
        "last_compiled_at": now,
        "last_entry_count": total_count,
        "last_new_count": new_count,
        "last_model": model,
    })

    logger.info("Wisdom compilation complete")
    return summary


async def check_and_compile() -> Optional[str]:
    """
    Async wrapper for scheduler integration.
    Checks threshold, compiles if needed, returns summary or None.
    """
    import asyncio

    needed, new_count = should_compile()
    if not needed:
        logger.debug(f"Wisdom compilation not needed ({new_count} new entries)")
        return None

    # Run the sync compile in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, compile_wisdom)
