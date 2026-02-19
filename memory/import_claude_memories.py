"""
Import Claude's exported memories.json into Doris memory store.

Run: python -m memory.import_claude_memories /path/to/memories.json
"""

import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.store import store_memory, get_all_active
from security.injection_scanner import scan_for_injection

logger = logging.getLogger(__name__)


def parse_section(text: str, section_name: str) -> str:
    """Extract a section from markdown-like text."""
    pattern = rf'\*\*{re.escape(section_name)}\*\*\s*\n+(.*?)(?=\n\*\*[A-Z]|\Z)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_facts_from_section(section_text: str, category: str, source: str) -> list[dict]:
    """Extract individual facts from a section of text."""
    facts = []

    # Split on sentences, keeping reasonable chunks
    sentences = re.split(r'(?<=[.!?])\s+', section_text)

    current_fact = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence keeps us under ~200 chars, combine
        if len(current_fact) + len(sentence) < 200:
            current_fact = f"{current_fact} {sentence}".strip()
        else:
            if current_fact:
                facts.append({
                    "content": current_fact,
                    "category": category,
                    "source": source
                })
            current_fact = sentence

    # Don't forget the last one
    if current_fact:
        facts.append({
            "content": current_fact,
            "category": category,
            "source": source
        })

    return facts


def parse_conversations_memory(text: str) -> list[dict]:
    """Parse the main conversations_memory into facts."""
    facts = []

    # Work context -> identity + project
    work = parse_section(text, "Work context")
    if work:
        facts.extend(extract_facts_from_section(work, "identity", "claude_export:work"))

    # Personal context -> family + preference
    personal = parse_section(text, "Personal context")
    if personal:
        # Split into family and preferences
        for fact in extract_facts_from_section(personal, "family", "claude_export:personal"):
            # Reclassify based on content
            content_lower = fact["content"].lower()
            if any(word in content_lower for word in ["wife", "husband", "son", "daughter", "dog", "children", "spouse"]):
                fact["category"] = "family"
            elif any(word in content_lower for word in ["prefers", "wants", "values", "uses", "exclusively"]):
                fact["category"] = "preference"
            facts.append(fact)

    # Top of mind -> project (current work)
    top_of_mind = parse_section(text, "Top of mind")
    if top_of_mind:
        facts.extend(extract_facts_from_section(top_of_mind, "project", "claude_export:current"))

    # Brief history sections
    history = parse_section(text, "Brief history")
    if history:
        facts.extend(extract_facts_from_section(history, "history", "claude_export:history"))

    # Other instructions -> preference
    instructions = parse_section(text, "Other instructions")
    if instructions:
        facts.extend(extract_facts_from_section(instructions, "preference", "claude_export:instructions"))

    return facts


def parse_project_memory(project_id: str, text: str) -> list[dict]:
    """Parse a project memory into facts."""
    facts = []

    # Try to extract project name from first paragraph
    first_para = text.split('\n\n')[0] if text else ""
    project_name = None

    # Look for "building X" or "X is a" patterns
    name_match = re.search(r'(?:building|developing)\s+(\w+)', first_para, re.IGNORECASE)
    if name_match:
        project_name = name_match.group(1)

    # Purpose & context section
    purpose = parse_section(text, "Purpose & context")
    if purpose:
        for fact in extract_facts_from_section(purpose, "project", f"claude_export:project:{project_id}"):
            if project_name:
                fact["subject"] = project_name
            facts.append(fact)

    # Tech stack section
    tech = parse_section(text, "Tech stack")
    if tech:
        for fact in extract_facts_from_section(tech, "project", f"claude_export:project:{project_id}"):
            if project_name:
                fact["subject"] = project_name
            facts.append(fact)

    # Architecture section
    arch = parse_section(text, "Architecture")
    if arch:
        for fact in extract_facts_from_section(arch, "decision", f"claude_export:project:{project_id}"):
            if project_name:
                fact["subject"] = project_name
            facts.append(fact)

    # Key decisions section
    decisions = parse_section(text, "Key decisions")
    if decisions:
        for fact in extract_facts_from_section(decisions, "decision", f"claude_export:project:{project_id}"):
            if project_name:
                fact["subject"] = project_name
            facts.append(fact)

    # If no sections matched, just parse the whole thing
    if not facts and text:
        for fact in extract_facts_from_section(text[:2000], "project", f"claude_export:project:{project_id}"):
            if project_name:
                fact["subject"] = project_name
            facts.append(fact)

    return facts


def import_memories(memories_path: str, dry_run: bool = False) -> dict:
    """Import memories from Claude export."""

    with open(memories_path) as f:
        data = json.load(f)

    # Handle array wrapper
    if isinstance(data, list):
        data = data[0]

    all_facts = []

    # Parse conversations memory
    conv_mem = data.get("conversations_memory", "")
    if conv_mem:
        print(f"Parsing conversations_memory ({len(conv_mem)} chars)...")
        facts = parse_conversations_memory(conv_mem)
        print(f"  Extracted {len(facts)} facts")
        all_facts.extend(facts)

    # Parse project memories
    proj_mems = data.get("project_memories", {})
    print(f"\nParsing {len(proj_mems)} project memories...")
    for proj_id, proj_text in proj_mems.items():
        facts = parse_project_memory(proj_id, proj_text)
        print(f"  Project {proj_id[:8]}...: {len(facts)} facts")
        all_facts.extend(facts)

    print(f"\nTotal facts extracted: {len(all_facts)}")

    # Group by category for summary
    by_cat = {}
    for f in all_facts:
        cat = f["category"]
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print("\nBy category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")

    if dry_run:
        print("\n[DRY RUN] Would import these facts:")
        for i, fact in enumerate(all_facts[:10], 1):
            print(f"  {i}. [{fact['category']}] {fact['content'][:60]}...")
        if len(all_facts) > 10:
            print(f"  ... and {len(all_facts) - 10} more")
        return {"extracted": len(all_facts), "imported": 0}

    # Import to memory store
    print("\nImporting to memory store...")
    imported = 0
    blocked = 0
    errors = []

    for fact in all_facts:
        # Scan for injection before storing
        scan_result = scan_for_injection(
            fact["content"], source=f"import_claude_memories:{fact.get('source', 'unknown')}"
        )
        if scan_result.risk_level == "high":
            logger.warning(
                f"Blocked high-risk import: {fact['content'][:100]!r} "
                f"(patterns: {scan_result.matched_patterns})"
            )
            print(f"  BLOCKED (high-risk injection): {fact['content'][:60]}...")
            blocked += 1
            continue

        try:
            source_tag = fact["source"]
            if scan_result.is_suspicious:
                source_tag += ":flagged"

            mem_id = store_memory(
                content=fact["content"],
                category=fact["category"],
                subject=fact.get("subject"),
                source=source_tag,
                confidence=0.85  # Slightly lower since auto-extracted
            )
            imported += 1
        except Exception as e:
            errors.append({"fact": fact["content"][:50], "error": str(e)})

    print(f"Imported {imported} memories")
    if blocked:
        print(f"Blocked (high-risk): {blocked}")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:3]:
            print(f"  - {err}")

    return {"extracted": len(all_facts), "imported": imported, "blocked": blocked, "errors": len(errors)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m memory.import_claude_memories <path/to/memories.json> [--dry-run]")
        sys.exit(1)

    memories_path = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    if not Path(memories_path).exists():
        print(f"Error: File not found: {memories_path}")
        sys.exit(1)

    result = import_memories(memories_path, dry_run=dry_run)
    print(f"\nDone: {result}")
