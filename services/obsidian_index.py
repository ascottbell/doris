"""
Obsidian vault indexer for Doris memory.

Indexes key content from the user's Obsidian vault:
- Project documentation
- Personal ideas and notes
- Properties and checklists
"""

import os
from pathlib import Path
import re

OBSIDIAN_PATH = Path(os.environ.get("OBSIDIAN_VAULT_PATH", str(Path.home() / "Obsidian")))

# Directories to index (relative to OBSIDIAN_PATH)
# Configure these to match your vault structure
INDEX_DIRS = [
    # "Projects/MyProject/00-Reference",
    # "Personal/Ideas",
    # "Personal/Properties",
]

# Skip these patterns
SKIP_PATTERNS = [
    r"Archive/",
    r"Sessions/",
    r"\.obsidian/",
]


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    path_str = str(path)
    return any(re.search(p, path_str) for p in SKIP_PATTERNS)


def extract_key_facts(content: str, filename: str) -> list[dict]:
    """
    Extract key facts from markdown content.

    Returns list of facts with content and category.
    """
    facts = []

    # Split into sections by headers
    sections = re.split(r'^#+\s+', content, flags=re.MULTILINE)

    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split('\n')
        if not lines:
            continue

        header = lines[0].strip() if lines else ""
        body = '\n'.join(lines[1:]).strip()

        # Skip empty or very short sections
        if len(body) < 20:
            continue

        # Extract bullet points as individual facts
        bullets = re.findall(r'^[-*]\s+(.+)$', body, flags=re.MULTILINE)

        for bullet in bullets:
            # Clean up the bullet
            bullet = bullet.strip()
            if len(bullet) < 10:
                continue

            # Determine category based on content
            category = "project"
            if any(w in bullet.lower() for w in ['prefer', 'always', 'never', 'like', 'want']):
                category = "preference"
            elif any(w in bullet.lower() for w in ['decide', 'chose', 'will use', 'going with']):
                category = "decision"
            elif any(w in bullet.lower() for w in ['idea:', 'could', 'might', 'maybe']):
                category = "idea"

            facts.append({
                'content': bullet,
                'category': category,
                'source_file': filename,
                'section': header
            })

    return facts


def index_file(filepath: Path) -> list[dict]:
    """Index a single markdown file."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"[ObsidianIndex] Error reading {filepath}: {e}")
        return []

    # Get relative path for context
    rel_path = filepath.relative_to(OBSIDIAN_PATH)
    filename = str(rel_path)

    return extract_key_facts(content, filename)


def index_vault() -> dict:
    """
    Index the Obsidian vault and extract key facts.

    Returns dict with facts and stats.
    """
    all_facts = []
    files_processed = 0

    for dir_path in INDEX_DIRS:
        full_path = OBSIDIAN_PATH / dir_path
        if not full_path.exists():
            continue

        for md_file in full_path.rglob("*.md"):
            if should_skip(md_file):
                continue

            facts = index_file(md_file)
            all_facts.extend(facts)
            files_processed += 1

    # Also index specific important files
    # Add paths relative to OBSIDIAN_PATH for key project files
    important_files = [
        # "Projects/MyProject/Status.md",
    ]

    for rel_path in important_files:
        full_path = OBSIDIAN_PATH / rel_path
        if full_path.exists() and not should_skip(full_path):
            facts = index_file(full_path)
            all_facts.extend(facts)
            files_processed += 1

    return {
        'facts': all_facts,
        'files_processed': files_processed,
        'total_facts': len(all_facts)
    }


def store_obsidian_facts():
    """Index Obsidian and store facts in memory."""
    from memory.store import store_memory

    result = index_vault()
    stored = 0

    # Deduplicate by content
    seen = set()

    for fact in result['facts']:
        content = fact['content']

        # Skip duplicates
        if content in seen:
            continue
        seen.add(content)

        # Determine subject from source file path
        source = fact['source_file']
        if 'Properties' in source:
            subject = 'Property'
        else:
            # Use first directory component as subject
            parts = Path(source).parts
            subject = parts[1] if len(parts) > 1 else None

        store_memory(
            content=content,
            category=fact['category'],
            subject=subject,
            source=f"obsidian:{source}",
            confidence=0.9
        )
        stored += 1

        if stored % 10 == 0:
            print(f"[ObsidianIndex] Stored {stored} facts...")

    print(f"[ObsidianIndex] Indexed {result['files_processed']} files, stored {stored} unique facts")
    return result


if __name__ == "__main__":
    result = store_obsidian_facts()

    print(f"\n=== Obsidian Index Summary ===")
    print(f"Files processed: {result['files_processed']}")
    print(f"Total facts found: {result['total_facts']}")

    # Show sample facts
    print("\nSample facts:")
    for fact in result['facts'][:10]:
        print(f"  [{fact['category']}] {fact['content'][:60]}...")
