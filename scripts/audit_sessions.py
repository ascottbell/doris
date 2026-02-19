#!/usr/bin/env python3
"""
Audit Claude Code sessions from Jan 13-24 to extract facts that should have been logged.
"""

import json
import os
from datetime import datetime
from pathlib import Path

SESSION_DIR = Path(os.environ.get(
    "CLAUDE_SESSION_DIR",
    str(Path.home() / ".claude" / "projects")
))
START_DATE = datetime(2026, 1, 13)

# Sessions to audit — add your session UUIDs here
# Find them in your Claude Code sessions directory
SESSIONS = [
    # "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",  # Example session
]


def extract_session_summary(session_id: str) -> dict:
    """Extract key info from a session."""
    path = SESSION_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return None

    messages = []
    commits = []
    files_modified = []
    first_user_msg = None
    last_timestamp = None

    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except:
                continue

            # Get timestamp
            ts = entry.get("timestamp")
            if ts:
                last_timestamp = ts

            # Get user messages
            if entry.get("type") == "user":
                msg = entry.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, str) and content and not first_user_msg:
                    first_user_msg = content[:200]
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            if not first_user_msg:
                                first_user_msg = item.get("text", "")[:200]

            # Look for git commits
            if entry.get("type") == "assistant":
                msg = entry.get("message", {})
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_name = block.get("name", "")
                            tool_input = block.get("input", {})

                            # Track commits
                            if tool_name == "Bash":
                                cmd = tool_input.get("command", "")
                                if "git commit" in cmd:
                                    commits.append(cmd[:200])

                            # Track file modifications
                            if tool_name in ("Write", "Edit"):
                                fp = tool_input.get("file_path", "")
                                if fp and fp not in files_modified:
                                    files_modified.append(fp)

    return {
        "session_id": session_id,
        "timestamp": last_timestamp,
        "first_message": first_user_msg,
        "commits": commits,
        "files_modified": files_modified[:20],  # Limit
    }


def main():
    print("=" * 60)
    print("SESSION AUDIT: Jan 13-24, 2026")
    print("=" * 60)

    all_summaries = []

    for session_id in SESSIONS:
        print(f"\nProcessing {session_id}...")
        summary = extract_session_summary(session_id)
        if summary:
            all_summaries.append(summary)
            print(f"  Timestamp: {summary['timestamp']}")
            print(f"  First msg: {summary['first_message'][:80] if summary['first_message'] else 'N/A'}...")
            print(f"  Commits: {len(summary['commits'])}")
            print(f"  Files modified: {len(summary['files_modified'])}")

    # Save full audit
    output_path = Path(__file__).parent.parent / "data" / "session_audit.json"
    with open(output_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n\nFull audit saved to: {output_path}")
    print(f"Total sessions: {len(all_summaries)}")
    print(f"Total commits: {sum(len(s['commits']) for s in all_summaries)}")


if __name__ == "__main__":
    main()
