"""
Doris Daemon

Background service that runs scouts and manages Doris's autonomous awareness.

Components:
- AwarenessDigest: Collects scout observations
- DorisScheduler: APScheduler-based job scheduling
- SessionManager: Persistent session state
- MemoryExtractor: Memory extraction before compaction
- DorisDaemon: Main orchestrator (in daemon.py)

Usage:
    from daemon import DorisScheduler, get_digest

    scheduler = DorisScheduler()
    scheduler.start()
"""

from daemon.digest import (
    AwarenessDigest,
    get_digest,
    save_digest,
)
from daemon.scheduler import DorisScheduler
from daemon.session import (
    SessionManager,
    SessionState,
    get_session_manager,
)
from daemon.extraction import (
    MemoryExtractor,
    get_extractor,
    extract_and_store,
)

__all__ = [
    # Digest
    "AwarenessDigest",
    "get_digest",
    "save_digest",
    # Scheduler
    "DorisScheduler",
    # Session
    "SessionManager",
    "SessionState",
    "get_session_manager",
    # Extraction
    "MemoryExtractor",
    "get_extractor",
    "extract_and_store",
]
