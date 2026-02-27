"""
Persistent Session for Doris.

Provides a single, living conversation context that persists across all entry points
(HTTP, channels, daemon). Includes thread-safe operations, disk checkpointing, and
automatic compaction when context approaches token limits.
"""

import json
import threading
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Any
from dataclasses import dataclass, field, asdict

from security.file_io import atomic_json_write, locked_json_read, locked_json_update
from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session checkpoint encryption
#
# Checkpoints are encrypted at rest using Fernet with a key derived from
# DORIS_API_TOKEN via PBKDF2-SHA256. Falls back to plaintext in dev mode.
# Uses a distinct salt from OAuth token encryption so the derived keys differ.
# ---------------------------------------------------------------------------

_SALT = b"doris-session-checkpoint-v1"


def _get_fernet():
    """Return a Fernet/MultiFernet keyed from DORIS_API_TOKEN, or None in dev mode.

    Supports key rotation: when DORIS_API_TOKEN is comma-separated, encrypt() uses
    the first token and decrypt() tries all tokens in order.
    """
    from config import settings
    from security.crypto import get_fernet
    return get_fernet(settings.doris_api_token, _SALT)

# Configuration
MAX_CONTEXT_TOKENS = 180_000  # Conservative limit for 200K model context
COMPACTION_THRESHOLD = 0.80  # Trigger compaction at 80% capacity
VERBATIM_TURNS = 20  # Keep last 20 exchanges verbatim
CHECKPOINT_PATH = settings.data_dir / "session_checkpoint.json"
SESSIONS_DIR = settings.data_dir / "sessions"

# Token estimation (rough approximation: ~4 chars per token for English)
CHARS_PER_TOKEN = 4


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_summary: bool = False  # True for compacted summary messages

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            is_summary=data.get("is_summary", False)
        )


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation)."""
    return len(text) // CHARS_PER_TOKEN


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        # Account for role overhead (~4 tokens) plus content
        total += 4 + estimate_tokens(msg.content)
    return total


class PersistentSession:
    """
    The living conversation context for Doris.

    Features:
    - Thread-safe operations via RLock
    - Automatic checkpointing to disk
    - Background compaction when approaching token limits
    - WORM-aware (never compacts WORM content)
    - Activity tracking for sleep-time compute
    - Memory extraction before compaction
    """

    def __init__(self, checkpoint_path: Optional[Path] = None, session_key: Optional[str] = None):
        self._messages: list[Message] = []
        self._token_count: int = 0
        self._lock = threading.RLock()
        self._compaction_callback: Optional[Callable] = None
        self._compaction_in_progress = False
        self._last_checkpoint_time: float = 0
        self._checkpoint_interval: float = 60.0  # Checkpoint every 60 seconds minimum

        # Session identity
        self._session_key: Optional[str] = session_key
        self._checkpoint_path: Path = checkpoint_path or CHECKPOINT_PATH

        # Activity tracking for sleep-time compute
        self._last_activity_time: float = time.time()

        # Memory extraction tracking
        self._last_extraction_time: float = 0
        self._last_extraction_message_idx: int = 0  # Track which messages we've extracted

        # Try to load existing checkpoint
        self._load_checkpoint()

    @property
    def messages(self) -> list[Message]:
        """Get a copy of current messages."""
        with self._lock:
            return list(self._messages)

    @property
    def token_count(self) -> int:
        """Get current estimated token count."""
        with self._lock:
            return self._token_count

    @property
    def last_activity_time(self) -> float:
        """Get timestamp of last activity (for idle detection)."""
        with self._lock:
            return self._last_activity_time

    @property
    def last_extraction_time(self) -> float:
        """Get timestamp of last memory extraction (for coordination with sleep review)."""
        with self._lock:
            return self._last_extraction_time

    def touch(self) -> None:
        """Update last activity time (call on any user interaction)."""
        with self._lock:
            self._last_activity_time = time.time()

    def set_compaction_callback(self, callback: Callable) -> None:
        """Set the callback function for compaction."""
        self._compaction_callback = callback

    def append(self, role: str, content: str, is_summary: bool = False) -> None:
        """
        Add a message to the session.

        Triggers compaction check after adding. Compaction runs in background
        to avoid blocking the hot path.
        """
        msg = Message(role=role, content=content, is_summary=is_summary)
        msg_tokens = 4 + estimate_tokens(content)

        with self._lock:
            self._messages.append(msg)
            self._token_count += msg_tokens
            self._last_activity_time = time.time()  # Update activity on message
            logger.debug(f"[Session] Added {role} message ({msg_tokens} tokens, total: {self._token_count})")

        # Check if we need compaction (non-blocking)
        self._maybe_trigger_compaction()

        # Periodic checkpoint (debounced)
        self._maybe_checkpoint()

    def get_context(self) -> list[dict]:
        """
        Get messages formatted for Claude API.

        Returns list of {"role": str, "content": str} dicts.
        Filters out system messages as they're handled separately.
        """
        with self._lock:
            return [
                {"role": msg.role, "content": msg.content}
                for msg in self._messages
                if msg.role in ("user", "assistant")
            ]

    def get_context_with_system(self) -> list[dict]:
        """
        Get all messages including system messages.

        Used for daemon context injection.
        """
        with self._lock:
            return [
                {"role": msg.role, "content": msg.content}
                for msg in self._messages
            ]

    def checkpoint(self) -> None:
        """Save current session state to disk (encrypted if DORIS_API_TOKEN is set)."""
        with self._lock:
            data = {
                "version": 1,
                "timestamp": datetime.now().isoformat(),
                "token_count": self._token_count,
                "session_key": self._session_key,
                "messages": [msg.to_dict() for msg in self._messages]
            }

        try:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(data, indent=2).encode()
            fernet = _get_fernet()
            if fernet:
                payload = fernet.encrypt(payload)

            # Write atomically via temp file
            temp_path = self._checkpoint_path.with_suffix(".tmp")
            temp_path.write_bytes(payload)
            temp_path.chmod(0o600)
            temp_path.rename(self._checkpoint_path)
            self._last_checkpoint_time = time.time()
            key_label = f" [{self._session_key}]" if self._session_key else ""
            logger.info(f"[Session{key_label}] Checkpointed {len(self._messages)} messages ({self._token_count} tokens)")
        except Exception as e:
            logger.error(f"[Session] Failed to checkpoint: {e}")

    def _load_checkpoint(self) -> None:
        """Load session state from disk checkpoint (decrypts if encrypted).

        Handles three cases:
        1. Encrypted checkpoint (normal production) — decrypt and parse
        2. Plaintext checkpoint (legacy / dev mode) — parse directly, migrate
           to encrypted if an API token is available
        3. No checkpoint — start fresh
        """
        if not self._checkpoint_path.exists():
            key_label = f" [{self._session_key}]" if self._session_key else ""
            logger.info(f"[Session{key_label}] No checkpoint found, starting fresh")
            return

        try:
            raw = self._checkpoint_path.read_bytes()
            fernet = _get_fernet()
            data = None

            # Try decrypting first (normal case)
            if fernet:
                try:
                    plaintext = fernet.decrypt(raw)
                    data = json.loads(plaintext)
                except Exception:
                    pass

            # Fallback: maybe it's still plaintext (legacy or dev mode)
            if data is None:
                try:
                    data = json.loads(raw)
                    # Migrate to encrypted on next checkpoint
                    if fernet:
                        logger.info("[Session] Migrating plaintext checkpoint to encrypted")
                except Exception:
                    raise ValueError("Checkpoint is neither valid encrypted nor valid JSON")

            with self._lock:
                self._messages = [Message.from_dict(m) for m in data.get("messages", [])]
                self._token_count = data.get("token_count", 0)

                # Recalculate token count for safety
                if self._messages:
                    self._token_count = estimate_messages_tokens(self._messages)

            key_label = f" [{self._session_key}]" if self._session_key else ""
            logger.info(f"[Session{key_label}] Loaded checkpoint: {len(self._messages)} messages ({self._token_count} tokens)")

            # If we loaded plaintext and have a key, re-save encrypted
            if fernet:
                self.checkpoint()

        except Exception as e:
            logger.error(f"[Session] Failed to load checkpoint: {e}")
            # Start fresh on error
            with self._lock:
                self._messages = []
                self._token_count = 0

    def _maybe_checkpoint(self) -> None:
        """Checkpoint if enough time has passed since last checkpoint."""
        now = time.time()
        if now - self._last_checkpoint_time >= self._checkpoint_interval:
            # Run in background to avoid blocking
            threading.Thread(target=self.checkpoint, daemon=True).start()

    def _maybe_trigger_compaction(self) -> None:
        """Check if compaction is needed and trigger if so."""
        threshold = int(MAX_CONTEXT_TOKENS * COMPACTION_THRESHOLD)

        with self._lock:
            if self._token_count < threshold:
                return
            if self._compaction_in_progress:
                logger.debug("[Session] Compaction already in progress")
                return
            if not self._compaction_callback:
                logger.warning("[Session] Over threshold but no compaction callback set")
                return

            self._compaction_in_progress = True

        logger.info(f"[Session] Triggering compaction ({self._token_count}/{MAX_CONTEXT_TOKENS} tokens)")

        # Run compaction in background thread
        threading.Thread(target=self._run_compaction, daemon=True).start()

    def _get_messages_for_extraction(self) -> list[dict]:
        """
        Get messages that haven't been extracted yet.

        Returns list of dicts formatted for the extraction module.
        Only returns messages added since the last extraction.
        """
        with self._lock:
            # Get messages we haven't extracted yet
            if self._last_extraction_message_idx >= len(self._messages):
                return []

            new_messages = self._messages[self._last_extraction_message_idx:]

            # Format for extraction (matches daemon/session.py Message format)
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                }
                for msg in new_messages
            ]

    def _run_compaction(self) -> None:
        """Execute compaction in background, with memory extraction first."""
        try:
            # CRITICAL: Extract memories BEFORE compaction
            # This preserves rich context that would otherwise be lost to summarization
            messages_to_extract = self._get_messages_for_extraction()

            if messages_to_extract and len(messages_to_extract) > 10:
                logger.info(f"[Memory:Extraction] Starting extraction of {len(messages_to_extract)} messages before compaction...")
                try:
                    from daemon.extraction import MemoryExtractor

                    extractor = MemoryExtractor()

                    # Build a minimal session-like object for the extractor
                    # The extractor expects SessionState from daemon/session.py
                    from daemon.session import SessionState, Message as DaemonMessage
                    from datetime import datetime

                    extraction_state = SessionState()
                    for msg_dict in messages_to_extract:
                        ts = msg_dict["timestamp"]
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts)
                        extraction_state.messages.append(DaemonMessage(
                            role=msg_dict["role"],
                            content=msg_dict["content"],
                            timestamp=ts,
                        ))
                    extraction_state.total_tokens = self._token_count

                    # Run extraction synchronously (we're already in a background thread)
                    result = extractor.extract_from_session_sync(extraction_state)

                    if result.get("status") == "success" and result.get("extraction"):
                        stored_ids = extractor.store_extraction(result["extraction"])
                        logger.info(f"[Memory:Extraction] Stored {len(stored_ids)} memories from {len(messages_to_extract)} messages")

                        # Update memory metrics
                        _update_memory_metrics("extraction", {
                            "message_count": len(messages_to_extract),
                            "memories_stored": len(stored_ids),
                        })
                    else:
                        logger.warning(f"[Memory:Extraction] Extraction returned: {result.get('status', 'unknown')}")

                    # Update extraction tracking
                    with self._lock:
                        self._last_extraction_time = time.time()
                        self._last_extraction_message_idx = len(self._messages)

                except Exception as e:
                    logger.error(f"[Memory:Extraction] Extraction failed (continuing with compaction): {e}")
                    # Don't let extraction failure block compaction

            # Now run the actual compaction
            logger.info(f"[Memory:Compaction] Starting compaction (extraction ran: {self._last_extraction_time > 0})")
            self._compaction_callback(self)
            logger.info("[Memory:Compaction] Compaction completed")

            # Update compaction metrics
            try:
                _update_memory_metrics("compaction", {
                    "extraction_ran_first": self._last_extraction_time > time.time() - 60,
                })
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[Session] Compaction failed: {e}")
        finally:
            with self._lock:
                self._compaction_in_progress = False

    def replace_messages(self, new_messages: list[Message]) -> None:
        """
        Replace session messages (used by compaction).

        Thread-safe replacement with token recalculation.
        """
        with self._lock:
            self._messages = new_messages
            self._token_count = estimate_messages_tokens(new_messages)
            logger.info(f"[Session] Replaced with {len(new_messages)} messages ({self._token_count} tokens)")

        # Checkpoint after compaction
        self.checkpoint()

    def clear(self) -> None:
        """Clear all session messages."""
        with self._lock:
            self._messages = []
            self._token_count = 0
        self.checkpoint()
        logger.info("[Session] Cleared")


# --- Memory Metrics ---

MEMORY_METRICS_PATH = settings.data_dir / "memory_metrics.json"


def _default_memory_metrics() -> dict:
    """Return default memory metrics structure."""
    return {
        "extractions": {
            "total": 0,
            "last_at": None,
            "last_message_count": 0,
            "memories_stored": 0,
            "entities_extracted": 0,
            "failures": 0
        },
        "compactions": {
            "total": 0,
            "last_at": None,
            "extraction_ran_first": 0,
            "extraction_skipped": 0
        },
        "wisdom": {
            "entries_logged": 0,
            "tools_covered": [],
            "tools_skipped": []
        },
        "graph": {
            "entities_created": 0,
            "relationships_created": 0,
            "last_population_at": None
        }
    }


def _load_memory_metrics() -> dict:
    """Load memory metrics from disk under a shared lock."""
    return locked_json_read(MEMORY_METRICS_PATH, default=_default_memory_metrics())


def _save_memory_metrics(metrics: dict) -> None:
    """Save memory metrics to disk atomically."""
    try:
        atomic_json_write(MEMORY_METRICS_PATH, metrics)
    except Exception as e:
        logger.warning(f"Failed to save memory metrics: {e}")


def _update_memory_metrics(category: str, updates: dict) -> None:
    """
    Update memory metrics atomically with file locking.

    Args:
        category: "extraction", "compaction", "wisdom", or "graph"
        updates: Dict of fields to update
    """
    def _apply(metrics: dict) -> dict:
        if category == "extraction":
            metrics.setdefault("extractions", _default_memory_metrics()["extractions"])
            metrics["extractions"]["total"] += 1
            metrics["extractions"]["last_at"] = datetime.now().isoformat()
            if "message_count" in updates:
                metrics["extractions"]["last_message_count"] = updates["message_count"]
            if "memories_stored" in updates:
                metrics["extractions"]["memories_stored"] += updates["memories_stored"]
            if "entities_extracted" in updates:
                metrics["extractions"]["entities_extracted"] += updates["entities_extracted"]
            if "failed" in updates and updates["failed"]:
                metrics["extractions"]["failures"] += 1

        elif category == "compaction":
            metrics.setdefault("compactions", _default_memory_metrics()["compactions"])
            metrics["compactions"]["total"] += 1
            metrics["compactions"]["last_at"] = datetime.now().isoformat()
            if updates.get("extraction_ran_first"):
                metrics["compactions"]["extraction_ran_first"] += 1
            else:
                metrics["compactions"]["extraction_skipped"] += 1

        elif category == "wisdom":
            metrics.setdefault("wisdom", _default_memory_metrics()["wisdom"])
            if "entries_logged" in updates:
                metrics["wisdom"]["entries_logged"] += updates["entries_logged"]
            if "tool" in updates:
                if updates["tool"] not in metrics["wisdom"]["tools_covered"]:
                    metrics["wisdom"]["tools_covered"].append(updates["tool"])

        elif category == "graph":
            metrics.setdefault("graph", _default_memory_metrics()["graph"])
            if "entities_created" in updates:
                metrics["graph"]["entities_created"] += updates["entities_created"]
            if "relationships_created" in updates:
                metrics["graph"]["relationships_created"] += updates["relationships_created"]
            metrics["graph"]["last_population_at"] = datetime.now().isoformat()

        return metrics

    try:
        locked_json_update(MEMORY_METRICS_PATH, _apply, default=_default_memory_metrics())
    except Exception as e:
        logger.warning(f"Failed to update memory metrics: {e}")


def get_memory_metrics() -> dict:
    """Get current memory metrics (for health endpoint)."""
    return _load_memory_metrics()


# Session registry — keyed by session_key (None = default/main session)
_sessions: dict[Optional[str], PersistentSession] = {}
_session_lock = threading.Lock()


def _checkpoint_path_for_key(key: Optional[str]) -> Path:
    """Return the checkpoint file path for a given session key."""
    if key is None:
        return CHECKPOINT_PATH  # backwards-compatible default path
    # Sanitize key for filesystem: replace unsafe chars
    safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_").replace("..", "_")
    return SESSIONS_DIR / f"{safe_key}.json"


def get_session(session_key: Optional[str] = None) -> PersistentSession:
    """
    Get a PersistentSession by key (thread-safe lazy initialization).

    Args:
        session_key: Session identifier. None returns the default/main session
                     (backwards compatible). Channel sessions use "channel:sender_id".
    """
    if session_key in _sessions:
        return _sessions[session_key]

    with _session_lock:
        # Double-check after acquiring lock
        if session_key not in _sessions:
            cp_path = _checkpoint_path_for_key(session_key)
            _sessions[session_key] = PersistentSession(
                checkpoint_path=cp_path,
                session_key=session_key,
            )
            key_label = session_key or "main"
            logger.info(f"[Session] Initialized session [{key_label}]")
    return _sessions[session_key]
