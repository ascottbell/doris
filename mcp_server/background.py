"""
Background worker for slow-path memory operations.

Runs graph extraction in a separate thread so it doesn't block
MCP tool responses. Uses a simple thread-safe queue pattern.

Why threading instead of asyncio:
1. MCP tools are sync functions - can't easily interact with async event loops
2. Threading works reliably regardless of event loop state
3. Simpler to implement and debug
"""

import threading
import queue
import logging
import atexit
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("doris.mcp.background")


@dataclass
class GraphExtractionJob:
    """A job to extract entities/relationships from conversation."""
    conversation: str
    source: str


class BackgroundWorker:
    """Thread-based background worker for slow-path operations."""

    def __init__(self):
        self._queue: queue.Queue[GraphExtractionJob] = queue.Queue(maxsize=100)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._started = False

    def _ensure_started(self):
        """Lazily start the worker thread on first job."""
        if self._started:
            return

        self._started = True
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[Background] Worker thread started")

        # Register cleanup
        atexit.register(self.stop)

    def stop(self):
        """Stop the worker thread."""
        if not self._running:
            return

        self._running = False
        # Put sentinel to unblock the queue.get()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        logger.info("[Background] Worker thread stopped")

    def queue_graph_extraction(self, conversation: str, source: str) -> bool:
        """
        Queue a graph extraction job (non-blocking).

        Returns True if queued, False if queue is full.
        """
        self._ensure_started()

        # Truncate to prevent OOM from huge conversations in the queue
        MAX_CONVERSATION_CHARS = 10_000
        if len(conversation) > MAX_CONVERSATION_CHARS:
            conversation = conversation[:MAX_CONVERSATION_CHARS] + "\n...[truncated]"

        job = GraphExtractionJob(conversation=conversation, source=source)
        try:
            self._queue.put_nowait(job)
            qsize = self._queue.qsize()
            logger.info(f"[Background] Queued graph extraction, queue size: {qsize}")
            return True
        except queue.Full:
            logger.warning("[Background] Queue full, dropping job")
            return False

    def _run(self):
        """Main worker loop - runs in background thread."""
        logger.info("[Background] Worker loop starting")

        while self._running:
            try:
                # Block with timeout so we can check _running flag
                job = self._queue.get(timeout=1.0)

                if job is None:  # Sentinel for shutdown
                    break

                self._process_job(job)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Background] Unexpected error: {e}", exc_info=True)

        logger.info("[Background] Worker loop exiting")

    def _process_job(self, job: GraphExtractionJob):
        """Process a single graph extraction job."""
        # Import here to avoid circular imports
        from mcp_server.server import extract_graph_from_conversation, process_graph_extraction

        try:
            logger.info(f"[Background] Processing extraction ({len(job.conversation)} chars)")

            # This is the slow part - LLM call via Ollama
            extraction = extract_graph_from_conversation(job.conversation, job.source)

            # Create entities/relationships in DB
            result = process_graph_extraction(extraction, source=job.source)

            entities = result.get("entities_created", 0)
            rels = result.get("relationships_created", 0)
            errors = result.get("errors")

            if errors:
                logger.warning(f"[Background] Extraction completed with errors: {errors}")
            else:
                logger.info(f"[Background] Extracted: {entities} entities, {rels} relationships")

        except Exception as e:
            logger.error(f"[Background] Extraction failed: {e}", exc_info=True)


# Singleton instance
_worker: Optional[BackgroundWorker] = None


def get_worker() -> BackgroundWorker:
    """Get the global background worker (creates if needed)."""
    global _worker
    if _worker is None:
        _worker = BackgroundWorker()
    return _worker
