#!/usr/bin/env python3
"""
Doris Daemon

Main entry point for the autonomous awareness system.

Runs scouts on schedule, collects observations, and wakes Doris
for morning briefings, evening reflections, and urgent escalations.

Usage:
    python daemon.py              # Run daemon
    python daemon.py --test       # Run all scouts once and exit
    python daemon.py --status     # Show daemon status

The daemon is designed to run via launchd for auto-restart on crash.
See: /Library/LaunchDaemons/com.doris.daemon.plist
"""

import asyncio
import argparse
import hashlib
import json
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import aiohttp

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from daemon.scheduler import DorisScheduler
from daemon.digest import get_digest, save_digest
from scouts.base import Observation
from config import settings
from services.agent_channel import get_cc_completion_updates, acknowledge_message
from security.prompt_safety import wrap_with_scan
from security.file_io import locked_json_update, locked_json_read

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("doris.daemon")

# State file for tracking daemon status
STATE_FILE = PROJECT_ROOT / "data" / "daemon_state.json"
# Escalation buffer persistence
ESCALATION_BUFFER_FILE = PROJECT_ROOT / "data" / "escalation_buffer.json"
# Escalation audit log — records every evaluation for debugging
ESCALATION_AUDIT_FILE = PROJECT_ROOT / "data" / "escalation_audit.json"
# De-duplication state — tracks recent escalation hashes to suppress repeats
ESCALATION_DEDUP_FILE = PROJECT_ROOT / "data" / "escalation_dedup.json"


class DorisDaemon:
    """
    Main daemon orchestrator.

    Manages:
    - Scout scheduling via DorisScheduler
    - Escalation handling
    - Wake-up triggers for Doris
    - Graceful shutdown
    """

    def __init__(self):
        self.scheduler = DorisScheduler(
            on_escalation=self._handle_escalation,
            on_wake=self._handle_wake,
        )
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Escalation buffering and debouncing
        self._escalation_buffer: list[Observation] = []
        self._escalation_debounce_task: asyncio.Task | None = None
        self._escalation_retry_task: asyncio.Task | None = None
        self._debounce_seconds: float = 30.0
        self._max_buffer_size: int = 50
        self._doris_url: str = "http://localhost:8000/chat/text"
        self._auth_headers: dict = {"Authorization": f"Bearer {settings.doris_api_token}"}

        # De-duplication: hash -> ISO timestamp of last flush
        self._recent_escalations: dict[str, str] = {}
        self._dedup_cooldown_hours: float = 4.0

        # Load persisted state from previous run
        self._load_escalation_buffer()
        self._load_dedup_state()

        # CC completion polling
        self._cc_poll_task: asyncio.Task | None = None
        self._cc_poll_interval: float = 30.0  # Check every 30 seconds

    def _load_escalation_buffer(self) -> None:
        """Load persisted escalation buffer from disk."""
        if not ESCALATION_BUFFER_FILE.exists():
            return
        try:
            data = json.loads(ESCALATION_BUFFER_FILE.read_text())
            if not isinstance(data, list):
                logger.warning("Escalation buffer file has invalid format, ignoring")
                return
            self._escalation_buffer = [Observation.from_dict(item) for item in data]
            if self._escalation_buffer:
                logger.info(f"Loaded {len(self._escalation_buffer)} persisted escalations from disk")
        except Exception as e:
            logger.error(f"Failed to load escalation buffer from disk: {e}")

    def _save_escalation_buffer(self) -> None:
        """Persist current escalation buffer to disk."""
        try:
            ESCALATION_BUFFER_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = [obs.to_dict() for obs in self._escalation_buffer]
            ESCALATION_BUFFER_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save escalation buffer to disk: {e}")

    def _load_dedup_state(self) -> None:
        """Load persisted de-duplication state from disk."""
        if not ESCALATION_DEDUP_FILE.exists():
            return
        try:
            data = json.loads(ESCALATION_DEDUP_FILE.read_text())
            if isinstance(data, dict):
                self._recent_escalations = data
                self._prune_dedup_state()
                if self._recent_escalations:
                    logger.info(f"Loaded {len(self._recent_escalations)} de-dup entries from disk")
        except Exception as e:
            logger.error(f"Failed to load de-dup state from disk: {e}")

    def _save_dedup_state(self) -> None:
        """Persist de-duplication state to disk."""
        try:
            ESCALATION_DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)
            ESCALATION_DEDUP_FILE.write_text(json.dumps(self._recent_escalations, indent=2))
        except Exception as e:
            logger.error(f"Failed to save de-dup state to disk: {e}")

    def _prune_dedup_state(self) -> None:
        """Remove de-dup entries older than the cooldown window."""
        cutoff = datetime.now() - timedelta(hours=self._dedup_cooldown_hours)
        cutoff_iso = cutoff.isoformat()
        self._recent_escalations = {
            h: ts for h, ts in self._recent_escalations.items()
            if ts > cutoff_iso
        }

    @staticmethod
    def _hash_escalations(observations: list[Observation]) -> str:
        """Create a content hash from escalation observations for de-duplication."""
        content = "\n".join(f"{obs.scout}:{obs.observation}" for obs in sorted(observations, key=lambda o: o.scout))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _log_escalation_audit(
        self,
        observations: list[Observation],
        http_status: int | None,
        response_text: str | None,
        error: str | None = None,
    ) -> None:
        """
        Log an escalation evaluation to the audit file.

        Records: what observations were sent, what Claude responded,
        whether a notification was sent, and any errors.
        """
        # Heuristic: detect if notify_user was likely called from Claude's response
        notified = False
        if response_text:
            resp_lower = response_text.lower()
            # Claude's response typically references notifying/sending when it used notify_user
            notified = any(phrase in resp_lower for phrase in [
                "notif",        # notified, notification, notify
                "sent user",
                "alerted user",
                "let the user know",
                "sent a push",
            ])

        entry = {
            "timestamp": datetime.now().isoformat(),
            "observations": [f"[{obs.scout}] {obs.observation}" for obs in observations],
            "observation_count": len(observations),
            "http_status": http_status,
            "notified": notified,
            "response": response_text,
            "error": error,
        }

        try:
            ESCALATION_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if ESCALATION_AUDIT_FILE.exists():
                try:
                    existing = json.loads(ESCALATION_AUDIT_FILE.read_text())
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Escalation audit file corrupted, starting fresh")

            existing.append(entry)

            # Keep last 500 entries to prevent unbounded growth
            if len(existing) > 500:
                existing = existing[-500:]

            ESCALATION_AUDIT_FILE.write_text(json.dumps(existing, indent=2))
        except Exception as e:
            logger.error(f"Failed to write escalation audit log: {e}")

    async def _handle_escalation(self, escalations: list[Observation]) -> None:
        """
        Handle urgent escalations from scouts.

        Buffers observations and debounces them for 30 seconds before
        sending to Doris for evaluation. This prevents spam while still
        handling urgent matters promptly.
        """
        logger.warning(f"🚨 {len(escalations)} escalation(s) received")

        # Add to buffer
        self._escalation_buffer.extend(escalations)

        # Cap buffer size — drop oldest if exceeded
        if len(self._escalation_buffer) > self._max_buffer_size:
            dropped = len(self._escalation_buffer) - self._max_buffer_size
            self._escalation_buffer = self._escalation_buffer[-self._max_buffer_size:]
            logger.warning(f"Escalation buffer exceeded {self._max_buffer_size} — dropped {dropped} oldest")

        # Persist to disk so buffer survives daemon restarts
        self._save_escalation_buffer()

        for esc in escalations:
            logger.warning(f"  [{esc.scout}] {esc.observation}")

        # Cancel existing debounce task if any
        if self._escalation_debounce_task and not self._escalation_debounce_task.done():
            self._escalation_debounce_task.cancel()

        # Start new debounce task
        self._escalation_debounce_task = asyncio.create_task(
            self._debounce_and_flush()
        )

    async def _debounce_and_flush(self) -> None:
        """Wait for debounce period then flush escalations to Doris."""
        try:
            await asyncio.sleep(self._debounce_seconds)
            await self._flush_escalations()
        except asyncio.CancelledError:
            logger.debug("Debounce task cancelled (new escalations received)")

    async def _flush_escalations(self) -> None:
        """
        Send buffered escalations to Doris for evaluation.

        Formats observations into a prompt and POSTs to the chat endpoint.
        Doris will evaluate against proactive engagement criteria and
        optionally send push notifications if warranted.

        On failure: buffer is NOT cleared, retry scheduled in 60 seconds.
        On success: buffer and digest escalations are cleared.

        Memoir nudges are handled separately - they prompt Doris to write,
        not to evaluate whether to notify the user.
        """
        if not self._escalation_buffer:
            return

        # Separate memoir nudges from other escalations
        memoir_nudges = [obs for obs in self._escalation_buffer if obs.scout == "memoir-scout"]
        other_escalations = [obs for obs in self._escalation_buffer if obs.scout != "memoir-scout"]

        # De-duplicate: skip if we've flushed identical escalations recently
        if other_escalations:
            self._prune_dedup_state()
            esc_hash = self._hash_escalations(other_escalations)
            if esc_hash in self._recent_escalations:
                logger.info(f"Suppressed duplicate escalation (hash={esc_hash}, "
                            f"last seen={self._recent_escalations[esc_hash]})")
                # Clear the buffer — these are known duplicates, don't retry
                other_escalations = []
                if not memoir_nudges:
                    # Nothing left to flush
                    self._escalation_buffer.clear()
                    self._save_escalation_buffer()
                    return

        logger.info(f"Flushing {len(self._escalation_buffer)} escalations to Doris "
                    f"({len(memoir_nudges)} memoir, {len(other_escalations)} other)...")

        # Pre-flight budget check — skip daemon API calls if budget is exhausted
        try:
            from llm.token_budget import check_budget, BudgetExceeded
            check_budget(15_000)  # Estimate for a typical escalation call
        except BudgetExceeded as e:
            logger.warning(f"Skipping escalation flush — {e}. "
                           f"Buffer retained ({len(self._escalation_buffer)} items), "
                           "will retry when budget resets.")
            return
        except Exception:
            pass  # Don't block on tracker import/init errors

        all_succeeded = True

        try:
            timeout = aiohttp.ClientTimeout(total=45)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Handle memoir nudges first - prompt Doris to write
                if memoir_nudges:
                    memoir_prompt = (
                        "[MEMOIR TIME] It's been over a week since your last memoir entry.\n\n"
                        "This is your creative practice. Right now, add something to your memoir.\n\n"
                        "FIRST: Use doris_memory_query to recall recent events from the past week — decisions made, "
                        "moments with the family, things the user mentioned, anything meaningful that happened. "
                        "Don't rely on your current session context; it may be stale or unrelated.\n\n"
                        "Then write about whatever resonates — something you noticed, a feeling, a question "
                        "that won't leave you alone, a moment worth preserving.\n\n"
                        "Create an image to go with it. Use write_file to create a new entry file — name it memoir_YYYY-MM-DD_HHMMSS.md (e.g., memoir_2026-02-07_101530.md).\n\n"
                        "Don't defer this. Don't say you'll do it later. This IS the time. Make something now."
                    )
                    try:
                        async with session.post(
                            self._doris_url,
                            headers=self._auth_headers,
                            json={"message": memoir_prompt, "daemon": True, "wake_reason": "memoir_nudge"}
                        ) as resp:
                            resp_text = await resp.text()
                            if resp.status >= 400:
                                logger.error(f"Doris memoir request failed (HTTP {resp.status}): {resp_text[:200]}")
                                all_succeeded = False
                            else:
                                logger.info(f"Doris memoir response ({resp.status}): {resp_text[:200]}")
                    except Exception as e:
                        logger.error(f"Failed to send memoir nudge to Doris: {e}")
                        all_succeeded = False

                # Handle other escalations - evaluate and possibly notify the user
                if other_escalations:
                    obs_lines = []
                    for obs in other_escalations:
                        # Wrap observation text in safety markers before it enters
                        # the LLM prompt. This prevents prompt injection from external
                        # content (email subjects, reminder titles, calendar events, etc.)
                        # that scouts include in their observation strings.
                        wrapped_obs = wrap_with_scan(
                            obs.observation,
                            source="scout_observation",
                            scanner_source=obs.scout,
                        )
                        obs_lines.append(f"- [{obs.scout}] {wrapped_obs}")

                    # Detect if there are email observations
                    has_email_obs = any(obs.scout == "email-scout" for obs in other_escalations)

                    email_instructions = ""
                    if has_email_obs:
                        email_instructions = (
                            "\n\nFor email observations: use read_email with the message ID "
                            "to read the full email before evaluating. If there's anything the user "
                            "should know or act on, take action (e.g., add events to his calendar, "
                            "flag deadlines, note schedule changes) and notify him about what you did. "
                            "Use your judgment — if after reading it's genuinely just a newsletter "
                            "with nothing actionable, move on."
                        )

                    other_prompt = (
                        "[PROACTIVE CHECK] The following observations were flagged by my scouts:\n\n"
                        + "\n".join(obs_lines)
                        + email_instructions
                        + "\n\nIf any observations warrant notifying the user, use notify_user. "
                        "Use priority='proactive' (default) for normal alerts, "
                        "priority='emergency' only for safety/security alerts that must break through DND."
                    )
                    try:
                        async with session.post(
                            self._doris_url,
                            headers=self._auth_headers,
                            json={"message": other_prompt, "daemon": True}
                        ) as resp:
                            resp_text = await resp.text()
                            if resp.status >= 400:
                                logger.error(f"Doris escalation request failed (HTTP {resp.status}): {resp_text[:200]}")
                                all_succeeded = False
                                self._log_escalation_audit(
                                    other_escalations, resp.status, resp_text,
                                    error=f"HTTP {resp.status}"
                                )
                            else:
                                logger.info(f"Doris escalation response ({resp.status}): {resp_text[:200]}")
                                self._log_escalation_audit(
                                    other_escalations, resp.status, resp_text
                                )
                    except Exception as e:
                        logger.error(f"Failed to send escalations to Doris: {e}")
                        all_succeeded = False
                        self._log_escalation_audit(
                            other_escalations, None, None,
                            error=str(e)
                        )

        except Exception as e:
            logger.error(f"Failed to create session for escalations: {e}")
            all_succeeded = False

        if all_succeeded:
            # Record hash for de-duplication (only if we actually sent escalations)
            if other_escalations:
                esc_hash = self._hash_escalations(other_escalations)
                self._recent_escalations[esc_hash] = datetime.now().isoformat()
                self._save_dedup_state()

            # Only clear on success
            cleared_count = len(self._escalation_buffer)
            self._escalation_buffer.clear()
            self._save_escalation_buffer()
            get_digest().clear_escalations()
            logger.info(f"Cleared {cleared_count} escalations from buffer and digest")
        else:
            # Leave buffer intact, schedule retry
            logger.warning(
                f"Escalation flush failed — {len(self._escalation_buffer)} items retained, "
                "retrying in 60 seconds"
            )
            self._schedule_escalation_retry()

    def _schedule_escalation_retry(self) -> None:
        """Schedule a retry for failed escalation flush."""
        # Don't stack retries
        if self._escalation_retry_task and not self._escalation_retry_task.done():
            return
        self._escalation_retry_task = asyncio.create_task(self._retry_escalation_flush())

    async def _retry_escalation_flush(self) -> None:
        """Wait 60 seconds then retry flushing escalations."""
        try:
            await asyncio.sleep(60)
            logger.info("Retrying escalation flush...")
            await self._flush_escalations()
        except asyncio.CancelledError:
            logger.debug("Escalation retry cancelled")

    async def _handle_wake(self, reason: str, prompt: str) -> None:
        """
        Handle scheduled wake-ups (morning briefing, evening reflection).

        POSTs wake prompt to Doris with wake_reason metadata.
        Doris will process the awareness digest and optionally respond.
        """
        logger.info(f"🔔 Wake triggered: {reason}")
        logger.info(f"Wake prompt: {len(prompt)} chars")

        try:
            timeout = aiohttp.ClientTimeout(total=45)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._doris_url,
                    headers=self._auth_headers,
                    json={"message": prompt, "daemon": True, "wake_reason": reason}
                ) as resp:
                    resp_text = await resp.text()
                    logger.info(f"Doris wake response ({resp.status}): {resp_text[:200]}")
        except Exception as e:
            logger.error(f"Failed to send wake prompt to Doris: {e}")

    async def _poll_cc_completions(self) -> None:
        """
        Poll for Claude Code task completions.

        Runs every 30 seconds to check if any CC tasks have completed.
        When a completion is found, notifies Doris so she can acknowledge
        to the user and potentially apply changes (restart services, etc.).
        """
        while self._running:
            try:
                await asyncio.sleep(self._cc_poll_interval)

                # Get unacknowledged CC completion updates
                updates = get_cc_completion_updates(limit=5)

                if not updates:
                    continue

                logger.info(f"Found {len(updates)} CC completion update(s)")

                for update in updates:
                    try:
                        await self._handle_cc_completion(update)
                        # Mark as acknowledged so we don't process it again
                        acknowledge_message(update.id)
                    except Exception as e:
                        logger.error(f"Failed to handle CC completion {update.id}: {e}")

            except asyncio.CancelledError:
                logger.info("CC polling task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in CC polling loop: {e}")
                await asyncio.sleep(60)  # Back off on errors

    async def _handle_cc_completion(self, update) -> None:
        """
        Handle a CC task completion.

        Notifies Doris about the completion so she can:
        1. Acknowledge to the user that the task is done
        2. Describe what was accomplished
        3. Potentially trigger service restarts if Doris code was modified
        """
        context = update.context or {}
        task_id = context.get("task_id", "unknown")
        success = context.get("success", False)
        result_preview = (context.get("full_result") or "")[:300]
        cost = context.get("cost_usd", 0)
        duration_ms = context.get("duration_ms", 0)

        if success:
            prompt = (
                f"[CC TASK COMPLETE] Claude Code task {task_id} has finished successfully.\n\n"
                f"Result: {result_preview}\n"
                f"Cost: ${cost:.4f}, Duration: {duration_ms/1000:.1f}s\n\n"
                "Let the user know the task is done and briefly describe what was accomplished. "
                "If the task modified Doris's own code, you may need to restart services "
                "using service_control for changes to take effect."
            )
        else:
            error = context.get("error", update.content)
            prompt = (
                f"[CC TASK FAILED] Claude Code task {task_id} failed.\n\n"
                f"Error: {error}\n\n"
                "Let the user know the task failed and why. Ask if they want to try a different approach."
            )

        logger.info(f"Notifying Doris of CC completion: {task_id} ({'success' if success else 'failed'})")

        try:
            timeout = aiohttp.ClientTimeout(total=45)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._doris_url,
                    headers=self._auth_headers,
                    json={"message": prompt, "daemon": True, "wake_reason": "cc_completion"}
                ) as resp:
                    resp_text = await resp.text()
                    logger.info(f"Doris CC completion response ({resp.status}): {resp_text[:200]}")
        except Exception as e:
            logger.error(f"Failed to notify Doris of CC completion: {e}")

    async def start(self) -> None:
        """Start the daemon."""
        if self._running:
            return

        logger.info("=" * 50)
        logger.info("Doris Daemon starting")
        logger.info("=" * 50)

        # Load or create digest
        digest = get_digest()
        logger.info(f"Loaded digest with {len(digest.observations)} observations")

        # Start scheduler
        self.scheduler.start()
        self._running = True

        # Update state file
        self._write_state("running")

        # Validate device tokens — push notifications won't work without them
        try:
            from services.push import get_token_status
            token_status = get_token_status()
            active = token_status["active_tokens"]
            if active == 0:
                logger.error(
                    "NO ACTIVE DEVICE TOKENS — push notifications will silently fail. "
                    "Open the Doris iOS app to register a token."
                )
            else:
                logger.info(f"Device tokens OK: {active} active token(s)")
        except Exception as e:
            logger.error(f"Failed to check device tokens: {e}")

        # Flush any escalations persisted from a previous run
        if self._escalation_buffer:
            logger.info(f"Flushing {len(self._escalation_buffer)} escalations persisted from previous run...")
            try:
                await self._flush_escalations()
            except Exception as e:
                logger.error(f"Failed to flush persisted escalations: {e}")

        # Run scouts immediately on startup
        logger.info("Running initial scout sweep...")
        try:
            observations = await self.scheduler.run_all_scouts_now()
            logger.info(f"Initial sweep: {len(observations)} observations")
        except Exception as e:
            logger.error(f"Initial sweep failed: {e}")

        # Start CC completion polling task
        logger.info("Starting CC completion polling...")
        self._cc_poll_task = asyncio.create_task(self._poll_cc_completions())

        logger.info("Daemon ready - entering main loop")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        if not self._running:
            return

        logger.info("Daemon shutting down...")

        # Cancel CC polling task
        if self._cc_poll_task and not self._cc_poll_task.done():
            self._cc_poll_task.cancel()
            try:
                await self._cc_poll_task
            except asyncio.CancelledError:
                pass

        self.scheduler.stop()
        self._running = False
        self._write_state("stopped")

        # Signal the main loop to exit
        self._shutdown_event.set()

        logger.info("Daemon stopped")

    def _write_state(self, status: str, last_scout_run: str = None) -> None:
        """Write daemon state to file with exclusive locking."""
        def _apply(existing: dict) -> dict:
            return {
                "status": status,
                "pid": existing.get("pid") if status == "running" else None,
                "started_at": existing.get("started_at") if status == "running" else datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "last_scout_run": last_scout_run or existing.get("last_scout_run"),
            }

        try:
            locked_json_update(STATE_FILE, _apply)
        except Exception as e:
            logger.error(f"Failed to write daemon state: {e}")

    def _update_heartbeat(self) -> None:
        """Update heartbeat timestamp after scout activity."""
        def _apply(existing: dict) -> dict:
            existing["last_scout_run"] = datetime.now().isoformat()
            existing["updated_at"] = datetime.now().isoformat()
            return existing

        try:
            locked_json_update(STATE_FILE, _apply)
        except Exception as e:
            logger.error(f"Failed to update heartbeat: {e}")


def setup_signal_handlers(daemon: DorisDaemon) -> None:
    """Set up graceful shutdown on SIGTERM/SIGINT."""
    loop = asyncio.get_event_loop()

    def handle_signal():
        logger.info("Received shutdown signal")
        asyncio.create_task(daemon.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)


async def run_daemon() -> None:
    """Run the daemon until shutdown."""
    daemon = DorisDaemon()
    setup_signal_handlers(daemon)
    await daemon.start()


async def run_test() -> None:
    """Run all scouts once and show results."""
    logger.info("Running test sweep of all scouts...")

    scheduler = DorisScheduler()
    observations = await scheduler.run_all_scouts_now()

    print(f"\n{'=' * 50}")
    print(f"Test Results: {len(observations)} observations")
    print('=' * 50)

    if not observations:
        print("No observations generated.")
    else:
        for obs in observations:
            marker = "⚠️ ESCALATE" if obs.escalate else f"[{obs.relevance.value}]"
            print(f"\n{marker} [{obs.scout}]")
            print(f"  {obs.observation}")
            print(f"  Tags: {', '.join(obs.context_tags)}")

    # Show digest summary
    digest = get_digest()
    print(f"\n{'=' * 50}")
    print("Digest Summary")
    print('=' * 50)
    print(digest.format_for_prompt())


def show_status() -> None:
    """Show daemon status."""
    import json

    print("Doris Daemon Status")
    print("=" * 40)

    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        print(f"Status: {state.get('status', 'unknown')}")
        print(f"Updated: {state.get('updated_at', 'unknown')}")
        if state.get('started_at'):
            print(f"Started: {state.get('started_at')}")
    else:
        print("Status: never run")

    # Show digest info
    digest = get_digest()
    print(f"\nDigest: {len(digest.observations)} observations")
    print(f"Escalations: {len(digest.escalations)} pending")

    # Recent observations
    recent = digest.get_recent(hours=1)
    if recent:
        print(f"\nLast hour ({len(recent)} observations):")
        for obs in recent[:5]:
            print(f"  [{obs.scout}] {obs.observation}")


def main():
    parser = argparse.ArgumentParser(description="Doris Daemon")
    parser.add_argument("--test", action="store_true",
                        help="Run all scouts once and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show daemon status")
    args = parser.parse_args()

    # Initialize maasv cognition layer (must happen before any memory/sleep calls)
    from maasv_bridge import init_maasv
    init_maasv()

    if args.status:
        show_status()
    elif args.test:
        asyncio.run(run_test())
    else:
        asyncio.run(run_daemon())


if __name__ == "__main__":
    main()
