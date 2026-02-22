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
import json
import signal
import sys
from pathlib import Path
from datetime import datetime
import logging
import aiohttp

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from daemon.scheduler import DorisScheduler
from daemon.digest import get_digest, save_digest
from daemon.sitrep import SitrepEngine, LedgerEntry, ReviewLogEntry
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


class DorisDaemon:
    """
    Main daemon orchestrator.

    Manages:
    - Scout scheduling via DorisScheduler
    - Sitrep-based observation routing (instant lane + sitrep lane)
    - Wake-up triggers for Doris
    - Graceful shutdown
    """

    def __init__(self):
        self.scheduler = DorisScheduler(
            on_escalation=self._handle_observation,
            on_wake=self._handle_wake,
            on_sitrep_review=self._run_sitrep_review,
        )
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Sitrep engine — replaces hash-based dedup
        self._sitrep = SitrepEngine()

        # Instant lane buffering (emergencies, memory-scout critical)
        self._instant_buffer: list[Observation] = []
        self._instant_debounce_task: asyncio.Task | None = None
        self._instant_retry_task: asyncio.Task | None = None
        self._debounce_seconds: float = 30.0
        self._doris_url: str = "http://localhost:8000/chat/text"
        self._auth_headers: dict = {"Authorization": f"Bearer {settings.doris_api_token}"}

        # CC completion polling
        self._cc_poll_task: asyncio.Task | None = None
        self._cc_poll_interval: float = 30.0  # Check every 30 seconds

    # --- Observation Routing ---

    async def _handle_observation(self, observations: list[Observation]) -> None:
        """
        Route observations through the sitrep engine.

        Instant lane observations (emergencies, memory-scout critical)
        are debounced and flushed immediately. Everything else goes to the
        sitrep buffer for periodic review.
        """
        instant_obs = []

        for obs in observations:
            # Memoir nudges bypass sitrep entirely — separate creative flow
            if obs.scout == "memoir-scout":
                instant_obs.append(obs)
                continue

            result = self._sitrep.ingest(obs)
            if result is not None:
                # Instant lane — needs immediate attention
                instant_obs.append(result)

        if instant_obs:
            logger.warning(f"Instant lane: {len(instant_obs)} observation(s)")
            for obs in instant_obs:
                logger.warning(f"  [{obs.scout}] {obs.observation}")

            self._instant_buffer.extend(instant_obs)

            # Cancel existing debounce task if any
            if self._instant_debounce_task and not self._instant_debounce_task.done():
                self._instant_debounce_task.cancel()

            # Start new debounce task
            self._instant_debounce_task = asyncio.create_task(
                self._debounce_and_flush_instant()
            )

        sitrep_count = len(observations) - len(instant_obs)
        if sitrep_count > 0:
            logger.info(f"Sitrep lane: {sitrep_count} observation(s) buffered "
                        f"(total buffer: {self._sitrep.buffer_count})")

    # --- Instant Lane (emergencies + memoir) ---

    async def _debounce_and_flush_instant(self) -> None:
        """Wait for debounce period then flush instant lane to Doris."""
        try:
            await asyncio.sleep(self._debounce_seconds)
            await self._flush_instant_lane()
        except asyncio.CancelledError:
            logger.debug("Instant debounce cancelled (new observations received)")

    async def _flush_instant_lane(self) -> None:
        """
        Send instant lane observations to Doris immediately.

        Memoir nudges get their own prompt. Emergency observations get
        a direct proactive check prompt. No dedup — these are urgent.
        """
        if not self._instant_buffer:
            return

        # Separate memoir nudges from emergencies
        memoir_nudges = [obs for obs in self._instant_buffer if obs.scout == "memoir-scout"]
        emergency_obs = [obs for obs in self._instant_buffer if obs.scout != "memoir-scout"]

        logger.info(f"Flushing instant lane: {len(memoir_nudges)} memoir, {len(emergency_obs)} emergency")

        # Pre-flight budget check
        try:
            from llm.token_budget import check_budget, BudgetExceeded
            check_budget(15_000)
        except BudgetExceeded as e:
            logger.warning(f"Skipping instant flush — {e}. "
                           f"Buffer retained ({len(self._instant_buffer)} items).")
            return
        except Exception:
            pass

        all_succeeded = True

        try:
            timeout = aiohttp.ClientTimeout(total=45)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Handle memoir nudges
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

                # Handle emergency observations — immediate notify
                if emergency_obs:
                    obs_lines = [f"- [{obs.scout}] {obs.observation}" for obs in emergency_obs]
                    emergency_prompt = (
                        "[EMERGENCY ESCALATION] The following observations require immediate attention:\n\n"
                        + "\n".join(obs_lines)
                        + "\n\nThese bypassed sitrep review because they are life-safety or critical alerts. "
                        "Notify the user immediately via notify_user with priority='emergency'."
                    )
                    try:
                        async with session.post(
                            self._doris_url,
                            headers=self._auth_headers,
                            json={"message": emergency_prompt, "daemon": True, "wake_reason": "instant_escalation"}
                        ) as resp:
                            resp_text = await resp.text()
                            if resp.status >= 400:
                                logger.error(f"Doris emergency request failed (HTTP {resp.status}): {resp_text[:200]}")
                                all_succeeded = False
                            else:
                                logger.info(f"Doris emergency response ({resp.status}): {resp_text[:200]}")
                                # Record in sitrep ledger
                                for obs in emergency_obs:
                                    self._sitrep.record_notification(LedgerEntry(
                                        timestamp=datetime.now().isoformat(),
                                        scout=obs.scout,
                                        summary=obs.observation[:200],
                                        priority="emergency",
                                    ))
                    except Exception as e:
                        logger.error(f"Failed to send emergency to Doris: {e}")
                        all_succeeded = False

        except Exception as e:
            logger.error(f"Failed to create session for instant lane: {e}")
            all_succeeded = False

        if all_succeeded:
            cleared_count = len(self._instant_buffer)
            self._instant_buffer.clear()
            get_digest().clear_escalations()
            logger.info(f"Cleared {cleared_count} instant lane observations")
        else:
            logger.warning(
                f"Instant flush failed — {len(self._instant_buffer)} items retained, "
                "retrying in 60 seconds"
            )
            self._schedule_instant_retry()

    def _schedule_instant_retry(self) -> None:
        """Schedule a retry for failed instant lane flush."""
        if self._instant_retry_task and not self._instant_retry_task.done():
            return
        self._instant_retry_task = asyncio.create_task(self._retry_instant_flush())

    async def _retry_instant_flush(self) -> None:
        """Wait 60 seconds then retry flushing instant lane."""
        try:
            await asyncio.sleep(60)
            logger.info("Retrying instant lane flush...")
            await self._flush_instant_lane()
        except asyncio.CancelledError:
            logger.debug("Instant retry cancelled")

    # --- Sitrep Review (called on schedule, not on observation arrival) ---

    async def _run_sitrep_review(self) -> None:
        """
        Run a consolidated sitrep review.

        Called every 30 minutes by the scheduler. Builds a sitrep prompt
        with all buffered observations, notification ledger, ongoing
        conditions, and time context. POSTs to Doris for editorial review.
        """
        if not self._sitrep.should_review():
            logger.debug("Sitrep review skipped — no buffered observations")
            return

        logger.info(f"Starting sitrep review ({self._sitrep.buffer_count} observations)")

        # Budget check (sitrep reviews are ~20k tokens)
        try:
            from llm.token_budget import check_budget, BudgetExceeded
            check_budget(20_000)
        except BudgetExceeded as e:
            logger.warning(f"Skipping sitrep review — {e}. "
                           f"Buffer retained ({self._sitrep.buffer_count} items).")
            return
        except Exception:
            pass

        sitrep_prompt = self._sitrep.build_sitrep()
        review_error = None
        decisions = []
        conditions = []
        summary = ""
        notifications_sent = 0

        try:
            timeout = aiohttp.ClientTimeout(total=90)  # Longer timeout for review
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._doris_url,
                    headers=self._auth_headers,
                    json={"message": sitrep_prompt, "daemon": True, "wake_reason": "sitrep_review"}
                ) as resp:
                    resp_text = await resp.text()

                    if resp.status >= 400:
                        logger.error(f"Sitrep review failed (HTTP {resp.status}): {resp_text[:200]}")
                        review_error = f"HTTP {resp.status}"
                    else:
                        logger.info(f"Sitrep review response ({resp.status}): {resp_text[:200]}")

                        # Parse the response for structured data
                        try:
                            resp_data = json.loads(resp_text)
                            claude_text = resp_data.get("response", "")
                            tools_called = resp_data.get("tools_called", [])

                            # Extract JSON decisions from Claude's response
                            parsed = self._parse_sitrep_response(claude_text)
                            decisions = parsed.get("decisions", [])
                            conditions = parsed.get("conditions", [])
                            summary = parsed.get("summary", "")

                            # Record notifications from tool call metadata
                            notify_calls = [t for t in tools_called if t.get("name") == "notify_user"]
                            for call in notify_calls:
                                args = call.get("args", {})
                                self._sitrep.record_notification(LedgerEntry(
                                    timestamp=datetime.now().isoformat(),
                                    scout="sitrep_review",
                                    summary=args.get("message", args.get("action", "notification"))[:200],
                                    priority=args.get("priority", "proactive"),
                                ))
                                notifications_sent += 1

                            # Fallback: if no tool metadata, check decisions for NOTIFY
                            if not notify_calls:
                                for d in decisions:
                                    if d.get("action") == "NOTIFY":
                                        notifications_sent += 1

                            # Update conditions from Doris's response
                            if conditions:
                                self._sitrep.update_conditions(conditions)

                            # Store review summary for next time
                            if summary:
                                self._sitrep.record_review_summary(summary)

                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Failed to parse sitrep response: {e}")

        except Exception as e:
            logger.error(f"Sitrep review failed: {e}")
            review_error = str(e)

        # Log the review
        obs_count = self._sitrep.buffer_count
        scout_count = len(set(obs.scout for obs in self._sitrep._buffer))
        self._sitrep.log_review(ReviewLogEntry(
            timestamp=datetime.now().isoformat(),
            observation_count=obs_count,
            scout_count=scout_count,
            decisions=decisions,
            conditions=conditions,
            summary=summary,
            notifications_sent=notifications_sent,
            error=review_error,
        ))

        # Clear buffer on success (Doris has reviewed everything)
        if review_error is None:
            cleared = self._sitrep.clear_buffer()
            logger.info(f"Sitrep review complete: {len(cleared)} observations reviewed, "
                        f"{notifications_sent} notifications sent")
        else:
            logger.warning(f"Sitrep review had errors — buffer retained "
                           f"({self._sitrep.buffer_count} items)")

    @staticmethod
    def _parse_sitrep_response(text: str) -> dict:
        """
        Extract JSON from Doris's sitrep review response.

        Doris responds with JSON followed by tool calls. The JSON may be
        embedded in markdown code blocks or mixed with prose.
        """
        import re

        # Try markdown code block first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try bare JSON object
        json_match = re.search(r'(\{"decisions".*\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        return {"decisions": [], "conditions": [], "summary": ""}

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

        # Log sitrep state from previous run (if any)
        if self._sitrep.buffer_count > 0:
            logger.info(f"Sitrep buffer has {self._sitrep.buffer_count} observations from previous run")
        ledger_count = len(self._sitrep._ledger)
        if ledger_count > 0:
            logger.info(f"Notification ledger has {ledger_count} entries")

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
