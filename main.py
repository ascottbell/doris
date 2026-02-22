import os
import time
import asyncio
from contextlib import asynccontextmanager
import json as _json
from typing import Annotated, Literal
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from config import settings
from security.auth import verify_token
from llm.brain import chat_claude, ClaudeResponse
from mcp_client import init_mcp, shutdown_mcp, get_mcp_manager
from tools.gmail import scan_recent_emails, summarize_important_emails, get_unread_count
from services.status import get_consolidated_status
from proactive import init_proactive_db, init_scheduler, shutdown_scheduler
from services.daemon_monitor import start_daemon_monitor, stop_daemon_monitor

# Session continuity - persistent conversation context
from session import get_session, compact_session

# Sleep-time compute - background consolidation during idle
from sleep import get_sleep_worker, start_idle_monitor, stop_idle_monitor
from sleep.worker import SleepJob, JobType


def _start_sleep_time_compute(session):
    """
    Initialize sleep-time compute with idle detection.

    When the session becomes idle (no activity for 30s), queues
    background jobs for inference, review, and graph reorganization.
    """
    from memory.store import get_entities_by_type, get_recent_memories

    def on_idle():
        """Called when session becomes idle."""
        worker = get_sleep_worker()
        worker.resume_work()

        # Get context for sleep jobs
        messages = session.messages
        if not messages:
            return

        # Format messages for jobs
        formatted_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in messages[-50:]  # Last 50 messages
        ]

        # Queue inference job (resolve vague references)
        try:
            entities = get_entities_by_type("person") + get_entities_by_type("place") + get_entities_by_type("restaurant")
            worker.queue_job(SleepJob(
                job_type=JobType.INFERENCE,
                data={
                    "messages": formatted_messages[-20:],  # Recent messages only
                    "entities": entities[:50]  # Limit entities
                },
                priority=2  # Medium priority
            ))
        except Exception as e:
            print(f"[Sleep] Failed to queue inference job: {e}")

        # Queue review job (second-pass analysis)
        # Pass extraction time so review can skip if extraction ran recently
        try:
            recent_mems = get_recent_memories(hours=24, limit=20)
            worker.queue_job(SleepJob(
                job_type=JobType.REVIEW,
                data={
                    "messages": formatted_messages,
                    "recent_memories": recent_mems,
                    "last_extraction_time": session.last_extraction_time,
                },
                priority=1  # Lower priority
            ))
        except Exception as e:
            print(f"[Sleep] Failed to queue review job: {e}")

        # Queue reorganize job (graph optimization)
        try:
            worker.queue_job(SleepJob(
                job_type=JobType.REORGANIZE,
                data={"mode": "incremental"},
                priority=0  # Lowest priority
            ))
        except Exception as e:
            print(f"[Sleep] Failed to queue reorganize job: {e}")

        # Queue memory hygiene job (dedup and prune)
        # Runs with dry_run=False to actually clean up, but no consolidation
        # (consolidation is expensive and should be run manually)
        try:
            worker.queue_job(SleepJob(
                job_type=JobType.MEMORY_HYGIENE,
                data={
                    "mode": "incremental",
                    "dry_run": False,
                    "dedup": True,
                    "prune": True,
                    "consolidate": False
                },
                priority=0  # Lowest priority (maintenance)
            ))
        except Exception as e:
            print(f"[Sleep] Failed to queue memory hygiene job: {e}")

    def on_active():
        """Called when session becomes active after idle."""
        worker = get_sleep_worker()
        worker.cancel_current_work()

    # Start idle monitor
    start_idle_monitor(
        get_last_activity=lambda: session.last_activity_time,
        on_idle=on_idle,
        on_active=on_active
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Security gate — refuse to start without auth tokens in production mode
    from config import validate_security_settings
    validate_security_settings()

    # Initialize maasv cognition layer (must happen before any memory/sleep calls)
    from maasv_bridge import init_maasv
    init_maasv()
    print("[maasv] Cognition layer initialized")

    # Initialize Sentry error tracking (before anything else)
    if settings.sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.logging import LoggingIntegration
            import logging as _logging

            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                environment="production",
                integrations=[
                    FastApiIntegration(transaction_style="endpoint"),
                    LoggingIntegration(
                        level=_logging.INFO,        # Breadcrumbs for INFO+
                        event_level=_logging.ERROR,  # Events for ERROR+
                    ),
                ],
                traces_sample_rate=0.1,  # 10% of transactions
                send_default_pii=False,
                # Tag every event with the Doris server identity
                release=f"doris@{settings.ollama_model}",
            )
            sentry_sdk.set_user({"username": os.getenv("DORIS_USER", "default")})
            sentry_sdk.set_tag("server_type", "doris")
            print("[Sentry] Initialized error tracking")
        except ImportError:
            print("[Sentry] sentry-sdk not installed, skipping")
        except Exception as e:
            print(f"[Sentry] Warning: Could not initialize: {e}")
    else:
        print("[Sentry] No DSN configured, skipping error tracking")

    # Record server start time for uptime tracking
    import services.status as _status_mod
    _status_mod.server_start_time = time.time()

    # Initialize conversations database (local SQLite)
    from api.conversations import init_conversations_db
    print("[Conversations] Initializing SQLite tables...")
    init_conversations_db()

    # Initialize persistent session with compaction callback
    print("[Session] Initializing persistent session...")
    session = get_session()
    session.set_compaction_callback(compact_session)
    print(f"[Session] Loaded {len(session.messages)} messages ({session.token_count} tokens)")

    # Initialize MCP connections (for tools like Apple Music, Home Assistant)
    try:
        await init_mcp(verbose=True)
    except Exception as e:
        print(f"[MCP] Warning: Could not initialize MCP servers: {e}")

    # Initialize channel adapters (Telegram, BlueBubbles, etc.)
    # Channels need MCP ready (brain uses tools) so this comes after init_mcp.
    active_channels = []
    try:
        from channels.registry import create_adapters_from_config, create_message_handler

        _channel_adapters = create_adapters_from_config(settings)
        if _channel_adapters:
            _channel_handler = create_message_handler()
            for adapter in _channel_adapters:
                try:
                    await adapter.start(_channel_handler)
                    active_channels.append(adapter)
                    print(f"[Channel] {adapter.name} adapter started")
                except Exception as e:
                    print(f"[Channel] Warning: {adapter.name} failed to start: {e}")
        else:
            print("[Channel] No channels configured (API-only mode)")
    except Exception as e:
        print(f"[Channel] Warning: Could not initialize channel adapters: {e}")

    # Initialize proactive system (calendar reminders, weather alerts, email monitoring)
    try:
        init_proactive_db()
        init_scheduler()
        print("[Proactive] Scheduler started")
    except Exception as e:
        print(f"[Proactive] Warning: Could not start proactive scheduler: {e}")

    # Initialize sleep-time compute (background consolidation during idle)
    try:
        _start_sleep_time_compute(session)
        print("[Sleep] Sleep-time compute started")
    except Exception as e:
        print(f"[Sleep] Warning: Could not start sleep-time compute: {e}")

    # Start daemon health monitor (auto-restarts daemon if unresponsive)
    try:
        start_daemon_monitor()
        print("[DaemonMonitor] Started - will auto-restart daemon if unhealthy")
    except Exception as e:
        print(f"[DaemonMonitor] Warning: Could not start daemon monitor: {e}")

    yield

    # Shutdown: stop channels, sleep worker, checkpoint session, stop proactive scheduler, health monitor, and MCP connections
    for adapter in active_channels:
        try:
            await adapter.stop()
        except Exception as e:
            print(f"[Channel] Warning: {adapter.name} failed to stop cleanly: {e}")
    if active_channels:
        print(f"[Channel] Stopped {len(active_channels)} adapter(s)")
    try:
        stop_idle_monitor()
        get_sleep_worker().stop()
        print("[Sleep] Sleep-time compute stopped")
    except Exception:
        pass
    try:
        print("[Session] Checkpointing session...")
        session.checkpoint()
    except Exception as e:
        print(f"[Session] Warning: Could not checkpoint session: {e}")
    try:
        shutdown_scheduler()
    except Exception:
        pass
    try:
        await shutdown_mcp(verbose=True)
    except Exception:
        pass
    try:
        stop_daemon_monitor()
    except Exception:
        pass


app = FastAPI(title="Doris", version="0.1.0", lifespan=lifespan)

# CORS: restrictive by default, configurable via env var
# Format: DORIS_CORS_ORIGINS=http://localhost:3000,https://example.com
_cors_origins = [o.strip() for o in os.getenv("DORIS_CORS_ORIGINS", "").split(",") if o.strip()]
if "*" in _cors_origins:
    import logging as _cors_log
    _cors_log.getLogger("doris.security").error(
        "DORIS_CORS_ORIGINS contains '*' which is incompatible with allow_credentials=True. "
        "Removing wildcard — set explicit origins instead."
    )
    _cors_origins = [o for o in _cors_origins if o != "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
)

# Rate limiting on chat endpoints (applied after CORS so preflight requests pass through)
from security.rate_limit import RateLimiter, RateLimitMiddleware
_chat_limiter = RateLimiter(max_per_minute=settings.chat_rate_limit)
app.add_middleware(RateLimitMiddleware, limiter=_chat_limiter, path_prefixes=["/chat/"])

# Rate limiting on expensive / sensitive API endpoints (separate from chat)
# Covers Gemini ($2-5/call for research), Codex, MCP proxy, push notifications,
# agent messaging, brain endpoints, and test-escalation.
_api_limiter = RateLimiter(max_per_minute=settings.api_rate_limit)
app.add_middleware(RateLimitMiddleware, limiter=_api_limiter, path_prefixes=[
    "/gemini/", "/codex/", "/mcp/", "/push/", "/agents/", "/brain/", "/test-escalation",
])

# Security headers on all responses (outermost middleware layer)
# CSRF note: Doris uses Bearer token auth (not cookies) with deny-by-default CORS,
# so CSRF is mitigated by design — no separate CSRF middleware needed.
from security.headers import SecurityHeadersMiddleware
_enable_hsts = bool(os.getenv("DORIS_ENABLE_HSTS", "").lower() in ("true", "1", "yes"))
app.add_middleware(SecurityHeadersMiddleware, enable_hsts=_enable_hsts)


# --- Context dict validation (size + depth caps) ---

_CONTEXT_MAX_SIZE = 50_000  # bytes serialized
_CONTEXT_MAX_DEPTH = 5


def validate_context(v: dict | None) -> dict | None:
    """Validate context dict: cap serialized size and nesting depth."""
    if v is None:
        return v

    def _check_depth(obj, depth=0):
        if depth > _CONTEXT_MAX_DEPTH:
            raise ValueError(f"context nesting exceeds max depth of {_CONTEXT_MAX_DEPTH}")
        if isinstance(obj, dict):
            for val in obj.values():
                _check_depth(val, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _check_depth(item, depth + 1)

    _check_depth(v)

    serialized = _json.dumps(v, default=str)
    if len(serialized) > _CONTEXT_MAX_SIZE:
        raise ValueError(f"context serialized size ({len(serialized)}) exceeds max of {_CONTEXT_MAX_SIZE}")

    return v


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=settings.chat_max_message_length)

class ClientLocation(BaseModel):
    lat: float
    lon: float
    accuracy: float | None = None

class ClientContext(BaseModel):
    device: str | None = None
    timestamp: str | None = None
    location: ClientLocation | None = None

class ChatRequest(BaseModel):
    message: str = Field(max_length=settings.chat_max_message_length)
    history: list[ChatMessage] | None = Field(default=None, max_length=settings.chat_max_history_size)
    client_context: ClientContext | None = None  # Device/location context
    daemon: bool = False  # True if request came from daemon
    wake_reason: str | None = None  # "wake_word" or "background" for daemon requests
    user: str | None = None  # Identified speaker name

class ChatResponse(BaseModel):
    response: str
    source: str
    latency_ms: int
    tools_called: list[dict] = []

@app.get("/status")
async def consolidated_status(_token: str = Depends(verify_token)):
    """
    Consolidated status endpoint — single view of all Doris subsystems.
    """
    loop = asyncio.get_running_loop()
    status = await loop.run_in_executor(None, get_consolidated_status)
    return status


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/status")
async def health_status(_token: str = Depends(verify_token)):
    """
    Get detailed system health status including circuit breaker states.

    Returns:
        - Overall health status
        - Claude API status
        - Tool health by category
        - Circuit breaker details for each tool
    """
    from tools.circuit_breaker import get_health_tracker, get_circuit_breaker

    health_tracker = get_health_tracker()
    circuit_breaker = get_circuit_breaker()

    full_status = health_tracker.get_full_status()

    return {
        "status": "healthy" if full_status["healthy"] else "degraded",
        "claude_api": full_status["claude_status"],
        "disabled_tools": full_status["disabled_tools"],
        "degraded_tools": full_status["degraded_tools"],
        "circuit_breaker": full_status["circuit_details"],
    }


@app.post("/health/reset-circuit")
async def reset_circuit(tool_name: str = None, _token: str = Depends(verify_token)):
    """
    Manually reset a circuit breaker.

    Args:
        tool_name: Specific tool to reset, or None to reset all

    Use this to recover from a circuit breaker lockout.
    """
    from tools.circuit_breaker import get_circuit_breaker

    circuit_breaker = get_circuit_breaker()

    if tool_name:
        circuit_breaker.reset_circuit(tool_name)
        return {"status": "ok", "message": f"Reset circuit for {tool_name}"}
    else:
        circuit_breaker.reset_all()
        return {"status": "ok", "message": "Reset all circuits"}


# --- Health Data Sync (for Apple Health from iOS app) ---

class WorkoutEntry(BaseModel):
    """Single workout from Apple Health / HealthKit."""
    type: str = Field(max_length=100)
    duration_min: float | None = None
    distance_mi: float | None = None
    calories: float | None = None


class SleepStages(BaseModel):
    """Sleep stage breakdown from HealthKit."""
    deep_hours: float | None = None
    rem_hours: float | None = None
    core_hours: float | None = None
    awake_hours: float | None = None


class HealthSyncRequest(BaseModel):
    """Health data from Apple Health / HealthKit."""
    date: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$')  # YYYY-MM-DD
    steps: int | None = None
    workouts: list[WorkoutEntry] | None = Field(default=None, max_length=100)
    sleep_hours: float | None = None
    active_calories: int | None = None
    resting_hr: int | None = None
    stand_hours: int | None = None
    hrv: int | None = None  # Heart rate variability in ms
    vo2_max: float | None = None  # mL/(kg·min)
    sleep_stages: SleepStages | None = None


@app.post("/health/sync")
async def health_sync(request: HealthSyncRequest, _token: str = Depends(verify_token)):
    """
    Receive health data from Apple Health via iOS app.

    This data is used by HealthScout for:
    - Sleep debt monitoring
    - Activity level tracking
    - Workout consistency
    - HRV trends (stress/recovery)
    """
    from services.health_db import store_health_data

    # Store to SQLite (also generates insights)
    store_health_data(request.model_dump())

    # Log for debugging
    print(f"[Health] Synced data for {request.date}: {request.steps or 0} steps, {len(request.workouts or [])} workouts")

    return {
        "status": "ok",
        "date": request.date,
        "message": "Health data stored"
    }


@app.get("/health/data")
async def get_health_data(date: str | None = None, days: int = Query(default=7, le=365), _token: str = Depends(verify_token)):
    """Get stored health data, optionally filtered by date or last N days."""
    from services.health_db import get_health_data as db_get_health_data
    return db_get_health_data(date=date, days=days)


# --- End-to-End Escalation Test ---

@app.post("/test-escalation")
async def test_escalation(_token: str = Depends(verify_token)):
    """
    End-to-end test of the escalation → notification pipeline.

    Creates a synthetic test observation, runs it through Claude's evaluation
    (same path as real escalations), and reports the full chain result.

    Expected outcome: a visible push notification arrives on the user's phone.
    """
    from llm.brain import chat_claude, ClaudeResponse
    from services.push import get_token_status

    start = time.time()

    # Pre-flight: check tokens exist
    token_status = get_token_status()
    if token_status["active_tokens"] == 0:
        return {
            "status": "failed",
            "stage": "preflight",
            "error": "No active device tokens — push will fail even if Claude calls notify_user",
        }

    # Build the same prompt the daemon would send
    test_prompt = (
        "[PROACTIVE CHECK] The following observations were flagged by my scouts:\n\n"
        "- [test-scout] SYNTHETIC TEST: This is an automated pipeline test. "
        "Please call notify_user with the message 'Pipeline test successful' "
        "and priority='proactive'. This is not a real alert.\n\n"
        "This is a test of the escalation pipeline. You MUST call notify_user "
        "to verify the full chain works. Use priority='proactive'."
    )

    # Run through Claude — same as a real escalation via /chat/text
    try:
        loop = asyncio.get_running_loop()
        claude_response = await loop.run_in_executor(
            None, lambda: chat_claude(test_prompt, return_usage=True, use_session=False)
        )

        if isinstance(claude_response, ClaudeResponse):
            response_text = claude_response.text
            tokens_used = claude_response.input_tokens + claude_response.output_tokens
        else:
            response_text = str(claude_response)
            tokens_used = 0

    except Exception as e:
        print(f"[Wisdom] Claude evaluation error: {e}")
        return {
            "status": "failed",
            "stage": "claude_evaluation",
            "error": "Internal evaluation error",
            "duration_ms": int((time.time() - start) * 1000),
        }

    duration_ms = int((time.time() - start) * 1000)

    # Detect if notify_user was called from Claude's response
    resp_lower = response_text.lower()
    notified = any(phrase in resp_lower for phrase in [
        "notification sent",
        "pipeline test",
        "notif",
        "sent",
    ])

    return {
        "status": "passed" if notified else "warning",
        "stages": {
            "preflight": "ok",
            "claude_evaluation": "ok",
            "notify_called": notified,
        },
        "claude_response": response_text,
        "tokens_used": tokens_used,
        "duration_ms": duration_ms,
        "note": "Check your phone for 'Pipeline test successful' notification" if notified else
                "Claude may not have called notify_user — check response text",
    }


# --- Device Registration (Push Notifications) ---

class DeviceRegisterRequest(BaseModel):
    """Register a device for push notifications."""
    device_token: str = Field(pattern=r'^[0-9a-fA-F]{64}$')
    device_type: Literal["ios", "macos"] = "ios"
    device_name: str | None = None

# Store device tokens (in-memory for now, persisted to encrypted file)
_device_tokens: dict[str, dict] = {}
_device_tokens_file = Path(__file__).parent / "data" / "device_tokens.json"

# Encryption for device tokens at rest (same pattern as session/persistent.py)
_DEVICE_SALT = b"doris-device-tokens-v1"

def _get_device_fernet():
    """Return a Fernet/MultiFernet keyed from DORIS_API_TOKEN, or None in dev mode.

    Supports key rotation: when DORIS_API_TOKEN is comma-separated, encrypt() uses
    the first token and decrypt() tries all tokens in order.
    """
    from security.crypto import get_fernet
    return get_fernet(settings.doris_api_token, _DEVICE_SALT)

def _load_device_tokens():
    """Load device tokens from file (decrypts if encrypted)."""
    global _device_tokens
    if not _device_tokens_file.exists():
        _device_tokens = {}
        return

    import json
    try:
        raw = _device_tokens_file.read_bytes()
        fernet = _get_device_fernet()

        # Try decrypting first (normal case)
        if fernet:
            try:
                plaintext = fernet.decrypt(raw)
                _device_tokens = json.loads(plaintext)
                return
            except Exception:
                pass

        # Fallback: plaintext (legacy or dev mode)
        _device_tokens = json.loads(raw)
        # Migrate to encrypted if we have a key
        if fernet:
            _save_device_tokens()
    except Exception:
        _device_tokens = {}

def _save_device_tokens():
    """Save device tokens to file (encrypted if DORIS_API_TOKEN is set)."""
    import json
    _device_tokens_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_device_tokens, indent=2).encode()
    fernet = _get_device_fernet()
    if fernet:
        payload = fernet.encrypt(payload)
    _device_tokens_file.write_bytes(payload)
    _device_tokens_file.chmod(0o600)

# Load on startup
_load_device_tokens()

@app.post("/devices/register")
async def register_device(request: DeviceRegisterRequest, _token: str = Depends(verify_token)):
    """
    Register a device for push notifications.

    The iOS/macOS app sends its APNs token here so we can send pushes later.
    """
    global _device_tokens

    is_new = request.device_token not in _device_tokens

    _device_tokens[request.device_token] = {
        "device_type": request.device_type,
        "device_name": request.device_name,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    _save_device_tokens()

    # Also register in push service SQLite (for device_type-aware sending)
    try:
        from services.push import register_device as push_register
        push_register(
            token=request.device_token,
            device_name=request.device_name,
            device_type=request.device_type,
        )
    except Exception as e:
        print(f"[Push] SQLite registration failed (non-critical): {e}")

    print(f"[Push] {'Registered' if is_new else 'Updated'} device: {request.device_type} - {request.device_name or 'unnamed'}")

    return {
        "status": "registered" if is_new else "updated",
        "message": "Device registered for push notifications"
    }

@app.get("/devices")
async def list_devices(_token: str = Depends(verify_token)):
    """List all registered devices."""
    return {
        "devices": [
            {"token": k[:16] + "...", **v}  # Truncate token for display
            for k, v in _device_tokens.items()
        ],
        "count": len(_device_tokens)
    }


class PushRequest(BaseModel):
    """Send a push notification."""
    title: str = Field(default="Doris", max_length=256)
    body: str = Field(max_length=4096)
    device_token: str | None = None  # None = send to all devices
    badge: int | None = None


@app.post("/push/send")
async def send_push_notification(request: PushRequest, _token: str = Depends(verify_token)):
    """
    Send a push notification to registered devices.

    If device_token is provided, sends to that device only.
    Otherwise sends to all registered devices.
    """
    from tools.push import send_push

    result = await send_push(
        title=request.title,
        body=request.body,
        device_token=request.device_token,
        badge=request.badge,
    )
    return result


# --- Wisdom Endpoints (Learning from Experience) ---

class WisdomFeedbackRequest(BaseModel):
    """Add feedback to a wisdom entry."""
    wisdom_id: str
    score: int  # 1-5
    notes: str | None = None


@app.get("/wisdom")
async def list_wisdom(limit: int = Query(default=20, le=1000), pending_only: bool = False, _token: str = Depends(verify_token)):
    """List recent wisdom entries."""
    from memory.wisdom import get_recent_wisdom, get_pending_feedback

    if pending_only:
        entries = get_pending_feedback(limit=limit)
    else:
        entries = get_recent_wisdom(limit=limit)

    return {"entries": entries, "count": len(entries)}


@app.get("/wisdom/stats")
async def wisdom_stats(_token: str = Depends(verify_token)):
    """Get wisdom statistics."""
    from memory.wisdom import get_stats
    return get_stats()


@app.get("/wisdom/{wisdom_id}")
async def get_wisdom(wisdom_id: str, _token: str = Depends(verify_token)):
    """Get a specific wisdom entry."""
    from memory.wisdom import get_wisdom_by_id

    entry = get_wisdom_by_id(wisdom_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Wisdom entry not found")
    return entry


@app.post("/wisdom/feedback")
async def add_wisdom_feedback(request: WisdomFeedbackRequest, _token: str = Depends(verify_token)):
    """Add feedback to a wisdom entry."""
    from memory.wisdom import add_feedback

    if not 1 <= request.score <= 5:
        raise HTTPException(status_code=400, detail="Score must be 1-5")

    success = add_feedback(request.wisdom_id, request.score, request.notes)
    if not success:
        raise HTTPException(status_code=404, detail="Wisdom entry not found")

    return {"status": "feedback recorded", "score": request.score}


@app.delete("/wisdom/{wisdom_id}")
async def delete_wisdom(wisdom_id: str, _token: str = Depends(verify_token)):
    """Delete a wisdom entry."""
    from memory.wisdom import delete_wisdom

    success = delete_wisdom(wisdom_id)
    if not success:
        raise HTTPException(status_code=404, detail="Wisdom entry not found")

    return {"status": "deleted"}


# --- Memory Health Endpoint ---

@app.get("/health/memory")
async def memory_health(_token: str = Depends(verify_token)):
    """
    Check memory system health.

    Returns:
        - Status: healthy, warning, or error
        - Warnings: List of detected issues
        - Metrics: Current memory system metrics
    """
    from session.persistent import get_memory_metrics
    from llm.brain import WISDOM_REQUIRED_TOOLS

    metrics = get_memory_metrics()
    warnings = []

    # Check extraction is running before compaction
    compactions_total = metrics.get("compactions", {}).get("total", 0)
    extractions_total = metrics.get("extractions", {}).get("total", 0)
    extraction_skipped = metrics.get("compactions", {}).get("extraction_skipped", 0)

    if extraction_skipped > 0:
        warnings.append(f"{extraction_skipped} compactions ran without extraction - memories may be lost")

    if compactions_total > 0 and extractions_total == 0:
        warnings.append("Compactions running but no extractions recorded")

    # Check graph is being populated
    entities_created = metrics.get("graph", {}).get("entities_created", 0)
    if extractions_total > 5 and entities_created == 0:
        warnings.append("Extractions running but no entities created in knowledge graph")

    # Check wisdom coverage
    tools_covered = set(metrics.get("wisdom", {}).get("tools_covered", []))
    uncovered_tools = WISDOM_REQUIRED_TOOLS - tools_covered

    # Don't warn about uncovered tools if we haven't logged much yet
    wisdom_entries = metrics.get("wisdom", {}).get("entries_logged", 0)
    if wisdom_entries > 20 and len(uncovered_tools) > len(WISDOM_REQUIRED_TOOLS) // 2:
        warnings.append(f"Wisdom not logging many tools - {len(uncovered_tools)} tools never used")

    # Check for extraction failures
    extraction_failures = metrics.get("extractions", {}).get("failures", 0)
    if extraction_failures > 3:
        warnings.append(f"{extraction_failures} extraction failures recorded")

    status = "healthy"
    if warnings:
        status = "warning"
    if extraction_skipped > compactions_total // 2 and compactions_total > 2:
        status = "error"  # Most compactions skipping extraction is critical

    return {
        "status": status,
        "warnings": warnings,
        "metrics": metrics,
        "tools_coverage": {
            "required": list(WISDOM_REQUIRED_TOOLS),
            "covered": list(tools_covered),
            "uncovered": list(uncovered_tools),
        }
    }


# --- Agent Channel Endpoints (CC ↔ Doris) ---

from services.agent_channel import (
    Agent, MessageType, AgentMessage, StoredMessage,
    send_message, get_pending_messages, get_agent_messages,
    respond_to_message, notify_user_via_doris, get_conversation_thread
)


class AgentMessageRequest(BaseModel):
    """Send a message between agents."""
    from_agent: str  # "claude_code" or "doris"
    to_agent: str
    message_type: str = "chat"  # notification, question, handoff, update, chat
    content: str
    context: dict | None = None
    priority: str = "normal"
    expects_response: bool = False
    related_to: int | None = None

    _validate_context = field_validator('context')(validate_context)


@app.post("/agents/message")
async def agent_send_message(request: AgentMessageRequest, _token: str = Depends(verify_token)):
    """
    Send a message from one agent to another.

    Used for CC ↔ Doris communication.
    """
    message = AgentMessage(
        from_agent=Agent(request.from_agent),
        to_agent=Agent(request.to_agent),
        message_type=MessageType(request.message_type),
        content=request.content,
        context=request.context,
        priority=request.priority,
        expects_response=request.expects_response,
        related_to=request.related_to,
    )
    message_id = send_message(message)

    # If this is a notification to Doris asking to notify the user, also send push
    if (request.to_agent == "doris" and
        request.message_type == "notification" and
        "notify" in request.content.lower()):
        from tools.push import send_push
        await send_push(
            title="Claude Code",
            body=request.content,
        )

    return {"status": "sent", "message_id": message_id}


@app.get("/agents/messages")
async def agent_get_messages(
    agent: str | None = None,
    limit: int = Query(default=50, le=1000),
    _token: str = Depends(verify_token)
):
    """Get recent messages, optionally filtered by agent."""
    try:
        agent_enum = Agent(agent) if agent else None
        messages = get_agent_messages(agent=agent_enum, limit=limit)
        return {"messages": [m.model_dump() for m in messages]}
    except Exception as e:
        print(f"[AgentChannel] Error getting messages: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "Failed to retrieve messages", "messages": []}


@app.get("/agents/messages/pending")
async def agent_get_pending(agent: str, limit: int = Query(default=20, le=1000), _token: str = Depends(verify_token)):
    """Get pending messages that need response for an agent."""
    messages = get_pending_messages(Agent(agent), limit=limit)
    return {"pending": [m.model_dump() for m in messages]}


@app.get("/agents/thread/{message_id}")
async def agent_get_thread(message_id: int, _token: str = Depends(verify_token)):
    """Get a conversation thread."""
    messages = get_conversation_thread(message_id)
    return {"thread": [m.model_dump() for m in messages]}


class NotifyUserRequest(BaseModel):
    """Simple notification to the user via Doris."""
    message: str
    context: dict | None = None

    _validate_context = field_validator('context')(validate_context)


@app.post("/agents/notify-user")
async def agent_notify_user(request: NotifyUserRequest, _token: str = Depends(verify_token)):
    """
    Convenience endpoint: Claude Code needs the user's attention.

    Sends push notification and logs to agent channel.
    """
    # Log to agent channel
    message_id = notify_user_via_doris(request.message, request.context)

    # Send push notification
    from tools.push import send_push
    await send_push(
        title="Claude Code",
        body=request.message,
    )

    return {
        "status": "notified",
        "message_id": message_id,
        "push_sent": True
    }


# --- Gemini Consultant Endpoints ---

from services.gemini import (
    TaskType, quick_lookup, code_review, brainstorm,
    deep_research, validate as gemini_validate
)


class GeminiQueryRequest(BaseModel):
    """Request for Gemini consultation."""
    query: str = Field(max_length=50000)
    context: dict | None = None

    _validate_context = field_validator('context')(validate_context)


class GeminiCodeReviewRequest(BaseModel):
    """Request for code review."""
    code: str = Field(max_length=50000)
    description: str = Field(default="", max_length=5000)
    context: dict | None = None

    _validate_context = field_validator('context')(validate_context)


class GeminiResearchRequest(BaseModel):
    """Request for deep research."""
    question: str = Field(max_length=50000)
    context: dict | None = None

    _validate_context = field_validator('context')(validate_context)


@app.post("/gemini/quick")
async def gemini_quick_lookup(request: GeminiQueryRequest, _token: str = Depends(verify_token)):
    """Quick lookup using Gemini Flash - fast, low-cost."""
    result = await quick_lookup(request.query, request.context)
    return result


@app.post("/gemini/code-review")
async def gemini_code_review_endpoint(request: GeminiCodeReviewRequest, _token: str = Depends(verify_token)):
    """Get a thorough code review from Gemini Pro."""
    result = await code_review(request.code, request.description, request.context)
    return result


@app.post("/gemini/brainstorm")
async def gemini_brainstorm(request: GeminiQueryRequest, _token: str = Depends(verify_token)):
    """Brainstorm solutions with Gemini Pro - creative, divergent thinking."""
    result = await brainstorm(request.query, None, request.context)
    return result


@app.post("/gemini/research")
async def gemini_deep_research(request: GeminiResearchRequest, _token: str = Depends(verify_token)):
    """
    Comprehensive research using Google's Deep Research API.

    This is async and takes 2-5 minutes. Consults 50-100+ sources
    and produces a detailed report with citations.

    Cost: ~$2-5 per query.

    Returns:
        - success: bool
        - report: Full research report (markdown)
        - interaction_id: For reference
        - elapsed_seconds: Time taken
        - status: completed/failed/timeout
    """
    result = await deep_research(request.question, request.context)
    return result


@app.post("/gemini/validate")
async def gemini_validate_endpoint(request: GeminiQueryRequest, _token: str = Depends(verify_token)):
    """Validate a claim or assumption - critical analysis mode."""
    result = await gemini_validate(request.query, request.context)
    return result


# --- Codex Consultant Endpoints ---

from services.codex import (
    quick as codex_quick,
    code_review as codex_code_review,
    brainstorm as codex_brainstorm,
    validate as codex_validate,
    research as codex_research,
)


class CodexQueryRequest(BaseModel):
    """Request for Codex consultation."""
    query: str = Field(max_length=50000)
    context: dict | None = None

    _validate_context = field_validator('context')(validate_context)


class CodexCodeReviewRequest(BaseModel):
    """Request for Codex code review."""
    code: str = Field(max_length=50000)
    description: str = Field(default="", max_length=5000)
    context: dict | None = None

    _validate_context = field_validator('context')(validate_context)


class CodexResearchRequest(BaseModel):
    """Request for Codex research."""
    question: str = Field(max_length=50000)
    context: dict | None = None

    _validate_context = field_validator('context')(validate_context)


@app.post("/codex/quick")
async def codex_quick_endpoint(request: CodexQueryRequest, _token: str = Depends(verify_token)):
    """Quick lookup using GPT-5.3-Codex - fast, concise."""
    return await codex_quick(request.query, request.context)


@app.post("/codex/code-review")
async def codex_code_review_endpoint(request: CodexCodeReviewRequest, _token: str = Depends(verify_token)):
    """Get a thorough code review from GPT-5.3-Codex."""
    return await codex_code_review(request.code, request.description, request.context)


@app.post("/codex/brainstorm")
async def codex_brainstorm_endpoint(request: CodexQueryRequest, _token: str = Depends(verify_token)):
    """Brainstorm solutions with GPT-5.3-Codex."""
    return await codex_brainstorm(request.query, None, request.context)


@app.post("/codex/research")
async def codex_research_endpoint(request: CodexResearchRequest, _token: str = Depends(verify_token)):
    """Deep research analysis using GPT-5.3-Codex."""
    return await codex_research(request.question, request.context)


@app.post("/codex/validate")
async def codex_validate_endpoint(request: CodexQueryRequest, _token: str = Depends(verify_token)):
    """Validate a claim or assumption with GPT-5.3-Codex."""
    return await codex_validate(request.query, request.context)


# --- Brain Dump Endpoints ---

from tools.brain import (
    process_inbox, get_inbox_status, get_active_goals,
    get_relevant_thoughts, get_surfacing_context, format_proactive_message
)


@app.get("/brain/status")
async def brain_status(_token: str = Depends(verify_token)):
    """
    Get status of the brain dump inbox.

    Returns count of unprocessed thoughts and preview.
    """
    loop = asyncio.get_running_loop()
    status = await loop.run_in_executor(None, get_inbox_status)
    return status


@app.post("/brain/process")
async def brain_process(_token: str = Depends(verify_token)):
    """
    Process unprocessed thoughts in the inbox.

    Uses LLM to categorize each thought and move to organized files.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, process_inbox)
    return result


@app.get("/brain/goals")
async def brain_goals(limit: int = Query(default=5, le=100), days: int = Query(default=30, le=365), _token: str = Depends(verify_token)):
    """
    Get active goals for proactive surfacing.

    Returns goals from the last N days.
    """
    loop = asyncio.get_running_loop()
    goals = await loop.run_in_executor(None, lambda: get_active_goals(limit=limit, days=days))
    return {"goals": goals, "count": len(goals)}


@app.get("/brain/relevant")
async def brain_relevant(context: str, limit: int = Query(default=3, le=100), _token: str = Depends(verify_token)):
    """
    Get thoughts relevant to a context.

    Uses semantic search to find matching thoughts.
    """
    loop = asyncio.get_running_loop()
    thoughts = await loop.run_in_executor(None, lambda: get_relevant_thoughts(context, limit=limit))
    return {"thoughts": thoughts, "context": context}


@app.get("/brain/surface")
async def brain_surface(personality: str = "balanced", _token: str = Depends(verify_token)):
    """
    Get proactive surfacing suggestions based on current context.

    Combines weather, time of day, and health data to find
    relevant thoughts to surface.
    """
    from zoneinfo import ZoneInfo
    from datetime import datetime
    from tools.weather import get_current_weather

    loop = asyncio.get_running_loop()

    # Get current time of day
    eastern = ZoneInfo("America/New_York")
    now = datetime.now(eastern)
    hour = now.hour
    if hour < 12:
        time_of_day = "morning"
    elif hour < 17:
        time_of_day = "afternoon"
    else:
        time_of_day = "evening"

    # Get weather
    try:
        weather_data = await loop.run_in_executor(None, get_current_weather)
        weather = {
            "temp": weather_data.get("temperature", 0) if weather_data else 0,
            "conditions": weather_data.get("conditions", "") if weather_data else "",
        }
    except Exception:
        weather = None

    # Get today's health data from database
    today = now.strftime("%Y-%m-%d")
    try:
        from services.health_db import get_health_data as db_get_health_data
        health_data = db_get_health_data(date=today) or {}
    except Exception:
        health_data = {}

    # Get surfacing suggestions
    suggestions = await loop.run_in_executor(
        None,
        lambda: get_surfacing_context(weather=weather, time_of_day=time_of_day, health_data=health_data)
    )

    # Format message
    message = format_proactive_message(suggestions, personality=personality)

    return {
        "suggestions": suggestions,
        "message": message,
        "context": {
            "time_of_day": time_of_day,
            "weather": weather,
            "has_health_data": bool(health_data),
        }
    }


@app.post("/chat/text", response_model=ChatResponse)
async def chat_text(request: ChatRequest, _token: str = Depends(verify_token)):
    import uuid
    request_id = str(uuid.uuid4())[:8]
    device = request.client_context.device if request.client_context else "unknown"

    # Log daemon context if present
    daemon_info = ""
    if request.daemon:
        daemon_info = f" [daemon:{request.wake_reason or 'unknown'}]"

    print(f"[API:{request_id}] /chat/text from {device}{daemon_info}: {request.message[:50]}...")

    start = time.time()

    # Track token usage
    tokens_input = 0
    tokens_output = 0
    success = True
    error_msg = None

    # Convert history to format expected by chat_claude
    # If client provides history, use it (backward compatibility)
    # Otherwise, use persistent session
    history = None
    use_session = False
    if request.history:
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]
    else:
        # No explicit history - use persistent session
        use_session = True

    # Extract location from client context and persist it
    location = None
    if request.client_context and request.client_context.location:
        location = {
            "lat": request.client_context.location.lat,
            "lon": request.client_context.location.lon
        }
        # Store location for tracking (non-blocking)
        try:
            from services.location_db import store_location
            store_location(location["lat"], location["lon"], source="ios")
        except Exception as e:
            print(f"[Location] Failed to store: {e}")

    # Handle daemon requests with session context
    message_text = request.message
    if request.daemon:
        if request.wake_reason:
            # Wake-up with reason (morning briefing, evening reflection)
            # These benefit from session context — keep use_session=True
            if use_session:
                session = get_session()
                session.append("system", f"[DAEMON WAKE: {request.wake_reason}]")
            message_text = (
                f"[PROACTIVE WAKE - {request.wake_reason}]\n"
                f"{request.message}"
            )
        else:
            # Escalation check (no wake_reason)
            # These don't need conversation history — run sessionless to avoid
            # bloating the persistent session and sending 86K+ tokens per call
            use_session = False
            history = []
            message_text = (
                "[PROACTIVE OBSERVATION - DO NOT RESPOND ALOUD]\n"
                f"{request.message}"
            )

    # Get speaker name from user ID if provided
    speaker = request.user.title() if request.user else None

    try:
        # Run Claude in thread pool to avoid blocking event loop
        loop = asyncio.get_running_loop()
        claude_response = await loop.run_in_executor(
            None, lambda: chat_claude(message_text, history=history, return_usage=True, location=location, speaker=speaker, use_session=use_session)
        )
        if isinstance(claude_response, ClaudeResponse):
            response = claude_response.text
            tokens_input = claude_response.input_tokens
            tokens_output = claude_response.output_tokens
            tools_called = claude_response.tools_called
        else:
            response = str(claude_response)
            tools_called = []

    except Exception as e:
        success = False
        error_msg = str(e)
        response = "Sorry, something went wrong processing your message."
        tools_called = []
        import logging
        logging.getLogger("doris.api").exception("Chat error")

    latency_ms = int((time.time() - start) * 1000)

    print(f"[API:{request_id}] Response ({latency_ms}ms): {response[:50]}...")
    return ChatResponse(response=response, source="claude", latency_ms=latency_ms, tools_called=tools_called)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, _token: str = Depends(verify_token)):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Returns text token-by-token as Claude generates it, with tool execution
    events interleaved. Clients see text appear incrementally instead of
    waiting for the full response.

    SSE event format:
        data: {"type": "text", "content": "Hello "}\n\n
        data: {"type": "tool_start", "tool": "get_weather"}\n\n
        data: {"type": "tool_complete"}\n\n
        data: {"type": "done", "latency_ms": 1234}\n\n
        data: {"type": "error", "message": "..."}\n\n
    """
    import uuid
    import json as json_mod

    from llm.brain import (
        get_system_prompt_cached, execute_tool,
        _get_wisdom_context_for_message, log_token_usage,
    )
    from llm.tools import TOOLS
    from llm.types import StopReason, TokenUsage, ToolCall, ToolResult
    from llm.providers import get_llm_provider

    request_id = str(uuid.uuid4())[:8]
    device = request.client_context.device if request.client_context else "unknown"
    print(f"[API:{request_id}] /chat/stream from {device}: {request.message[:50]}...")

    start = time.time()

    # --- Same request preprocessing as /chat/text ---

    history = None
    use_session = False
    if request.history:
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]
    else:
        use_session = True

    location = None
    if request.client_context and request.client_context.location:
        location = {
            "lat": request.client_context.location.lat,
            "lon": request.client_context.location.lon
        }
        try:
            from services.location_db import store_location
            store_location(location["lat"], location["lon"], source="ios")
        except Exception as e:
            print(f"[Location] Failed to store: {e}")

    message_text = request.message
    speaker = request.user.title() if request.user else None

    # --- Build messages for API ---

    session = None
    if use_session:
        session = get_session()
        session.append("user", message_text)
        messages = session.get_context()
    elif history is None:
        messages = [{"role": "user", "content": message_text}]
    else:
        messages = list(history)
        messages.append({"role": "user", "content": message_text})

    # Inject wisdom context
    try:
        wisdom_context = _get_wisdom_context_for_message(message_text)
        if wisdom_context:
            last_msg = messages[-1]
            if isinstance(last_msg.get("content"), str):
                messages[-1] = {
                    "role": "user",
                    "content": f"{wisdom_context}\n\n{last_msg['content']}"
                }
    except Exception as e:
        print(f"[API:{request_id}] Wisdom context failed: {e}")

    system_prompt = get_system_prompt_cached(location=location, speaker=speaker)
    provider = get_llm_provider()

    async def event_generator():
        full_text = ""
        total_usage = TokenUsage()
        nonlocal messages

        try:
            max_tool_rounds = 5
            for tool_round in range(max_tool_rounds):
                # Stream from LLM provider
                tool_calls_acc = []
                current_tool_name = ""
                current_tool_id = ""
                current_tool_input = ""
                stop_reason = None
                raw_content = None

                async for event in provider.astream(
                    messages,
                    system=system_prompt,
                    tools=TOOLS,
                    max_tokens=1024,
                    source="stream",
                ):
                    if event.type == "text":
                        full_text += event.text
                        yield f"data: {json_mod.dumps({'type': 'text', 'content': event.text})}\n\n"

                    elif event.type == "tool_start":
                        # Finalize previous tool if any (multiple tools in one response)
                        if current_tool_name:
                            try:
                                args = json_mod.loads(current_tool_input) if current_tool_input else {}
                            except json_mod.JSONDecodeError:
                                args = {}
                            tool_calls_acc.append(ToolCall(
                                id=current_tool_id, name=current_tool_name, arguments=args,
                            ))
                        current_tool_name = event.tool_name
                        current_tool_id = event.tool_call_id
                        current_tool_input = ""
                        yield f"data: {json_mod.dumps({'type': 'tool_start', 'tool': event.tool_name})}\n\n"

                    elif event.type == "tool_input":
                        current_tool_input += event.tool_input_json

                    elif event.type == "done":
                        if event.usage:
                            total_usage = total_usage + event.usage
                        stop_reason = event.stop_reason
                        raw_content = event.raw_content

                        # Finalize any in-progress tool block
                        if current_tool_name:
                            try:
                                args = json_mod.loads(current_tool_input) if current_tool_input else {}
                            except json_mod.JSONDecodeError:
                                args = {}
                            tool_calls_acc.append(ToolCall(
                                id=current_tool_id,
                                name=current_tool_name,
                                arguments=args,
                            ))
                            current_tool_name = ""
                            current_tool_input = ""

                # Handle tool calls
                if stop_reason == StopReason.TOOL_USE and tool_calls_acc:
                    tool_results = []
                    for tc in tool_calls_acc:
                        loop = asyncio.get_running_loop()
                        tool_result_str = await loop.run_in_executor(
                            None,
                            lambda tc=tc: execute_tool(tc.name, tc.arguments, context=message_text)
                        )
                        tool_results.append(ToolResult(tool_call_id=tc.id, content=tool_result_str))

                    # Thread tool results back via provider
                    threading_msgs = provider.build_tool_result_messages(raw_content, tool_results)
                    messages.extend(threading_msgs)

                    yield f"data: {json_mod.dumps({'type': 'tool_complete'})}\n\n"
                    continue
                else:
                    # No more tools — done
                    break

            # Save assistant response to session
            if session and full_text.strip():
                session.append("assistant", full_text.strip())

            latency_ms = int((time.time() - start) * 1000)

            # Log metrics
            log_token_usage(total_usage.input_tokens, total_usage.output_tokens, source="stream")

            yield f"data: {json_mod.dumps({'type': 'done', 'latency_ms': latency_ms})}\n\n"
            print(f"[API:{request_id}] Stream done ({latency_ms}ms, {total_usage.output_tokens} tokens)")

        except Exception as e:
            print(f"[API:{request_id}] Stream error: {e}")
            import traceback
            traceback.print_exc()

            # Save partial response if we have any
            if session and full_text.strip():
                session.append("assistant", full_text.strip())

            yield f"data: {json_mod.dumps({'type': 'error', 'message': 'Sorry, something went wrong processing your message.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/emails")
async def emails(hours: int = 24, _token: str = Depends(verify_token)):
    """Get important emails from the last N hours."""
    return {
        "unread_count": get_unread_count(),
        "important_emails": scan_recent_emails(hours=hours)
    }


@app.get("/briefing")
async def briefing(include_brain: bool = True, _token: str = Depends(verify_token)):
    """
    Morning briefing with calendar, emails, reminders, and proactive surfacing.

    Args:
        include_brain: Whether to include brain dump goals/suggestions
    """
    from tools.cal import get_todays_events
    from tools.reminders import list_reminders

    loop = asyncio.get_running_loop()

    result = {
        "calendar": get_todays_events(),
        "emails": scan_recent_emails(hours=24),
        "unread_count": get_unread_count(),
        "reminders": list_reminders(),
    }

    # Add brain context if enabled
    if include_brain:
        try:
            from zoneinfo import ZoneInfo
            from datetime import datetime
            from tools.weather import get_current_weather

            # Get weather for context
            try:
                weather_data = await loop.run_in_executor(None, get_current_weather)
                weather = {
                    "temp": weather_data.get("temperature", 0) if weather_data else 0,
                    "conditions": weather_data.get("conditions", "") if weather_data else "",
                }
            except Exception:
                weather = None

            # Get today's health data from database
            eastern = ZoneInfo("America/New_York")
            today = datetime.now(eastern).strftime("%Y-%m-%d")
            try:
                from services.health_db import get_health_data as db_get_health_data_brief
                health_data = db_get_health_data_brief(date=today) or {}
            except Exception:
                health_data = {}

            # Get active goals
            goals = await loop.run_in_executor(None, lambda: get_active_goals(limit=3, days=14))

            # Get surfacing suggestions
            suggestions = await loop.run_in_executor(
                None,
                lambda: get_surfacing_context(
                    weather=weather,
                    time_of_day="morning",
                    health_data=health_data
                )
            )

            # Format proactive message
            proactive_message = format_proactive_message(suggestions, personality="jewish_mother")

            result["brain"] = {
                "active_goals": goals,
                "suggestions": suggestions,
                "proactive_message": proactive_message,
            }

        except Exception as e:
            print(f"[Briefing] Brain analysis error: {e}")
            result["brain"] = {"error": "Brain analysis failed"}

    return result


@app.get("/mcp/health")
async def mcp_health(_token: str = Depends(verify_token)):
    """Ping all MCP servers, auto-reconnect failures, return status dict.

    Used by the daemon's system scout to check MCP health without needing
    its own MCP connections (separate process boundary).

    Returns dict mapping server name to: "healthy", "reconnected", "dead", "missing".
    """
    manager = get_mcp_manager()
    statuses = await manager.health_check_and_reconnect()
    return statuses


@app.get("/mcp/status")
async def mcp_status(_token: str = Depends(verify_token)):
    """Get MCP server connection status and available tools."""
    manager = get_mcp_manager()
    tools = await manager.list_tools()

    return {
        "connected_servers": manager.connected_servers,
        "tools": [
            {
                "server": t.server,
                "name": t.name,
                "description": t.description,
                "qualified_name": t.qualified_name,
            }
            for t in tools
        ],
        "tool_count": len(tools),
    }


@app.post("/mcp/call")
async def mcp_call_tool(server: str, tool: str, arguments: dict = None, _token: str = Depends(verify_token)):
    """Call an MCP tool directly.

    Deny-by-default: only tools listed in MCP_CALL_ALLOWED_TOOLS may be
    invoked. This endpoint bypasses the LLM brain's safety layers (path
    validation, shortcut allowlist, etc.), so access is restricted.
    """
    from security.audit import audit

    # --- Allowlist check (deny-by-default) ---
    allowed = {
        s.strip() for s in settings.mcp_call_allowed_tools.split(",") if s.strip()
    }
    tool_key = f"{server}:{tool}"
    if not allowed or tool_key not in allowed:
        audit.tool_action(
            tool="mcp_call",
            detail=f"Blocked: '{tool_key}' not in MCP_CALL_ALLOWED_TOOLS",
            blocked=True,
            server=server,
            mcp_tool=tool,
        )
        return {
            "success": False,
            "error": f"Tool '{tool_key}' is not in the allowed list for /mcp/call. "
                     f"Add it to MCP_CALL_ALLOWED_TOOLS or use the chat endpoint instead.",
        }

    audit.tool_action(
        tool="mcp_call",
        detail=f"Calling {tool_key}",
        blocked=False,
        server=server,
        mcp_tool=tool,
    )

    manager = get_mcp_manager()

    try:
        result = await manager.call_tool(server, tool, arguments or {})

        # Format result for JSON response
        content = []
        for item in result.content:
            if hasattr(item, 'text'):
                content.append({"type": "text", "text": item.text})
            elif hasattr(item, 'data'):
                content.append({"type": "binary", "mime_type": getattr(item, 'mimeType', 'unknown')})

        return {
            "success": True,
            "content": content,
            "is_error": result.isError if hasattr(result, 'isError') else False,
        }
    except Exception as e:
        print(f"[MCP] Tool call error: {e}")
        return {
            "success": False,
            "error": "MCP tool call failed",
        }


# =============================================================================
# Conversation Sync Endpoints
# =============================================================================

from api.conversations import (
    save_message, get_messages, get_recent_messages,
    search_messages as search_conversation_messages,
    delete_all_messages, MessageCreate
)


class ConversationMessageCreate(BaseModel):
    """Message to save to conversation history."""
    id: str
    content: str
    role: Literal["user", "assistant"]
    device_id: str
    created_at: str | None = None
    conversation_id: str | None = None
    metadata: dict | None = None


class ConversationMessageResponse(BaseModel):
    """Message returned from API."""
    id: str
    content: str
    role: str
    device_id: str | None
    created_at: str
    conversation_id: str | None
    metadata: dict | None = None


@app.get("/conversations/messages")
async def get_conversation_messages(
    since: str | None = None,
    limit: int = Query(default=100, le=1000),
    device_id: str | None = None,
    _token: str = Depends(verify_token)
):
    """
    Get messages for conversation sync.

    Args:
        since: ISO timestamp to fetch messages after (for incremental sync)
        limit: Maximum number of messages to return (default 100)
        device_id: Optional filter for specific device

    Returns messages in chronological order (oldest first).
    """
    loop = asyncio.get_running_loop()

    def fetch():
        return get_messages(since=since, limit=limit, device_id=device_id)

    messages, has_more = await loop.run_in_executor(None, fetch)

    return {
        "messages": messages,
        "count": len(messages),
        "has_more": has_more
    }


@app.post("/conversations/messages")
async def save_conversation_message(message: ConversationMessageCreate, _token: str = Depends(verify_token)):
    """
    Save a new message to conversation history.

    Called by iOS/macOS clients after each exchange.
    Server assigns conversation_id if not provided.
    """
    loop = asyncio.get_running_loop()

    def save():
        msg = MessageCreate(
            id=message.id,
            content=message.content,
            role=message.role,
            device_id=message.device_id,
            created_at=message.created_at,
            conversation_id=message.conversation_id,
            metadata=message.metadata
        )
        return save_message(msg)

    try:
        saved = await loop.run_in_executor(None, save)
        return {"success": True, "message": saved}
    except Exception as e:
        print(f"[Conversations] Failed to save conversation message: {e}")
        return {"success": False, "error": "Failed to save message"}


@app.get("/conversations/recent")
async def get_recent_conversation(limit: int = Query(default=50, le=1000), _token: str = Depends(verify_token)):
    """
    Get recent conversation messages across all devices.

    Used by clients on launch to load recent history.
    Returns messages in chronological order.
    """
    loop = asyncio.get_running_loop()

    try:
        messages = await loop.run_in_executor(None, lambda: get_recent_messages(limit=limit))
        return {
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        print(f"[Conversations] Failed to get recent messages: {e}")
        return {
            "messages": [],
            "count": 0,
            "error": "Failed to retrieve messages"
        }


@app.get("/conversations/search")
async def search_conversation(q: str, limit: int = Query(default=20, le=1000), _token: str = Depends(verify_token)):
    """
    Search conversation history using full-text search.

    Args:
        q: Search query
        limit: Maximum results (default 20)

    Returns matching messages ordered by relevance.
    """
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        lambda: search_conversation_messages(q, limit=limit)
    )

    return {
        "query": q,
        "results": results,
        "count": len(results)
    }


@app.delete("/conversations/messages")
async def clear_conversation_history(_token: str = Depends(verify_token)):
    """
    Delete all conversation history.

    Use with caution - this is irreversible.
    """
    loop = asyncio.get_running_loop()
    count = await loop.run_in_executor(None, delete_all_messages)

    return {
        "success": True,
        "deleted_count": count,
        "message": f"Deleted {count} messages"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
