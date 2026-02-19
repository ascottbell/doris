"""
Generic webhook channel adapter for Doris.

Provides a platform-agnostic HTTP interface. Any system that can make
HTTP POST requests can talk to Doris — Slack, n8n, Zapier, Home Assistant,
custom apps, anything.

Inbound:
    POST /webhook with JSON body → synchronous JSON response.
    Auth via Authorization: Bearer <token> header.

Proactive:
    Doris POSTs to a pre-configured outbound URL (set in .env).
    No per-request callback URLs — prevents SSRF.

Required env var:
    WEBHOOK_SECRET — Secret token for authenticating inbound requests.

Optional env vars:
    WEBHOOK_HOST         — Bind address. Default: 127.0.0.1 (local only).
    WEBHOOK_PORT         — Port for the webhook listener. Default: 8766.
    WEBHOOK_OUTBOUND_URL — URL to POST proactive messages to. Empty = proactive disabled.
    WEBHOOK_RATE_LIMIT   — Max requests per minute per IP. Default: 30.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from uvicorn import Config, Server

from channels.base import ChannelAdapter, IncomingMessage, collect_response

if TYPE_CHECKING:
    from channels.base import MessageHandler

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Simple fixed-window rate limiter per source IP.

    Tracks request counts in 60-second windows. Stale entries are cleaned
    up on each check to prevent unbounded memory growth.
    """

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        # ip -> (window_start_monotonic, count)
        self._windows: dict[str, tuple[float, int]] = {}

    def check(self, ip: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        now = time.monotonic()
        self._cleanup(now)

        if ip in self._windows:
            window_start, count = self._windows[ip]
            if now - window_start < 60:
                if count >= self._max:
                    return False
                self._windows[ip] = (window_start, count + 1)
                return True

        # New window
        self._windows[ip] = (now, 1)
        return True

    def _cleanup(self, now: float) -> None:
        """Remove expired windows (older than 60s)."""
        expired = [
            ip
            for ip, (start, _) in self._windows.items()
            if now - start >= 60
        ]
        for ip in expired:
            del self._windows[ip]


class WebhookAdapter(ChannelAdapter):
    """Generic webhook adapter — the universal spigot.

    Runs a lightweight HTTP server that accepts messages and returns
    responses synchronously. Proactive messages go to a pre-configured
    outbound URL.

    Lifecycle:
        adapter = WebhookAdapter(secret="...", outbound_url="https://...")
        await adapter.start(handler)
        await adapter.stop()
    """

    def __init__(
        self,
        secret: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8766,
        outbound_url: str = "",
        rate_limit: int = 30,
    ) -> None:
        super().__init__()
        if not secret:
            raise ValueError("WEBHOOK_SECRET is required")
        self._secret = secret
        self._host = host
        self._port = port
        self._outbound_url = outbound_url.rstrip("/") if outbound_url else ""
        self._rate_limiter = _RateLimiter(rate_limit)
        self._handler: MessageHandler | None = None
        self._http: httpx.AsyncClient | None = None
        self._server: Server | None = None
        self._serve_task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return "webhook"

    # -- Lifecycle -----------------------------------------------------------

    async def start(self, handler: MessageHandler) -> None:
        """Start the webhook listener.

        Binds to the configured port and begins accepting requests.
        """
        self._handler = handler
        self._http = httpx.AsyncClient(timeout=30.0)

        app = FastAPI(docs_url=None, redoc_url=None)

        @app.post("/webhook")
        async def webhook_endpoint(request: Request) -> JSONResponse:
            return await self._handle_request(request)

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok", "adapter": "webhook"}

        config = Config(
            app=app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._server = Server(config)

        self._running = True
        logger.info(
            "Webhook adapter started (%s:%d, outbound: %s, rate limit: %d/min)",
            self._host,
            self._port,
            self._outbound_url or "disabled",
            self._rate_limiter._max,
        )

        self._serve_task = asyncio.create_task(self._server.serve())

    async def stop(self) -> None:
        """Shut down the webhook server and HTTP client."""
        self._running = False

        if self._server:
            self._server.should_exit = True
        if self._serve_task:
            try:
                await asyncio.wait_for(self._serve_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._serve_task.cancel()
            self._serve_task = None
        self._server = None

        if self._http:
            await self._http.aclose()
            self._http = None

        self._handler = None
        logger.info("Webhook adapter stopped.")

    async def send_message(self, conversation_id: str, text: str) -> None:
        """Send a proactive message to the configured outbound URL.

        POSTs a JSON payload to WEBHOOK_OUTBOUND_URL. If no outbound URL
        is configured, logs a warning and returns (does not raise).
        """
        self._check_running()

        if not self._outbound_url:
            logger.warning(
                "Webhook proactive message dropped — no WEBHOOK_OUTBOUND_URL configured. "
                "conversation_id=%s, text_length=%d",
                conversation_id,
                len(text),
            )
            return

        assert self._http is not None

        payload = {
            "text": text,
            "conversation_id": conversation_id,
            "channel": "webhook",
            "type": "proactive",
        }

        try:
            resp = await self._http.post(
                self._outbound_url,
                json=payload,
            )
            resp.raise_for_status()
            logger.debug(
                "Proactive message sent to %s (conversation: %s, status: %d)",
                self._outbound_url,
                conversation_id,
                resp.status_code,
            )
        except httpx.HTTPError as e:
            logger.error(
                "Failed to send proactive message to %s: %s",
                self._outbound_url,
                e,
            )

    # -- Request handling ----------------------------------------------------

    async def _handle_request(self, request: Request) -> JSONResponse:
        """Process an inbound webhook request.

        Flow: rate limit → auth → parse → handle → respond.
        Every rejection returns a clear JSON error with appropriate status code.
        """
        client_ip = request.client.host if request.client else "unknown"

        # Rate limiting
        if not self._rate_limiter.check(client_ip):
            logger.warning("Rate limited webhook request from %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again in a minute."},
            )

        # Authentication: Bearer token in Authorization header
        if not self._authenticate(request):
            logger.warning("Unauthorized webhook request from %s", client_ip)
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized. Provide secret via Authorization: Bearer <token> header."},
            )

        # Parse body
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON body."},
            )

        text = body.get("text", "").strip()
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing or empty 'text' field."},
            )

        sender_id = str(body.get("sender_id", "webhook")).strip() or "webhook"
        conversation_id = str(body.get("conversation_id", "default")).strip() or "default"

        incoming = IncomingMessage(
            text=text,
            sender_id=f"webhook:{sender_id}",
            conversation_id=conversation_id,
            channel="webhook",
            sender_name=body.get("sender_name"),
            metadata={
                "source_ip": client_ip,
            },
        )

        if not self._handler:
            logger.error("Webhook message received but no handler registered")
            return JSONResponse(
                status_code=503,
                content={"error": "Service not ready."},
            )

        # Process message and collect response (synchronous from caller's perspective)
        start_time = time.monotonic()
        response_text = await collect_response(
            self._safe_handle(self._handler, incoming)
        )
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(
            "[Channel:webhook] %s -> %d chars (%dms)",
            incoming.sender_id,
            len(response_text),
            elapsed_ms,
        )

        return JSONResponse(
            status_code=200,
            content={
                "text": response_text,
                "conversation_id": conversation_id,
                "channel": "webhook",
                "elapsed_ms": elapsed_ms,
            },
        )

    def _authenticate(self, request: Request) -> bool:
        """Verify the request carries a valid Bearer token.

        Only accepts: Authorization: Bearer <secret>

        Query parameter auth (?secret=) was removed because query params
        are logged by proxies, load balancers, WAFs, and browser history,
        risking credential exposure.
        """
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            from security.crypto import token_matches_any
            if token_matches_any(token, self._secret):
                return True

        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_webhook_adapter(
    secret: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    outbound_url: str = "",
    rate_limit: int = 30,
) -> WebhookAdapter:
    """Factory that creates a WebhookAdapter from config strings.

    Args:
        secret: Secret token for authenticating inbound requests.
        host: Bind address. Default 127.0.0.1 (local only).
        port: Port for the webhook listener. Default 8766.
        outbound_url: URL to POST proactive messages to. Empty = disabled.
        rate_limit: Max requests per minute per IP. Default 30.
    """
    return WebhookAdapter(
        secret=secret,
        host=host,
        port=port,
        outbound_url=outbound_url,
        rate_limit=rate_limit,
    )
