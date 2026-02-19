"""
BlueBubbles channel adapter for Doris.

Connects to a BlueBubbles server (macOS) for iMessage integration.
Receives messages via webhook (BlueBubbles POSTs to us), sends replies
via the BlueBubbles REST API. Messages are atomic — iMessage doesn't
support editing, so we accumulate the full streaming response and send once.

Required env vars:
    BLUEBUBBLES_URL              — BlueBubbles server URL (e.g., http://localhost:1234)
    BLUEBUBBLES_PASSWORD         — Server password for API authentication
    BLUEBUBBLES_WEBHOOK_SECRET   — Bearer token for webhook auth
    BLUEBUBBLES_ALLOWED_SENDERS  — Comma-separated phone numbers/emails (empty = deny all)

Optional env vars:
    BLUEBUBBLES_WEBHOOK_PORT — Port for the webhook listener. Default: 8765.
    BLUEBUBBLES_MAX_LENGTH   — Max chars per message before splitting. Default: 4000.

Setup:
    1. Run BlueBubbles Server on your Mac
    2. In BB Server → API & Webhooks → Add Webhook
    3. URL: http://<your-doris-host>:<webhook_port>/webhook
    4. Subscribe to "New Message" events
    5. Set BLUEBUBBLES_URL and BLUEBUBBLES_PASSWORD in .env

Docs: https://docs.bluebubbles.app/server/developer-guides/rest-api-and-webhooks
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from uvicorn import Config, Server

from channels.base import ChannelAdapter, IncomingMessage, collect_response

if TYPE_CHECKING:
    from channels.base import MessageHandler

logger = logging.getLogger(__name__)

# iMessage doesn't have a hard limit like Telegram, but very long messages
# get truncated or split weirdly by the client. 4000 is a safe default.
_DEFAULT_MAX_LENGTH = 4000

# Security note: The BlueBubbles REST API only supports authentication via
# query parameter (?guid=<password>). There is no header-based alternative.
# This means the server password appears in request URLs and can be logged
# by HTTP proxies, load balancers, and web server access logs.
#
# Mitigations applied:
#   1. Logging filter redacts ?guid= from httpx debug output
#   2. Startup warning if BB server is non-localhost (higher proxy risk)
#   3. Recommendation to keep BB communication on localhost/LAN only

_CREDENTIAL_PARAM_RE = re.compile(r"([\?&])(guid|password|token)=[^&\s'\"]+", re.IGNORECASE)


class _RedactCredentialFilter(logging.Filter):
    """Logging filter that redacts BlueBubbles query-param credentials.

    Attached to the 'httpx' and 'httpcore' loggers to prevent the BB server
    password from appearing in debug output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.args = tuple(
                _CREDENTIAL_PARAM_RE.sub(r"\1\2=[REDACTED]", str(a))
                if isinstance(a, str) else a
                for a in (record.args if isinstance(record.args, tuple) else (record.args,))
            )
        record.msg = _CREDENTIAL_PARAM_RE.sub(r"\1\2=[REDACTED]", str(record.msg))
        return True


# Install the filter on httpx/httpcore loggers so BB passwords never leak
_cred_filter = _RedactCredentialFilter()
logging.getLogger("httpx").addFilter(_cred_filter)
logging.getLogger("httpcore").addFilter(_cred_filter)


class BlueBubblesAdapter(ChannelAdapter):
    """BlueBubbles iMessage adapter.

    Receives messages via webhook, sends replies via REST API.
    All responses are collected (not streamed) since iMessage messages
    are atomic — you can't edit a sent message.

    Lifecycle:
        adapter = BlueBubblesAdapter(url="...", password="...")
        await adapter.start(handler)   # starts webhook server
        await adapter.stop()           # shuts down
    """

    def __init__(
        self,
        url: str,
        password: str,
        *,
        webhook_port: int = 8765,
        webhook_secret: str = "",
        webhook_host: str = "127.0.0.1",
        allowed_senders: set[str] | None = None,
        max_message_length: int = _DEFAULT_MAX_LENGTH,
    ) -> None:
        super().__init__()
        if not url:
            raise ValueError("BLUEBUBBLES_URL is required")
        if not password:
            raise ValueError("BLUEBUBBLES_PASSWORD is required")

        self._url = url.rstrip("/")
        self._password = password
        self._webhook_port = webhook_port
        self._webhook_secret = webhook_secret.strip()
        self._webhook_host = webhook_host
        self._allowed_senders = allowed_senders
        self._max_length = max_message_length
        self._handler: MessageHandler | None = None
        self._http: httpx.AsyncClient | None = None
        self._server: Server | None = None
        self._serve_task: asyncio.Task | None = None
        self._processing_semaphore = asyncio.Semaphore(5)  # max concurrent webhook tasks

    @property
    def name(self) -> str:
        return "bluebubbles"

    # -- Lifecycle -----------------------------------------------------------

    async def start(self, handler: MessageHandler) -> None:
        """Start the webhook listener and HTTP client.

        Validates connectivity to the BlueBubbles server before accepting
        messages. Raises on failure — does not silently degrade.
        """
        self._handler = handler
        self._http = httpx.AsyncClient(timeout=30.0)

        # Warn if BlueBubbles server is not on localhost — the BB API sends
        # the password as a query parameter, which proxies/LBs may log.
        if not self._is_localhost_url(self._url):
            logger.warning(
                "SECURITY: BlueBubbles server URL (%s) is not localhost. "
                "The BB API sends credentials as URL query parameters, which "
                "may be logged by proxies and load balancers. Use localhost or "
                "a private network connection to minimize exposure.",
                self._url.split("//")[0] + "//[redacted]",
            )

        # Validate connection to BlueBubbles server
        await self._ping()

        # Require webhook secret unless in dev mode
        if not self._webhook_secret:
            dev_mode = os.environ.get("DORIS_DEV_MODE", "").strip().lower() in ("true", "1", "yes")
            if dev_mode:
                logger.warning(
                    "BlueBubbles webhook has no secret configured (BLUEBUBBLES_WEBHOOK_SECRET). "
                    "Running without auth because DORIS_DEV_MODE=true. "
                    "DO NOT run like this in production.",
                )
            else:
                raise ValueError(
                    "BLUEBUBBLES_WEBHOOK_SECRET is required. Anyone with network access to "
                    f"port {self._webhook_port} can inject messages without it. "
                    "Set BLUEBUBBLES_WEBHOOK_SECRET in .env, or set DORIS_DEV_MODE=true for local dev."
                )

        # Build a minimal FastAPI app for the webhook endpoint
        app = FastAPI(docs_url=None, redoc_url=None)

        @app.post("/webhook")
        async def webhook_handler(request: Request) -> dict:
            # Verify Bearer token if a webhook secret is configured
            if self._webhook_secret:
                auth_header = request.headers.get("authorization", "")
                if not auth_header.startswith("Bearer "):
                    logger.warning(
                        "BlueBubbles webhook request rejected — missing Bearer token from %s",
                        request.client.host if request.client else "unknown",
                    )
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Unauthorized — Bearer token required"},
                    )
                token = auth_header[7:].strip()
                from security.crypto import token_matches_any
                if not token_matches_any(token, self._webhook_secret):
                    logger.warning(
                        "BlueBubbles webhook request rejected — invalid Bearer token from %s",
                        request.client.host if request.client else "unknown",
                    )
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Unauthorized — invalid token"},
                    )

            body = await request.json()

            # Backpressure: reject if too many tasks are already processing
            if self._processing_semaphore.locked():
                logger.warning(
                    "BlueBubbles webhook rejected — processing queue full (5 concurrent tasks)"
                )
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too many requests — processing queue full"},
                )

            asyncio.create_task(self._process_webhook_guarded(body))
            return {"status": "ok"}

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok", "adapter": "bluebubbles"}

        config = Config(
            app=app,
            host=self._webhook_host,
            port=self._webhook_port,
            log_level="warning",
        )
        self._server = Server(config)

        self._running = True
        logger.info(
            "BlueBubbles adapter started (server: %s, webhook: %s:%d, auth: %s)",
            self._url,
            self._webhook_host,
            self._webhook_port,
            "enabled" if self._webhook_secret else "DISABLED",
        )

        # Run the uvicorn server in a background task so start() doesn't block
        # if the caller needs to start multiple adapters
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
        logger.info("BlueBubbles adapter stopped.")

    async def send_message(self, conversation_id: str, text: str) -> None:
        """Send a proactive message to an iMessage chat.

        conversation_id is the BlueBubbles chat GUID
        (e.g., 'iMessage;-;+15555550123').
        """
        self._check_running()
        for chunk in _split_message(text, self._max_length):
            await self._send_text(conversation_id, chunk)

    # -- Security helpers ----------------------------------------------------

    @staticmethod
    def _is_localhost_url(url: str) -> bool:
        """Check if a URL points to localhost / loopback."""
        from urllib.parse import urlparse

        host = urlparse(url).hostname or ""
        return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

    def _is_allowed_sender(self, sender_id: str) -> bool:
        """Check if the sender is in the allowlist.

        Deny-by-default: if no allowlist is configured, all senders are denied.
        Set BLUEBUBBLES_ALLOWED_SENDERS in .env to allow specific senders.
        """
        if not self._allowed_senders:
            logger.warning(
                "BlueBubbles message denied — no allowlist configured. "
                "Sender %s tried to send a message. "
                "Set BLUEBUBBLES_ALLOWED_SENDERS in .env to allow specific senders.",
                sender_id or "unknown",
            )
            return False

        # Normalize: strip whitespace, lowercase for case-insensitive email matching
        normalized = sender_id.strip().lower()
        if normalized not in self._allowed_senders:
            logger.warning(
                "Unauthorized BlueBubbles sender %s attempted access",
                sender_id,
            )
            return False
        return True

    # -- Webhook processing --------------------------------------------------

    async def _process_webhook_guarded(self, body: dict) -> None:
        """Wrap _process_webhook with semaphore-based backpressure."""
        async with self._processing_semaphore:
            await self._process_webhook(body)

    async def _process_webhook(self, body: dict) -> None:
        """Process an incoming BlueBubbles webhook event.

        Webhook envelope: {"type": "new-message", "data": {<Message>}}
        """
        event_type = body.get("type")

        if event_type != "new-message":
            logger.debug("Ignoring BlueBubbles event: %s", event_type)
            return

        data = body.get("data", {})

        # Ignore our own outgoing messages
        if data.get("isFromMe"):
            return

        # Ignore system/service messages
        if data.get("isServiceMessage") or data.get("isSystemMessage"):
            return

        text = data.get("text", "")
        if not text or not text.strip():
            logger.debug("Ignoring empty BlueBubbles message")
            return

        # Extract chat GUID from the chats array
        chats = data.get("chats", [])
        if not chats:
            logger.warning("BlueBubbles message with no chats array: %s", data.get("guid"))
            return
        chat_guid = chats[0].get("guid", "")

        # Extract sender info from handle
        handle = data.get("handle", {})
        sender_id = handle.get("address", "") if isinstance(handle, dict) else ""

        # Sender allowlist check (deny-by-default, matching Telegram/Discord pattern)
        if not self._is_allowed_sender(sender_id):
            return

        incoming = IncomingMessage(
            text=text.strip(),
            sender_id=sender_id,
            conversation_id=chat_guid,
            channel="bluebubbles",
            sender_name=handle.get("displayName") if isinstance(handle, dict) else None,
            timestamp=None,  # BB uses epoch ms — could parse but not critical
            metadata={
                "message_guid": data.get("guid", ""),
                "chat_identifier": chats[0].get("chatIdentifier", ""),
                "group_name": chats[0].get("displayName", ""),
            },
        )

        if not self._handler:
            logger.warning("BlueBubbles message received but no handler registered")
            return

        # Collect full response (iMessage is atomic — no streaming)
        response = await collect_response(self._safe_handle(self._handler, incoming))

        if response.strip():
            try:
                await self.send_message(chat_guid, response)
            except Exception:
                logger.exception(
                    "Failed to send reply to %s",
                    chat_guid,
                )

    # -- BlueBubbles API calls -----------------------------------------------

    async def _ping(self) -> None:
        """Verify connectivity to the BlueBubbles server.

        Raises httpx.HTTPError or RuntimeError on failure.
        """
        assert self._http is not None
        try:
            resp = await self._http.get(
                f"{self._url}/api/v1/ping",
                params={"guid": self._password},
            )
            resp.raise_for_status()
            logger.debug("BlueBubbles ping OK: %s", resp.json())
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Cannot reach BlueBubbles server at {self._url}: {e}"
            ) from e

    async def _send_text(self, chat_guid: str, text: str) -> dict:
        """Send a text message via the BlueBubbles REST API.

        POST /api/v1/message/text
        Body: {"chatGuid": "...", "message": "...", "tempGuid": "...", "method": "private-api"}
        """
        assert self._http is not None
        resp = await self._http.post(
            f"{self._url}/api/v1/message/text",
            params={"guid": self._password},
            json={
                "chatGuid": chat_guid,
                "message": text,
                "tempGuid": f"doris-{uuid.uuid4().hex[:12]}",
                "method": "private-api",
            },
        )
        resp.raise_for_status()
        result = resp.json()
        logger.debug("Sent message to %s: %s", chat_guid, result.get("status"))
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _split_message(text: str, max_length: int) -> list[str]:
    """Split a long message for iMessage delivery.

    Tries to split on paragraph breaks, then newlines, then hard-splits.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Try paragraph break
        split_at = remaining.rfind("\n\n", 0, max_length)
        if split_at > max_length // 4:
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")
            continue

        # Try single newline
        split_at = remaining.rfind("\n", 0, max_length)
        if split_at > max_length // 4:
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")
            continue

        # Hard split at word boundary
        split_at = remaining.rfind(" ", 0, max_length)
        if split_at > max_length // 4:
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at + 1 :]
            continue

        # Last resort: hard character split
        chunks.append(remaining[:max_length])
        remaining = remaining[max_length:]

    return chunks


def create_bluebubbles_adapter(
    url: str,
    password: str,
    *,
    webhook_port: int = 8765,
    webhook_secret: str = "",
    webhook_host: str = "127.0.0.1",
    allowed_senders_csv: str = "",
    max_message_length: int = _DEFAULT_MAX_LENGTH,
) -> BlueBubblesAdapter:
    """Factory that creates a BlueBubblesAdapter from config strings.

    Args:
        url: BlueBubbles server URL (e.g., http://localhost:1234).
        password: Server password for API auth.
        webhook_port: Port for webhook listener. Default 8765.
        webhook_secret: Bearer token for webhook auth. Empty = no auth.
        webhook_host: Bind address. Default 127.0.0.1 (localhost only).
        allowed_senders_csv: Comma-separated phone numbers/emails. Empty = deny all.
        max_message_length: Max chars per message. Default 4000.
    """
    allowed: set[str] | None = None
    if allowed_senders_csv.strip():
        allowed = {s.strip().lower() for s in allowed_senders_csv.split(",") if s.strip()}
    else:
        logger.critical(
            "BLUEBUBBLES_ALLOWED_SENDERS is empty — ALL messages will be denied. "
            "Set BLUEBUBBLES_ALLOWED_SENDERS to a comma-separated list of phone numbers/emails."
        )

    return BlueBubblesAdapter(
        url=url,
        password=password,
        webhook_port=webhook_port,
        webhook_secret=webhook_secret,
        webhook_host=webhook_host,
        allowed_senders=allowed,
        max_message_length=max_message_length,
    )
