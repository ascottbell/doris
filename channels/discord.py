"""
Discord channel adapter for Doris.

Uses discord.py (v2+) to connect as a Discord bot. Streams responses by
sending an initial message and editing it every ~500ms as chunks arrive,
giving users real-time feedback as Doris thinks.

Responds to:
    - Direct messages (always, if user is in allowlist)
    - Guild messages where the bot is @mentioned

Required env var:
    DISCORD_BOT_TOKEN — Bot token from Discord Developer Portal

Optional env vars:
    DISCORD_ALLOWED_USERS — Comma-separated list of Discord user IDs.
        If set, only these users can interact with the bot. If empty, the
        bot is open to anyone (not recommended for production).

    DISCORD_EDIT_INTERVAL — Seconds between message edits during streaming.
        Default: 0.5. Lower = snappier but more API calls.

Setup:
    1. Go to https://discord.com/developers/applications
    2. Create a New Application → go to Bot → Reset Token → copy the token
    3. Enable "Message Content Intent" under Privileged Gateway Intents
    4. Go to OAuth2 → URL Generator → select "bot" scope
    5. Select permissions: Send Messages, Read Message History, View Channels
    6. Copy the generated URL and open it to invite the bot to your server
    7. Set DISCORD_BOT_TOKEN in .env
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import discord

from channels.base import ChannelAdapter, IncomingMessage

if TYPE_CHECKING:
    from channels.base import MessageHandler

logger = logging.getLogger(__name__)

# Discord message length limit
_MAX_MESSAGE_LENGTH = 2000

# Placeholder while Doris is generating
_THINKING_TEXT = "..."


class DiscordAdapter(ChannelAdapter):
    """Discord bot adapter with streaming message edits.

    Lifecycle:
        adapter = DiscordAdapter(token="...", allowed_users={123, 456})
        await adapter.start(handler)   # validates token, starts gateway
        await adapter.stop()           # graceful shutdown
    """

    def __init__(
        self,
        token: str,
        *,
        allowed_users: set[int] | None = None,
        edit_interval: float = 0.5,
    ) -> None:
        super().__init__()
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN is required")
        self._token = token
        self._allowed_users = allowed_users
        self._edit_interval = edit_interval
        self._client: discord.Client | None = None
        self._handler: MessageHandler | None = None
        self._connect_task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return "discord"

    # -- Lifecycle -----------------------------------------------------------

    async def start(self, handler: MessageHandler) -> None:
        """Start the Discord bot.

        Validates the bot token via login(), then connects to the gateway
        in a background task. If the token is invalid, raises immediately.
        """
        self._handler = handler

        intents = discord.Intents.default()
        intents.message_content = True  # Privileged intent — must enable in Dev Portal

        self._client = discord.Client(intents=intents)

        # Register event handlers as closures over self
        @self._client.event
        async def on_ready() -> None:
            logger.info(
                "Discord gateway ready — %s (id: %s)",
                self._client.user.name,
                self._client.user.id,
            )

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            await self._handle_message(message)

        # login() validates the token (raises LoginFailure if invalid)
        await self._client.login(self._token)

        self._running = True
        logger.info(
            "Discord adapter started as %s (id: %s)",
            self._client.user.name,
            self._client.user.id,
        )

        # connect() runs the gateway loop — background task so start() returns
        self._connect_task = asyncio.create_task(
            self._client.connect(reconnect=True)
        )

    async def stop(self) -> None:
        """Gracefully shut down the Discord bot.

        Closes the gateway connection and cleans up.
        """
        if not self._client:
            return

        logger.info("Discord adapter shutting down...")
        self._running = False

        if not self._client.is_closed():
            await self._client.close()

        if self._connect_task:
            try:
                await asyncio.wait_for(self._connect_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._connect_task.cancel()
            self._connect_task = None

        self._client = None
        self._handler = None
        logger.info("Discord adapter stopped.")

    async def send_message(self, conversation_id: str, text: str) -> None:
        """Send a proactive message to a Discord channel or DM.

        conversation_id is the Discord channel ID as a string (works for
        both DM channels and guild text channels).
        """
        self._check_running()
        assert self._client is not None

        channel = self._client.get_channel(int(conversation_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(conversation_id))

        for chunk in _split_message(text):
            await channel.send(chunk)

    # -- Message handling ----------------------------------------------------

    async def _handle_message(self, message: discord.Message) -> None:
        """Handle an incoming Discord message with streaming response edits."""
        # Ignore our own messages
        if message.author == self._client.user:
            return

        # Determine if we should respond:
        #   - DMs: always (if user is allowed)
        #   - Guild channels: only when @mentioned
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = (
            self._client.user in message.mentions if message.guild else False
        )

        if not is_dm and not is_mentioned:
            return

        if not self._is_allowed(message.author):
            return

        if not self._handler:
            return

        # Strip the @mention from guild messages to get clean input
        text = message.content
        if is_mentioned and self._client.user:
            text = text.replace(f"<@{self._client.user.id}>", "").strip()

        if not text:
            return

        incoming = IncomingMessage(
            text=text,
            sender_id=str(message.author.id),
            conversation_id=str(message.channel.id),
            channel="discord",
            sender_name=message.author.display_name,
            timestamp=message.created_at,
            metadata={
                "message_id": message.id,
                "guild_id": message.guild.id if message.guild else None,
                "is_dm": is_dm,
            },
        )

        # Send "thinking" placeholder
        sent = await message.channel.send(_THINKING_TEXT)

        # Stream response with periodic edits
        accumulated = ""
        last_edit_time = 0.0
        edit_count = 0

        async for chunk in self._safe_handle(self._handler, incoming):
            if isinstance(chunk, dict):
                continue
            accumulated += chunk

            now = time.monotonic()
            if now - last_edit_time >= self._edit_interval:
                display = _truncate(accumulated)
                try:
                    await sent.edit(content=display)
                    edit_count += 1
                    last_edit_time = now
                except Exception as e:
                    # edit can fail if content unchanged or rate limited
                    logger.debug("Edit failed (will retry on next chunk): %s", e)

        # Final edit with complete response
        if accumulated:
            chunks = _split_message(accumulated)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    try:
                        await sent.edit(content=chunk)
                    except Exception:
                        pass
                else:
                    await message.channel.send(chunk)
        else:
            # Handler yielded nothing — remove the thinking placeholder
            try:
                await sent.delete()
            except Exception:
                try:
                    await sent.edit(content="(No response)")
                except Exception:
                    pass

        logger.debug(
            "Message from %s: %d edits, %d chars",
            incoming.sender_id,
            edit_count,
            len(accumulated),
        )

    def _is_allowed(self, user: discord.User | discord.Member) -> bool:
        """Check if the user is in the allowlist.

        Deny-by-default: if no allowlist is configured, all users are denied.
        Set DISCORD_ALLOWED_USERS in .env to allow specific users.
        """
        if not self._allowed_users:
            logger.warning(
                "Discord message denied — no allowlist configured. "
                "User %s (%s) tried to send a message. "
                "Set DISCORD_ALLOWED_USERS in .env to allow specific users.",
                user.id,
                user.display_name,
            )
            return False

        if user.id not in self._allowed_users:
            logger.warning(
                "Unauthorized Discord user %s (%s) attempted access",
                user.id,
                user.display_name,
            )
            return False
        return True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _truncate(text: str) -> str:
    """Truncate text to fit Discord's message limit."""
    if len(text) <= _MAX_MESSAGE_LENGTH:
        return text
    return text[: _MAX_MESSAGE_LENGTH - 3] + "..."


def _split_message(text: str) -> list[str]:
    """Split a long message into chunks that fit Discord's 2000-char limit.

    Tries to split on newlines to preserve formatting. Falls back to
    hard splits at the character limit.
    """
    if len(text) <= _MAX_MESSAGE_LENGTH:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= _MAX_MESSAGE_LENGTH:
            chunks.append(remaining)
            break

        # Try to find a good split point (newline near the limit)
        split_at = remaining.rfind("\n", 0, _MAX_MESSAGE_LENGTH)
        if split_at <= 0 or split_at < _MAX_MESSAGE_LENGTH // 2:
            # No good newline — try word boundary
            split_at = remaining.rfind(" ", 0, _MAX_MESSAGE_LENGTH)
        if split_at <= 0 or split_at < _MAX_MESSAGE_LENGTH // 4:
            # Last resort — hard split
            split_at = _MAX_MESSAGE_LENGTH

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n ")

    return chunks


def create_discord_adapter(
    token: str,
    *,
    allowed_users_csv: str = "",
    edit_interval: float = 0.5,
) -> DiscordAdapter:
    """Factory that creates a DiscordAdapter from config strings.

    Args:
        token: Bot token from Discord Developer Portal.
        allowed_users_csv: Comma-separated Discord user IDs. Empty = open.
        edit_interval: Seconds between streaming edits. Default 0.5.
    """
    allowed: set[int] | None = None
    if allowed_users_csv.strip():
        try:
            allowed = {
                int(uid.strip())
                for uid in allowed_users_csv.split(",")
                if uid.strip()
            }
        except ValueError as e:
            raise ValueError(
                f"DISCORD_ALLOWED_USERS must be comma-separated integers, got: {allowed_users_csv!r}"
            ) from e
    else:
        logger.critical(
            "DISCORD_ALLOWED_USERS is empty — ALL messages will be denied. "
            "Set DISCORD_ALLOWED_USERS to a comma-separated list of Discord user IDs."
        )

    return DiscordAdapter(
        token=token,
        allowed_users=allowed,
        edit_interval=edit_interval,
    )
