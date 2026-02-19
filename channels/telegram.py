"""
Telegram channel adapter for Doris.

Uses python-telegram-bot (v20+) in polling mode. Streams responses by
sending an initial message and editing it every ~500ms as chunks arrive,
giving users real-time feedback as Doris thinks.

Required env var:
    TELEGRAM_BOT_TOKEN — Bot token from @BotFather

Optional env vars:
    TELEGRAM_ALLOWED_USERS — Comma-separated list of Telegram user IDs.
        If set, only these users can interact with the bot. If empty, the
        bot is open to anyone (not recommended for production).

    TELEGRAM_EDIT_INTERVAL — Seconds between message edits during streaming.
        Default: 0.5. Lower = snappier but more API calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler as TGMessageHandler,
    filters,
)

from channels.base import ChannelAdapter, IncomingMessage, collect_response

if TYPE_CHECKING:
    from channels.base import MessageHandler

logger = logging.getLogger(__name__)

# Telegram message length limit (UTF-8)
_MAX_MESSAGE_LENGTH = 4096

# Placeholder while Doris is generating
_THINKING_TEXT = "..."


class TelegramAdapter(ChannelAdapter):
    """Telegram Bot API adapter with streaming message edits.

    Lifecycle:
        adapter = TelegramAdapter(token="...", allowed_users={123, 456})
        await adapter.start(handler)   # blocks until stop() is called
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
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        self._token = token
        self._allowed_users = allowed_users
        self._edit_interval = edit_interval
        self._app: Application | None = None
        self._handler: MessageHandler | None = None

    @property
    def name(self) -> str:
        return "telegram"

    # -- Lifecycle -----------------------------------------------------------

    async def start(self, handler: MessageHandler) -> None:
        """Start the Telegram bot in polling mode.

        This method blocks until stop() is called from another task.
        If the bot token is invalid or the network is unreachable, this
        raises immediately — it does not silently fail.
        """
        self._handler = handler

        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(
            TGMessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # Initialize validates the token (get_me call)
        await self._app.initialize()
        self._running = True

        bot_info = await self._app.bot.get_me()
        logger.info("Telegram adapter started as @%s (id: %s)", bot_info.username, bot_info.id)

        # start() begins processing the update queue
        await self._app.start()

        # run_polling manages the updater loop. We use start_polling/stop
        # manually so we can integrate with our own async lifecycle instead
        # of handing control to Application.run_polling().
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        """Gracefully shut down the Telegram bot.

        Stops accepting new updates, finishes in-flight handlers, then
        shuts down the application.
        """
        if not self._app:
            return

        logger.info("Telegram adapter shutting down...")
        self._running = False

        if self._app.updater and self._app.updater.running:
            await self._app.updater.stop()
        if self._app.running:
            await self._app.stop()
        await self._app.shutdown()

        self._app = None
        self._handler = None
        logger.info("Telegram adapter stopped.")

    async def send_message(self, conversation_id: str, text: str) -> None:
        """Send a proactive message to a Telegram chat.

        conversation_id is the Telegram chat_id as a string.
        """
        self._check_running()
        assert self._app is not None

        for chunk in _split_message(text):
            await self._app.bot.send_message(
                chat_id=int(conversation_id),
                text=chunk,
            )

    # -- Telegram handlers ---------------------------------------------------

    async def _handle_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /start command."""
        if not self._is_allowed(update):
            return

        await update.message.reply_text(
            "Hey! I'm Doris. Send me a message and I'll help you out."
        )

    async def _handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle incoming text messages with streaming response edits."""
        if not self._is_allowed(update):
            return
        if not self._handler or not update.message or not update.message.text:
            return

        incoming = IncomingMessage(
            text=update.message.text,
            sender_id=str(update.effective_user.id),
            conversation_id=str(update.effective_chat.id),
            channel="telegram",
            sender_name=update.effective_user.full_name,
            timestamp=update.message.date,
            metadata={
                "message_id": update.message.message_id,
                "chat_type": update.effective_chat.type,
            },
        )

        # Send "thinking" placeholder
        sent = await update.message.reply_text(_THINKING_TEXT)

        # Stream response with periodic edits
        accumulated = ""
        last_edit_time = 0.0
        edit_count = 0

        async for chunk in self._safe_handle(self._handler, incoming):
            accumulated += chunk

            now = time.monotonic()
            if now - last_edit_time >= self._edit_interval:
                display = _truncate(accumulated)
                try:
                    await sent.edit_text(display)
                    edit_count += 1
                    last_edit_time = now
                except Exception as e:
                    # edit_message_text can fail if text unchanged or rate limited
                    logger.debug("Edit failed (will retry on next chunk): %s", e)

        # Final edit with complete response
        if accumulated:
            for i, chunk in enumerate(_split_message(accumulated)):
                if i == 0:
                    # Edit the original message with the first chunk
                    try:
                        await sent.edit_text(chunk)
                    except Exception:
                        # If edit fails (e.g., text identical), that's fine
                        pass
                else:
                    # Send overflow as new messages
                    await self._app.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=chunk,
                    )
        else:
            # Handler yielded nothing — remove the thinking placeholder
            try:
                await sent.delete()
            except Exception:
                # If delete fails, edit to indicate empty response
                try:
                    await sent.edit_text("(No response)")
                except Exception:
                    pass

        logger.debug(
            "Message from %s: %d chunks, %d edits, %d chars",
            incoming.sender_id,
            edit_count,
            edit_count,
            len(accumulated),
        )

    # -- Helpers -------------------------------------------------------------

    def _is_allowed(self, update: Update) -> bool:
        """Check if the user is in the allowlist.

        Deny-by-default: if no allowlist is configured, all users are denied.
        Set TELEGRAM_ALLOWED_USERS in .env to allow specific users.
        """
        if not self._allowed_users:
            user_id = update.effective_user.id if update.effective_user else None
            logger.warning(
                "Telegram message denied — no allowlist configured. "
                "User %s (%s) tried to send a message. "
                "Set TELEGRAM_ALLOWED_USERS in .env to allow specific users.",
                user_id,
                update.effective_user.full_name if update.effective_user else "unknown",
            )
            return False

        user_id = update.effective_user.id if update.effective_user else None
        if user_id not in self._allowed_users:
            logger.warning(
                "Unauthorized Telegram user %s (%s) attempted access",
                user_id,
                update.effective_user.full_name if update.effective_user else "unknown",
            )
            return False
        return True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _truncate(text: str) -> str:
    """Truncate text to fit Telegram's message limit."""
    if len(text) <= _MAX_MESSAGE_LENGTH:
        return text
    return text[: _MAX_MESSAGE_LENGTH - 3] + "..."


def _split_message(text: str) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit.

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
            # No good newline — hard split
            split_at = _MAX_MESSAGE_LENGTH

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")

    return chunks


def create_telegram_adapter(
    token: str,
    *,
    allowed_users_csv: str = "",
    edit_interval: float = 0.5,
) -> TelegramAdapter:
    """Factory that creates a TelegramAdapter from config strings.

    Args:
        token: Bot token from @BotFather.
        allowed_users_csv: Comma-separated Telegram user IDs. Empty = open.
        edit_interval: Seconds between streaming edits. Default 0.5.
    """
    allowed: set[int] | None = None
    if allowed_users_csv.strip():
        try:
            allowed = {int(uid.strip()) for uid in allowed_users_csv.split(",") if uid.strip()}
        except ValueError as e:
            raise ValueError(
                f"TELEGRAM_ALLOWED_USERS must be comma-separated integers, got: {allowed_users_csv!r}"
            ) from e
    else:
        logger.critical(
            "TELEGRAM_ALLOWED_USERS is empty — ALL messages will be denied. "
            "Set TELEGRAM_ALLOWED_USERS to a comma-separated list of Telegram user IDs."
        )

    return TelegramAdapter(
        token=token,
        allowed_users=allowed,
        edit_interval=edit_interval,
    )
