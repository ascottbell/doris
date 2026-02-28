"""
Telegram channel adapter for Doris.

Uses python-telegram-bot (v20+) in polling mode. Streams responses by
sending an initial message and editing it every ~500ms as chunks arrive,
giving users real-time feedback as Doris thinks.

When users send media/files via Telegram (documents, photos, video, audio,
etc.), the files are automatically downloaded and saved to ``data/telegram_files/``.
The message text then includes the file path(s), so Doris can process or
reference them if needed.

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
from pathlib import Path
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
from config import settings

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
            TGMessageHandler(
                (filters.TEXT | filters.Document.ALL | filters.PHOTO | filters.VIDEO
                 | filters.AUDIO | filters.VOICE | filters.ANIMATION)
                & ~filters.COMMAND,
                self._handle_message,
            )
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

    async def send_message(
        self,
        conversation_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        """Send a proactive message to a Telegram chat.

        conversation_id is the Telegram chat_id as a string.

        `metadata` is an optional dict that may contain a ``media`` key
        (a list of media specifications).  Each media entry should include
        at least ``type`` (e.g. ``photo``, ``document``) and ``file`` which
        may be a file-like object, a local path, a URL, or an existing
        Telegram ``file_id``.  A ``caption`` can accompany media.

        If media is present we send it before any leftover text; otherwise
        the behaviour is identical to the previous implementation.
        """
        self._check_running()
        assert self._app is not None

        # handle media attachments if provided
        media_list = []
        if metadata:
            media_list = metadata.get("media", []) or []

        if media_list:
            # send each media item; caption may consume the text for the
            # first item
            remaining_text = text
            for i, m in enumerate(media_list):
                typ = m.get("type")
                file_obj = m.get("file")
                caption = m.get("caption")
                # the first media item can carry the overall text if no
                # explicit caption is provided
                if i == 0 and not caption and remaining_text:
                    caption = remaining_text
                    remaining_text = ""

                send_kwargs = {"chat_id": int(conversation_id)}
                if caption:
                    send_kwargs["caption"] = caption

                if typ == "photo":
                    await self._app.bot.send_photo(photo=file_obj, **send_kwargs)
                elif typ == "document":
                    await self._app.bot.send_document(document=file_obj, **send_kwargs)
                elif typ == "video":
                    await self._app.bot.send_video(video=file_obj, **send_kwargs)
                elif typ == "animation":
                    await self._app.bot.send_animation(animation=file_obj, **send_kwargs)
                elif typ == "audio":
                    await self._app.bot.send_audio(audio=file_obj, **send_kwargs)
                elif typ == "voice":
                    await self._app.bot.send_voice(voice=file_obj, **send_kwargs)
                elif typ == "sticker":
                    await self._app.bot.send_sticker(sticker=file_obj, **send_kwargs)
                elif typ == "video_note":
                    await self._app.bot.send_video_note(video_note=file_obj, **send_kwargs)
                else:
                    # unknown media type, fall back to send_message for safety
                    await self._app.bot.send_message(
                        chat_id=int(conversation_id),
                        text=f"[unsupported media type: {typ}]",
                    )

            # send any leftover text after media
            if remaining_text:
                for chunk in _split_message(remaining_text):
                    await self._app.bot.send_message(
                        chat_id=int(conversation_id),
                        text=chunk,
                    )
            return

        # no media: behave as before
        for chunk in _split_message(text):
            await self._app.bot.send_message(
                chat_id=int(conversation_id),
                text=chunk,
            )

    async def _download_file(self, file_id: str, filename: str) -> str:
        """Download a file from Telegram and save it locally.

        Returns the path to the saved file (relative to data directory).
        """
        assert self._app is not None

        try:
            # Get file info and download
            file_obj = await self._app.bot.get_file(file_id)
            
            # Create telegram_files directory
            media_dir = settings.data_dir / "telegram_files"
            media_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file
            filepath = media_dir / filename
            await file_obj.download_to_drive(filepath)
            
            # Return relative path for display
            return str(filepath.relative_to(settings.data_dir))
        except Exception as e:
            logger.warning("Failed to download Telegram file %s: %s", file_id, e)
            return f"[file-download-failed: {filename}]"

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
        """Handle incoming messages with streaming response edits.

        Supports both text and media. Media files are downloaded and saved
        to disk; filenames are included in the outgoing message text.
        """
        if not self._is_allowed(update):
            return
        if not self._handler or not update.message:
            return

        msg = update.message

        # Determine primary text: prefer text, then caption
        text = msg.text or msg.caption or ""

        # Download and save any media files
        saved_files: list[str] = []
        if msg.document:
            fpath = await self._download_file(
                msg.document.file_id,
                msg.document.file_name or "document",
            )
            saved_files.append(fpath)
        elif msg.photo:
            # Take the largest photo
            fpath = await self._download_file(
                msg.photo[-1].file_id,
                f"photo_{msg.message_id}.jpg",
            )
            saved_files.append(fpath)
        elif msg.video:
            fpath = await self._download_file(
                msg.video.file_id,
                msg.video.file_name or f"video_{msg.message_id}.mp4",
            )
            saved_files.append(fpath)
        elif msg.audio:
            fpath = await self._download_file(
                msg.audio.file_id,
                msg.audio.file_name or f"audio_{msg.message_id}.mp3",
            )
            saved_files.append(fpath)
        elif msg.voice:
            fpath = await self._download_file(
                msg.voice.file_id,
                f"voice_{msg.message_id}.ogg",
            )
            saved_files.append(fpath)
        elif msg.animation:
            fpath = await self._download_file(
                msg.animation.file_id,
                msg.animation.file_name or f"animation_{msg.message_id}.gif",
            )
            saved_files.append(fpath)

        # Append file paths to message text
        if saved_files:
            file_refs = " ".join(saved_files)
            text = f"{text} {file_refs}".strip() if text else file_refs

        if not text:
            # nothing for the handler to process
            return

        metadata = {
            "message_id": msg.message_id,
            "chat_type": update.effective_chat.type,
        }

        incoming = IncomingMessage(
            text=text,
            sender_id=str(update.effective_user.id),
            conversation_id=str(update.effective_chat.id),
            channel="telegram",
            sender_name=update.effective_user.full_name,
            timestamp=msg.date,
            metadata=metadata,
            media=[],  # media already downloaded and included in text
        )

        # Send "thinking" placeholder
        sent = await update.message.reply_text(_THINKING_TEXT)

        # Stream response with periodic edits
        accumulated = ""
        last_edit_time = 0.0
        edit_count = 0
        pending_media: list[dict] = []

        async for chunk in self._safe_handle(self._handler, incoming):
            # Dict chunks carry side-channel data (e.g., media from MCP tools)
            if isinstance(chunk, dict):
                pending_media.extend(chunk.get("media", []))
                continue

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

        # Send any media files accumulated during tool execution
        if pending_media:
            logger.info(
                "Sending %d media item(s) to %s",
                len(pending_media),
                incoming.conversation_id,
            )
            try:
                await self.send_message(
                    incoming.conversation_id,
                    "",
                    metadata={"media": pending_media},
                )
            except Exception:
                logger.exception(
                    "Failed to send media to %s",
                    incoming.conversation_id,
                )

        logger.debug(
            "Message from %s: %d chunks, %d edits, %d chars, %d media",
            incoming.sender_id,
            edit_count,
            edit_count,
            len(accumulated),
            len(pending_media),
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

    The adapter now supports sending and receiving arbitrary media
    types.  To send media proactively, provide a ``metadata`` dict with a
    ``media`` key when calling ``send_message`` on the adapter.

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
