"""
Channel adapter registry.

Creates adapters from config and provides the message handler that bridges
channel messages to the LLM brain.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator

from channels.base import ChannelAdapter, IncomingMessage, MessageHandler

logger = logging.getLogger(__name__)


def create_adapters_from_config(settings) -> list[ChannelAdapter]:
    """Create channel adapters based on settings.channel.

    Returns an empty list if no channels are configured (API-only mode).
    Raises ValueError on invalid channel names or missing required config.
    """
    names = [c.strip().lower() for c in settings.channel.split(",") if c.strip()]
    if not names:
        return []

    adapters: list[ChannelAdapter] = []

    for name in names:
        if name == "telegram":
            from channels.telegram import create_telegram_adapter

            adapter = create_telegram_adapter(
                token=settings.telegram_bot_token,
                allowed_users_csv=settings.telegram_allowed_users,
                edit_interval=settings.telegram_edit_interval,
            )
            adapters.append(adapter)

        elif name == "bluebubbles":
            from channels.bluebubbles import create_bluebubbles_adapter

            adapter = create_bluebubbles_adapter(
                url=settings.bluebubbles_url,
                password=settings.bluebubbles_password,
                webhook_port=settings.bluebubbles_webhook_port,
                webhook_secret=settings.bluebubbles_webhook_secret,
                webhook_host=settings.bluebubbles_webhook_host,
                allowed_senders_csv=settings.bluebubbles_allowed_senders,
                max_message_length=settings.bluebubbles_max_length,
            )
            adapters.append(adapter)

        elif name == "discord":
            from channels.discord import create_discord_adapter

            adapter = create_discord_adapter(
                token=settings.discord_bot_token,
                allowed_users_csv=settings.discord_allowed_users,
                edit_interval=settings.discord_edit_interval,
            )
            adapters.append(adapter)

        elif name == "webhook":
            from channels.webhook import create_webhook_adapter

            adapter = create_webhook_adapter(
                secret=settings.webhook_secret,
                host=settings.webhook_host,
                port=settings.webhook_port,
                outbound_url=settings.webhook_outbound_url,
                rate_limit=settings.webhook_rate_limit,
            )
            adapters.append(adapter)

        else:
            raise ValueError(
                f"Unknown channel: {name!r}. Valid options: telegram, bluebubbles, discord, webhook"
            )

    return adapters


def create_message_handler() -> MessageHandler:
    """Create the async message handler that bridges channels to the brain.

    The handler:
    - Runs chat() in an executor (sync → async bridge)
    - Logs request metrics
    - Yields text to the adapter
    """
    from llm.brain import chat, ClaudeResponse

    async def handler(message: IncomingMessage) -> AsyncGenerator[str, None]:
        start = time.time()

        # Isolate session by channel + sender to prevent cross-channel context leakage
        session_key = f"{message.channel}:{message.sender_id}"

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: chat(
                message.text,
                return_usage=True,
                use_session=True,
                session_key=session_key,
                speaker=message.sender_name,
            ),
        )

        if isinstance(response, ClaudeResponse):
            text = response.text
            tokens_in = response.input_tokens
            tokens_out = response.output_tokens
        else:
            text = str(response)
            tokens_in = 0
            tokens_out = 0

        latency_ms = int((time.time() - start) * 1000)

        logger.info(
            "[Channel:%s] %s -> %d chars (%dms, %d out tokens)",
            message.channel,
            message.sender_id,
            len(text),
            latency_ms,
            tokens_out,
        )

        yield text

    return handler
