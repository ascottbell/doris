"""
Channel adapters for Doris.

Each adapter translates between a platform (Telegram, iMessage, CLI, etc.)
and Doris's internal message format.
"""

from channels.base import (
    ChannelAdapter,
    IncomingMessage,
    MessageHandler,
    OutgoingMessage,
    collect_response,
)
from channels.bluebubbles import BlueBubblesAdapter, create_bluebubbles_adapter
from channels.discord import DiscordAdapter, create_discord_adapter
from channels.registry import create_adapters_from_config, create_message_handler
from channels.telegram import TelegramAdapter, create_telegram_adapter
from channels.webhook import WebhookAdapter, create_webhook_adapter

__all__ = [
    "ChannelAdapter",
    "IncomingMessage",
    "MessageHandler",
    "OutgoingMessage",
    "collect_response",
    "BlueBubblesAdapter",
    "DiscordAdapter",
    "TelegramAdapter",
    "WebhookAdapter",
    "create_adapters_from_config",
    "create_bluebubbles_adapter",
    "create_discord_adapter",
    "create_message_handler",
    "create_telegram_adapter",
    "create_webhook_adapter",
]
