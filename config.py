"""
Doris configuration settings.

Loads configuration from environment variables via pydantic-settings.
All API keys and sensitive values should be set in .env file.
"""

from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    anthropic_api_key: str
    google_api_key: str = ""
    groq_api_key: str = ""
    ha_token: str = ""
    ha_url: str = ""  # e.g., http://your-ha-host:8123

    # APNS (Apple Push Notifications)
    apns_key_id: str = ""
    apns_team_id: str = ""
    apns_bundle_id: str = ""

    # Third-party APIs
    github_personal_access_token: str = ""
    brave_api_key: str = ""

    # API Authentication
    doris_api_token: str = ""  # Bearer token for API authentication (REQUIRED in production)
    doris_dev_mode: bool = False  # Set True to allow running without auth tokens (local dev only)

    # Observability
    sentry_dsn: str = ""  # Sentry DSN for error tracking

    db_path: Path = Path(__file__).parent / "data" / "doris.db"
    ollama_model: str = "qwen3-coder:30b"  # Non-thinking instruct model, clean tool outputs
    claude_model: str = "claude-opus-4-6"  # Backward compat — use default_model for new code

    # Personalization
    owner_name: str = "User"           # Your name — used in system prompts and bootstrap context

    # LLM provider abstraction
    llm_provider: str = "claude"       # claude | openai | ollama
    embed_provider: str = "ollama"     # ollama (more providers in future)
    default_model: str = ""            # Main brain — empty = falls back to claude_model
    mid_model: str = ""                # Mid-tier (code gen, creative) — empty = provider default
    utility_model: str = ""            # Cheap/fast tasks — empty = provider default
    openai_api_key: str = ""           # Required when llm_provider=openai

    # Channel adapters
    channel: str = ""                          # "" = API only; "telegram", "bluebubbles", or comma-separated
    telegram_bot_token: str = ""               # Bot token from @BotFather
    telegram_allowed_users: str = ""           # Comma-separated Telegram user IDs (REQUIRED — empty = deny all)
    telegram_edit_interval: float = 0.5        # Seconds between streaming edits
    bluebubbles_url: str = ""                  # BlueBubbles server URL (e.g., http://localhost:1234)
    bluebubbles_password: str = ""             # BlueBubbles server password
    bluebubbles_webhook_port: int = 8765       # Port for webhook listener
    bluebubbles_webhook_secret: str = ""       # Bearer token for webhook auth (recommended)
    bluebubbles_webhook_host: str = "127.0.0.1"  # Bind address (127.0.0.1 = local only, 0.0.0.0 = all interfaces)
    bluebubbles_allowed_senders: str = ""      # Comma-separated phone numbers/emails (REQUIRED — empty = deny all)
    bluebubbles_max_length: int = 4000         # Max chars per message before splitting
    discord_bot_token: str = ""                # Bot token from Discord Developer Portal
    discord_allowed_users: str = ""            # Comma-separated Discord user IDs (REQUIRED — empty = deny all)
    discord_edit_interval: float = 0.5         # Seconds between streaming edits
    webhook_secret: str = ""                   # Secret token for webhook auth (required if channel=webhook)
    webhook_port: int = 8766                   # Port for inbound webhook listener
    webhook_host: str = "127.0.0.1"            # Bind address (127.0.0.1 = local only, 0.0.0.0 = all interfaces)
    webhook_outbound_url: str = ""             # URL to POST proactive messages to (empty = disabled)
    webhook_rate_limit: int = 30               # Max requests per minute per IP

    # Chat endpoint protection
    chat_rate_limit: int = 20                  # Max chat requests per minute per IP
    chat_max_message_length: int = 50_000      # Max chars in a single message (~12K tokens)
    chat_max_history_size: int = 100           # Max messages in history array

    # API endpoint rate limiting (Gemini, Codex, MCP, push, agents, brain)
    api_rate_limit: int = 30                   # Max requests per minute per IP for expensive/sensitive endpoints

    # Outbound message safety (anti-exfiltration)
    email_allowed_recipients: str = ""         # Comma-separated emails always allowed (in addition to contacts)
    email_allow_any_recipient: bool = False    # Set True to disable EMAIL recipient validation (NOT recommended)
    imessage_allow_any_recipient: bool = False # Set True to disable IMESSAGE recipient validation (NOT recommended)

    # Filesystem access control (defense-in-depth, on top of MCP server's own directory restrictions)
    filesystem_allowed_dirs: str = ""          # Comma-separated allowed base dirs (empty = ~/Desktop,~/Downloads,~/Documents,~/Projects)

    # Apple Shortcuts access control
    allowed_shortcuts: str = ""                # Comma-separated shortcut names the LLM may run (REQUIRED — empty = deny all)

    # /mcp/call endpoint access control (direct MCP tool invocation bypassing LLM safety layers)
    mcp_call_allowed_tools: str = ""           # Comma-separated "server:tool" pairs (REQUIRED — empty = deny all)

    class Config:
        env_file = str(Path(__file__).parent / ".env")
        extra = "ignore"  # Allow extra env vars without errors

settings = Settings()


def validate_security_settings() -> None:
    """Fail fast if critical auth tokens are missing in production mode.

    Called at startup by main.py and mcp_server. Skipped when DORIS_DEV_MODE=true.
    """
    if settings.doris_dev_mode:
        return

    missing = []
    if not settings.doris_api_token:
        missing.append("DORIS_API_TOKEN")

    if missing:
        print("\n" + "=" * 70)
        print("FATAL: Missing required security configuration")
        print("=" * 70)
        for var in missing:
            print(f"  - {var} is not set")
        print()
        print("Doris refuses to start without authentication in production mode.")
        print("Either:")
        print("  1. Set the missing env var(s) in your .env file")
        print("  2. Set DORIS_DEV_MODE=true for local development (NOT for production)")
        print("=" * 70 + "\n")
        raise SystemExit(1)

    # Warn loudly about bypass flags that weaken anti-exfiltration
    import logging
    _log = logging.getLogger(__name__)
    if settings.email_allow_any_recipient:
        _log.warning(
            "SECURITY: EMAIL_ALLOW_ANY_RECIPIENT=true — email recipient validation DISABLED. "
            "Prompt injection can send emails to arbitrary addresses."
        )
    if settings.imessage_allow_any_recipient:
        _log.warning(
            "SECURITY: IMESSAGE_ALLOW_ANY_RECIPIENT=true — iMessage recipient validation DISABLED. "
            "Prompt injection can send iMessages to arbitrary contacts."
        )
