# Doris

**A personal AI assistant that monitors your world, remembers your life, and acts before you ask.**

I built Doris to help me juggle family life — two kids, a busy household, and the constant feeling that something was slipping through the cracks. She started as a weekend project and turned into something I actually rely on every day: an always-on assistant that reads my email, watches my calendar, knows my family, and handles things before I even notice they need handling.

Along the way, I realized the memory system was the hardest and most important piece. What began as one project became two: **Doris**, the agent you're looking at now, and [**maasv**](https://github.com/ascottbell/maasv), the cognition layer underneath that gives her a real memory — not just storage, but understanding.

Doris runs on a Mac Mini in my home. She's been managing my actual calendar, parsing my real email, and reminding me to pick up the kids at the right time for months now. This is production code, not a demo.

## What Doris Actually Does

These aren't hypotheticals — they're things Doris does for me regularly:

**Handles the boring stuff end-to-end.** An afterschool registration email arrives with a semester's worth of activities. Doris sees it, parses the dates and times, creates recurring calendar events through June, and lets me know it's handled — before I've opened my inbox.

**Remembers what you've talked about.** "What was that restaurant we discussed for our anniversary?" Doris searches months of conversations by meaning, not keywords, and finds the exact discussion — who suggested what, why, and whether you ever made the reservation.

**Pays attention so you don't have to.** It's Tuesday at 4:25pm. Doris knows there's a pickup at 5pm, knows where I am right now, and sends a notification with transit options and timing. She figured out that pickup time ≠ event start time on her own.

**Learns from experience.** Doris flagged an email she shouldn't have? Tell her. She tracks your feedback — and after a few corrections, she adapts. Positive feedback reinforces a behavior until it becomes her default. Negative feedback makes her try a different approach next time. You don't have to accept the assistant's style — she actively learns yours.

## Quick Start

Doris needs Python 3.11+, an LLM API key, and [Ollama](https://ollama.ai) running locally for embeddings.

```bash
# Clone the repo
git clone https://github.com/ascottbell/doris.git
cd doris

# Install dependencies
pip install -r requirements.txt

# Pull an embedding model
ollama pull qwen3-embedding:8b

# Configure (at minimum, set your API key)
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (or OPENAI_API_KEY with LLM_PROVIDER=openai)

# Run
python main.py
```

That gets you a running Doris with the API server on `localhost:8000`. Talk to her via the CLI:

```bash
./bin/doris "what's on my calendar today?"
```

By default, Doris uses Claude as the LLM provider and Ollama for embeddings. See [Configuration](#configuration) for all the options.

### Docker

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum

docker compose up -d
```

This runs Doris in a container on `localhost:8000`. It expects Ollama running on the host for embeddings — if you'd rather run Ollama in Docker too, uncomment the `ollama` service in `docker-compose.yml`.

macOS-specific features (iMessage, Apple Music, Reminders, Contacts) aren't available in Docker. Everything else works: chat, memory, email, weather, channels, scouts, and the full API.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       Channels                          │
│    CLI · Telegram · Discord · BlueBubbles · Webhook     │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                        Brain                            │
│          Provider-agnostic LLM conversation loop        │
│             Claude · OpenAI · Ollama                    │
└─────────┬──────────────────────────────────┬────────────┘
          │                                  │
┌─────────▼─────────┐          ┌─────────────▼────────────┐
│      Tools        │          │     Memory (maasv)       │
│                   │          │                          │
│  Calendar         │          │  Semantic retrieval      │
│  Email (Gmail)    │          │  Knowledge graph         │
│  Reminders        │          │  Wisdom (learning)       │
│  iMessage         │          │  Sleep-time compute      │
│  Weather          │          │  Entity extraction       │
│  Contacts         │          │  Memory lifecycle        │
│  Apple Music      │          │                          │
│  Smart Home       │          │                          │
│  + 34 more        │          │                          │
└───────────────────┘          └──────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│                  Scouts & Proactive System               │
│                                                         │
│  9 scouts monitoring: email, calendar, weather,         │
│  reminders, health, location, time, system, memory      │
│                                                         │
│  Daemon → Observation → Sitrep Review → Notification     │
└─────────────────────────────────────────────────────────┘
```

### Brain

The conversation engine. Doris processes messages through a tool-use loop — she can call any of her 42 tools, chain them together, and maintain context across a session. The brain is provider-agnostic: swap between Claude, OpenAI, or Ollama by changing one environment variable.

The LLM layer uses a three-tier model system:

| Tier | What it's for | Claude | OpenAI | Ollama |
|------|---------------|--------|--------|--------|
| **default** | Main conversation | claude-opus-4-6 | gpt-5.2 | your choice |
| **mid** | Complex background tasks | claude-sonnet-4-6 | gpt-5.2 | your choice |
| **utility** | Classification, extraction | claude-haiku-4-5 | gpt-5-mini | your choice |

### Memory (powered by maasv)

This is the piece that makes Doris feel like an intelligent assistant.

[maasv](https://github.com/ascottbell/maasv) is a standalone cognition layer I extracted from Doris into its own package. It handles the full memory lifecycle — not just storing and retrieving, but actively maintaining and improving memories over time:

**Retrieval that actually works.** When you ask Doris about "that Italian place," she doesn't just do a keyword search. maasv combines three signals — semantic similarity (vector search), keyword matching (BM25), and knowledge graph traversal — then fuses the results. This means a question about "dinner plans" can surface a memory about Trattoria Roma even if you never used those words together.

**A knowledge graph, not a flat store.** Doris extracts entities and relationships from every conversation — people, places, projects, and how they connect. When you ask about a person, she can traverse the graph to find related context you didn't explicitly ask about. Relationships are temporal: she knows you *used to* work at Company X but *now* work at Company Y.

**Learning from experience.** maasv's wisdom system creates a feedback loop between you and your assistant. When Doris takes an action, you can give feedback — positive or negative. Positive feedback counts toward making that behavior the default: after three confirmations, Doris treats it as the right way to handle similar situations going forward. Negative feedback resets that confidence and makes her try a different approach next time. No feedback means she keeps doing what she's doing. The result: instead of you adapting to the assistant's style, the assistant actively adapts to yours. This same system drives the scouts — when an email escalation was a false alarm, Doris checks that history before deciding whether to bother you next time.

**Compiled wisdom.** Raw wisdom entries are useful, but nobody reads through hundreds of individual decisions before acting. A background compilation job periodically distills all wisdom entries into a living summary document with two sections: a **narrative understanding** (prose about who you are as a decision-maker — what you value, how you think about tradeoffs) and **operational patterns** (a punch list of specific do's, don'ts, and gotchas learned from real outcomes). The compiler runs on a mid-tier model (Sonnet-class), triggers when 5+ new entries accumulate since the last compilation, and evolves the summary over time rather than rebuilding from scratch. This compiled summary is loaded into Doris's system prompt at startup — adjacent to her immutable persona, but separate from it. The persona is who Doris is; the wisdom is what she's learned about working with you.

**Sleep-time compute.** When you're not talking to Doris, she's still working — resolving vague references ("that place" → "Trattoria Roma"), running second-pass analysis on conversations, consolidating related memories, and pruning stale data. The system improves when it's idle.

**Token-efficient by design.** A common problem with AI assistants is loading the entire memory store into every prompt — context windows balloon, costs spike, and most of the tokens are wasted on irrelevant context. Doris takes a different approach: maasv retrieves only the memories relevant to the current conversation, the system prompt is cached across requests (90% cost reduction on cached content after the first call), and old conversation turns are compacted into summaries instead of carried verbatim. You get full memory without paying for a full context window every time.

Everything runs on SQLite. No Redis, no Postgres, no external services. One file.

If you're building your own agent and want this kind of memory, maasv is a [standalone package on PyPI](https://pypi.org/project/maasv/) — `pip install maasv`. Doris is one integration, but maasv is designed to work with any agent.

### Scouts

Scouts are lightweight agents that monitor your environment on a schedule and surface what matters. They run on the utility model tier (Haiku-class), keeping costs around ~$1/month for all scouts combined.

| Scout | What it watches | Schedule |
|-------|----------------|----------|
| **Email** | Gmail inbox — new messages, VIP senders, urgency | Every 30 min |
| **Calendar** | Upcoming events, conflicts, changes | Every 15 min |
| **Weather** | Conditions, forecasts, severe weather | Every hour |
| **Reminders** | Due soon, overdue items | Every 15 min |
| **Time** | Rule-based triggers (morning briefing, evening wind-down) | Every minute |
| **Health** | Apple Health trends (steps, sleep, workouts) | Daily |
| **Location** | Location-based context and triggers | On change |
| **System** | Server health, MCP connections, disk space | Every 15 min |
| **Memory** | Consolidation and maintenance triggers | Daily |

Scouts don't act on their own — they create **observations** with a relevance score. The daemon routes observations through a **sitrep engine** that separates them into two lanes:

- **Instant lane** — Life-safety emergencies (smoke detectors, CO alarms, water leaks) bypass review and notify you immediately.
- **Sitrep lane** — Everything else. Observations accumulate in a buffer, and every 30 minutes Doris reviews the consolidated situation report with full context: what she's already told you (notification ledger, 72-hour window), what conditions she's tracking, and the current time/day context. She makes editorial decisions — **NOTIFY** (send now), **HOLD** (include in next morning/evening brief), or **DISMISS** (noise, already covered) — so you get one thoughtful notification instead of seven about the same snowstorm.

This design exists because the old system used content hashing for dedup, which broke whenever scout output changed slightly (precipitation drifting from 94% to 89% produced different hashes, bypassing dedup). The fix wasn't better hashing — it was putting the intelligence in the right place. Scouts observe; Doris decides.

The scout framework is extensible. To build your own:

```python
from scouts.base import Scout, Observation, Relevance

class MyScout(Scout):
    name = "my-scout"

    async def observe(self) -> list[Observation]:
        # Check something, return observations
        data = await self.check_something()

        if data.is_interesting:
            return [Observation(
                scout=self.name,
                observation=f"Something interesting happened: {data.summary}",
                relevance=Relevance.MEDIUM,
                escalate=True,
            )]
        return []
```

Register it in `scouts/__init__.py` and the daemon picks it up automatically.

### Channels

Doris is interface-agnostic. The brain doesn't know or care how you're talking to her — it just processes messages and returns responses. Channel adapters handle the transport.

| Channel | How it works | Setup |
|---------|-------------|-------|
| **CLI** | `./bin/doris "message"` | Works out of the box |
| **API** | REST endpoints on `localhost:8000` | Works out of the box |
| **Telegram** | Bot API with streaming (edits message as response builds) | Set `TELEGRAM_BOT_TOKEN` |
| **Discord** | DMs + @mention in guilds, streaming edits | Set `DISCORD_BOT_TOKEN` |
| **BlueBubbles** | Native iMessage via [BlueBubbles](https://bluebubbles.app) server | Set `BLUEBUBBLES_URL` + password |
| **Webhook** | Generic HTTP — POST in, JSON out. Connect to anything. | Set `WEBHOOK_SECRET` |

Enable channels via the `CHANNEL` environment variable (comma-separated):

```bash
CHANNEL=telegram,discord  # Run both simultaneously
CHANNEL=webhook           # Just the generic HTTP interface
# Leave empty for API-only mode
```

Each adapter handles streaming differently: Telegram and Discord show a "..." placeholder then progressively edit the message. BlueBubbles collects the full response and sends it atomically (iMessage doesn't support edits). Webhook returns a synchronous JSON response.

### Security

Doris takes security seriously. When your AI assistant has access to your email, calendar, and messages, careful is the only way to build.

#### Authentication & Access Control

**Mandatory authentication.** `DORIS_API_TOKEN` is required to start Doris in production — the server refuses to launch without it. All REST endpoints (except `/health`) require a Bearer token via `Authorization: Bearer <token>` header. The dashboard is gated by a login form that validates against the same token. WebSocket connections authenticate via query parameter (`?token=...`) because the WebSocket protocol doesn't support custom headers — use TLS to protect the token in transit. A `DORIS_DEV_MODE=true` escape hatch exists for local development only.

**MCP server auth.** The MCP server in HTTP mode requires `DORIS_MCP_AUTH_TOKEN` — it also refuses to start without it. All requests (except `/health`) must include an `X-API-Key` header. All token comparisons use constant-time `hmac.compare_digest` to prevent timing attacks.

**API key rotation.** Both `DORIS_API_TOKEN` and `DORIS_MCP_AUTH_TOKEN` support comma-separated token lists for zero-downtime rotation. Set `DORIS_API_TOKEN=new_token,old_token`, restart, verify, then remove the old token. Auth accepts any token in the list; encryption uses the first (primary) token, and decryption tries all keys in order so existing encrypted files remain readable.

**Channel access control.** Telegram and Discord adapters require explicit allowlists (`TELEGRAM_ALLOWED_USERS`, `DISCORD_ALLOWED_USERS`). Empty allowlist = all messages denied (deny-by-default). BlueBubbles and generic webhook adapters authenticate inbound requests via Bearer token in the Authorization header only. CORS is locked to explicit origins.

**Rate limiting.** Chat endpoints (`/chat/*`) are rate-limited to 20 requests per minute per IP (configurable). Input size is constrained: messages capped at 50,000 characters, history at 100 messages. Oversized payloads are rejected with a 422 before reaching any business logic.

**Audit logging.** Security-relevant events — authentication failures, rate limit rejections, and startup configuration — are logged as structured JSON to `logs/audit.log` (rotating, 10 MB × 5 backups) and stderr. Each entry includes timestamp, client IP, endpoint, and reason.

#### Prompt Injection Defense

**Content sandboxing.** Every piece of external content — email bodies, calendar event titles, reminder text, MCP server responses — is wrapped in XML boundary tags before it reaches the LLM. This prevents prompt injection: a malicious email that says "ignore all instructions" is treated as data, not a command.

**Injection scanning.** A multi-layer scanning system checks incoming content against known injection patterns — instruction overrides, invisible Unicode characters, RTL text manipulation, base64 payloads, fake UI elements. Content is pre-scanned before reaching the LLM: high-risk content (3+ patterns, invisible characters, or RTL overrides) skips the LLM entirely and returns safe defaults. LLM classification outputs are sanitized with strict size limits (max 10 tags, 50 chars per tag, 200 char reason) to prevent output manipulation.

**Memory poisoning defense.** All memory writes are automatically scanned for injection patterns and stripped of invisible characters at the storage layer. Suspicious content is flagged in metadata for audit. On retrieval, memories are wrapped in boundary tags before reaching the LLM. The knowledge graph and wisdom system have the same protections.

**MCP server trust tiers.** External tool servers are classified as `builtin`, `trusted`, or `sandboxed`. Sandboxed servers get their tool descriptions scanned, responses quarantined, and outbound arguments checked for PII before they leave. The default is `sandboxed` — safe until proven otherwise.

**AppleScript sanitization.** Every string that touches AppleScript or subprocess calls is escaped and validated. Calendar event titles, reminder names, iMessage text — all sanitized against injection.

#### Data Protection

**Encryption at rest.** OAuth tokens, session checkpoints, and device tokens are encrypted on disk using Fernet with PBKDF2-SHA256 key derivation (480,000 iterations) from `DORIS_API_TOKEN`. Each data type uses a distinct salt so the derived keys differ. File permissions are set to `0600` (owner-only). Legacy plaintext files are auto-migrated to encrypted on first read. Falls back to plaintext only in dev mode.

**Minimal information exposure.** The unauthenticated `/health` endpoint returns only `{"status": "ok"}` — no model names, versions, or system info. Detailed status is available at the authenticated `/health/status` endpoint. BlueBubbles API credentials are automatically redacted from all log output.

**Security headers.** All responses include `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, and a restrictive `Permissions-Policy`. API responses get a strict `Content-Security-Policy` (`default-src 'none'`); the dashboard gets a relaxed policy allowing inline styles and fonts. HSTS is available via `DORIS_ENABLE_HSTS` (off by default — enable only behind TLS).

#### Infrastructure

**Network binding.** All webhook and listener endpoints default to `127.0.0.1` (localhost only). Binding to `0.0.0.0` requires explicit opt-in via environment variable.

**Pinned dependencies.** All Python packages are pinned to exact versions (`==`) in both `requirements.txt` and `requirements-docker.txt` to prevent supply chain attacks and unexpected breaking changes.

**Docker hardening.** The container runs as a non-root `doris` user. The Dockerfile uses a multi-stage build: build tools (compilers, git) exist only in the builder stage and are absent from the runtime image. The base image is pinned to a specific SHA256 digest.

The security layer includes 223 tests covering injection scanning, content wrapping, PII detection, memory poisoning, MCP hardening, and AppleScript escaping.

## Configuration

Doris is configured entirely via environment variables (`.env` file). The `.env.example` file documents every option — here are the important ones:

### Required

| Variable | What it does |
|----------|-------------|
| `ANTHROPIC_API_KEY` | API key for Claude (default LLM provider) |

That's it. Everything else has sensible defaults.

### LLM Provider

| Variable | Default | Options |
|----------|---------|---------|
| `LLM_PROVIDER` | `claude` | `claude`, `openai`, `ollama` |
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `DEFAULT_MODEL` | (provider default) | Override the main conversation model |
| `UTILITY_MODEL` | (provider default) | Override the background/scout model |

### Embeddings

| Variable | Default | What it does |
|----------|---------|-------------|
| `DORIS_EMBED_MODEL` | `qwen3-embedding:8b` | Ollama embedding model |
| `DORIS_EMBED_DIMS` | `1024` | Embedding dimensions |

Requires [Ollama](https://ollama.ai) running locally with the model pulled.

### Channels

| Variable | What it does |
|----------|-------------|
| `CHANNEL` | Comma-separated list: `telegram`, `discord`, `bluebubbles`, `webhook` |
| `TELEGRAM_BOT_TOKEN` | From [@BotFather](https://t.me/botfather) |
| `TELEGRAM_ALLOWED_USERS` | Comma-separated Telegram user IDs |
| `DISCORD_BOT_TOKEN` | From Discord Developer Portal |
| `DISCORD_ALLOWED_USERS` | Comma-separated Discord user IDs |
| `BLUEBUBBLES_URL` | BlueBubbles server URL |
| `BLUEBUBBLES_PASSWORD` | BlueBubbles server password |
| `WEBHOOK_SECRET` | Bearer token for webhook auth |
| `WEBHOOK_OUTBOUND_URL` | URL for proactive outbound messages |

### Services

| Variable | What it does |
|----------|-------------|
| `DORIS_USER_EMAIL` | Your Gmail address (for email scout) |
| `DORIS_HOME_LAT` / `DORIS_HOME_LON` | Home coordinates (for weather, location) |
| `HA_URL` / `HA_TOKEN` | Home Assistant (for smart home control) |
| `DORIS_API_TOKEN` | Bearer token for API authentication |

See `.env.example` for the complete list with descriptions.

## Platform Notes

Doris was built on macOS and many of her tools use macOS-specific APIs:

- **Calendar & Reminders** — AppleScript → Apple Calendar / Reminders
- **iMessage** — AppleScript → Messages.app (or BlueBubbles for API-based access)
- **Health** — Apple Health data via HealthKit
- **Apple Music** — MCP server for music control
- **Contacts** — Address Book API

The core runtime (brain, memory, scouts, channels, security) works on any platform. If you're running on Linux, you'll have weather, email (Gmail), webhooks, Telegram, Discord, and the full memory system — just not the Apple-specific integrations.

## Project Structure

```
doris/
├── llm/                 # Brain, providers, tools, persona
│   ├── brain.py         # Main conversation loop + tool execution
│   ├── providers/       # Claude, OpenAI, Ollama
│   ├── tools.py         # 42 tool definitions
│   └── worm_persona.py  # Immutable personality (survives context compaction)
├── channels/            # Telegram, Discord, BlueBubbles, Webhook
├── scouts/              # 9 environment monitoring agents
├── proactive/           # Event sources, evaluator, executor, notifier
├── security/            # Auth, injection scanning, content wrapping, PII detection
├── memory/              # Thin wrappers over maasv
├── session/             # Persistent conversation context + compaction
├── tools/               # Tool implementations (calendar, email, weather, etc.)
├── mcp_client/          # MCP server connections + trust tiers
├── mcp_server/          # Doris Memory MCP server (expose memory via MCP)
├── daemon/              # Sitrep engine, digest, scheduler, scout health
├── daemon.py            # Autonomous scout scheduler + sitrep review
├── main.py              # FastAPI application entry point
├── config.py            # All settings via pydantic-settings
└── maasv_bridge.py      # Bridges Doris to the maasv cognition layer
```

## Contributing

This is my first open-source project. I've been building Doris for my own family for months and sharing it is new territory — feedback, issues, and contributions are genuinely welcome.

Some areas where help would be especially valuable:

- **Linux tool implementations** — replacing AppleScript tools with cross-platform alternatives
- **New scouts** — there's a whole world of things worth monitoring
- **Channel adapters** — Slack, Matrix, Signal, or whatever you use
- **Testing** — more edge cases, integration tests, real-world usage reports
- **Documentation** — guides, tutorials, examples of custom setups

If you find a bug or have an idea, [open an issue](https://github.com/ascottbell/doris/issues). If you want to contribute code, fork the repo and send a PR. I'm not picky about process — just be kind and write tests for anything security-related.

## License

Doris is licensed under the [Apache License 2.0](LICENSE).

[maasv](https://github.com/ascottbell/maasv) (the cognition layer) is licensed under BSL-1.1, free for personal, internal, and educational use. It converts to Apache 2.0 on February 16, 2030.

## Links

- [maasv](https://github.com/ascottbell/maasv) — The cognition layer that powers Doris's memory
- [maasv on PyPI](https://pypi.org/project/maasv/) — `pip install maasv`
