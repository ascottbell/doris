# Doris API Reference

Complete reference for the Doris REST API, WebSocket protocol, and extensibility interfaces (channels, scouts, LLM providers).

Doris runs a FastAPI server on `localhost:8000` by default.

## Authentication

All endpoints require a bearer token except where noted.

```
Authorization: Bearer YOUR_TOKEN
```

Set the token via `DORIS_API_TOKEN` in your `.env`. If unset, auth is disabled (development only — don't do this in production).

WebSocket connections pass the token as a query parameter: `?token=YOUR_TOKEN`.

---

## Chat

The primary interface. Send a message, get a response. Doris runs up to 5 rounds of tool use per request.

### POST /chat/text

Synchronous chat. Returns when the full response is ready.

**Request:**
```json
{
  "message": "what's on my calendar today?",
  "history": null,
  "client_context": null,
  "daemon": false,
  "wake_reason": null,
  "user": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | yes | User message |
| `history` | array | no | Conversation history. If null, uses persistent server-side session. |
| `client_context` | object | no | Device context (location, timezone, etc.) |
| `daemon` | bool | no | True when called by the proactive daemon |
| `wake_reason` | string | no | `"wake_word"` or `"background"` |
| `user` | string | no | Speaker name (for multi-user setups) |

`history` entries:
```json
{"role": "user", "content": "hello"}
{"role": "assistant", "content": "hey there"}
```

**Response:**
```json
{
  "response": "You have 3 events today...",
  "source": "claude",
  "latency_ms": 2341
}
```

### POST /chat/stream

Server-Sent Events (SSE) streaming. Same request body as `/chat/text`.

**Response:** `text/event-stream`

Events arrive as JSON lines:

```
data: {"type": "text", "content": "You have "}
data: {"type": "text", "content": "3 events "}
data: {"type": "tool_start", "tool": "get_calendar"}
data: {"type": "tool_complete"}
data: {"type": "text", "content": "today..."}
data: {"type": "done", "latency_ms": 2341}
```

| Event type | Fields | Description |
|------------|--------|-------------|
| `text` | `content` | Token(s) of response text |
| `tool_start` | `tool` | Tool invocation beginning |
| `tool_complete` | — | Tool execution finished |
| `done` | `latency_ms` | Response complete |
| `error` | `message` | Error during generation |

---

## Health & Status

### GET /health

**No auth required.** Quick health check for load balancers.

```json
{
  "status": "ok",
  "ollama_model": "qwen3-coder:30b",
  "claude_model": "claude-opus-4-6"
}
```

### GET /status

Consolidated system status — channels, MCP servers, proactive system, sleep compute, etc.

### GET /health/status

Detailed health with circuit breaker state.

```json
{
  "status": "healthy",
  "claude_api": "ok",
  "disabled_tools": [],
  "degraded_tools": ["get_weather"],
  "circuit_breaker": {
    "get_weather": {"state": "half_open", "failures": 3}
  }
}
```

### POST /health/reset-circuit

Reset a circuit breaker.

| Query param | Type | Description |
|-------------|------|-------------|
| `tool_name` | string | Tool to reset, or omit for all |

### GET /health/memory

Memory system health — warnings, metrics, wisdom coverage stats.

---

## Health Data

Receive health data from an iOS/watchOS app via HealthKit sync.

### POST /health/sync

```json
{
  "date": "2026-02-17",
  "steps": 8432,
  "workouts": [
    {"type": "running", "duration_min": 35, "distance_mi": 3.2, "calories": 340}
  ],
  "sleep_hours": 7.2,
  "active_calories": 520,
  "resting_hr": 58,
  "stand_hours": 10,
  "hrv": 42,
  "vo2_max": 44.5,
  "sleep_stages": {
    "deep_hours": 1.5,
    "rem_hours": 1.8,
    "core_hours": 3.2,
    "awake_hours": 0.7
  }
}
```

All fields except `date` are optional.

### GET /health/data

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `date` | string | — | Specific date (YYYY-MM-DD) |
| `days` | int | 7 | Number of days to return (used when `date` is omitted) |

---

## Push Notifications

### POST /devices/register

Register an iOS/macOS device for APNs push notifications.

```json
{
  "device_token": "abc123...",
  "device_type": "ios",
  "device_name": "My iPhone"
}
```

### GET /devices

List all registered devices.

### POST /push/send

```json
{
  "title": "Doris",
  "body": "You have a pickup in 30 minutes",
  "device_token": null,
  "badge": 1
}
```

`device_token` null = send to all registered devices.

---

## Wisdom (Learning System)

Doris tracks outcomes and learns from feedback.

### GET /wisdom

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `limit` | int | 20 | Max entries |
| `pending_only` | bool | false | Only show entries awaiting feedback |

### GET /wisdom/stats

Aggregate statistics.

### GET /wisdom/{wisdom_id}

Single entry by ID.

### POST /wisdom/feedback

```json
{
  "wisdom_id": "abc-123",
  "score": 4,
  "notes": "Good catch, but too early in the morning"
}
```

`score`: 1-5. Positive scores (4-5) reinforce the behavior. Low scores (1-2) make Doris try differently next time.

### DELETE /wisdom/{wisdom_id}

Delete a wisdom entry.

---

## Brain Dump

Capture and process unstructured thoughts.

### GET /brain/status

Unprocessed thoughts count + preview.

### POST /brain/process

Process all unprocessed thoughts (categorize, extract tasks, link to memories).

### GET /brain/goals

Active goals for proactive surfacing.

| Query param | Type | Default |
|-------------|------|---------|
| `limit` | int | 5 |
| `days` | int | 30 |

### GET /brain/relevant

Find thoughts relevant to a given context string.

| Query param | Type | Default |
|-------------|------|---------|
| `context` | string | (required) |
| `limit` | int | 3 |

### GET /brain/surface

Proactive surfacing suggestions based on time of day, weather, health data, and goals.

| Query param | Type | Default |
|-------------|------|---------|
| `personality` | string | `"balanced"` |

Returns `suggestions`, `message` (formatted for display), and `context`.

---

## Morning Briefing

### GET /briefing

Aggregated morning briefing — calendar, emails, reminders, brain state.

| Query param | Type | Default |
|-------------|------|---------|
| `include_brain` | bool | true |

```json
{
  "calendar": [...],
  "emails": [...],
  "unread_count": 12,
  "reminders": [...],
  "brain": {
    "goals": [...],
    "suggestions": [...],
    "message": "Good morning..."
  }
}
```

---

## Email

### GET /emails

| Query param | Type | Default |
|-------------|------|---------|
| `hours` | int | 24 |

```json
{
  "unread_count": 12,
  "important_emails": [...]
}
```

---

## Agent Communication

Internal message bus between Doris and other agents (e.g., Claude Code).

### POST /agents/message

```json
{
  "from_agent": "claude_code",
  "to_agent": "doris",
  "message_type": "notification",
  "content": "Build complete — 0 errors, 3 warnings",
  "context": {"project": "doris-public"},
  "priority": "normal",
  "expects_response": false,
  "related_to": null
}
```

| `message_type` | Description |
|-----------------|-------------|
| `chat` | General conversation |
| `notification` | One-way alert |
| `question` | Expects a response |
| `handoff` | Transfer context between agents |
| `update` | Status update |

### GET /agents/messages

| Query param | Type | Default |
|-------------|------|---------|
| `agent` | string | — |
| `limit` | int | 50 |

### GET /agents/messages/pending

Unread messages for a specific agent.

| Query param | Type | Required |
|-------------|------|----------|
| `agent` | string | yes |
| `limit` | int | 20 |

### GET /agents/thread/{message_id}

Full conversation thread from a message.

### POST /agents/notify-user

Quick notification to the user via Doris + push.

```json
{
  "message": "Your deploy finished successfully",
  "context": {"project": "doris-public", "commit": "abc123"}
}
```

---

## Gemini Consultant

External LLM consultant endpoints using Google's Gemini models.

| Endpoint | Model | Cost | Latency |
|----------|-------|------|---------|
| `POST /gemini/quick` | Flash | ~$0.001 | <1s |
| `POST /gemini/brainstorm` | Pro | ~$0.01 | 2-5s |
| `POST /gemini/code-review` | Pro | ~$0.01 | 2-5s |
| `POST /gemini/validate` | Pro | ~$0.01 | 2-5s |
| `POST /gemini/research` | Deep Research | **$2-5** | **2-5 min** |

**Quick/Brainstorm/Validate request:**
```json
{
  "query": "What's the best way to handle rate limiting in FastAPI?",
  "context": {}
}
```

**Code review request:**
```json
{
  "code": "def process(data):\n    ...",
  "description": "Data processing function for health sync",
  "context": {}
}
```

**Research request:**
```json
{
  "question": "Compare BlueBubbles vs Beeper for iMessage integration",
  "context": {}
}
```

Research response:
```json
{
  "success": true,
  "report": "...",
  "interaction_id": "abc-123",
  "elapsed_seconds": 187,
  "status": "completed"
}
```

---

## Codex Consultant

Same interface pattern as Gemini but uses OpenAI's GPT-5.3-Codex model.

| Endpoint | Description |
|----------|-------------|
| `POST /codex/quick` | Fast lookup |
| `POST /codex/brainstorm` | Creative ideation |
| `POST /codex/code-review` | Code analysis |
| `POST /codex/research` | Deep research |
| `POST /codex/validate` | Critical analysis |

Same request/response shapes as Gemini equivalents.

---

## MCP (Model Context Protocol)

### GET /mcp/health

Health check all connected MCP servers. Auto-reconnects dead servers.

```json
{
  "apple-music": "healthy",
  "brave-search": "reconnected",
  "filesystem": "dead"
}
```

### GET /mcp/status

```json
{
  "connected_servers": ["apple-music", "brave-search"],
  "tools": [
    {
      "server": "apple-music",
      "name": "itunes_play",
      "description": "Play a song or playlist",
      "qualified_name": "apple-music:itunes_play"
    }
  ],
  "tool_count": 15
}
```

### POST /mcp/call

Call an MCP tool directly.

| Query param | Type | Required |
|-------------|------|----------|
| `server` | string | yes |
| `tool` | string | yes |
| `arguments` | object | no |

```json
{
  "success": true,
  "content": [...],
  "is_error": false
}
```

---

## Conversation Sync

Sync conversation history across devices.

### GET /conversations/messages

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `since` | string | — | ISO timestamp — only messages after this time |
| `limit` | int | 100 | Max messages |
| `device_id` | string | — | Filter by source device |

### POST /conversations/messages

```json
{
  "id": "msg-abc-123",
  "content": "What's the weather like?",
  "role": "user",
  "device_id": "iphone-14",
  "created_at": "2026-02-17T15:30:00Z",
  "conversation_id": "conv-001",
  "metadata": {}
}
```

### GET /conversations/recent

Recent messages across all devices.

| Query param | Type | Default |
|-------------|------|---------|
| `limit` | int | 50 |

### GET /conversations/search

Full-text search across conversation history.

| Query param | Type | Required |
|-------------|------|----------|
| `q` | string | yes |
| `limit` | int | 20 |

### DELETE /conversations/messages

**Destructive.** Clears all conversation history.

```json
{
  "success": true,
  "deleted_count": 1523,
  "message": "All conversation history cleared"
}
```

---

## Test Escalation

### POST /test-escalation

End-to-end test of the proactive escalation pipeline. Sends a synthetic observation through the full path: scout → daemon → brain → notification.

---

## Extensibility Interfaces

### Channel Adapters

Build a custom channel by implementing the `ChannelAdapter` ABC.

```python
from channels.base import ChannelAdapter, IncomingMessage, MessageHandler

class MyAdapter(ChannelAdapter):
    @property
    def name(self) -> str:
        return "my-channel"

    async def start(self, handler: MessageHandler) -> None:
        """Begin listening. Store handler for incoming messages."""
        self._running = True
        self._handler = handler
        # Start your listener (polling loop, webhook server, etc.)

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        # Clean up connections, finish in-flight messages

    async def send_message(self, conversation_id: str, text: str) -> None:
        """Send a proactive message (Doris-initiated)."""
        self._check_running()  # Raises if not started
        # Send via your platform's API
```

**Message types:**

```python
@dataclass
class IncomingMessage:
    text: str                           # Message content
    sender_id: str                      # Platform user ID
    conversation_id: str                # Platform chat/thread ID
    channel: str                        # "telegram", "discord", etc.
    sender_name: str | None = None      # Display name
    timestamp: datetime | None = None   # Platform timestamp
    metadata: dict = {}                 # Platform-specific extras

@dataclass
class OutgoingMessage:
    text: str                           # Response content
    conversation_id: str                # Where to send
    metadata: dict = {}                 # Platform-specific extras
```

**Handler type:**

```python
MessageHandler = Callable[[IncomingMessage], AsyncGenerator[str, None]]
```

The handler yields text chunks as the LLM generates. Your adapter decides how to consume them:

- **Stream progressively** (Telegram, Discord): Show placeholder, edit with accumulated text every N ms
- **Collect and send** (iMessage, SMS): `text = await collect_response(handler(msg))`
- **Return synchronously** (Webhook): Collect full response, return as JSON

Helper: `collect_response(stream)` accumulates all chunks into a single string.

Error handling: Call `self._safe_handle(handler, message)` instead of `handler(message)` directly — it catches exceptions and yields a user-facing error message.

**Registration:** Add your adapter to `channels/registry.py`:

```python
def create_adapters_from_config(settings) -> list[ChannelAdapter]:
    # Add: "my-channel": lambda: MyAdapter(settings.my_setting)
```

### Scouts

Build a custom scout by extending the `Scout` base class.

```python
from scouts.base import Scout, Observation, Relevance

class StockScout(Scout):
    name = "stocks"

    async def observe(self) -> list[Observation]:
        """Called on schedule. Return observations or empty list."""
        prices = await self._fetch_portfolio()

        observations = []
        for stock in prices:
            if abs(stock.change_pct) > 5:
                observations.append(Observation(
                    scout=self.name,
                    timestamp=datetime.now(),
                    observation=f"{stock.symbol} moved {stock.change_pct:+.1f}% to ${stock.price:.2f}",
                    relevance=Relevance.HIGH if abs(stock.change_pct) > 10 else Relevance.MEDIUM,
                    escalate=abs(stock.change_pct) > 10,
                    context_tags=["finance", "portfolio"],
                    raw_data={"symbol": stock.symbol, "price": stock.price},
                ))
        return observations
```

**Observation model:**

```python
@dataclass
class Observation:
    scout: str                              # Scout identifier
    timestamp: datetime                     # When observed
    observation: str                        # Human-readable description
    relevance: Relevance                    # LOW, MEDIUM, or HIGH
    escalate: bool = False                  # Wake Doris for immediate evaluation
    context_tags: list[str] = []           # Categorization tags
    raw_data: dict | None = None           # Optional structured data
```

**Relevance levels:**

| Level | What happens |
|-------|-------------|
| `LOW` | Logged and discarded |
| `MEDIUM` | Added to awareness digest |
| `HIGH` | Considered for immediate escalation to Doris |

Setting `escalate=True` with `HIGH` relevance wakes the brain for a real-time decision. Use this for time-sensitive situations.

**Optional override:**

```python
def should_escalate(self, observation: Observation) -> bool:
    """Custom escalation logic. Default: HIGH + escalate=True."""
    # Example: always escalate during market hours
    if 9 <= datetime.now().hour < 16:
        return observation.relevance in (Relevance.HIGH, Relevance.MEDIUM)
    return observation.relevance == Relevance.HIGH and observation.escalate
```

**Using LLM for classification:** Extend `HaikuScout` instead of `Scout` to get helper methods:

```python
from scouts.base import HaikuScout

class SmartScout(HaikuScout):
    name = "smart"

    async def observe(self) -> list[Observation]:
        data = await self.fetch_data()

        # Use the utility model to classify relevance
        relevance, escalate, tags = await self.classify_relevance(
            f"Stock {data.symbol} dropped {data.change_pct}%",
            context="User has a diversified portfolio, not a day trader"
        )

        # Or use it for free-form analysis
        analysis = await self.analyze_with_haiku(
            f"Summarize this earnings report: {data.report[:2000]}",
            system_prompt="Be concise. Focus on surprises vs expectations."
        )

        return [Observation(
            scout=self.name,
            timestamp=datetime.now(),
            observation=analysis,
            relevance=relevance,
            escalate=escalate,
            context_tags=tags,
        )]
```

**Registration:** Add your scout to `scouts/__init__.py` and the daemon picks it up automatically.

### LLM Providers

Add a new LLM backend by implementing the `LLMProvider` protocol.

```python
from llm.providers.base import LLMProvider
from llm.types import LLMResponse, StreamEvent, ToolDef, ToolCall, ToolResult, TokenUsage, StopReason

class MyProvider:
    """Implements the LLMProvider protocol."""

    def complete(
        self,
        messages: list[dict],
        *,
        system: str | list | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 1024,
        model: str | None = None,
        source: str = "",
    ) -> LLMResponse:
        """Synchronous completion. Must return an LLMResponse."""
        # Convert messages to your API format
        # Make the API call
        # Return LLMResponse(text=..., tool_calls=..., stop_reason=..., usage=...)

    def stream(
        self,
        messages: list[dict],
        *,
        system: str | list | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 1024,
        model: str | None = None,
        source: str = "",
    ) -> Generator[StreamEvent, None, None]:
        """Synchronous streaming. Yield StreamEvent objects."""

    async def astream(
        self,
        messages: list[dict],
        *,
        system: str | list | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 1024,
        model: str | None = None,
        source: str = "",
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming. Yield StreamEvent objects."""

    def build_tool_result_messages(
        self,
        assistant_content: Any,
        tool_results: list[ToolResult],
    ) -> list[dict]:
        """Build messages to thread tool results back into the conversation.

        This is provider-specific because tool result format varies:
        - Claude: user message with tool_result content blocks
        - OpenAI: separate tool-role messages per result
        """
```

**StreamEvent types your provider should yield:**

| Type | When | Required fields |
|------|------|-----------------|
| `text` | Text token generated | `text` |
| `tool_start` | Tool call begins | `tool_name`, `tool_call_id` |
| `tool_input` | Partial tool arguments | `tool_input_json` |
| `tool_stop` | Tool call definition complete | — |
| `done` | Generation complete | `usage`, `stop_reason`, `raw_content` |
| `error` | Something went wrong | `text` (error message) |

**Registration:** Add your provider to `llm/providers/__init__.py`:

```python
def get_llm_provider(provider_name: str = None) -> LLMProvider:
    # Add: "my-provider": MyProvider
```

And add model tier defaults to `resolve_model()`:

```python
def resolve_model(tier: str = "default", provider: str = None) -> str:
    DEFAULTS = {
        "my-provider": {
            "default": "my-model-large",
            "mid": "my-model-medium",
            "utility": "my-model-small",
        },
    }
```

### MCP Server Configuration

Doris connects to external MCP servers defined in `mcp_client/servers.yaml`.

```yaml
apple-music:
  type: stdio
  command: uvx
  args: ["mcp-applemusic"]
  trust_level: trusted

brave-search:
  type: stdio
  command: uvx
  args: ["mcp-brave-search"]
  env:
    BRAVE_API_KEY: "${BRAVE_API_KEY}"
  trust_level: sandboxed

my-server:
  type: http
  url: http://localhost:9000/mcp
  headers:
    Authorization: "Bearer ${MY_SERVER_TOKEN}"
  trust_level: sandboxed
```

**Trust levels:**

| Level | Description | Behavior |
|-------|-------------|----------|
| `builtin` | Part of Doris itself | No scanning, no PII filtering |
| `trusted` | Verified first-party servers | Responses scanned, PII logged but not redacted |
| `sandboxed` | Third-party / unknown | Tool descriptions scanned, responses quarantined, PII redacted from outbound args |

Default is `sandboxed`. Promote to `trusted` only after reviewing the server's code.

Environment variables in config (`${VAR}`) are expanded at load time.

---

## Error Responses

All endpoints return standard HTTP status codes. Error bodies:

```json
{
  "detail": "Human-readable error message"
}
```

| Code | Meaning |
|------|---------|
| 401 | Missing or invalid auth token |
| 404 | Resource not found |
| 422 | Validation error (bad request body) |
| 429 | Rate limited (webhook adapter) |
| 500 | Server error |
