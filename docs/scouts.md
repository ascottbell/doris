# Building Scouts

Scouts are lightweight agents that monitor your environment on a schedule and surface what matters. This guide covers everything you need to build a production-quality scout from scratch.

## How Scouts Work

```
Scout.observe()
    │
    ▼
Observations (with relevance + escalate flag)
    │
    ▼
Awareness Digest (collects, deduplicates)
    │
    ▼
Daemon checks: any escalations?
    │ yes
    ▼
Brain evaluates (default model) + wisdom check
    │ worth notifying?
    ▼
Notification (push, channel message, etc.)
```

Scouts don't act on their own. They create **observations** — structured descriptions of something noteworthy. The daemon collects observations from all scouts, deduplicates them, and escalates the important ones to Doris's brain. The brain (running on the default model tier) decides what's worth notifying you about, consulting the wisdom system for past outcomes of similar situations.

This separation matters: scouts run on the cheap utility model (or no model at all), while the expensive thinking only happens when something actually needs attention.

## Minimal Scout

```python
from datetime import datetime
from scouts.base import Scout, Observation, Relevance


class DiskScout(Scout):
    name = "disk"

    async def observe(self) -> list[Observation]:
        import shutil

        usage = shutil.disk_usage("/")
        free_gb = usage.free / (1024 ** 3)

        if free_gb < 10:
            return [Observation(
                scout=self.name,
                timestamp=datetime.now(),
                observation=f"Low disk space: {free_gb:.1f} GB free",
                relevance=Relevance.HIGH,
                escalate=True,
                context_tags=["system", "disk"],
            )]
        return []
```

That's a working scout. It checks disk space, returns an observation if it's low, and returns nothing otherwise. No LLM calls, no API keys, no dependencies.

## Observation Model

Every observation has these fields:

```python
@dataclass
class Observation:
    scout: str                           # Your scout's name
    timestamp: datetime                  # When you observed this
    observation: str                     # Human-readable description
    relevance: Relevance                 # LOW, MEDIUM, or HIGH
    escalate: bool = False               # Should this wake Doris?
    context_tags: list[str] = []         # Categorization tags
    raw_data: dict | None = None         # Structured data for the brain
```

### Relevance Levels

| Level | What happens | Use when |
|-------|-------------|----------|
| `LOW` | Logged, discarded | Informational. No action needed. |
| `MEDIUM` | Added to awareness digest | Worth noting. Include in daily briefing. |
| `HIGH` | Considered for escalation | Time-sensitive. Needs attention soon. |

### Escalation

Setting `escalate=True` with `HIGH` relevance sends the observation to the brain for a real-time decision. The brain may notify the user, take action, or decide it's not worth interrupting.

Use `escalate=True` for:
- Time-sensitive situations (departure alerts, appointment reminders)
- Safety/security concerns
- Financial alerts above a threshold
- School/family communications requiring action
- Deadlines within 24-48 hours

Use `escalate=False` even with `HIGH` relevance when:
- It's important but not time-sensitive
- It should appear in the next briefing rather than interrupt now

### Raw Data

The `raw_data` field carries structured data that the brain can use for deeper analysis. The `observation` string is what humans see in logs and dashboards; `raw_data` is what the brain gets for decision-making.

```python
Observation(
    scout="stocks",
    timestamp=datetime.now(),
    observation="AAPL dropped 8.2% to $187.50",
    relevance=Relevance.HIGH,
    escalate=True,
    context_tags=["finance", "portfolio"],
    raw_data={
        "symbol": "AAPL",
        "price": 187.50,
        "change_pct": -8.2,
        "volume": 98_000_000,
        "market_cap_b": 2890,
    },
)
```

## Scout Patterns

### Pattern 1: Pure Rules (No LLM)

The cheapest and fastest pattern. Check a condition, return an observation if it matches. No API calls, no tokens, no latency.

Good for: thresholds, schedules, state comparisons, system health.

```python
class UptimeScout(Scout):
    name = "uptime"

    def __init__(self):
        super().__init__()
        self._alerted_services: set[str] = set()

    async def observe(self) -> list[Observation]:
        import httpx

        services = {
            "api": "https://api.example.com/health",
            "dashboard": "https://dashboard.example.com/health",
        }

        observations = []
        now = datetime.now()

        for name, url in services.items():
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        raise httpx.HTTPStatusError("", request=resp.request, response=resp)

                # Service recovered — clear alert
                self._alerted_services.discard(name)

            except Exception as e:
                if name not in self._alerted_services:
                    self._alerted_services.add(name)
                    observations.append(Observation(
                        scout=self.name,
                        timestamp=now,
                        observation=f"{name} is down: {str(e)[:100]}",
                        relevance=Relevance.HIGH,
                        escalate=True,
                        context_tags=["system", "uptime", name],
                    ))

        return observations
```

**Key details:**
- `_alerted_services` prevents re-alerting for the same outage
- `.discard()` clears the alert when the service recovers
- Alert once per outage, not every poll

### Pattern 2: LLM Classification

When rules aren't enough and you need judgment. Extend `HaikuScout` to get `classify_relevance()` and `analyze_with_haiku()`.

Good for: email triage, news filtering, content analysis — anything where "is this important?" requires understanding.

```python
from scouts.base import HaikuScout, Observation, Relevance


class NewsScout(HaikuScout):
    name = "news"

    def __init__(self):
        super().__init__()
        self._seen_ids: set[str] = set()

    async def observe(self) -> list[Observation]:
        articles = await self._fetch_headlines()
        observations = []
        now = datetime.now()

        for article in articles:
            if article["id"] in self._seen_ids:
                continue
            self._seen_ids.add(article["id"])

            # LLM decides relevance
            relevance, escalate, tags = await self.classify_relevance(
                f"Headline: {article['title']}\nSource: {article['source']}",
                context="User is a tech professional interested in AI, finance, "
                        "and NYC local news. Not interested in celebrity gossip."
            )

            if relevance != Relevance.LOW:
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation=f"{article['source']}: {article['title']}",
                    relevance=relevance,
                    escalate=escalate,
                    context_tags=tags + ["news"],
                    raw_data=article,
                ))

        # Cap memory
        if len(self._seen_ids) > 200:
            self._seen_ids = set(list(self._seen_ids)[-200:])

        return observations
```

**`classify_relevance()` returns a tuple:**
```python
(Relevance, bool, list[str])
#  relevance, escalate, tags
```

It handles JSON parsing, markdown stripping, validation, and falls back to `(MEDIUM, False, [])` on any failure. You don't need to handle LLM errors yourself.

**`analyze_with_haiku()` for free-form analysis:**

```python
summary = await self.analyze_with_haiku(
    f"Summarize this earnings report in 2 sentences: {report[:2000]}",
    system_prompt="Be concise. Focus on surprises vs expectations."
)
```

Returns the model's text response. Uses the utility model tier (Haiku-class).

### Pattern 3: External API + Security Scanning

When your scout pulls data from an external source (APIs, RSS feeds, scraped content), that content is untrusted. It could contain prompt injection payloads — especially if anyone besides you can write to that data source.

```python
from scouts.base import Scout, Observation, Relevance
from security.injection_scanner import scan_for_injection


class RSSScout(Scout):
    name = "rss"

    async def observe(self) -> list[Observation]:
        import feedparser

        feed = feedparser.parse("https://example.com/feed.xml")
        observations = []
        now = datetime.now()

        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")[:500]

            # REQUIRED: scan external content
            title_scan = scan_for_injection(title, source=self.name)
            if title_scan.is_suspicious:
                logger.warning(
                    f"[{self.name}] Suspicious title (risk={title_scan.risk_level}): "
                    f"{title[:100]!r}"
                )

            summary_scan = scan_for_injection(summary, source=self.name)
            if summary_scan.is_suspicious:
                logger.warning(
                    f"[{self.name}] Suspicious summary (risk={summary_scan.risk_level})"
                )

            # Continue processing — scan-and-tag, not scan-and-block
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation=f"{title}: {summary[:200]}",
                relevance=Relevance.MEDIUM,
                context_tags=["rss"],
                raw_data={"title": title, "summary": summary},
            ))

        return observations
```

**The security model is scan-and-tag, not scan-and-block.** Blocking on suspicion would cause false positives to silently drop legitimate content. Instead:

1. Scan with `scan_for_injection(content, source="your-scout")`
2. Log warnings for suspicious content
3. Continue processing — the daemon wraps all observations in `<untrusted_scout_observation>` tags before they reach the brain

The daemon's wrapping at `daemon.py:345` is the single defense point where ALL scout observations enter the LLM. Your scout's scanning is early detection and logging.

### Pattern 4: State Comparison

Many scouts need to detect *changes*, not just current state. Track previous state and compare.

```python
class GitHubScout(Scout):
    name = "github"

    def __init__(self):
        super().__init__()
        self._last_counts: dict[str, int] = {}

    async def observe(self) -> list[Observation]:
        repos = ["myorg/api", "myorg/frontend"]
        observations = []
        now = datetime.now()

        for repo in repos:
            issues = await self._fetch_open_issues(repo)
            current_count = len(issues)
            last_count = self._last_counts.get(repo, current_count)

            new_issues = current_count - last_count
            if new_issues > 0:
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation=f"{repo}: {new_issues} new issue(s) (total: {current_count})",
                    relevance=Relevance.MEDIUM,
                    escalate=new_issues >= 5,  # Spike = something broke
                    context_tags=["github", "issues", repo.split("/")[-1]],
                ))

            self._last_counts[repo] = current_count

        return observations
```

**Key details:**
- First run: `last_count = current_count` — no false alert on startup
- Tracks per-repo state
- Escalates on spikes (5+ new issues = likely incident)

### Pattern 5: Time-Based Triggers

For daily rhythms — morning briefing, evening wind-down, workday end. Fire once per day at the right time, never spam.

```python
class RitualScout(Scout):
    name = "rituals"

    def __init__(self):
        super().__init__()
        self._triggered_today: set[str] = set()
        self._last_date: str | None = None

    async def observe(self) -> list[Observation]:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Reset at midnight
        if self._last_date != today:
            self._triggered_today = set()
            self._last_date = today

        observations = []

        # Lunch reminder
        if self._should_trigger("lunch", now, hour=12, minute=30):
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation="Lunch time — step away from the screen",
                relevance=Relevance.LOW,
                context_tags=["time", "ritual", "health"],
            ))

        # End of workday
        if self._should_trigger("eod", now, hour=18, minute=0):
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation="End of workday — time to wrap up",
                relevance=Relevance.LOW,
                context_tags=["time", "ritual", "work"],
            ))

        return observations

    def _should_trigger(
        self, name: str, now: datetime, hour: int, minute: int, window: int = 2
    ) -> bool:
        if name in self._triggered_today:
            return False
        current = now.hour * 60 + now.minute
        target = hour * 60 + minute
        if target <= current < target + window:
            self._triggered_today.add(name)
            return True
        return False
```

**Key details:**
- `_triggered_today` prevents duplicate fires
- Reset at midnight
- 2-minute window (scout runs every minute, so this guarantees exactly one fire)
- LOW relevance — these go to the digest, not push notifications

## Error Handling

Every scout MUST handle its own errors. An unhandled exception in `observe()` kills that scout's run — the daemon catches it, but you lose the entire observation cycle.

```python
async def observe(self) -> list[Observation]:
    observations = []
    now = datetime.now()

    try:
        data = await self._fetch_external_data()
        # ... process data, create observations ...

    except Exception as e:
        logger.error(f"[{self.name}] Error: {e}")
        observations.append(Observation(
            scout=self.name,
            timestamp=now,
            observation=f"Error checking {self.name}: {str(e)[:100]}",
            relevance=Relevance.HIGH,
            escalate=True,
            context_tags=["scout_error", self.name],
        ))

    return observations
```

Scout errors are `HIGH` + `escalate=True` because a broken scout means a blind spot. The daemon also tracks consecutive failures per scout — after 3 in a row, it auto-escalates regardless.

## Memory Management

Scouts persist in memory across runs. State tracking sets (`_seen_ids`, `_alerted_events`, etc.) grow over time. Cap them.

```python
# Bad — unbounded growth
self._seen_ids.add(item_id)

# Good — capped at 200 most recent
self._seen_ids.add(item_id)
if len(self._seen_ids) > 200:
    self._seen_ids = set(list(self._seen_ids)[-200:])
```

Pick a cap that covers your scout's typical volume between daemon restarts. Email scout uses 100, calendar uses unbounded (events are naturally finite per day), reminders uses 500.

## Scheduling

Scouts are scheduled in `daemon/scheduler.py`. Each scout gets an interval or cron trigger.

```python
# Every 15 minutes
self.scheduler.add_job(
    self._run_scout,
    IntervalTrigger(minutes=15),
    args=[MyScout()],
    id="my_scout",
    name="My Scout (15 min)",
)

# Daily at 8 AM
self.scheduler.add_job(
    self._run_scout,
    CronTrigger(hour=8, minute=0),
    args=[DailyScout()],
    id="daily_scout",
    name="Daily Scout (8 AM)",
)

# Every minute (for time-based triggers)
self.scheduler.add_job(
    self._run_scout,
    IntervalTrigger(minutes=1),
    args=[TimeScout()],
    id="time_scout",
    name="Time Scout (1 min)",
)
```

**Schedule guidelines:**

| Data source | Interval | Why |
|-------------|----------|-----|
| Time-of-day triggers | Every minute | Needs precision, but observe() is <1ms |
| Reminders, calendar | Every 15 min | Events don't change faster than this |
| Email | Every 30 min | Balance between responsiveness and API quota |
| Weather | Every hour | Forecasts update hourly |
| Health, memory | Daily | Trends, not real-time |
| System health | Every 15 min | Catch degradation before it compounds |

## Registration

Two places to register a new scout:

### 1. `scouts/__init__.py`

```python
from scouts.my_scout import MyScout

__all__ = [
    # ... existing scouts ...
    "MyScout",
]


def get_all_scouts() -> list[Scout]:
    return [
        # ... existing scouts ...
        MyScout(),
    ]
```

### 2. `daemon/scheduler.py`

Add a job in `setup_jobs()`:

```python
self.my_scout = MyScout()

self.scheduler.add_job(
    self._run_scout,
    IntervalTrigger(minutes=15),
    args=[self.my_scout],
    id="my_scout",
    name="My Scout (15 min)",
)
```

## Testing

Every scout should have a `__main__` block for quick manual testing:

```python
if __name__ == "__main__":
    import asyncio

    async def test():
        scout = MyScout()
        print(f"Running {scout.name}...")

        observations = await scout.run()

        if not observations:
            print("No observations")
        else:
            for obs in observations:
                marker = "!" if obs.escalate else f"[{obs.relevance.value}]"
                print(f"{marker} {obs.observation}")
                print(f"  Tags: {obs.context_tags}")
                if obs.raw_data:
                    print(f"  Data: {obs.raw_data}")
                print()

    asyncio.run(test())
```

Run with:
```bash
python -m scouts.my_scout
```

For automated tests, mock the external data source and verify observation output:

```python
import pytest
from scouts.my_scout import MyScout
from scouts.base import Relevance


@pytest.mark.asyncio
async def test_my_scout_detects_anomaly(monkeypatch):
    scout = MyScout()

    # Mock external data
    async def mock_fetch():
        return {"value": 999, "threshold": 100}
    monkeypatch.setattr(scout, "_fetch_data", mock_fetch)

    observations = await scout.observe()

    assert len(observations) == 1
    assert observations[0].relevance == Relevance.HIGH
    assert observations[0].escalate is True
    assert "999" in observations[0].observation


@pytest.mark.asyncio
async def test_my_scout_handles_api_failure(monkeypatch):
    scout = MyScout()

    async def mock_fetch():
        raise ConnectionError("API down")
    monkeypatch.setattr(scout, "_fetch_data", mock_fetch)

    observations = await scout.observe()

    # Should report the error, not crash
    assert len(observations) == 1
    assert "Error" in observations[0].observation
    assert observations[0].escalate is True
```

## Cost

Scouts run on the utility model tier (Haiku-class). At current pricing:

- **Rule-based scouts** (time, weather, system, disk): $0/month — no LLM calls
- **LLM scouts** (email, calendar): ~$0.10-0.15/month each — one classification per new item
- **All 9 built-in scouts combined**: ~$1/month

If your scout uses `classify_relevance()` or `analyze_with_haiku()`, estimate:
- ~500 input tokens + ~100 output tokens per call
- Haiku: $1/MTok input, $5/MTok output
- So each classification costs ~$0.001 (a tenth of a cent)

If your scout runs every 15 minutes and makes 1 LLM call per run: 96 calls/day × 30 days × $0.001 = ~$2.88/month. That's on the high end — most scouts only call the LLM when they find something new.

## Checklist

Before calling your scout done:

- [ ] Inherits from `Scout` or `HaikuScout`
- [ ] `name` is set (lowercase, hyphenated)
- [ ] `observe()` returns `list[Observation]` (empty list = nothing noteworthy)
- [ ] External content scanned with `scan_for_injection()`
- [ ] State tracked to prevent re-alerting (ID sets, last-seen values)
- [ ] State sets capped to prevent memory leaks
- [ ] Exceptions caught — scout errors become HIGH/escalate observations
- [ ] `__main__` test block works
- [ ] Registered in `scouts/__init__.py`
- [ ] Scheduled in `daemon/scheduler.py`
- [ ] Schedule interval documented in class docstring

## Existing Scouts

For reference, here's what ships with Doris:

| Scout | Pattern | LLM? | Schedule |
|-------|---------|------|----------|
| **Email** | API + Haiku classification | Yes | Every 30 min |
| **Calendar** | API + rules + Haiku for new events | Yes | Every hour |
| **Weather** | API + rule thresholds | No | Every hour |
| **Time** | Pure rules (daily rhythm triggers) | No | Every minute |
| **Health** | API + rule thresholds | No | Daily 8 AM |
| **Location** | API + rules | No | Every 15 min |
| **Reminders** | API + rules | No | Every 15 min |
| **System** | Local checks (disk, errors, MCP) | No | Every 15 min |
| **Memory** | Internal checks (extraction health) | No | Every hour |

Read their implementations in `scouts/` for production patterns. The email and calendar scouts are the most complex; the time and system scouts are the simplest.
