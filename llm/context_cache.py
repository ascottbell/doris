"""
Context caching for Doris.

The build_context() function is expensive - it makes API calls to:
- Weather API (~200-500ms)
- Gmail API (~300-800ms)  
- Calendar CLI (~100-200ms)
- Reminders CLI (~100-200ms)

This module provides caching to avoid rebuilding context on every message,
especially during conversation mode where follow-up questions shouldn't
require fresh email/calendar data.

Cache Strategy:
- Weather: 15 minute TTL (changes slowly)
- Calendar: 5 minute TTL (might add events via phone)
- Email: 5 minute TTL (new emails might arrive)
- Reminders: 5 minute TTL 
- Memory: No cache (cheap SQLite query)

During conversation mode, we can use even longer TTLs since we just
checked everything when the conversation started.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from functools import wraps


@dataclass
class CacheEntry:
    """Single cache entry with TTL."""
    value: Any
    expires_at: float
    
    @property
    def is_valid(self) -> bool:
        return time.time() < self.expires_at


class ContextCache:
    """
    Cache for expensive context data.
    
    Usage:
        cache = ContextCache()
        
        # First call - actually fetches weather
        weather = cache.get_or_fetch('weather', fetch_weather, ttl=900)
        
        # Second call within TTL - returns cached
        weather = cache.get_or_fetch('weather', fetch_weather, ttl=900)
        
        # Force refresh
        cache.invalidate('weather')
    """
    
    # Default TTLs in seconds
    DEFAULT_TTLS = {
        'weather': 900,      # 15 minutes
        'calendar': 300,     # 5 minutes
        'email': 300,        # 5 minutes
        'reminders': 300,    # 5 minutes
        'memory': 60,        # 1 minute (cheap anyway)
    }
    
    # Extended TTLs for conversation mode
    CONVERSATION_TTLS = {
        'weather': 1800,     # 30 minutes
        'calendar': 600,     # 10 minutes
        'email': 600,        # 10 minutes
        'reminders': 600,    # 10 minutes
        'memory': 300,       # 5 minutes
    }
    
    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._conversation_mode = False
        self._stats = {
            'hits': 0,
            'misses': 0,
            'fetch_times': {},  # key -> list of fetch times
        }
    
    def enter_conversation_mode(self):
        """Enable extended TTLs for follow-up questions."""
        self._conversation_mode = True
    
    def exit_conversation_mode(self):
        """Return to normal TTLs."""
        self._conversation_mode = False
    
    def get_ttl(self, key: str) -> int:
        """Get appropriate TTL for key based on mode."""
        ttls = self.CONVERSATION_TTLS if self._conversation_mode else self.DEFAULT_TTLS
        return ttls.get(key, 300)  # Default 5 min
    
    def get_or_fetch(
        self, 
        key: str, 
        fetch_fn: Callable[[], Any],
        ttl: Optional[int] = None,
        force_refresh: bool = False
    ) -> Any:
        """
        Get cached value or fetch if expired/missing.
        
        Args:
            key: Cache key
            fetch_fn: Function to call if cache miss
            ttl: Override TTL (seconds), or None for default
            force_refresh: If True, always fetch fresh
        
        Returns:
            Cached or freshly fetched value
        """
        if ttl is None:
            ttl = self.get_ttl(key)

        # Check cache (under lock to prevent stale reads from bg refresh)
        with self._lock:
            if not force_refresh and key in self._cache:
                entry = self._cache[key]
                if entry.is_valid:
                    self._stats['hits'] += 1
                    return entry.value
            self._stats['misses'] += 1

        # Fetch outside lock to avoid blocking readers during slow API calls
        start = time.perf_counter()
        value = fetch_fn()
        elapsed = (time.perf_counter() - start) * 1000  # ms

        # Store result under lock
        with self._lock:
            if key not in self._stats['fetch_times']:
                self._stats['fetch_times'][key] = []
            self._stats['fetch_times'][key].append(elapsed)

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + ttl
            )

        return value
    
    def invalidate(self, key: str):
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def invalidate_all(self):
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total if total > 0 else 0

            avg_fetch_times = {}
            for key, times in self._stats['fetch_times'].items():
                if times:
                    avg_fetch_times[key] = sum(times) / len(times)

            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': f"{hit_rate:.1%}",
                'avg_fetch_times_ms': avg_fetch_times,
                'cached_keys': list(self._cache.keys()),
                'conversation_mode': self._conversation_mode,
            }


# Global cache instance
_cache: Optional[ContextCache] = None


def get_cache() -> ContextCache:
    """Get or create global cache."""
    global _cache
    if _cache is None:
        _cache = ContextCache()
    return _cache


def cached_build_context() -> str:
    """
    Build context with caching.
    
    This is a drop-in replacement for build_context() in llm/local.py
    that uses caching to avoid expensive API calls on every message.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from tools import cal, reminders, gmail, weather
    from memory.store import get_tiered_memory_context
    cache = get_cache()
    eastern = ZoneInfo("America/New_York")
    now = datetime.now(eastern)

    parts = []

    # Current time context (never cached - always fresh)
    time_of_day = "morning" if now.hour < 12 else "afternoon" if now.hour < 17 else "evening"
    parts.append(f"Current time: {now.strftime('%A, %B %d at %-I:%M %p')} ({time_of_day})")

    # Memory context - tiered approach: core memories always, others by priority
    # No semantic search on hot path (saves 400ms)
    try:
        memory_context = cache.get_or_fetch(
            'memory',
            lambda: get_tiered_memory_context(core_limit=10, relevant_limit=5)
        )
        if memory_context:
            parts.append(memory_context)
    except Exception:
        pass
    
    # Weather (cached - API call)
    try:
        weather_summary = cache.get_or_fetch(
            'weather',
            weather.get_weather_summary
        )
        parts.append(weather_summary)
    except Exception:
        pass
    
    # Today's calendar (cached - subprocess call)
    try:
        def fetch_calendar():
            events = cal.get_todays_events()
            if not events:
                return "Today's calendar: No events scheduled"
            
            event_strs = []
            for e in events[:5]:
                start = datetime.fromisoformat(e['start'].replace('Z', '+00:00'))
                start_local = start.astimezone(eastern)
                if e['isAllDay']:
                    event_strs.append(f"- {e['title']} (all day)")
                else:
                    time_str = start_local.strftime("%-I:%M %p").lower()
                    event_strs.append(f"- {e['title']} at {time_str}")
            return "Today's calendar:\n" + "\n".join(event_strs)
        
        calendar_context = cache.get_or_fetch('calendar', fetch_calendar)
        parts.append(calendar_context)
    except Exception:
        pass
    
    # Important emails (cached - Gmail API)
    try:
        def fetch_emails():
            emails = gmail.scan_recent_emails(hours=24)
            if not emails:
                return None
            email_strs = []
            for e in emails[:3]:
                sender = e['sender_name'] or e['sender'].split('@')[0]
                email_strs.append(f"- From {sender}: {e['subject']}")
            return "Recent important emails:\n" + "\n".join(email_strs)
        
        email_context = cache.get_or_fetch('email', fetch_emails)
        if email_context:
            parts.append(email_context)
    except Exception:
        pass
    
    # Active reminders (cached - subprocess call)
    try:
        def fetch_reminders():
            reminder_items = reminders.list_reminders()
            if not reminder_items:
                return None
            reminder_strs = [f"- {r['title']}" for r in reminder_items[:5]]
            return f"Active reminders ({len(reminder_items)} total):\n" + "\n".join(reminder_strs)
        
        reminders_context = cache.get_or_fetch('reminders', fetch_reminders)
        if reminders_context:
            parts.append(reminders_context)
    except Exception:
        pass
    
    return "\n\n".join(parts)


def print_cache_stats():
    """Print cache statistics."""
    stats = get_cache().get_stats()
    print("\n=== CONTEXT CACHE STATS ===")
    print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']}")
    print(f"Conversation mode: {stats['conversation_mode']}")
    print("Average fetch times:")
    for key, ms in stats['avg_fetch_times_ms'].items():
        print(f"  {key}: {ms:.1f}ms")
    print("=" * 30)


# Background refresh state
_refresh_thread: Optional[threading.Thread] = None
_refresh_stop_event: Optional[threading.Event] = None


def prewarm_context(verbose: bool = True) -> dict:
    """
    Pre-warm the context cache by fetching all data sources.

    Args:
        verbose: If True, print timing for each source

    Returns:
        Dict with timing results for each source
    """
    from tools import cal, reminders, gmail, weather
    from memory.store import get_tiered_memory_context
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if verbose:
        print("[Pre-warming context cache...]")

    cache = get_cache()
    eastern = ZoneInfo("America/New_York")
    timings = {}
    total_start = time.perf_counter()

    # Weather
    try:
        start = time.perf_counter()
        cache.get_or_fetch('weather', weather.get_weather_summary, force_refresh=True)
        elapsed = (time.perf_counter() - start) * 1000
        timings['weather'] = elapsed
        if verbose:
            print(f"  ✓ Weather: {elapsed:.0f}ms")
    except Exception as e:
        timings['weather'] = None
        if verbose:
            print(f"  ✗ Weather: failed ({e})")

    # Email
    try:
        start = time.perf_counter()
        def fetch_emails():
            emails = gmail.scan_recent_emails(hours=24)
            if not emails:
                return None
            email_strs = []
            for e in emails[:3]:
                sender = e['sender_name'] or e['sender'].split('@')[0]
                email_strs.append(f"- From {sender}: {e['subject']}")
            return "Recent important emails:\n" + "\n".join(email_strs)
        cache.get_or_fetch('email', fetch_emails, force_refresh=True)
        elapsed = (time.perf_counter() - start) * 1000
        timings['email'] = elapsed
        if verbose:
            print(f"  ✓ Email: {elapsed:.0f}ms")
    except Exception as e:
        timings['email'] = None
        if verbose:
            print(f"  ✗ Email: failed ({e})")

    # Calendar
    try:
        start = time.perf_counter()
        def fetch_calendar():
            events = cal.get_todays_events()
            if not events:
                return "Today's calendar: No events scheduled"
            event_strs = []
            for e in events[:5]:
                start_dt = datetime.fromisoformat(e['start'].replace('Z', '+00:00'))
                start_local = start_dt.astimezone(eastern)
                if e['isAllDay']:
                    event_strs.append(f"- {e['title']} (all day)")
                else:
                    time_str = start_local.strftime("%-I:%M %p").lower()
                    event_strs.append(f"- {e['title']} at {time_str}")
            return "Today's calendar:\n" + "\n".join(event_strs)
        cache.get_or_fetch('calendar', fetch_calendar, force_refresh=True)
        elapsed = (time.perf_counter() - start) * 1000
        timings['calendar'] = elapsed
        if verbose:
            print(f"  ✓ Calendar: {elapsed:.0f}ms")
    except Exception as e:
        timings['calendar'] = None
        if verbose:
            print(f"  ✗ Calendar: failed ({e})")

    # Reminders
    try:
        start = time.perf_counter()
        def fetch_reminders():
            reminder_items = reminders.list_reminders()
            if not reminder_items:
                return None
            reminder_strs = [f"- {r['title']}" for r in reminder_items[:5]]
            return f"Active reminders ({len(reminder_items)} total):\n" + "\n".join(reminder_strs)
        cache.get_or_fetch('reminders', fetch_reminders, force_refresh=True)
        elapsed = (time.perf_counter() - start) * 1000
        timings['reminders'] = elapsed
        if verbose:
            print(f"  ✓ Reminders: {elapsed:.0f}ms")
    except Exception as e:
        timings['reminders'] = None
        if verbose:
            print(f"  ✗ Reminders: failed ({e})")

    total_elapsed = (time.perf_counter() - total_start) * 1000
    timings['total'] = total_elapsed

    if verbose:
        print(f"[Context ready - total: {total_elapsed:.0f}ms]")

    return timings


def _background_refresh_loop(interval: int, stop_event: threading.Event):
    """
    Background thread loop that refreshes cache entries before they expire.

    Checks every `interval` seconds and refreshes entries expiring within 60s.
    """
    from tools import cal, reminders, gmail, weather
    from memory.store import get_tiered_memory_context
    from datetime import datetime
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo("America/New_York")

    # Fetch functions for each source
    def fetch_weather():
        return weather.get_weather_summary()

    def fetch_emails():
        emails = gmail.scan_recent_emails(hours=24)
        if not emails:
            return None
        email_strs = []
        for e in emails[:3]:
            sender = e['sender_name'] or e['sender'].split('@')[0]
            email_strs.append(f"- From {sender}: {e['subject']}")
        return "Recent important emails:\n" + "\n".join(email_strs)

    def fetch_calendar():
        events = cal.get_todays_events()
        if not events:
            return "Today's calendar: No events scheduled"
        event_strs = []
        for e in events[:5]:
            start_dt = datetime.fromisoformat(e['start'].replace('Z', '+00:00'))
            start_local = start_dt.astimezone(eastern)
            if e['isAllDay']:
                event_strs.append(f"- {e['title']} (all day)")
            else:
                time_str = start_local.strftime("%-I:%M %p").lower()
                event_strs.append(f"- {e['title']} at {time_str}")
        return "Today's calendar:\n" + "\n".join(event_strs)

    def fetch_reminders():
        reminder_items = reminders.list_reminders()
        if not reminder_items:
            return None
        reminder_strs = [f"- {r['title']}" for r in reminder_items[:5]]
        return f"Active reminders ({len(reminder_items)} total):\n" + "\n".join(reminder_strs)

    sources = [
        ('weather', fetch_weather),
        ('email', fetch_emails),
        ('calendar', fetch_calendar),
        ('reminders', fetch_reminders),
    ]

    while not stop_event.is_set():
        # Wait for interval or until stop is requested
        if stop_event.wait(timeout=interval):
            break

        cache = get_cache()
        now = time.time()

        for key, fetch_fn in sources:
            # Check if entry expires within 60 seconds (read under lock)
            with cache._lock:
                entry = cache._cache.get(key)
                time_until_expiry = (entry.expires_at - now) if entry else -1

            if entry is not None and time_until_expiry <= 60:
                    # Refresh this entry
                    try:
                        start = time.perf_counter()
                        cache.get_or_fetch(key, fetch_fn, force_refresh=True)
                        elapsed = (time.perf_counter() - start) * 1000
                        print(f"[bg] Refreshed {key} ({elapsed:.0f}ms)")
                    except Exception as e:
                        print(f"[bg] Failed to refresh {key}: {e}")

                    # Small stagger between refreshes to avoid hammering APIs
                    time.sleep(0.5)


def start_background_refresh(interval: int = 120):
    """
    Start background thread that refreshes cache before entries expire.

    Args:
        interval: Seconds between refresh checks (default 120)
    """
    global _refresh_thread, _refresh_stop_event

    # Stop existing thread if running
    if _refresh_thread is not None and _refresh_thread.is_alive():
        stop_background_refresh()

    _refresh_stop_event = threading.Event()
    _refresh_thread = threading.Thread(
        target=_background_refresh_loop,
        args=(interval, _refresh_stop_event),
        daemon=True,
        name="context-cache-refresh"
    )
    _refresh_thread.start()
    print(f"[Background refresh started (interval: {interval}s)]")


def stop_background_refresh():
    """Stop the background refresh thread cleanly."""
    global _refresh_thread, _refresh_stop_event

    if _refresh_stop_event is not None:
        _refresh_stop_event.set()

    if _refresh_thread is not None and _refresh_thread.is_alive():
        _refresh_thread.join(timeout=2.0)

    _refresh_thread = None
    _refresh_stop_event = None


# Quick test
if __name__ == "__main__":
    import time
    
    print("Testing context cache...")
    cache = get_cache()
    
    # Simulate expensive fetch
    def slow_fetch():
        time.sleep(0.5)
        return "expensive result"
    
    # First call - should be slow
    start = time.time()
    result = cache.get_or_fetch('test', slow_fetch, ttl=10)
    print(f"First call: {(time.time() - start)*1000:.1f}ms - {result}")
    
    # Second call - should be fast (cached)
    start = time.time()
    result = cache.get_or_fetch('test', slow_fetch, ttl=10)
    print(f"Second call: {(time.time() - start)*1000:.1f}ms - {result}")
    
    print_cache_stats()
    
    # Test with real context
    print("\nBuilding real context (first call)...")
    start = time.time()
    context = cached_build_context()
    print(f"First build: {(time.time() - start)*1000:.1f}ms")
    print(f"Context length: {len(context)} chars")
    
    print("\nBuilding real context (second call - cached)...")
    start = time.time()
    context = cached_build_context()
    print(f"Second build: {(time.time() - start)*1000:.1f}ms")
    
    print_cache_stats()
