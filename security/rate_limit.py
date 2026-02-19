"""Rate limiting middleware for Doris API endpoints.

Implements a fixed-window rate limiter per client IP, applied as ASGI
middleware to configurable path prefixes. Returns 429 with Retry-After
header when the limit is exceeded.

Supports X-Forwarded-For extraction when running behind a trusted proxy
(Docker, nginx, Tailscale). Configure TRUSTED_PROXIES env var as a
comma-separated list of proxy IPs/CIDRs.
"""

import ipaddress
import logging
import os
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from security.audit import audit

logger = logging.getLogger(__name__)


def _parse_trusted_proxies() -> set[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Parse TRUSTED_PROXIES env var into a set of IP networks.

    Accepts comma-separated IPs or CIDRs:
        TRUSTED_PROXIES=127.0.0.1,10.0.0.0/8,::1
    """
    raw = os.environ.get("TRUSTED_PROXIES", "")
    if not raw.strip():
        return set()
    networks = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.add(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning(f"TRUSTED_PROXIES: invalid entry '{entry}', skipping")
    return networks


# Parse once at module load
TRUSTED_PROXY_NETWORKS = _parse_trusted_proxies()


def _is_trusted_proxy(ip: str) -> bool:
    """Check if an IP address belongs to a trusted proxy network."""
    if not TRUSTED_PROXY_NETWORKS:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in network for network in TRUSTED_PROXY_NETWORKS)


def get_client_ip(request: Request) -> str:
    """Extract the real client IP, respecting X-Forwarded-For from trusted proxies.

    If the direct connection is from a trusted proxy and X-Forwarded-For is present,
    walks the chain right-to-left and returns the first IP not in the trusted set.
    This prevents spoofing: untrusted clients can't forge their position in the chain.
    """
    direct_ip = request.client.host if request.client else "unknown"

    if not _is_trusted_proxy(direct_ip):
        return direct_ip

    xff = request.headers.get("x-forwarded-for", "")
    if not xff:
        return direct_ip

    # Walk right-to-left: rightmost entries are added by proxies closest to us
    ips = [ip.strip() for ip in xff.split(",") if ip.strip()]
    for ip in reversed(ips):
        if not _is_trusted_proxy(ip):
            return ip

    # All IPs in the chain are trusted proxies — use the leftmost
    return ips[0] if ips else direct_ip


class RateLimiter:
    """Fixed-window rate limiter per source IP.

    Tracks request counts in 60-second windows. Stale entries are cleaned
    up on each check to prevent unbounded memory growth.
    """

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        # ip -> (window_start_monotonic, count)
        self._windows: dict[str, tuple[float, int]] = {}

    def check(self, ip: str) -> tuple[bool, int]:
        """Check if request is allowed.

        Returns (allowed, seconds_until_reset).
        """
        now = time.monotonic()
        self._cleanup(now)

        if ip in self._windows:
            window_start, count = self._windows[ip]
            elapsed = now - window_start
            if elapsed < 60:
                if count >= self._max:
                    return False, int(60 - elapsed) + 1
                self._windows[ip] = (window_start, count + 1)
                return True, 0

        # New window
        self._windows[ip] = (now, 1)
        return True, 0

    def _cleanup(self, now: float) -> None:
        """Remove expired windows (older than 60s)."""
        expired = [
            ip
            for ip, (start, _) in self._windows.items()
            if now - start >= 60
        ]
        for ip in expired:
            del self._windows[ip]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that rate-limits requests to specific path prefixes.

    Args:
        app: The ASGI application.
        limiter: RateLimiter instance.
        path_prefixes: Only apply rate limiting to paths starting with these.
    """

    def __init__(self, app, limiter: RateLimiter, path_prefixes: list[str]) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._prefixes = tuple(path_prefixes)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith(self._prefixes):
            return await call_next(request)

        client_ip = get_client_ip(request)
        allowed, retry_after = self._limiter.check(client_ip)

        if not allowed:
            audit.rate_limit(ip=client_ip, path=path, limit=self._limiter._max)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
