"""Security headers middleware for Doris API.

Adds standard security headers to all HTTP responses:
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- Referrer-Policy: strict-origin-when-cross-origin
- Permissions-Policy: restrict unused browser features
- Strict-Transport-Security: enforce HTTPS (production only)
- Content-Security-Policy: strict API-only policy

CSRF note: Doris uses Bearer token authentication (not cookies), and CORS is
deny-by-default. Browsers cannot send custom Authorization headers cross-origin
without CORS approval, so CSRF attacks are mitigated by design. No separate
CSRF token middleware is needed.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


# Strict CSP for API responses (no HTML rendering expected)
_API_CSP = "default-src 'none'; frame-ancestors 'none'"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that adds security headers to all responses.

    Args:
        app: The ASGI application.
        enable_hsts: Whether to add Strict-Transport-Security header.
            Should be False for local dev (plain HTTP), True behind TLS.
    """

    def __init__(self, app, enable_hsts: bool = False) -> None:
        super().__init__(app)
        self._enable_hsts = enable_hsts

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Universal security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        # Modern recommendation: disable XSS Auditor (rely on CSP instead)
        response.headers["X-XSS-Protection"] = "0"

        response.headers["Content-Security-Policy"] = _API_CSP

        # HSTS: only when running behind TLS (not local dev)
        if self._enable_hsts:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        return response
