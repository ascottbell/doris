"""
API Authentication for Doris.

Bearer token authentication using FastAPI's HTTPBearer security scheme.
Protects sensitive endpoints from unauthorized access while keeping
health checks publicly accessible.
"""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from security.audit import audit
from security.crypto import token_matches_any


# HTTPBearer security scheme
# auto_error=True means it will return 401 if no token provided
security = HTTPBearer(auto_error=True)


def verify_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Verify the Bearer token from the Authorization header.

    Args:
        request: The incoming request (for audit logging)
        credentials: HTTPBearer credentials from the request

    Returns:
        The validated token string

    Raises:
        HTTPException: 401 if token is missing or invalid
        HTTPException: 503 if API token not configured
    """
    client_ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path

    if not settings.doris_api_token:
        # Should never happen — validate_security_settings() blocks startup without a token.
        # If we're here in dev mode, reject with 401 (not 503) to avoid leaking config state.
        audit.auth_failure(ip=client_ip, reason="token_not_configured", endpoint=endpoint)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API authentication not configured — set DORIS_API_TOKEN",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not token_matches_any(credentials.credentials, settings.doris_api_token):
        audit.auth_failure(ip=client_ip, reason="invalid_token", endpoint=endpoint)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials


def get_token_dependency():
    """
    Get the token verification dependency for use in route definitions.

    Usage:
        @app.post("/protected", dependencies=[Depends(get_token_dependency())])
        async def protected_endpoint():
            ...
    """
    return verify_token
