"""
Cryptographic utilities for Doris.

Centralized token comparison and encryption helpers that support key rotation.
During rotation, set comma-separated tokens: DORIS_API_TOKEN=new_token,old_token

- Auth: accepts any token in the list
- Encrypt: always uses the first (primary) token
- Decrypt: tries all tokens in order (via MultiFernet)
"""

import base64
import hmac
import logging
from typing import Optional, Union

from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

logger = logging.getLogger(__name__)


def token_matches_any(provided: str, configured_tokens: str) -> bool:
    """Check if provided token matches any configured token.

    Supports comma-separated token lists for zero-downtime key rotation.
    All comparisons use hmac.compare_digest to prevent timing attacks.

    Args:
        provided: The token from the incoming request.
        configured_tokens: Single token or comma-separated list (e.g. "new,old").

    Returns:
        True if provided matches any configured token.
    """
    if not provided or not configured_tokens:
        return False

    for token in configured_tokens.split(","):
        token = token.strip()
        if token and hmac.compare_digest(provided, token):
            return True

    return False


def get_fernet(
    token_csv: str,
    salt: bytes,
    iterations: int = 480_000,
) -> Optional[Union[Fernet, MultiFernet]]:
    """Return a Fernet/MultiFernet for encrypting/decrypting with key rotation.

    When token_csv contains multiple comma-separated tokens:
    - encrypt() uses the first (primary) token
    - decrypt() tries all tokens in order

    Returns None if token_csv is empty (dev mode).
    """
    if not token_csv:
        logger.warning(
            "Encryption disabled — no DORIS_API_TOKEN set. "
            "Sensitive data (OAuth tokens, sessions, device tokens) will be stored as plaintext."
        )
        return None

    tokens = [t.strip() for t in token_csv.split(",") if t.strip()]
    if not tokens:
        logger.warning(
            "Encryption disabled — DORIS_API_TOKEN is empty. "
            "Sensitive data will be stored as plaintext."
        )
        return None

    fernets = []
    for token in tokens:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        key = base64.urlsafe_b64encode(kdf.derive(token.encode()))
        fernets.append(Fernet(key))

    if len(fernets) == 1:
        return fernets[0]

    return MultiFernet(fernets)
