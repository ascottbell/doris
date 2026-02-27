"""
Shared pytest configuration for Doris tests.

Sets required environment variables at module level BEFORE any project imports.
pydantic-settings reads env vars when config.py is first imported, so these must
be set before test collection triggers project imports.

This replaces the per-file os.environ.setdefault() calls that previously scattered
placeholder credentials across individual test modules.
"""

import os
from pathlib import Path

# Required by config.py (no default — pydantic-settings raises without it)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-placeholder")

# Required for security validation (validate_security_settings)
os.environ.setdefault("DORIS_API_TOKEN", "test-placeholder")

# Centralised path defaults — ensure fresh Settings() instances resolve correctly
# even when tests run from an arbitrary working directory.
os.environ.setdefault("DORIS_DATA_DIR", str(Path(__file__).parent.parent / "data"))
os.environ.setdefault("DORIS_CONFIG_DIR", str(Path(__file__).parent.parent))
