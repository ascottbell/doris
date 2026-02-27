"""
Runtime platform detection.

Provides constants for gating platform-specific code paths.
Used as a failsafe: even if services.yaml configures a macosx service
on Linux, the system refuses to load it.
"""

import sys

IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform == "linux"
PLATFORM = "macos" if IS_MACOS else "linux" if IS_LINUX else sys.platform
