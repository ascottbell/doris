"""
Log Rotation for Doris

Rotates launchd-managed log files using copytruncate semantics:
1. Copy current log to timestamped file
2. Truncate the original (launchd holds the fd)
3. Compress the copy with gzip
4. Delete copies older than retention period

Called daily by the daemon health scout.
"""

import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"

# Files to rotate with size thresholds (MB)
ROTATION_CONFIG = {
    "server.error.log": 10,   # Rotate when > 10MB
    "server.log": 50,         # Rotate when > 50MB
    "daemon.error.log": 10,
    "daemon.log": 10,
}

RETENTION_DAYS = 7


def rotate_logs() -> list[str]:
    """
    Rotate any log files that exceed their size threshold.

    Returns list of actions taken.
    """
    actions = []

    for filename, threshold_mb in ROTATION_CONFIG.items():
        log_path = LOGS_DIR / filename
        if not log_path.exists():
            continue

        size_mb = log_path.stat().st_size / (1024 * 1024)
        if size_mb < threshold_mb:
            continue

        try:
            # Generate rotated filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            stem = log_path.stem  # e.g. "server.error" or "server"
            suffix = log_path.suffix  # e.g. ".log"
            rotated_name = f"{stem}-{timestamp}{suffix}.gz"
            rotated_path = LOGS_DIR / rotated_name

            # Copy and compress
            with open(log_path, 'rb') as f_in:
                with gzip.open(rotated_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Truncate the original (launchd keeps the fd open)
            with open(log_path, 'w') as f:
                f.write("")

            actions.append(f"Rotated {filename} ({size_mb:.0f}MB) -> {rotated_name}")

        except Exception as e:
            actions.append(f"Failed to rotate {filename}: {e}")

    # Clean up old rotated files
    cleanup_actions = _cleanup_old_logs()
    actions.extend(cleanup_actions)

    return actions


def _cleanup_old_logs() -> list[str]:
    """Delete compressed log files older than RETENTION_DAYS."""
    actions = []
    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)

    for gz_file in LOGS_DIR.glob("*.log.gz"):
        try:
            mtime = datetime.fromtimestamp(gz_file.stat().st_mtime)
            if mtime < cutoff:
                gz_file.unlink()
                actions.append(f"Deleted old log: {gz_file.name}")
        except Exception as e:
            actions.append(f"Failed to clean {gz_file.name}: {e}")

    return actions


def get_log_sizes() -> dict[str, float]:
    """Get current log file sizes in MB."""
    sizes = {}
    for filename in ROTATION_CONFIG:
        log_path = LOGS_DIR / filename
        if log_path.exists():
            sizes[filename] = round(log_path.stat().st_size / (1024 * 1024), 1)
    return sizes
