"""
Atomic JSON file I/O with cross-process locking.

Prevents race conditions on read-modify-write cycles for JSON state files.
Uses fcntl.flock() for cross-process locking and atomic write (tmp + fsync + rename)
to prevent corruption from concurrent access between daemon, API server, and monitors.

Usage:
    from security.file_io import atomic_json_write, locked_json_read, locked_json_update

    # Write JSON atomically (no locking, just crash-safe):
    atomic_json_write(path, {"key": "value"})

    # Read JSON under a shared lock:
    data = locked_json_read(path, default={})

    # Read-modify-write under an exclusive lock:
    def updater(data):
        data["counter"] += 1
        return data
    locked_json_update(path, updater, default={"counter": 0})
"""

import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("doris.security.file_io")


def atomic_json_write(path: Path, data: Any, *, mode: int = 0o600) -> None:
    """
    Write JSON data to a file atomically.

    Writes to a .tmp sibling, fsyncs, then renames over the target.
    On any OS-level failure the original file is left intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")

    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        payload = json.dumps(data, indent=2).encode()
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)

    os.rename(str(tmp_path), str(path))


def locked_json_read(path: Path, *, default: Any = None) -> Any:
    """
    Read a JSON file under a shared (read) lock.

    Returns `default` if the file doesn't exist or can't be parsed.
    """
    path = Path(path)
    if not path.exists():
        return default if default is not None else {}

    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Corrupt JSON in {path}: {e}")
                return default if default is not None else {}
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except OSError as e:
        logger.error(f"Failed to lock/read {path}: {e}")
        return default if default is not None else {}


def locked_json_update(
    path: Path,
    update_fn: Callable[[Any], Any],
    *,
    default: Any = None,
) -> Any:
    """
    Atomic read-modify-write on a JSON file with exclusive locking.

    1. Acquires an exclusive lock (fcntl.flock LOCK_EX)
    2. Reads existing JSON (or uses `default` if missing/corrupt)
    3. Calls update_fn(data) — must return the new data
    4. Writes atomically (tmp + fsync + rename)
    5. Releases lock

    Returns the updated data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = path.with_suffix(".lock")
    if default is None:
        default = {}

    try:
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                # Read current state
                if path.exists():
                    try:
                        current = json.loads(path.read_text())
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Corrupt JSON in {path}, using default: {e}")
                        current = default
                else:
                    current = default

                # Apply update
                updated = update_fn(current)

                # Write atomically
                atomic_json_write(path, updated)

                return updated
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except OSError as e:
        logger.error(f"Failed to lock/update {path}: {e}")
        raise
