"""
Graph Reorganization — thin wrapper over maasv.lifecycle.reorganize.

All reorganization logic now lives in the maasv package.
This module re-exports everything for backward compatibility.
"""

from maasv.lifecycle.reorganize import (  # noqa: F401
    run_reorganize_job,
    get_cached_path,
    _update_access_stats,
    _cache_common_paths,
    _store_cached_path,
    _cleanup_orphans,
    _strengthen_frequent_connections,
)
