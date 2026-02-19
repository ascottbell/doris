"""
Memory Hygiene — thin wrapper over maasv.lifecycle.memory_hygiene.

All memory hygiene logic now lives in the maasv package.
This module re-exports everything for backward compatibility.
"""

from maasv.lifecycle.memory_hygiene import (  # noqa: F401
    HygieneStats,
    run_memory_hygiene_job,
    run_hygiene,
)
