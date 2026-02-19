"""
Inference Generation — thin wrapper over maasv.lifecycle.inference.

All inference logic now lives in the maasv package.
This module re-exports everything for backward compatibility.
"""

from maasv.lifecycle.inference import (  # noqa: F401
    run_inference_job,
    _format_messages,
    _format_entities,
    _extract_inferences,
    _store_inferences,
)
