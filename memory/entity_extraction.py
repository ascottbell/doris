"""
Entity Extraction — thin wrapper over maasv.extraction.entity_extraction.

All entity extraction logic now lives in the maasv package.
This module re-exports everything for backward compatibility.
"""

from maasv.extraction.entity_extraction import (  # noqa: F401
    EntityExtractor,
    get_entity_extractor,
    extract_and_store_entities,
    BLOCKED_ENTITY_NAMES,
    PREDICATE_OBJECT_TYPE,
    PREDICATE_SUBJECT_TYPE,
)
