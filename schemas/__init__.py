"""Document extraction schemas for Doris."""

from .document_schemas import (
    INSURANCE_SCHEMA,
    VEHICLE_REGISTRATION_SCHEMA,
    RECEIPT_SCHEMA,
    CONTRACT_SCHEMA,
    GENERIC_DOCUMENT_SCHEMA,
    ALL_SCHEMAS,
    get_schema_for_query,
    get_search_patterns_for_schema,
    build_extraction_prompt
)

__all__ = [
    "INSURANCE_SCHEMA",
    "VEHICLE_REGISTRATION_SCHEMA",
    "RECEIPT_SCHEMA",
    "CONTRACT_SCHEMA",
    "GENERIC_DOCUMENT_SCHEMA",
    "ALL_SCHEMAS",
    "get_schema_for_query",
    "get_search_patterns_for_schema",
    "build_extraction_prompt"
]
