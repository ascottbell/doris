"""
Wisdom System — thin wrapper over maasv.core.wisdom.

All wisdom operations now live in the maasv package.
This module re-exports everything for backward compatibility.
"""

from maasv.core.wisdom import (  # noqa: F401
    WisdomEntry,
    ensure_wisdom_tables,
    log_reasoning,
    record_outcome,
    add_feedback,
    delete_wisdom,
    update_wisdom,
    get_recent_wisdom,
    get_wisdom_by_id,
    get_relevant_wisdom,
    search_wisdom,
    format_wisdom_for_prompt,
    get_stats,
    get_action_family,
    get_family_actions,
    should_query_wisdom,
    get_smart_wisdom,
    format_smart_wisdom_for_prompt,
    get_escalation_patterns,
    should_escalate_based_on_wisdom,
    log_escalation_miss,
    log_escalation_correct,
    get_pending_feedback,
)

# Doris-specific: ACTION_FAMILIES moved to config-driven in maasv, kept here for backward compat
ACTION_FAMILIES = {
    "calendar": ["create_calendar_event", "move_calendar_event", "delete_calendar_event"],
    "reminders": ["create_reminder", "complete_reminder"],
    "messaging": ["send_imessage", "send_email"],
    "home": ["control_music"],
    "memory": ["store_memory"],
    "notifications": ["notify_user"],
    "creative": ["create_note"],
    "escalation": ["email_escalation_miss", "email_escalation_correct", "calendar_escalation_miss"],
    "development": [
        "architecture_decision",
        "debugging_resolution",
        "dependency_choice",
        "config_change",
        "gotcha",
        "user_preference",
        "approach_validated",
        "approach_rejected",
    ],
}
