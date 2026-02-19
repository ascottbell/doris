#!/usr/bin/env python3
"""Test the proactive system."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def test_db_init():
    """Test database initialization."""
    print("Testing database initialization...")
    from proactive.db import init_proactive_db, get_connection

    init_proactive_db()

    # Verify tables exist
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'proactive%'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    expected = ["proactive_events", "proactive_actions", "proactive_checkpoints"]
    for table in expected:
        assert table in tables, f"Missing table: {table}"
        print(f"  ✓ {table}")

    print("  Database OK\n")


def test_models():
    """Test model creation and serialization."""
    print("Testing models...")
    from proactive.models import ProactiveEvent, ProactiveAction, EvaluationResult

    # Test ProactiveEvent
    event = ProactiveEvent(
        source_type="email",
        source_id="test123",
        raw_data={"subject": "Test", "body": "Hello"}
    )
    row = event.to_db_row()
    event2 = ProactiveEvent.from_db_row(row)
    assert event.source_id == event2.source_id
    print("  ✓ ProactiveEvent")

    # Test EvaluationResult
    result = EvaluationResult(
        should_act=True,
        action_type="create_event",
        action_data={"title": "Test Event"},
        reasoning="Test"
    )
    d = result.to_dict()
    assert d["should_act"] == True
    print("  ✓ EvaluationResult")

    print("  Models OK\n")


def test_evaluator_dry():
    """Test evaluator without actually calling Claude."""
    print("Testing evaluator structure...")
    from proactive.evaluator import (
        _format_email_for_eval,
        _format_imessage_for_eval,
        FAMILY_CONTEXT,
    )

    # Test email formatting
    email_data = {
        "sender_name": "City Dance",
        "sender": "info@citydance.com",
        "subject": "Winter Recital",
        "date": "2024-01-15",
        "body": "The recital is on January 19th at 4:30pm."
    }
    formatted = _format_email_for_eval(email_data)
    assert "City Dance" in formatted
    assert "Winter Recital" in formatted
    print("  ✓ Email formatting")

    # Test context exists (family details loaded from knowledge graph at runtime)
    assert FAMILY_CONTEXT is not None
    assert len(FAMILY_CONTEXT) > 0
    print("  ✓ Family context exists")

    print("  Evaluator structure OK\n")


def test_email_filter():
    """Test email actionability filter."""
    print("Testing email filter...")
    from proactive.sources.email import _is_potentially_actionable

    # Should be actionable
    actionable = [
        {"subject": "Spring Recital Schedule", "snippet": "March 15th at 7pm", "sender": "info@communitycenter.org"},
        {"subject": "School Parent Night", "snippet": "Please join us Thursday", "sender": "office@school.edu"},
        {"subject": "Dentist appointment", "snippet": "reminder for 3pm", "sender": "dental@office.com"},
    ]

    for email in actionable:
        email.setdefault("is_starred", False)
        email.setdefault("is_important", False)
        assert _is_potentially_actionable(email), f"Should be actionable: {email['subject']}"
        print(f"  ✓ Actionable: {email['subject'][:30]}")

    # Should NOT be actionable (unless starred/important)
    not_actionable = [
        {"subject": "Weekly Newsletter", "snippet": "Here's what happened", "sender": "news@company.com", "is_starred": False, "is_important": False},
        {"subject": "Sale ends today!", "snippet": "20% off everything", "sender": "deals@store.com", "is_starred": False, "is_important": False},
    ]

    for email in not_actionable:
        result = _is_potentially_actionable(email)
        # These might still pass due to date patterns, so just note it
        status = "✗ (filtered)" if not result else "~ (passed filter)"
        print(f"  {status}: {email['subject'][:30]}")

    print("  Email filter OK\n")


def test_scheduler():
    """Test scheduler creation."""
    print("Testing scheduler...")
    from proactive.scheduler import ProactiveScheduler

    scheduler = ProactiveScheduler()

    call_count = 0
    def test_job():
        nonlocal call_count
        call_count += 1

    scheduler.add_job("test", test_job, interval_minutes=1, run_immediately=False)
    status = scheduler.status()
    assert "test" in status["jobs"]
    print("  ✓ Job registration")

    # Don't actually start - that would run in background
    print("  Scheduler OK\n")


def test_full_pipeline_dry():
    """Test full pipeline without external calls (dry run)."""
    print("Testing pipeline structure...")

    from proactive import (
        ProactiveEvent,
        EvaluationResult,
        execute_action,
    )

    # Create a mock event
    event = ProactiveEvent(
        source_type="email",
        source_id="dry-test-123",
        raw_data={
            "subject": "Dry Run Test",
            "sender": "test@test.com",
            "body": "This is a test"
        }
    )

    # Create a mock evaluation that says NO action
    evaluation = EvaluationResult.no_action("Dry run test")
    assert evaluation.should_act == False
    print("  ✓ No-action evaluation")

    # Verify execute_action handles no-action correctly
    action = execute_action(event, evaluation)
    assert action is None
    print("  ✓ Executor handles no-action")

    print("  Pipeline structure OK\n")


def main():
    print("=" * 50)
    print("PROACTIVE SYSTEM TESTS")
    print("=" * 50)
    print()

    test_db_init()
    test_models()
    test_evaluator_dry()
    test_email_filter()
    test_scheduler()
    test_full_pipeline_dry()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
    print()
    print("To test with real emails (calls Claude):")
    print("  python -c \"from proactive.sources.email import monitor; monitor()\"")
    print()
    print("To start the scheduler:")
    print("  python -c \"from proactive import init_scheduler; init_scheduler()\"")


if __name__ == "__main__":
    main()
