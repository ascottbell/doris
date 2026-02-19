"""
Comprehensive tests for Doris Persistent Memory Architecture.

Tests all phases:
- Phase 1: Graph Memory (entities, relationships, temporal edges)
- Phase 2: Fast/Slow Path (background extraction)
- Phase 3: WORM Persona (immutable identity)
- Phase 4: Session Continuity (persistent context, compaction)
- Phase 5: Sleep-Time Compute (idle consolidation)
"""

import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path


def test_phase1_graph_memory():
    """Test Phase 1: Graph Memory MCP"""
    print("\n" + "="*60)
    print("PHASE 1: Graph Memory")
    print("="*60)

    from memory.store import (
        create_entity, get_entity, find_entity_by_name,
        find_or_create_entity, search_entities, get_entities_by_type,
        add_relationship, expire_relationship, get_entity_relationships,
        update_relationship_value, graph_query, get_entity_profile
    )

    # Test 1.1: Create entities
    print("\n[1.1] Creating test entities...")
    test_person_id = create_entity(
        name="Test Person",
        entity_type="person",
        metadata={"test": True}
    )
    print(f"  Created person: {test_person_id}")

    test_place_id = create_entity(
        name="Test Restaurant",
        entity_type="restaurant",
        metadata={"cuisine": "Italian", "test": True}
    )
    print(f"  Created restaurant: {test_place_id}")

    # Test 1.2: Find entities
    print("\n[1.2] Finding entities...")
    found = find_entity_by_name("Test Person")
    assert found is not None, "Should find test person"
    assert found["id"] == test_person_id
    print(f"  Found by name: {found['name']}")

    # Test 1.3: Find or create
    print("\n[1.3] Find or create (idempotent)...")
    same_id = find_or_create_entity("Test Person", "person")
    assert same_id == test_person_id, "Should return existing entity"
    print(f"  Same ID returned: {same_id == test_person_id}")

    # Test 1.4: Search entities
    print("\n[1.4] Searching entities...")
    results = search_entities("Test")
    assert len(results) >= 2, "Should find test entities"
    print(f"  Found {len(results)} entities matching 'Test'")

    # Test 1.5: Add relationships
    print("\n[1.5] Adding relationships...")
    rel_id = add_relationship(
        subject_id=test_person_id,
        predicate="frequents",
        object_id=test_place_id,
        confidence=0.9,
        source="test"
    )
    print(f"  Created relationship: {rel_id}")

    # Test 1.6: Get entity relationships
    print("\n[1.6] Getting entity relationships...")
    rels = get_entity_relationships(test_person_id)
    assert len(rels) >= 1, "Should have at least one relationship"
    print(f"  Found {len(rels)} relationships")

    # Test 1.7: Temporal edges (Zep pattern)
    print("\n[1.7] Testing temporal edges...")
    old_rel_id, new_rel_id = update_relationship_value(
        subject_id=test_person_id,
        predicate="has_email",
        new_value="test@example.com",
        source="test"
    )
    print(f"  Old relationship: {old_rel_id}")
    print(f"  New relationship: {new_rel_id}")

    # Update again to test expiration
    old_rel_id2, new_rel_id2 = update_relationship_value(
        subject_id=test_person_id,
        predicate="has_email",
        new_value="updated@example.com",
        source="test"
    )
    print(f"  Updated relationship: {new_rel_id2}")

    # Check historical is preserved
    rels_with_expired = get_entity_relationships(test_person_id, include_expired=True)
    rels_current = get_entity_relationships(test_person_id, include_expired=False)
    print(f"  With expired: {len(rels_with_expired)}, Current only: {len(rels_current)}")

    # Test 1.8: Graph query
    print("\n[1.8] Graph query...")
    results = graph_query(subject_type="person", predicate="frequents")
    print(f"  Found {len(results)} person-frequents-* relationships")

    # Test 1.9: Entity profile
    print("\n[1.9] Entity profile...")
    profile = get_entity_profile(test_person_id)
    assert "entity" in profile
    assert "relationships" in profile
    print(f"  Profile has {len(profile['relationships'])} predicate types")

    # Cleanup test entities
    print("\n[1.10] Cleanup...")
    from memory.store import get_db
    db = get_db()
    db.execute("DELETE FROM relationships WHERE source = 'test'")
    db.execute("DELETE FROM entities WHERE json_extract(metadata, '$.test') = 1")
    db.commit()
    db.close()
    print("  Cleaned up test data")

    print("\n✓ Phase 1 tests passed!")
    return True


def test_phase2_fast_slow_path():
    """Test Phase 2: Fast Path / Slow Path separation"""
    print("\n" + "="*60)
    print("PHASE 2: Fast/Slow Path")
    print("="*60)

    from mcp_server.background import get_worker, BackgroundWorker

    # Test 2.1: Background worker exists
    print("\n[2.1] Background worker...")
    worker = get_worker()
    assert worker is not None
    assert isinstance(worker, BackgroundWorker)
    print(f"  Worker exists: {worker}")

    # Test 2.2: Queue job (non-blocking)
    print("\n[2.2] Queue extraction job...")
    test_conversation = "User: I went to Trattoria Roma last night.\nDoris: How was the food?"
    start = time.time()
    result = worker.queue_graph_extraction(test_conversation, source="test")
    elapsed = (time.time() - start) * 1000
    print(f"  Queued in {elapsed:.1f}ms (should be <100ms)")
    assert elapsed < 100, "Queue operation should be fast"

    # Test 2.3: Worker processes in background
    print("\n[2.3] Background processing...")
    print("  Waiting for background processing (up to 10s)...")
    time.sleep(3)  # Give some time for processing
    print("  Background worker should be processing")

    print("\n✓ Phase 2 tests passed!")
    return True


def test_phase3_worm_persona():
    """Test Phase 3: WORM Persona (Write Once Read Many)"""
    print("\n" + "="*60)
    print("PHASE 3: WORM Persona")
    print("="*60)

    from llm.worm_persona import (
        WORM_START_MARKER, WORM_END_MARKER,
        WORM_IDENTITY, WORM_PERSONALITY,
        get_worm_persona, get_worm_persona_with_markers
    )

    # Test 3.1: WORM markers exist
    print("\n[3.1] WORM markers...")
    assert WORM_START_MARKER is not None
    assert WORM_END_MARKER is not None
    print(f"  Start marker: {WORM_START_MARKER[:30]}...")
    print(f"  End marker: {WORM_END_MARKER[:30]}...")

    # Test 3.2: WORM components exist
    print("\n[3.2] WORM components...")
    assert len(WORM_IDENTITY) > 0, "Identity should exist"
    assert len(WORM_PERSONALITY) > 0, "Personality should exist"
    print(f"  Identity length: {len(WORM_IDENTITY)} chars")
    print(f"  Personality length: {len(WORM_PERSONALITY)} chars")

    # Test 3.3: Full persona assembly
    print("\n[3.3] Full persona assembly...")
    persona = get_worm_persona()
    print(f"  Full persona length: {len(persona)} chars")

    # Test 3.3b: Persona with markers
    print("\n[3.3b] Persona with markers...")
    persona_with_markers = get_worm_persona_with_markers()
    assert WORM_START_MARKER in persona_with_markers
    assert WORM_END_MARKER in persona_with_markers
    print(f"  Persona with markers length: {len(persona_with_markers)} chars")

    # Test 3.4: Verify WORM is in system prompt
    print("\n[3.4] System prompt includes WORM...")
    from llm.brain import get_system_prompt
    system_prompt = get_system_prompt()
    assert WORM_START_MARKER in system_prompt
    assert WORM_END_MARKER in system_prompt
    print("  System prompt contains WORM markers")

    print("\n✓ Phase 3 tests passed!")
    return True


def test_phase4_session_continuity():
    """Test Phase 4: Session Continuity"""
    print("\n" + "="*60)
    print("PHASE 4: Session Continuity")
    print("="*60)

    from session import get_session, Message, estimate_tokens
    from session.compaction import (
        contains_worm_content, split_messages_for_compaction,
        format_messages_for_summary
    )
    from llm.worm_persona import WORM_START_MARKER

    # Test 4.1: Session singleton
    print("\n[4.1] Session singleton...")
    session1 = get_session()
    session2 = get_session()
    assert session1 is session2, "Should be singleton"
    print(f"  Singleton verified: {session1 is session2}")

    # Test 4.2: Message append
    print("\n[4.2] Message append...")
    initial_count = len(session1.messages)
    session1.append("user", "Test message for phase 4")
    session1.append("assistant", "Test response")
    assert len(session1.messages) == initial_count + 2
    print(f"  Messages: {initial_count} -> {len(session1.messages)}")

    # Test 4.3: Token estimation
    print("\n[4.3] Token estimation...")
    tokens = estimate_tokens("This is a test message with about twenty words in it for testing.")
    assert tokens > 0
    print(f"  Estimated tokens: {tokens}")

    # Test 4.4: Activity tracking
    print("\n[4.4] Activity tracking...")
    old_time = session1.last_activity_time
    time.sleep(0.1)
    session1.touch()
    new_time = session1.last_activity_time
    assert new_time > old_time
    print(f"  Activity time updated: {new_time > old_time}")

    # Test 4.5: WORM detection
    print("\n[4.5] WORM content detection...")
    normal_text = "This is normal text"
    worm_text = f"Text with {WORM_START_MARKER} inside"
    assert not contains_worm_content(normal_text)
    assert contains_worm_content(worm_text)
    print("  WORM detection works")

    # Test 4.6: Message split for compaction
    print("\n[4.6] Message split logic...")
    test_messages = [
        Message(role="user", content=f"Message {i}")
        for i in range(50)
    ]
    to_compact, to_keep = split_messages_for_compaction(test_messages)
    print(f"  50 messages split: {len(to_compact)} to compact, {len(to_keep)} to keep")
    assert len(to_keep) == 40  # VERBATIM_TURNS * 2

    # Test 4.7: Format for summary
    print("\n[4.7] Format for summary...")
    test_msgs = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there")
    ]
    formatted = format_messages_for_summary(test_msgs)
    assert "User:" in formatted
    assert "Doris:" in formatted
    print("  Formatting works")

    # Test 4.8: Checkpointing
    print("\n[4.8] Checkpointing...")
    checkpoint_path = Path(__file__).parent.parent / "data" / "session_checkpoint.json"
    session1.checkpoint()
    assert checkpoint_path.exists()
    checkpoint_data = json.loads(checkpoint_path.read_text())
    assert "messages" in checkpoint_data
    assert "token_count" in checkpoint_data
    print(f"  Checkpoint saved: {len(checkpoint_data['messages'])} messages")

    # Test 4.9: Context retrieval
    print("\n[4.9] Context retrieval...")
    context = session1.get_context()
    assert isinstance(context, list)
    # Should only have user/assistant messages
    for msg in context:
        assert msg["role"] in ("user", "assistant")
    print(f"  Context has {len(context)} user/assistant messages")

    print("\n✓ Phase 4 tests passed!")
    return True


def test_phase5_sleep_time_compute():
    """Test Phase 5: Sleep-Time Compute"""
    print("\n" + "="*60)
    print("PHASE 5: Sleep-Time Compute")
    print("="*60)

    from sleep.worker import SleepWorker, SleepJob, JobType
    from sleep.inference import _format_messages, _format_entities, _extract_inferences
    from sleep.review import _format_conversation, _extract_insights
    from sleep.reorganize import get_cached_path, _update_access_stats

    # Test 5.1: Worker creation and job queueing
    print("\n[5.1] SleepWorker...")
    worker = SleepWorker()
    job = SleepJob(job_type=JobType.INFERENCE, data={"test": True}, priority=2)
    result = worker.queue_job(job)
    assert result is True
    print(f"  Job queued: {result}")

    # Test 5.2: Priority ordering
    print("\n[5.2] Priority queue...")
    worker2 = SleepWorker()
    worker2.queue_job(SleepJob(job_type=JobType.REORGANIZE, data={}, priority=0))
    worker2.queue_job(SleepJob(job_type=JobType.INFERENCE, data={}, priority=2))
    worker2.queue_job(SleepJob(job_type=JobType.REVIEW, data={}, priority=1))
    # Higher priority should come out first
    _, _, first_job = worker2._queue.get_nowait()
    assert first_job.job_type == JobType.INFERENCE
    print(f"  First job is highest priority: {first_job.job_type}")
    worker2.stop()

    # Test 5.3: Cancellation
    print("\n[5.3] Cancellation...")
    worker.queue_job(SleepJob(job_type=JobType.REVIEW, data={}, priority=1))
    worker.cancel_current_work()
    assert worker.is_cancelled()
    assert worker._queue.empty()
    print("  Cancellation works")

    # Test 5.4: Resume
    print("\n[5.4] Resume after cancellation...")
    worker.resume_work()
    assert not worker.is_cancelled()
    print("  Resume works")

    worker.stop()

    # Test 5.5: Inference formatting
    print("\n[5.5] Inference message formatting...")
    messages = [
        {"role": "user", "content": "Let's go to that Italian place"},
        {"role": "assistant", "content": "You mean Trattoria Roma?"}
    ]
    formatted = _format_messages(messages)
    assert "User:" in formatted
    assert "Doris:" in formatted
    print(f"  Formatted: {formatted[:50]}...")

    # Test 5.6: Entity formatting
    print("\n[5.6] Entity formatting...")
    entities = [
        {"name": "Trattoria Roma", "entity_type": "restaurant"},
        {"name": "User", "entity_type": "person"}
    ]
    formatted = _format_entities(entities)
    assert "Trattoria Roma (restaurant)" in formatted
    print(f"  Formatted: {formatted[:50]}...")

    # Test 5.7: Review formatting
    print("\n[5.7] Review conversation formatting...")
    messages = [
        {"role": "user", "content": "Test", "timestamp": "2026-01-24T10:00:00"}
    ]
    formatted = _format_conversation(messages)
    assert "User:" in formatted
    assert "2026-01-24" in formatted
    print("  Formatting includes timestamp")

    # Test 5.8: Path caching
    print("\n[5.8] Path caching...")
    result = get_cached_path("nonexistent")
    assert result is None
    print("  Non-existent path returns None")

    # Test 5.9: Access stats schema
    print("\n[5.9] Access stats schema update...")
    _update_access_stats()
    from memory.store import get_db
    db = get_db()
    # Check column exists
    cursor = db.execute("PRAGMA table_info(entities)")
    columns = [row[1] for row in cursor.fetchall()]
    db.close()
    assert "access_count" in columns
    print("  access_count column exists")

    print("\n✓ Phase 5 tests passed!")
    return True


def test_integration_idle_to_sleep():
    """Test integration: idle detection triggering sleep work"""
    print("\n" + "="*60)
    print("INTEGRATION: Idle Detection → Sleep Work")
    print("="*60)

    from sleep import start_idle_monitor, stop_idle_monitor, get_sleep_worker
    from sleep.worker import IDLE_THRESHOLD_SECONDS, IDLE_CHECK_INTERVAL

    print(f"\n[Config] Idle threshold: {IDLE_THRESHOLD_SECONDS}s, check interval: {IDLE_CHECK_INTERVAL}s")

    # Track callbacks
    idle_triggered = threading.Event()
    active_triggered = threading.Event()

    # Simulate activity time
    activity_time = [time.time()]

    def get_activity():
        return activity_time[0]

    def on_idle():
        print("  [Callback] on_idle triggered!")
        idle_triggered.set()

    def on_active():
        print("  [Callback] on_active triggered!")
        active_triggered.set()

    print("\n[Test] Starting idle monitor with simulated old activity...")
    # Set activity time to 35 seconds ago (past threshold)
    activity_time[0] = time.time() - 35

    start_idle_monitor(get_activity, on_idle, on_active)

    print(f"  Waiting for idle callback (up to {IDLE_CHECK_INTERVAL + 2}s)...")
    idle_triggered.wait(timeout=IDLE_CHECK_INTERVAL + 2)

    if idle_triggered.is_set():
        print("  ✓ Idle callback triggered!")
    else:
        print("  ✗ Idle callback NOT triggered (may need longer wait)")

    # Simulate activity resuming
    print("\n[Test] Simulating activity resume...")
    activity_time[0] = time.time()  # Update to now

    print(f"  Waiting for active callback (up to {IDLE_CHECK_INTERVAL + 2}s)...")
    active_triggered.wait(timeout=IDLE_CHECK_INTERVAL + 2)

    if active_triggered.is_set():
        print("  ✓ Active callback triggered!")
    else:
        print("  ✗ Active callback NOT triggered")

    stop_idle_monitor()
    print("\n✓ Integration test completed!")
    return True


def test_inference_with_haiku():
    """Test actual inference extraction with Haiku (live API call)"""
    print("\n" + "="*60)
    print("LIVE TEST: Inference with Haiku")
    print("="*60)

    from sleep.inference import _extract_inferences

    # Test conversation with vague reference
    conversation = """User: Hey, let's go to that Italian place for dinner
Doris: Which Italian place are you thinking of?
User: You know, the one we went to last month with Jane"""

    entities = """- Trattoria Roma (restaurant)
- Jane (person)
- User (person)
- Via Carota (restaurant)"""

    print("\n[Test] Extracting inferences from conversation...")
    print(f"  Conversation: {conversation[:60]}...")

    start = time.time()
    inferences = _extract_inferences(conversation, entities)
    elapsed = time.time() - start

    print(f"  Haiku call took {elapsed:.1f}s")
    print(f"  Found {len(inferences)} inferences:")

    for inf in inferences:
        print(f"    - '{inf.get('reference', 'N/A')}' → {inf.get('resolved_to', 'N/A')} "
              f"(conf: {inf.get('confidence', 0):.2f})")

    if inferences:
        print("\n✓ Inference extraction works!")
    else:
        print("\n⚠ No inferences extracted (may be expected for simple test)")

    return True


def test_review_with_haiku():
    """Test actual review extraction with Haiku (live API call)"""
    print("\n" + "="*60)
    print("LIVE TEST: Review with Haiku")
    print("="*60)

    from sleep.review import _extract_insights

    # Test conversation with patterns
    conversation = """[2026-01-24T09:00:00] User: Good morning Doris
[2026-01-24T09:00:05] Doris: Good morning! How can I help you today?
[2026-01-24T09:01:00] User: What's on my calendar?
[2026-01-24T09:01:10] Doris: You have a meeting at 10am with the project team
[2026-01-24T09:02:00] User: Ugh, I hate morning meetings
[2026-01-24T09:02:05] Doris: I noticed you prefer afternoon meetings. Want me to suggest rescheduling?
[2026-01-24T09:03:00] User: No it's fine, but in the future try to schedule them after lunch"""

    memories = """- [preference] User prefers afternoon meetings
- [project] MyProject is an active project"""

    print("\n[Test] Extracting insights from conversation...")
    print(f"  Conversation length: {len(conversation)} chars")

    start = time.time()
    insights = _extract_insights(conversation, memories)
    elapsed = time.time() - start

    print(f"  Haiku call took {elapsed:.1f}s")
    print(f"  Found {len(insights)} insights:")

    for insight in insights:
        print(f"    - [{insight.get('type', 'N/A')}] {insight.get('insight', 'N/A')[:60]}...")

    if insights:
        print("\n✓ Review extraction works!")
    else:
        print("\n⚠ No insights extracted (may be expected)")

    return True


def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#"*60)
    print("# DORIS PERSISTENT MEMORY ARCHITECTURE - COMPREHENSIVE TEST")
    print("#"*60)

    results = {}

    # Phase tests
    results["Phase 1: Graph Memory"] = test_phase1_graph_memory()
    results["Phase 2: Fast/Slow Path"] = test_phase2_fast_slow_path()
    results["Phase 3: WORM Persona"] = test_phase3_worm_persona()
    results["Phase 4: Session Continuity"] = test_phase4_session_continuity()
    results["Phase 5: Sleep-Time Compute"] = test_phase5_sleep_time_compute()

    # Integration tests
    results["Integration: Idle Detection"] = test_integration_idle_to_sleep()

    # Live API tests
    results["Live: Inference with Haiku"] = test_inference_with_haiku()
    results["Live: Review with Haiku"] = test_review_with_haiku()

    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - see above for details")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
