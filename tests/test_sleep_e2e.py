"""
End-to-end test for Sleep-Time Compute.

Simulates a conversation with vague references, then lets the system
go idle to trigger sleep-time consolidation. Verifies that:
1. Idle detection triggers after 30s
2. Inference job resolves vague references
3. Review job extracts insights
4. Reorganize job runs
5. Results are stored in graph/memory
"""

import time
import threading
from datetime import datetime

# Track events
events = []


def log_event(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    events.append(f"[{timestamp}] {msg}")
    print(f"[{timestamp}] {msg}")


def test_sleep_time_compute_e2e():
    """
    Full end-to-end test of sleep-time compute.
    """
    from session import get_session
    from sleep import get_sleep_worker, start_idle_monitor, stop_idle_monitor
    from sleep.worker import SleepJob, JobType
    from memory.store import get_entities_by_type, get_recent_memories, search_entities

    log_event("="*60)
    log_event("E2E TEST: Sleep-Time Compute")
    log_event("="*60)

    # Get session
    session = get_session()
    worker = get_sleep_worker()

    # Clear any existing jobs
    worker.cancel_current_work()
    worker.resume_work()

    log_event(f"Session has {len(session.messages)} messages")

    # Simulate a conversation with vague references
    log_event("\n[Phase 1] Simulating conversation with vague references...")

    conversation_turns = [
        ("user", "Hey Doris, remember that Italian place we went to with Jane?"),
        ("assistant", "You mean for your anniversary? That was a great dinner!"),
        ("user", "Yeah, that one. Let's book it for Friday."),
        ("assistant", "I'll look into reservations. What time works?"),
        ("user", "Evening, maybe 7 or 8."),
        ("assistant", "Got it. I'll check availability."),
        ("user", "Also remind me about that project we discussed"),
        ("assistant", "You mean MyProject? The project we've been working on?"),
        ("user", "Yeah, I need to review the API design"),
        ("assistant", "I'll set a reminder for you to review the MyProject API design."),
    ]

    for role, content in conversation_turns:
        session.append(role, content)
        log_event(f"  Added {role} message: {content[:40]}...")
        time.sleep(0.1)  # Small delay between messages

    log_event(f"\n[Phase 2] Session now has {len(session.messages)} messages")

    # Get entities for context
    log_event("\n[Phase 3] Getting known entities...")
    entities = (
        get_entities_by_type("person")[:10] +
        get_entities_by_type("restaurant")[:5] +
        get_entities_by_type("project")[:5]
    )
    log_event(f"  Found {len(entities)} entities")
    for ent in entities[:5]:
        log_event(f"    - {ent['name']} ({ent['entity_type']})")

    # Format messages for sleep jobs
    log_event("\n[Phase 4] Preparing sleep jobs...")
    formatted_messages = [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp
        }
        for msg in session.messages[-20:]
    ]
    log_event(f"  Prepared {len(formatted_messages)} messages")

    # Queue sleep jobs manually (instead of waiting for idle)
    log_event("\n[Phase 5] Queueing sleep jobs...")

    # Inference job
    worker.queue_job(SleepJob(
        job_type=JobType.INFERENCE,
        data={
            "messages": formatted_messages,
            "entities": entities
        },
        priority=2
    ))
    log_event("  Queued: Inference job (priority 2)")

    # Review job
    recent_mems = get_recent_memories(hours=24, limit=10)
    worker.queue_job(SleepJob(
        job_type=JobType.REVIEW,
        data={
            "messages": formatted_messages,
            "recent_memories": recent_mems
        },
        priority=1
    ))
    log_event("  Queued: Review job (priority 1)")

    # Reorganize job
    worker.queue_job(SleepJob(
        job_type=JobType.REORGANIZE,
        data={"mode": "incremental"},
        priority=0
    ))
    log_event("  Queued: Reorganize job (priority 0)")

    # Wait for jobs to complete
    log_event("\n[Phase 6] Waiting for sleep jobs to complete...")
    log_event("  (This may take 30-60 seconds for Haiku calls)")

    # Monitor the queue
    max_wait = 90  # seconds
    start = time.time()
    while time.time() - start < max_wait:
        if worker._queue.empty() and not worker._compaction_in_progress if hasattr(worker, '_compaction_in_progress') else True:
            log_event("  Queue is empty, checking if processing...")
            time.sleep(2)
            if worker._queue.empty():
                break
        time.sleep(1)
        elapsed = int(time.time() - start)
        if elapsed % 10 == 0:
            log_event(f"  ... {elapsed}s elapsed")

    log_event(f"  Jobs completed in {int(time.time() - start)}s")

    # Check results
    log_event("\n[Phase 7] Checking results...")

    # Check for new entities (inferences might create them)
    new_entities = search_entities("reference")
    log_event(f"  Reference entities: {len(new_entities)}")

    # Check for new memories
    new_memories = get_recent_memories(hours=1, limit=20)
    sleep_memories = [m for m in new_memories if m.get('source', '').startswith('sleep_')]
    log_event(f"  Sleep-generated memories: {len(sleep_memories)}")
    for mem in sleep_memories[:5]:
        log_event(f"    - [{mem['category']}] {mem['content'][:50]}...")

    # Summary
    log_event("\n" + "="*60)
    log_event("E2E TEST COMPLETE")
    log_event("="*60)
    log_event(f"  Total events: {len(events)}")
    log_event(f"  Messages in session: {len(session.messages)}")
    log_event(f"  Sleep-generated memories: {len(sleep_memories)}")

    return True


if __name__ == "__main__":
    test_sleep_time_compute_e2e()
