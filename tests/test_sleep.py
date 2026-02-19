"""
Tests for sleep-time compute module.
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock

from sleep.worker import SleepWorker, SleepJob, JobType, IDLE_THRESHOLD_SECONDS


class TestSleepWorker:
    """Tests for SleepWorker class."""

    def test_worker_creation(self):
        """Worker can be created."""
        worker = SleepWorker()
        assert worker is not None
        assert not worker._started

    def test_queue_job(self):
        """Jobs can be queued."""
        worker = SleepWorker()
        job = SleepJob(job_type=JobType.INFERENCE, data={"test": True})

        result = worker.queue_job(job)
        assert result is True
        assert worker._started

        worker.stop()

    def test_cancel_work(self):
        """Work can be cancelled."""
        worker = SleepWorker()
        worker._ensure_started()

        # Queue some jobs
        for i in range(3):
            worker.queue_job(SleepJob(job_type=JobType.REORGANIZE, data={}))

        # Cancel
        worker.cancel_current_work()
        assert worker.is_cancelled()
        assert worker._queue.empty()

        worker.stop()

    def test_resume_work(self):
        """Work can be resumed after cancellation."""
        worker = SleepWorker()
        worker._ensure_started()

        worker.cancel_current_work()
        assert worker.is_cancelled()

        worker.resume_work()
        assert not worker.is_cancelled()

        worker.stop()


class TestSleepJobTypes:
    """Tests for different job types."""

    def test_inference_job_creation(self):
        """Inference job can be created with proper data."""
        job = SleepJob(
            job_type=JobType.INFERENCE,
            data={
                "messages": [{"role": "user", "content": "test"}],
                "entities": [{"name": "Test", "entity_type": "person"}]
            },
            priority=2
        )
        assert job.job_type == JobType.INFERENCE
        assert job.priority == 2

    def test_review_job_creation(self):
        """Review job can be created."""
        job = SleepJob(
            job_type=JobType.REVIEW,
            data={
                "messages": [],
                "recent_memories": []
            },
            priority=1
        )
        assert job.job_type == JobType.REVIEW

    def test_reorganize_job_creation(self):
        """Reorganize job can be created."""
        job = SleepJob(
            job_type=JobType.REORGANIZE,
            data={"mode": "incremental"},
            priority=0
        )
        assert job.job_type == JobType.REORGANIZE


class TestIdleDetection:
    """Tests for idle detection."""

    def test_idle_threshold_configured(self):
        """Idle threshold is configured."""
        assert IDLE_THRESHOLD_SECONDS == 30

    def test_idle_callback_structure(self):
        """Idle callbacks can be registered."""
        from sleep import start_idle_monitor, stop_idle_monitor

        on_idle_called = threading.Event()
        on_active_called = threading.Event()
        activity_time = time.time()

        def get_activity():
            return activity_time

        def on_idle():
            on_idle_called.set()

        def on_active():
            on_active_called.set()

        # Start monitor
        start_idle_monitor(get_activity, on_idle, on_active)

        # Clean up
        stop_idle_monitor()


class TestInferenceModule:
    """Tests for inference job logic."""

    def test_format_messages(self):
        """Messages are formatted correctly."""
        from sleep.inference import _format_messages

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]

        result = _format_messages(messages)
        assert "User: Hello" in result
        assert "Doris: Hi there" in result

    def test_format_entities(self):
        """Entities are formatted correctly."""
        from sleep.inference import _format_entities

        entities = [
            {"name": "User", "entity_type": "person"},
            {"name": "Trattoria Roma", "entity_type": "restaurant"}
        ]

        result = _format_entities(entities)
        assert "User (person)" in result
        assert "Trattoria Roma (restaurant)" in result


class TestReviewModule:
    """Tests for review job logic."""

    def test_format_conversation(self):
        """Conversation is formatted correctly."""
        from sleep.review import _format_conversation

        messages = [
            {"role": "user", "content": "Test", "timestamp": "2026-01-24T10:00:00"},
            {"role": "assistant", "content": "Response", "timestamp": "2026-01-24T10:00:01"}
        ]

        result = _format_conversation(messages)
        assert "User: Test" in result
        assert "Doris: Response" in result


class TestReorganizeModule:
    """Tests for reorganize job logic."""

    def test_cache_path_functions(self):
        """Cache path functions exist."""
        from sleep.reorganize import get_cached_path, _store_cached_path

        # Should return None for non-existent path
        result = get_cached_path("nonexistent_path")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
