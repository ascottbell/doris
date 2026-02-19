"""
Tests for the session continuity module.

Tests:
- Singleton behavior
- Token estimation
- Checkpoint/load cycle
- WORM content never in compaction
- Compaction preserves recent turns
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock the config before importing session module
with patch.dict('sys.modules', {'config': MagicMock()}):
    from session.persistent import (
        PersistentSession, Message, get_session,
        estimate_tokens, estimate_messages_tokens,
        MAX_CONTEXT_TOKENS, COMPACTION_THRESHOLD, VERBATIM_TURNS,
        CHECKPOINT_PATH
    )


class TestTokenEstimation:
    """Test token estimation utilities."""

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        # "hello" = 5 chars, ~1 token
        tokens = estimate_tokens("hello")
        assert tokens == 1  # 5 // 4 = 1

    def test_estimate_tokens_longer(self):
        # 100 chars should be ~25 tokens
        text = "a" * 100
        tokens = estimate_tokens(text)
        assert tokens == 25  # 100 // 4

    def test_estimate_messages_tokens(self):
        messages = [
            Message(role="user", content="hello"),  # 4 + 1 = 5
            Message(role="assistant", content="world"),  # 4 + 1 = 5
        ]
        tokens = estimate_messages_tokens(messages)
        assert tokens == 10  # 2 messages * 5 tokens each


class TestMessage:
    """Test Message dataclass."""

    def test_message_creation(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.is_summary is False
        assert msg.timestamp is not None

    def test_message_to_dict(self):
        msg = Message(role="user", content="hello", is_summary=True)
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"
        assert d["is_summary"] is True

    def test_message_from_dict(self):
        data = {"role": "assistant", "content": "hi there", "is_summary": True}
        msg = Message.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "hi there"
        assert msg.is_summary is True


class TestPersistentSession:
    """Test PersistentSession class."""

    @pytest.fixture
    def session(self, tmp_path):
        """Create a fresh session with temp checkpoint path."""
        with patch('session.persistent.CHECKPOINT_PATH', tmp_path / "checkpoint.json"):
            s = PersistentSession()
            yield s

    def test_append_message(self, session):
        session.append("user", "hello")
        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "hello"

    def test_token_count_increases(self, session):
        initial_tokens = session.token_count
        session.append("user", "hello world this is a test")
        assert session.token_count > initial_tokens

    def test_get_context_filters_system(self, session):
        session.append("system", "context note")
        session.append("user", "hello")
        session.append("assistant", "hi")

        context = session.get_context()
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"

    def test_get_context_with_system_includes_all(self, session):
        session.append("system", "context note")
        session.append("user", "hello")

        context = session.get_context_with_system()
        assert len(context) == 2
        assert context[0]["role"] == "system"

    def test_checkpoint_and_load(self, session, tmp_path):
        checkpoint_path = tmp_path / "checkpoint.json"

        with patch('session.persistent.CHECKPOINT_PATH', checkpoint_path):
            session.append("user", "test message")
            session.append("assistant", "test response")
            session.checkpoint()

            assert checkpoint_path.exists()

            # Create new session and verify it loads
            new_session = PersistentSession()
            assert len(new_session.messages) == 2
            assert new_session.messages[0].content == "test message"

    def test_replace_messages(self, session):
        session.append("user", "old message")
        session.append("assistant", "old response")

        new_messages = [
            Message(role="system", content="summary", is_summary=True),
            Message(role="user", content="new message"),
        ]
        session.replace_messages(new_messages)

        assert len(session.messages) == 2
        assert session.messages[0].is_summary is True

    def test_clear(self, session):
        session.append("user", "hello")
        session.clear()
        assert len(session.messages) == 0
        assert session.token_count == 0


class TestSingleton:
    """Test singleton behavior."""

    def test_get_session_returns_same_instance(self, tmp_path):
        # Reset the session registry
        import session.persistent as sp
        sp._sessions.clear()

        with patch('session.persistent.CHECKPOINT_PATH', tmp_path / "checkpoint.json"):
            s1 = get_session()
            s2 = get_session()
            assert s1 is s2

    def test_get_session_isolates_by_key(self, tmp_path):
        """Different session keys should return different instances."""
        import session.persistent as sp
        sp._sessions.clear()

        with patch('session.persistent.SESSIONS_DIR', tmp_path / "sessions"):
            s_default = get_session()
            s_tg = get_session("telegram:user1")
            s_dc = get_session("discord:user2")
            s_tg2 = get_session("telegram:user1")

            assert s_default is not s_tg
            assert s_tg is not s_dc
            assert s_tg is s_tg2  # same key returns same instance


class TestCompactionTrigger:
    """Test that compaction is triggered at threshold."""

    def test_compaction_not_triggered_below_threshold(self, tmp_path):
        with patch('session.persistent.CHECKPOINT_PATH', tmp_path / "checkpoint.json"):
            session = PersistentSession()
            mock_callback = MagicMock()
            session.set_compaction_callback(mock_callback)

            # Add a small message
            session.append("user", "hello")

            # Callback should not be called
            mock_callback.assert_not_called()
