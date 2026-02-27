"""Tests for centralized path management in Settings."""

from pathlib import Path

# Project root is one level up from the tests/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_data_dir_default():
    """data_dir defaults to project_root/data."""
    from config import Settings

    s = Settings()
    assert s.data_dir == PROJECT_ROOT / "data"


def test_config_dir_default():
    """config_dir defaults to project_root (where config.py lives)."""
    from config import Settings

    s = Settings()
    assert s.config_dir == PROJECT_ROOT


def test_db_path_derives_from_data_dir():
    """db_path should be data_dir / 'doris.db' when not explicitly overridden."""
    from config import Settings

    s = Settings()
    assert s.db_path == s.data_dir / "doris.db"


def test_data_dir_env_override(monkeypatch, tmp_path):
    """DORIS_DATA_DIR env var overrides the default data_dir."""
    monkeypatch.setenv("DORIS_DATA_DIR", str(tmp_path))

    from config import Settings

    s = Settings()
    assert s.data_dir == tmp_path
    assert s.db_path == tmp_path / "doris.db"


def test_config_dir_env_override(monkeypatch, tmp_path):
    """DORIS_CONFIG_DIR env var overrides the default config_dir."""
    monkeypatch.setenv("DORIS_CONFIG_DIR", str(tmp_path))

    from config import Settings

    s = Settings()
    assert s.config_dir == tmp_path


def test_db_modules_use_settings_data_dir(monkeypatch, tmp_path):
    """All DB_PATH constants should derive from settings.data_dir."""
    monkeypatch.setenv("DORIS_DATA_DIR", str(tmp_path))

    import importlib
    import config
    importlib.reload(config)

    import api.conversations
    importlib.reload(api.conversations)
    assert api.conversations.DB_PATH == tmp_path / "doris.db"

    import proactive.db
    importlib.reload(proactive.db)
    assert proactive.db.DB_PATH == tmp_path / "doris.db"

    import services.agent_channel
    importlib.reload(services.agent_channel)
    assert services.agent_channel.DB_PATH == tmp_path / "agent_channel.db"

    import tools.documents
    importlib.reload(tools.documents)
    assert tools.documents.CACHE_DB_PATH == tmp_path / "document_cache.db"


def test_state_modules_use_settings_data_dir(monkeypatch, tmp_path):
    """State file constants should derive from settings.data_dir."""
    monkeypatch.setenv("DORIS_DATA_DIR", str(tmp_path))

    import importlib
    import sys
    import types
    import config
    importlib.reload(config)

    # Stub the daemon package to avoid its __init__.py import chain
    # (which pulls in scouts -> google, unavailable in test env).
    if "daemon" not in sys.modules or hasattr(sys.modules["daemon"], "__all__"):
        stub = types.ModuleType("daemon")
        stub.__path__ = [str(Path(__file__).parent.parent / "daemon")]
        monkeypatch.setitem(sys.modules, "daemon", stub)

    import daemon.scout_health
    importlib.reload(daemon.scout_health)
    assert daemon.scout_health.HEALTH_FILE == tmp_path / "scout_health.json"

    import llm.token_budget
    importlib.reload(llm.token_budget)
    assert llm.token_budget.STATE_FILE == tmp_path / "token_budget_state.json"
