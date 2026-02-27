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
