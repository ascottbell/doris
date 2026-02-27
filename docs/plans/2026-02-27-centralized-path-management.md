# Centralized Path Management — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace ~25 hardcoded `data/` path computations with `settings.data_dir`, add `DORIS_DATA_DIR` and `DORIS_CONFIG_DIR` env vars, update MCP config resolution, and remove symlinks.

**Architecture:** Add `data_dir` and `config_dir` fields to the existing pydantic-settings `Settings` class. Every module that references the data directory imports `settings.data_dir` instead of computing its own `Path(__file__).parent.parent / "data"`. MCP server config loads from `config_dir / "mcp_client.yaml"` with fallback to the bundled `servers.yaml`.

**Tech Stack:** Python 3.12, pydantic-settings, pytest

---

### Task 1: Add `data_dir` and `config_dir` to Settings

**Files:**
- Modify: `config.py:11-98`
- Modify: `tests/conftest.py`
- Test: `tests/test_config_paths.py` (create)

**Step 1: Write the failing test**

Create `tests/test_config_paths.py`:

```python
"""Tests for centralized path management via Settings."""
import os
import tempfile
from pathlib import Path


def test_data_dir_default():
    """data_dir defaults to <project_root>/data."""
    from config import settings
    expected = Path(__file__).parent.parent / "data"
    assert settings.data_dir == expected


def test_config_dir_default():
    """config_dir defaults to <project_root>."""
    from config import settings
    expected = Path(__file__).parent.parent
    assert settings.config_dir == expected


def test_db_path_derives_from_data_dir():
    """db_path default should be data_dir / 'doris.db'."""
    from config import settings
    assert settings.db_path == settings.data_dir / "doris.db"


def test_data_dir_env_override(monkeypatch):
    """DORIS_DATA_DIR env var overrides data_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("DORIS_DATA_DIR", tmp)
        # Force re-creation of Settings to pick up new env
        from config import Settings
        s = Settings()
        assert s.data_dir == Path(tmp)
        assert s.db_path == Path(tmp) / "doris.db"


def test_config_dir_env_override(monkeypatch):
    """DORIS_CONFIG_DIR env var overrides config_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("DORIS_CONFIG_DIR", tmp)
        from config import Settings
        s = Settings()
        assert s.config_dir == Path(tmp)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_paths.py -v`
Expected: FAIL — `settings` has no attribute `data_dir`

**Step 3: Write minimal implementation**

In `config.py`, add `data_dir` and `config_dir` fields to `Settings`, and change `db_path` to derive from `data_dir` via `model_validator`:

```python
# Add these imports at top of config.py
from pydantic import model_validator

class Settings(BaseSettings):
    anthropic_api_key: str
    # ... existing fields ...

    # --- Centralized paths ---
    data_dir: Path = Path(__file__).parent / "data"
    config_dir: Path = Path(__file__).parent

    db_path: Path = Path("")  # Derived from data_dir below

    # ... rest of existing fields ...

    @model_validator(mode="after")
    def _derive_paths(self):
        """Derive db_path from data_dir when not explicitly set."""
        if self.db_path == Path(""):
            self.db_path = self.data_dir / "doris.db"
        return self
```

Also update `tests/conftest.py` — add env var defaults so tests that create fresh `Settings()` don't fail:

```python
os.environ.setdefault("DORIS_DATA_DIR", str(Path(__file__).parent.parent / "data"))
os.environ.setdefault("DORIS_CONFIG_DIR", str(Path(__file__).parent.parent))
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_paths.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add config.py tests/test_config_paths.py tests/conftest.py
git commit -m "Add data_dir and config_dir settings with env var overrides"
```

---

### Task 2: Migrate data path references — database files

Replace hardcoded data paths in modules that reference `doris.db`, `agent_channel.db`, or `document_cache.db`.

**Files:**
- Modify: `api/conversations.py:20`
- Modify: `proactive/db.py:10`
- Modify: `services/health_db.py:12`
- Modify: `services/location_db.py:14`
- Modify: `services/push.py:52`
- Modify: `services/behavioral_patterns.py:15`
- Modify: `services/agent_channel.py:70`
- Modify: `tools/documents.py:37`

**Step 1: Write the failing test**

Add to `tests/test_config_paths.py`:

```python
def test_db_modules_use_settings_data_dir(monkeypatch, tmp_path):
    """All DB_PATH constants should derive from settings.data_dir."""
    monkeypatch.setenv("DORIS_DATA_DIR", str(tmp_path))

    # Force reimport with new env
    import importlib
    import config
    importlib.reload(config)

    # These modules define DB_PATH at module level — reimport them
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_paths.py::test_db_modules_use_settings_data_dir -v`
Expected: FAIL — paths still hardcoded

**Step 3: Apply mechanical replacements**

In each file, replace the hardcoded path with `settings.data_dir`. Pattern:

```python
# BEFORE (example: api/conversations.py:20)
DB_PATH = Path(__file__).parent.parent / "data" / "doris.db"

# AFTER
from config import settings
DB_PATH = settings.data_dir / "doris.db"
```

Apply to all 8 files. Remove unused `Path` imports where applicable (only if Path is no longer used elsewhere in the file).

Specific replacements:
- `api/conversations.py:20` → `settings.data_dir / "doris.db"`
- `proactive/db.py:10` → `settings.data_dir / "doris.db"`
- `services/health_db.py:12` → `settings.data_dir / "doris.db"`
- `services/location_db.py:14` → `settings.data_dir / "doris.db"`
- `services/push.py:52` → `settings.data_dir / "doris.db"`
- `services/behavioral_patterns.py:15` → `settings.data_dir / "doris.db"`
- `services/agent_channel.py:70` → `settings.data_dir / "agent_channel.db"`
- `tools/documents.py:37` → `settings.data_dir / "document_cache.db"`

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_paths.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add api/conversations.py proactive/db.py services/health_db.py services/location_db.py services/push.py services/behavioral_patterns.py services/agent_channel.py tools/documents.py tests/test_config_paths.py
git commit -m "Migrate database path references to settings.data_dir"
```

---

### Task 3: Migrate data path references — state and JSON files

Replace hardcoded data paths in modules that reference JSON state files, logs, and directories.

**Files:**
- Modify: `daemon/scheduler.py:33`
- Modify: `daemon/session.py:23`
- Modify: `daemon/digest.py:23`
- Modify: `daemon/scout_health.py:18`
- Modify: `daemon/sitrep.py:35`
- Modify: `daemon.py:51`
- Modify: `services/daemon_monitor.py:21`
- Modify: `services/status.py:20-22,149`
- Modify: `services/push.py:147`
- Modify: `scouts/system_scout.py:22,147`
- Modify: `llm/brain.py:61,213,1577,1731-1732`
- Modify: `llm/token_budget.py:36`
- Modify: `memory/wisdom_compiler.py:22-24`
- Modify: `memory/batch_extract.py:72`
- Modify: `session/persistent.py:48-49,446`
- Modify: `tools/google_cal.py:23-24`
- Modify: `tools/push.py:33`
- Modify: `main.py:595`
- Modify: `maasv_bridge.py:92,138`
- Modify: `scripts/audit_sessions.py:109`

**Step 1: Write the failing test**

Add to `tests/test_config_paths.py`:

```python
def test_state_modules_use_settings_data_dir(monkeypatch, tmp_path):
    """State file constants should derive from settings.data_dir."""
    monkeypatch.setenv("DORIS_DATA_DIR", str(tmp_path))

    import importlib
    import config
    importlib.reload(config)

    import daemon.scheduler
    importlib.reload(daemon.scheduler)
    assert daemon.scheduler.STATE_FILE == tmp_path / "daemon_state.json"

    import daemon.digest
    importlib.reload(daemon.digest)
    assert daemon.digest.DIGEST_FILE == tmp_path / "awareness_digest.json"

    import daemon.scout_health
    importlib.reload(daemon.scout_health)
    assert daemon.scout_health.HEALTH_FILE == tmp_path / "scout_health.json"

    import daemon.sitrep
    importlib.reload(daemon.sitrep)
    assert daemon.sitrep.SITREP_DIR == tmp_path / "sitrep"

    import llm.token_budget
    importlib.reload(llm.token_budget)
    assert llm.token_budget.STATE_FILE == tmp_path / "token_budget_state.json"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_paths.py::test_state_modules_use_settings_data_dir -v`
Expected: FAIL — paths still hardcoded

**Step 3: Apply mechanical replacements**

Same pattern as Task 2. For each file, add `from config import settings` (if not already imported) and replace the path computation.

Specific replacements:

| File | Line(s) | Before | After |
|---|---|---|---|
| `daemon/scheduler.py` | 33 | `Path(__file__).parent.parent / "data" / "daemon_state.json"` | `settings.data_dir / "daemon_state.json"` |
| `daemon/session.py` | 23 | `Path(__file__).parent.parent / "data" / "session_state.json"` | `settings.data_dir / "session_state.json"` |
| `daemon/digest.py` | 23 | `Path(__file__).parent.parent / "data" / "awareness_digest.json"` | `settings.data_dir / "awareness_digest.json"` |
| `daemon/scout_health.py` | 18 | `Path(__file__).parent.parent / "data" / "scout_health.json"` | `settings.data_dir / "scout_health.json"` |
| `daemon/sitrep.py` | 35 | `Path(__file__).parent.parent / "data" / "sitrep"` | `settings.data_dir / "sitrep"` |
| `daemon.py` | 51 | `PROJECT_ROOT / "data" / "daemon_state.json"` | `settings.data_dir / "daemon_state.json"` |
| `services/daemon_monitor.py` | 21 | `Path(__file__).parent.parent / "data" / "daemon_state.json"` | `settings.data_dir / "daemon_state.json"` |
| `services/status.py` | 20-22 | `DATA_DIR = Path(__file__).parent.parent / "data"` | `DATA_DIR = settings.data_dir` |
| `services/push.py` | 147 | `Path(__file__).parent.parent / "data" / "device_tokens.json"` | `settings.data_dir / "device_tokens.json"` |
| `scouts/system_scout.py` | 22 | `DATA_DIR = PROJECT_ROOT / "data"` | `DATA_DIR = settings.data_dir` |
| `llm/brain.py` | 61 | `Path(__file__).parent.parent / "data" / "wisdom_summary.md"` | `settings.data_dir / "wisdom_summary.md"` |
| `llm/brain.py` | 213 | `str(Path(__file__).parent.parent / "data" / "token_usage.jsonl")` | `str(settings.data_dir / "token_usage.jsonl")` |
| `llm/brain.py` | 1577 | `Path(__file__).parent.parent / "data" / "undelivered_notifications.json"` | `settings.data_dir / "undelivered_notifications.json"` |
| `llm/brain.py` | 1731-1732 | Two `Path(__file__).parent.parent / "data" / ...` lines | `settings.data_dir / "daemon_state.json"` and `settings.data_dir / "awareness_digest.json"` |
| `llm/token_budget.py` | 36 | `Path(__file__).parent.parent / "data" / "token_budget_state.json"` | `settings.data_dir / "token_budget_state.json"` |
| `memory/wisdom_compiler.py` | 22-24 | `DATA_DIR = Path(__file__).parent.parent / "data"` | `DATA_DIR = settings.data_dir` |
| `memory/batch_extract.py` | 72 | `checkpoint_path: str = "data/extraction_checkpoint.json"` | `checkpoint_path: str = str(settings.data_dir / "extraction_checkpoint.json")` |
| `session/persistent.py` | 48-49,446 | Three `Path(__file__).parent.parent / "data" / ...` lines | `settings.data_dir / ...` |
| `tools/google_cal.py` | 23-24 | `DATA_DIR = Path(__file__).parent.parent / "data"` | `DATA_DIR = settings.data_dir` |
| `tools/push.py` | 33 | `Path(__file__).parent.parent / "data" / "device_tokens.json"` | `settings.data_dir / "device_tokens.json"` |
| `main.py` | 595 | `Path(__file__).parent / "data" / "device_tokens.json"` | `settings.data_dir / "device_tokens.json"` |
| `maasv_bridge.py` | 92 | `Path(__file__).parent / "data" / "backups"` | `settings.data_dir / "backups"` |
| `maasv_bridge.py` | 138 | `Path(__file__).parent / "data" / "memory_hygiene_log.json"` | `settings.data_dir / "memory_hygiene_log.json"` |
| `scripts/audit_sessions.py` | 109 | `Path(__file__).parent.parent / "data" / "session_audit.json"` | `settings.data_dir / "session_audit.json"` |

**Note on `llm/brain.py:1577`:** This is inside a function body, not module level. The import `from config import settings` may need to be added at the top of the file, or it may already be there — check before adding a duplicate.

**Note on `llm/brain.py:1731-1732`:** These are inside a function body (`get_scout_status` tool handler). Same import consideration.

**Step 4: Run tests to verify**

Run: `python -m pytest tests/test_config_paths.py -v`
Expected: All tests PASS

Run: `python -m pytest tests/ -v`
Expected: Full suite PASS (no regressions)

**Step 5: Commit**

```bash
git add daemon/ services/ scouts/ llm/ memory/ session/ tools/ main.py maasv_bridge.py scripts/ tests/test_config_paths.py
git commit -m "Migrate all state/JSON path references to settings.data_dir"
```

---

### Task 4: Update MCP client config resolution

**Files:**
- Modify: `mcp_client/config.py:113-129`
- Test: `tests/test_config_paths.py` (append)

**Step 1: Write the failing test**

Add to `tests/test_config_paths.py`:

```python
def test_mcp_config_uses_config_dir(monkeypatch, tmp_path):
    """MCP config should check config_dir/mcp_client.yaml first."""
    # Create external config
    external_yaml = tmp_path / "mcp_client.yaml"
    external_yaml.write_text("""
mcp_servers:
  test-server:
    type: stdio
    command: echo
    args: ["hello"]
""")
    monkeypatch.setenv("DORIS_CONFIG_DIR", str(tmp_path))

    import importlib
    import config
    importlib.reload(config)

    from mcp_client.config import load_server_configs
    servers = load_server_configs()
    assert "test-server" in servers


def test_mcp_config_falls_back_to_bundled(monkeypatch, tmp_path):
    """When no external config exists, fall back to bundled servers.yaml."""
    # Point to empty dir — no mcp_client.yaml
    monkeypatch.setenv("DORIS_CONFIG_DIR", str(tmp_path))

    import importlib
    import config
    importlib.reload(config)

    from mcp_client.config import load_server_configs
    # Should not crash, should load defaults or bundled yaml
    servers = load_server_configs()
    assert isinstance(servers, dict)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_paths.py::test_mcp_config_uses_config_dir -v`
Expected: FAIL — MCP config doesn't look at `config_dir`

**Step 3: Update `mcp_client/config.py`**

In `load_server_configs()` (around line 113-129), change the default config path resolution:

```python
def load_server_configs(config_path: Optional[Path] = None) -> dict[str, ServerConfig]:
    """
    Load MCP server configurations.

    Looks for configuration in this order:
    1. Provided config_path
    2. <config_dir>/mcp_client.yaml (external deployment config)
    3. <mcp_client_package>/servers.yaml (bundled default)
    4. Falls back to DEFAULT_SERVERS
    """
    from config import settings

    servers = {}
    raw_configs = DEFAULT_SERVERS.copy()

    # Determine config file path
    if config_path is None:
        external = settings.config_dir / "mcp_client.yaml"
        bundled = Path(__file__).parent / "servers.yaml"
        config_path = external if external.exists() else bundled

    if config_path.exists():
        _check_yaml_permissions(config_path)
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config and "mcp_servers" in yaml_config:
                    raw_configs.update(yaml_config["mcp_servers"])
        except Exception as e:
            print(f"[MCP] Warning: Could not load {config_path}: {e}")

    # ... rest of function unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_paths.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add mcp_client/config.py tests/test_config_paths.py
git commit -m "Update MCP config to resolve from DORIS_CONFIG_DIR"
```

---

### Task 5: Update Dockerfile and .gitignore

**Files:**
- Modify: `Dockerfile:35`
- Modify: `.gitignore:20-23`

**Step 1: No test needed** (infrastructure files, not testable with pytest)

**Step 2: Update Dockerfile**

After the `ENV PATH=...` line (line 35), add:

```dockerfile
ENV DORIS_DATA_DIR=/app/data
```

**Step 3: Update .gitignore**

Replace the current data line:
```
# Logs and data
logs/
*.log
data/*.db
```

With:
```
# Logs and data
logs/
*.log
data/
```

This ignores the entire `data/` directory (it's all runtime state, nothing should be committed).

**Step 4: Commit**

```bash
git add Dockerfile .gitignore
git commit -m "Set DORIS_DATA_DIR in Dockerfile, gitignore entire data/"
```

---

### Task 6: Add startup mkdir and remove symlinks

**Files:**
- Modify: `main.py` (near top of `lifespan` or startup)
- Remove: `data` symlink (local only, not in git)
- Remove: `mcp_client` symlink (local only, restore real files)

**Step 1: Find the startup location**

Look for the FastAPI `lifespan` context manager or startup event in `main.py`. Add `data_dir.mkdir()` as the first operation.

**Step 2: Add startup mkdir**

```python
# Near the top of the lifespan function or startup sequence:
settings.data_dir.mkdir(parents=True, exist_ok=True)
```

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add main.py
git commit -m "Ensure data_dir exists at startup"
```

**Step 5: Remove symlinks (local operation, not committed)**

```bash
# Remove mcp_client symlink, restore real files from git
rm mcp_client
git checkout -- mcp_client/

# Remove data symlink (data will be at /config/doris/data via env var)
rm data

# Verify
ls -la mcp_client/  # Should show real files
ls -la data          # Should not exist (env var points elsewhere)
```

---

### Task 7: Final verification

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: All tests PASS

**Step 2: Verify import works end-to-end**

```bash
DORIS_DATA_DIR=/tmp/test-doris-data python -c "
from config import settings
print(f'data_dir: {settings.data_dir}')
print(f'config_dir: {settings.config_dir}')
print(f'db_path: {settings.db_path}')
assert str(settings.data_dir) == '/tmp/test-doris-data'
assert settings.db_path == settings.data_dir / 'doris.db'
print('OK')
"
```

**Step 3: Verify no remaining hardcoded data paths**

```bash
grep -rn "parent.parent.*['\"]data['\"]" --include="*.py" | grep -v test | grep -v __pycache__ | grep -v plans/
grep -rn "parent.*['\"]data['\"].*/" --include="*.py" | grep -v test | grep -v __pycache__ | grep -v plans/
```
Expected: No matches (all converted)

**Step 4: Commit any remaining changes**

If grep found stragglers, fix them and commit.
