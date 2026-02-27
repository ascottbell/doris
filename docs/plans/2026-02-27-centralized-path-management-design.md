# Centralized Path Management

## Problem

Doris has ~30 modules that independently compute paths to the `data/` directory using `Path(__file__).parent.parent / "data"`. This forces local deployments to use symlinks (`data/` and `mcp_client/`) to redirect storage to persistent locations. Two installations (production at `/Users/_shared` and dev at `/Users/matthew/_projects`) need to share the same data, which the symlink approach makes fragile.

## Design

### New Settings

Two env vars added to `config.py`:

| Setting | Env var | Default | Docker |
|---|---|---|---|
| `data_dir` | `DORIS_DATA_DIR` | `<project_root>/data` | `ENV DORIS_DATA_DIR=/app/data` |
| `config_dir` | `DORIS_CONFIG_DIR` | `<project_root>` | `/app` (default works) |

### Data Directory

All modules replace hardcoded `Path(__file__).parent.parent / "data" / ...` with `settings.data_dir / ...`. The change is mechanical:

```python
# Before
DB_PATH = Path(__file__).parent.parent / "data" / "doris.db"

# After
from config import settings
DB_PATH = settings.data_dir / "doris.db"
```

### MCP Client Config

`mcp_client/config.py` resolution order for server definitions:
1. `settings.config_dir / "mcp_client.yaml"` (external deployment config)
2. `Path(__file__).parent / "servers.yaml"` (bundled in-repo default)

The `.py` files in `mcp_client/` remain in the repo as real files.

### Symlink Removal

- Remove `data/` symlink — env var replaces it
- Remove `mcp_client/` symlink — was unnecessary, real files are in git
- Add `data/` to `.gitignore`

### Dockerfile

Add `ENV DORIS_DATA_DIR=/app/data` so the existing volume mount (`doris_data:/app/data`) works without changes.

### Startup Safety

`settings.data_dir.mkdir(parents=True, exist_ok=True)` at app startup ensures the directory exists regardless of deployment method.

### Special Cases

- `config.py` `db_path` field: default derived from `data_dir` via validator
- `memory/batch_extract.py`: uses bare string `"data/..."` — update to use `settings.data_dir`

## Affected Files (~30)

config.py, main.py, maasv_bridge.py, api/conversations.py, daemon/sitrep.py, daemon/session.py, daemon/scheduler.py, daemon/digest.py, daemon/scout_health.py, llm/brain.py, llm/token_budget.py, memory/batch_extract.py, memory/wisdom_compiler.py, mcp_client/config.py, proactive/db.py, scripts/audit_sessions.py, security/file_io.py, services/agent_channel.py, services/behavioral_patterns.py, services/daemon_monitor.py, services/health_db.py, services/location_db.py, services/push.py, services/status.py, session/persistent.py, tools/documents.py, tools/google_cal.py, tools/push.py, Dockerfile
