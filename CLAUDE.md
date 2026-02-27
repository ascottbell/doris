# Doris — CLAUDE.md

## Project
Python 3.12 FastAPI app. Config via pydantic-settings in `config.py`.
Data paths controlled by `DORIS_DATA_DIR` env var (default: `./data`).
Config paths controlled by `DORIS_CONFIG_DIR` env var (default: project root).

## Deployment
Production: /Users/_shared/dromos-inc/doris
Dev iteration: /Users/matthew/_projects/dromos-inc/doris
Shared data at: /config/doris/data (set DORIS_DATA_DIR in .env)
MCP config at: /config/doris/mcp_client.yaml (set DORIS_CONFIG_DIR=/config/doris)

## Testing
`python -m pytest tests/ -v` — full suite
Exclude missing-dep tests: `--ignore=tests/test_calendar_scout_timing.py --ignore=tests/test_setup.py --ignore=tests/test_sleep.py --ignore=tests/test_external_content_security.py`
~13 pre-existing failures from optional deps (google, ollama) — not regressions.

## Git
Commit style: imperative, no conventional-commits prefix (e.g., "Add platform detection...")
Do NOT put `Fixes: #N` in commit messages — only in PR body.

## GitHub
Repo: dromos-inc/doris
Uses GitHub App auth — see ~/.claude/CLAUDE.md for token patterns.
GraphQL sub-issues: use `-H 'GraphQL-Features:sub_issues'` header.
Shell quoting tip: avoid double-quote escaping in `gh api graphql` loops — use single-quote splicing.

## Environment Quirks
`.git/objects/` may have root-owned files from Docker — fix with `sudo chown -R $(whoami) .git/objects/`
`git config user.email` / `user.name` may not be set — check before first commit.

## Architecture
- `config.py` — pydantic-settings, singleton `settings` object
- `mcp_client/` — MCP server connections; `servers.yaml` is deployment config, `.py` files are app code
- `data/` — runtime state (DBs, JSON state files). Gitignored. Created at startup.
- `llm/brain.py` — main LLM interface, has function-body imports (not just module-level)
- `daemon.py` / `daemon/` — background scout scheduling
