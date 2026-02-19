"""
MCP Server Configuration for Doris.

Defines which MCP servers Doris can connect to and their connection parameters.
Servers can use stdio (local subprocess) or streamable HTTP (remote) transports.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import logging
import os
import re
import stat
import yaml
from dotenv import load_dotenv

logger = logging.getLogger("doris.mcp")

# Load .env so token_env values (like HA_TOKEN) are available in os.environ
load_dotenv()

VALID_TRUST_LEVELS = frozenset({"builtin", "trusted", "sandboxed"})


def _expand_env_vars(value: str) -> str:
    """Expand ${VAR} patterns in a string from environment variables."""
    pattern = re.compile(r'\$\{([^}]+)\}')
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    return pattern.sub(replacer, value)


@dataclass
class StdioServerConfig:
    """Configuration for an MCP server using stdio transport."""
    type: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    trust_level: str = "sandboxed"


@dataclass
class HttpServerConfig:
    """Configuration for an MCP server using streamable HTTP transport."""
    type: str = "http"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    token_env: Optional[str] = None  # Environment variable name for auth token
    enabled: bool = True
    trust_level: str = "sandboxed"


ServerConfig = StdioServerConfig | HttpServerConfig


# Default server configurations
DEFAULT_SERVERS: dict[str, dict] = {
    "apple-music": {
        "type": "stdio",
        "command": "uvx",
        "args": ["mcp-applemusic"],
        "enabled": False,  # Enable when mcp-applemusic is installed
    },
    # Add more servers here as needed
    # "home-assistant": {
    #     "type": "http",
    #     "url": "http://homeassistant.local:8123/api/mcp",
    #     "token_env": "HASS_TOKEN",
    #     "enabled": False,
    # },
}


def _check_yaml_permissions(config_path: Path) -> None:
    """Warn if servers.yaml has unsafe file permissions.

    servers.yaml defines subprocess commands — write access means arbitrary
    code execution on next Doris restart. We check:
    1. File is not world/group-writable (should be 0o600 or 0o400)
    2. File is owned by the current user
    """
    try:
        file_stat = config_path.stat()
    except OSError as e:
        logger.warning(f"[MCP] Cannot stat {config_path}: {e}")
        return

    mode = file_stat.st_mode
    # Check for group or other write/read permissions
    unsafe_bits = stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
    if mode & unsafe_bits:
        actual_perms = oct(stat.S_IMODE(mode))
        logger.warning(
            f"[MCP] SECURITY: {config_path} has permissions {actual_perms} — "
            f"expected 0o600. This file defines subprocess commands. "
            f"Fix with: chmod 600 {config_path}"
        )

    # Check ownership
    current_uid = os.getuid()
    if file_stat.st_uid != current_uid:
        logger.warning(
            f"[MCP] SECURITY: {config_path} is owned by UID {file_stat.st_uid}, "
            f"but Doris runs as UID {current_uid}. This file defines subprocess "
            f"commands — a different owner could inject malicious servers."
        )


def load_server_configs(config_path: Optional[Path] = None) -> dict[str, ServerConfig]:
    """
    Load MCP server configurations.

    Looks for configuration in this order:
    1. Provided config_path
    2. <project_root>/mcp/servers.yaml
    3. Falls back to DEFAULT_SERVERS

    Returns:
        Dict mapping server name to ServerConfig
    """
    servers = {}

    # Try to load from YAML file
    if config_path is None:
        config_path = Path(__file__).parent / "servers.yaml"

    raw_configs = DEFAULT_SERVERS.copy()

    if config_path.exists():
        _check_yaml_permissions(config_path)
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config and "mcp_servers" in yaml_config:
                    raw_configs.update(yaml_config["mcp_servers"])
        except Exception as e:
            print(f"[MCP] Warning: Could not load {config_path}: {e}")

    # Convert raw configs to dataclass instances
    for name, config in raw_configs.items():
        server_type = config.get("type", "stdio")

        # Parse and validate trust_level
        raw_trust = config.get("trust_level", "sandboxed")
        if raw_trust not in VALID_TRUST_LEVELS:
            logger.warning(
                f"[MCP] Invalid trust_level '{raw_trust}' for server '{name}', "
                f"falling back to 'sandboxed'"
            )
            raw_trust = "sandboxed"

        if server_type == "stdio":
            # Expand environment variables in env dict
            raw_env = config.get("env", {})
            expanded_env = {k: _expand_env_vars(v) for k, v in raw_env.items()}

            servers[name] = StdioServerConfig(
                type="stdio",
                command=config.get("command", ""),
                args=config.get("args", []),
                env=expanded_env,
                enabled=config.get("enabled", True),
                trust_level=raw_trust,
            )
        elif server_type in ("http", "sse"):
            # Resolve token from environment if specified
            headers = config.get("headers", {})
            token_env = config.get("token_env")
            if token_env:
                token = os.environ.get(token_env, "")
                if token:
                    headers["Authorization"] = f"Bearer {token}"

            servers[name] = HttpServerConfig(
                type="http",
                url=config.get("url", ""),
                headers=headers,
                token_env=token_env,
                enabled=config.get("enabled", True),
                trust_level=raw_trust,
            )

    return servers


def get_enabled_servers() -> dict[str, ServerConfig]:
    """Get only enabled server configurations."""
    all_servers = load_server_configs()
    return {name: config for name, config in all_servers.items() if config.enabled}
