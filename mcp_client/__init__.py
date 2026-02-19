"""
MCP (Model Context Protocol) Client Support for Doris.

This module enables Doris to connect to MCP servers and use their tools.
Supports both stdio (local subprocess) and HTTP (remote) transports.

Usage:
    from mcp_client import init_mcp, shutdown_mcp, get_mcp_manager

    # At startup
    await init_mcp()

    # Get available tools
    manager = get_mcp_manager()
    tools = await manager.list_tools()

    # Call a tool
    result = await manager.call_tool("apple-music", "play", {"query": "jazz"})

    # At shutdown
    await shutdown_mcp()
"""

from .manager import (
    MCPManager,
    AggregatedTool,
    ConnectedServer,
    get_mcp_manager,
    get_event_loop,
    init_mcp,
    shutdown_mcp,
)

from .config import (
    ServerConfig,
    StdioServerConfig,
    HttpServerConfig,
    load_server_configs,
    get_enabled_servers,
)

__all__ = [
    # Manager
    "MCPManager",
    "AggregatedTool",
    "ConnectedServer",
    "get_mcp_manager",
    "get_event_loop",
    "init_mcp",
    "shutdown_mcp",
    # Config
    "ServerConfig",
    "StdioServerConfig",
    "HttpServerConfig",
    "load_server_configs",
    "get_enabled_servers",
]
