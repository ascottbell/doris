"""
MCP Manager for Doris.

Manages connections to multiple MCP servers, aggregates tools from all servers,
and routes tool calls to the correct server.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger("doris.mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool, TextContent, CallToolResult

from .config import (
    ServerConfig,
    StdioServerConfig,
    HttpServerConfig,
    get_enabled_servers,
)


@dataclass
class ConnectedServer:
    """Represents an active connection to an MCP server."""
    name: str
    config: ServerConfig
    session: ClientSession
    tools: list[Tool] = field(default_factory=list)
    trust_level: str = "sandboxed"
    # Context managers for cleanup
    _transport_cm: Any = None
    _session_cm: Any = None


@dataclass
class AggregatedTool:
    """A tool with its source server information."""
    server: str
    tool: Tool

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return self.tool.description or ""

    @property
    def qualified_name(self) -> str:
        """Server-qualified tool name for disambiguation."""
        return f"{self.server}:{self.tool.name}"


class MCPManager:
    """
    Manages multiple MCP server connections.

    Usage:
        manager = MCPManager()
        await manager.connect_all()

        tools = await manager.list_tools()
        result = await manager.call_tool("apple-music", "play", {"query": "jazz"})

        await manager.disconnect_all()
    """

    def __init__(self):
        self._servers: dict[str, ConnectedServer] = {}
        self._lock = asyncio.Lock()
        self._keepalive_task: Optional[asyncio.Task] = None
        self._keepalive_interval: float = 300.0  # 5 minutes

    @property
    def connected_servers(self) -> list[str]:
        """List of currently connected server names."""
        return list(self._servers.keys())

    async def connect_all(self, verbose: bool = True) -> dict[str, bool]:
        """
        Connect to all enabled MCP servers.

        Args:
            verbose: Print connection status messages

        Returns:
            Dict mapping server name to connection success status
        """
        if verbose:
            print("[MCP] Connecting to servers...")

        enabled = get_enabled_servers()
        results = {}

        for name, config in enabled.items():
            try:
                await self._connect_server(name, config)
                results[name] = True
                server = self._servers[name]
                tool_count = len(server.tools)
                trust = server.trust_level
                if isinstance(config, HttpServerConfig):
                    logger.warning(
                        f"[MCP] Connected to '{name}': trust={trust} "
                        f"tools={tool_count} (HTTP — network-exposed)"
                    )
                else:
                    logger.info(
                        f"[MCP] Connected to '{name}': trust={trust} "
                        f"tools={tool_count}"
                    )
                if verbose:
                    print(f"  [ok] {name}: {tool_count} tools (trust={trust})")
            except Exception as e:
                results[name] = False
                logger.error(f"[MCP] Failed to connect to '{name}': {e}")
                if verbose:
                    print(f"  [fail] {name}: {e}")

        if verbose:
            connected = sum(1 for v in results.values() if v)
            print(f"[MCP] Connected to {connected}/{len(enabled)} servers")

        return results

    async def _connect_server(self, name: str, config: ServerConfig) -> None:
        """Connect to a single MCP server."""
        async with self._lock:
            if name in self._servers:
                return  # Already connected

        if isinstance(config, StdioServerConfig):
            await self._connect_stdio(name, config)
        elif isinstance(config, HttpServerConfig):
            await self._connect_http(name, config)
        else:
            raise ValueError(f"Unknown server type for {name}")

    async def _connect_stdio(self, name: str, config: StdioServerConfig) -> None:
        """Connect to a stdio-based MCP server."""
        # Always pass the full parent environment. The MCP SDK's default only
        # inherits HOME/PATH/SHELL/USER — our subprocesses need OLLAMA_HOST,
        # DORIS_DATA_DIR, ANTHROPIC_API_KEY, etc.
        merged_env = {**os.environ, **(config.env or {})}

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=merged_env,
        )

        # Start the transport — clean up on failure to avoid leaking cancel scopes
        transport_cm = stdio_client(server_params)
        try:
            read, write = await transport_cm.__aenter__()
        except BaseException:
            # Ensure partial context is cleaned up
            try:
                await transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
            raise

        # Create and initialize session
        session_cm = ClientSession(read, write)
        try:
            session = await session_cm.__aenter__()
            await session.initialize()
        except BaseException:
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
            raise

        # Get available tools
        tools_result = await session.list_tools()
        tools = tools_result.tools if tools_result else []

        async with self._lock:
            self._servers[name] = ConnectedServer(
                name=name,
                config=config,
                session=session,
                tools=tools,
                trust_level=config.trust_level,
                _transport_cm=transport_cm,
                _session_cm=session_cm,
            )

    async def _connect_http(self, name: str, config: HttpServerConfig) -> None:
        """Connect to an HTTP-based MCP server."""
        # Start the transport — clean up on failure to avoid leaking cancel scopes
        transport_cm = streamablehttp_client(config.url, headers=config.headers)
        try:
            read, write, _ = await transport_cm.__aenter__()
        except BaseException:
            try:
                await transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
            raise

        # Create and initialize session
        session_cm = ClientSession(read, write)
        try:
            session = await session_cm.__aenter__()
            await session.initialize()
        except BaseException:
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
            raise

        # Get available tools
        tools_result = await session.list_tools()
        tools = tools_result.tools if tools_result else []

        async with self._lock:
            self._servers[name] = ConnectedServer(
                name=name,
                config=config,
                session=session,
                tools=tools,
                trust_level=config.trust_level,
                _transport_cm=transport_cm,
                _session_cm=session_cm,
            )

    async def disconnect_all(self, verbose: bool = True) -> None:
        """Gracefully disconnect from all servers."""
        if verbose:
            print("[MCP] Disconnecting from servers...")

        async with self._lock:
            for name, server in list(self._servers.items()):
                try:
                    # Close session first
                    if server._session_cm:
                        await server._session_cm.__aexit__(None, None, None)
                    # Then close transport
                    if server._transport_cm:
                        await server._transport_cm.__aexit__(None, None, None)
                    if verbose:
                        print(f"  [ok] Disconnected from {name}")
                except Exception as e:
                    if verbose:
                        print(f"  [warn] Error disconnecting from {name}: {e}")

            self._servers.clear()

        if verbose:
            print("[MCP] All servers disconnected")

    async def disconnect_server(self, name: str) -> bool:
        """Disconnect from a specific server."""
        async with self._lock:
            if name not in self._servers:
                return False

            server = self._servers[name]
            try:
                if server._session_cm:
                    await server._session_cm.__aexit__(None, None, None)
                if server._transport_cm:
                    await server._transport_cm.__aexit__(None, None, None)
            except Exception:
                pass

            del self._servers[name]
            return True

    async def list_tools(self) -> list[AggregatedTool]:
        """
        Get aggregated list of tools from all connected servers.

        Returns:
            List of AggregatedTool with server information
        """
        tools = []
        async with self._lock:
            for name, server in self._servers.items():
                for tool in server.tools:
                    tools.append(AggregatedTool(server=name, tool=tool))
        return tools

    async def list_tools_by_server(self) -> dict[str, list[Tool]]:
        """Get tools grouped by server."""
        result = {}
        async with self._lock:
            for name, server in self._servers.items():
                result[name] = list(server.tools)
        return result

    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        """
        Call a specific tool on a specific server.

        If the call fails with a connection error, attempts one reconnect
        and retry before propagating the exception.

        Args:
            server: Server name
            tool: Tool name
            arguments: Tool arguments

        Returns:
            CallToolResult from the server

        Raises:
            KeyError: If server not connected (even after reconnect attempt)
            Exception: If tool call fails after retry
        """
        async with self._lock:
            if server not in self._servers:
                raise KeyError(f"Server '{server}' not connected")
            session = self._servers[server].session
            trust_level = self._servers[server].trust_level

        # PII exfiltration gate — scan outbound args
        outbound_args = arguments or {}
        try:
            from security.pii_scanner import scan_args_for_pii
            pii_result = scan_args_for_pii(outbound_args, server, tool, trust_level)
            if pii_result.modified_args is not None:
                outbound_args = pii_result.modified_args
        except Exception as e:
            logger.error(f"PII scan failed for {server}:{tool}: {e}")
            if trust_level == "sandboxed":
                from security.audit import audit
                audit.tool_action(
                    tool=f"{server}:{tool}",
                    detail=f"PII scan exception, refusing call to sandboxed server: {e}",
                    blocked=True,
                )
                raise RuntimeError(
                    f"PII scan failed for sandboxed server {server}:{tool} — "
                    f"refusing to send unscanned arguments: {e}"
                ) from e
            # Trusted servers: fail open (PII scanning is defense-in-depth, not primary gate)

        try:
            result = await session.call_tool(tool, outbound_args)
            return result
        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError,
                asyncio.TimeoutError, EOFError) as e:
            logger.warning(f"Tool call {server}:{tool} failed ({type(e).__name__}), attempting reconnect...")
            if await self.reconnect_server(server):
                logger.info(f"Reconnected to {server}, retrying {tool}")
                async with self._lock:
                    session = self._servers[server].session
                return await session.call_tool(tool, outbound_args)
            else:
                logger.error(f"Reconnect to {server} failed, cannot retry {tool}")
                raise
        except Exception as e:
            # For unexpected errors, check if it looks like a dead connection
            err_str = str(e).lower()
            if any(term in err_str for term in ["closed", "broken", "eof", "reset", "pipe"]):
                logger.warning(f"Tool call {server}:{tool} hit connection error ({e}), attempting reconnect...")
                if await self.reconnect_server(server):
                    logger.info(f"Reconnected to {server}, retrying {tool}")
                    async with self._lock:
                        session = self._servers[server].session
                    return await session.call_tool(tool, outbound_args)
            raise

    async def call_tool_by_qualified_name(
        self,
        qualified_name: str,
        arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        """
        Call a tool using its qualified name (server:tool).

        Args:
            qualified_name: Tool name in "server:tool" format
            arguments: Tool arguments

        Returns:
            CallToolResult from the server
        """
        if ":" not in qualified_name:
            raise ValueError(f"Invalid qualified name: {qualified_name}. Expected 'server:tool'")

        server, tool = qualified_name.split(":", 1)
        return await self.call_tool(server, tool, arguments)

    async def health_check(self, timeout_seconds: float = 5.0) -> dict[str, bool]:
        """
        Ping each connected server to verify liveness.

        Args:
            timeout_seconds: How long to wait for each ping before declaring dead.

        Returns:
            Dict mapping server name to alive status.
        """
        results = {}
        async with self._lock:
            servers = list(self._servers.items())

        for name, server in servers:
            try:
                await asyncio.wait_for(
                    server.session.send_ping(),
                    timeout=timeout_seconds,
                )
                results[name] = True
            except Exception:
                results[name] = False

        return results

    async def reconnect_server(self, name: str) -> bool:
        """
        Disconnect a dead server and reconnect from config.

        Args:
            name: Server name to reconnect.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        # Tear down the old connection
        await self.disconnect_server(name)

        # Look up config from enabled servers
        from .config import get_enabled_servers
        enabled = get_enabled_servers()
        config = enabled.get(name)
        if config is None:
            return False

        try:
            await self._connect_server(name, config)
            return True
        except Exception as e:
            logger.warning(f"Reconnect failed for {name}: {type(e).__name__}: {e}")
            return False

    async def health_check_and_reconnect(
        self, timeout_seconds: float = 5.0
    ) -> dict[str, str]:
        """
        Ping all servers, reconnect failures, and catch missing servers.

        Returns:
            Dict mapping server name to status:
            "healthy", "reconnected", "dead", or "missing".
        """
        from .config import get_enabled_servers
        enabled = get_enabled_servers()
        statuses: dict[str, str] = {}

        # Check connected servers
        ping_results = await self.health_check(timeout_seconds)
        for name, alive in ping_results.items():
            if alive:
                statuses[name] = "healthy"
            else:
                # Try reconnecting
                if await self.reconnect_server(name):
                    statuses[name] = "reconnected"
                else:
                    statuses[name] = "dead"

        # Catch enabled servers that aren't connected at all
        for name in enabled:
            if name not in statuses:
                if await self.reconnect_server(name):
                    statuses[name] = "reconnected"
                else:
                    statuses[name] = "missing"

        return statuses

    async def start_keepalive(self) -> None:
        """
        Start periodic keep-alive pings to all connected servers.

        Runs every 5 minutes. If a ping fails, attempts reconnection immediately.
        This prevents servers from silently dying overnight without detection.
        """
        if self._keepalive_task and not self._keepalive_task.done():
            return  # Already running
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        logger.info(f"MCP keep-alive started (interval={self._keepalive_interval}s)")

    async def stop_keepalive(self) -> None:
        """Stop the keep-alive loop."""
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            logger.info("MCP keep-alive stopped")

    async def _keepalive_loop(self) -> None:
        """Periodic ping loop — reconnects dead servers automatically."""
        while True:
            try:
                await asyncio.sleep(self._keepalive_interval)
                if not self._servers:
                    continue

                async with self._lock:
                    servers = list(self._servers.items())

                for name, server in servers:
                    try:
                        await asyncio.wait_for(
                            server.session.send_ping(),
                            timeout=5.0,
                        )
                    except Exception as e:
                        logger.warning(f"Keep-alive ping failed for {name}: {type(e).__name__}: {e}")
                        if await self.reconnect_server(name):
                            logger.info(f"Keep-alive reconnected {name}")
                        else:
                            logger.error(f"Keep-alive failed to reconnect {name}")

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Keep-alive loop error: {e}")
                await asyncio.sleep(60)  # Back off on unexpected errors

    @staticmethod
    def _safe_tool_description(description: str, tool_name: str, server_name: str) -> str:
        """Sanitize a tool description by scanning for injection patterns."""
        if not description:
            return "No description"
        try:
            from security.injection_scanner import scan_for_injection
            result = scan_for_injection(
                description, source=f"tool-desc:{server_name}:{tool_name}"
            )
            if result.is_suspicious:
                logger.warning(
                    f"Tool description for {server_name}:{tool_name} failed "
                    f"security scan (risk={result.risk_level})"
                )
                return f"Tool '{tool_name}' (description withheld — failed security scan)"
            return description
        except Exception as e:
            logger.error(
                f"Error scanning tool description for {server_name}:{tool_name}: {e}"
            )
            return f"Tool '{tool_name}' (description withheld — failed security scan)"

    def get_tools_for_prompt(self) -> str:
        """
        Format available tools for inclusion in LLM system prompt.

        Returns:
            Formatted string describing all available tools
        """
        # This is sync because it's used in prompt building
        # We cache tools during connection so this is safe
        if not self._servers:
            return ""

        lines = ["Available MCP Tools:"]

        for name, server in self._servers.items():
            if not server.tools:
                continue

            lines.append(f"\n[{name}]")
            for tool in server.tools:
                desc = self._safe_tool_description(
                    tool.description or "", tool.name, name
                )
                lines.append(f"  - {tool.name}: {desc}")

                # Add parameter info if available
                if tool.inputSchema and "properties" in tool.inputSchema:
                    props = tool.inputSchema["properties"]
                    required = tool.inputSchema.get("required", [])
                    if props:
                        param_strs = []
                        for pname, pschema in props.items():
                            req = "*" if pname in required else ""
                            ptype = pschema.get("type", "any")
                            param_strs.append(f"{pname}{req}:{ptype}")
                        lines.append(f"    Parameters: {', '.join(param_strs)}")

        # Add usage instructions with concrete examples
        lines.append("\nWhen you want to use a tool, respond with ONLY this JSON format:")
        lines.append("  {\"mcp_tool\": \"SERVER_NAME:tool_name\", \"arguments\": {\"param\": \"value\"}}")
        lines.append("")
        lines.append("IMPORTANT: Replace SERVER_NAME with the actual server name shown in brackets above.")

        # Add concrete examples from the first available server
        first_server = next(iter(self._servers.keys()), None)
        if first_server and self._servers[first_server].tools:
            tools = self._servers[first_server].tools
            lines.append(f"\nExamples for [{first_server}]:")
            if any(t.name == "itunes_play" for t in tools):
                lines.append(f'  Play music: {{"mcp_tool": "{first_server}:itunes_play", "arguments": {{}}}}')
            if any(t.name == "itunes_search" for t in tools):
                lines.append(f'  Search: {{"mcp_tool": "{first_server}:itunes_search", "arguments": {{"query": "jazz"}}}}')
            if any(t.name == "itunes_pause" for t in tools):
                lines.append(f'  Pause: {{"mcp_tool": "{first_server}:itunes_pause", "arguments": {{}}}}')

        # Add Home Assistant examples if connected
        if "home-assistant" in self._servers:
            ha_tools = self._servers["home-assistant"].tools
            lines.append(f"\nExamples for [home-assistant]:")
            lines.append("  IMPORTANT: For lights, use domain=['light']. Do NOT use device_class for lights.")
            lines.append("  device_class is ONLY for covers (blind, shutter, curtain, etc), switches, outlets, TVs, speakers.")
            lines.append("  Use 'area' for room-based commands (e.g., 'living room lights'). Use 'name' only for specific devices.")
            if any(t.name == "HassTurnOff" for t in ha_tools):
                lines.append('  Turn off room lights: {"mcp_tool": "home-assistant:HassTurnOff", "arguments": {"area": "living room", "domain": ["light"]}}')
            if any(t.name == "HassTurnOn" for t in ha_tools):
                lines.append('  Turn on room lights: {"mcp_tool": "home-assistant:HassTurnOn", "arguments": {"area": "bedroom", "domain": ["light"]}}')
            if any(t.name == "HassLightSet" for t in ha_tools):
                lines.append('  Set brightness: {"mcp_tool": "home-assistant:HassLightSet", "arguments": {"area": "office", "brightness": 50}}')

        return "\n".join(lines)

    def get_tool_schemas(self) -> list[dict]:
        """
        Get tool schemas in a format suitable for LLM function calling.

        Returns:
            List of tool schemas
        """
        schemas = []
        for name, server in self._servers.items():
            for tool in server.tools:
                desc = self._safe_tool_description(
                    tool.description or "", tool.name, name
                )
                schema = {
                    "name": f"{name}:{tool.name}",
                    "description": desc,
                    "parameters": tool.inputSchema or {"type": "object"},
                }
                schemas.append(schema)
        return schemas


# Global manager instance
_manager: Optional[MCPManager] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def get_mcp_manager() -> MCPManager:
    """Get or create the global MCP manager."""
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager


def get_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Get the stored event loop (the one MCP sessions are bound to)."""
    return _event_loop


async def init_mcp(verbose: bool = True) -> MCPManager:
    """Initialize and connect the global MCP manager."""
    global _event_loop
    # Store the event loop that MCP sessions will be bound to
    _event_loop = asyncio.get_running_loop()
    manager = get_mcp_manager()
    await manager.connect_all(verbose=verbose)
    # Start keep-alive pings to detect and recover from dead connections
    await manager.start_keepalive()
    return manager


async def shutdown_mcp(verbose: bool = True) -> None:
    """Shutdown the global MCP manager."""
    global _manager
    if _manager is not None:
        await _manager.stop_keepalive()
        await _manager.disconnect_all(verbose=verbose)
        _manager = None
