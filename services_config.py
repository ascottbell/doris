"""
Service configuration loader.

Loads config/services.yaml, normalizes the schema, and provides:
- ServiceRegistry: knows which service categories are enabled
- get_available_tools(): filters the tool list based on configured services
- is_service_enabled(): check if a category has any active backends

Platform safety: macosx backends are automatically disabled on non-macOS.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from platform_detect import IS_MACOS

logger = logging.getLogger("doris.services")

# Types that require macOS
_MACOS_ONLY_TYPES = frozenset({"macosx"})

# Types that require Linux
_LINUX_ONLY_TYPES = frozenset({"linux", "systemd", "sysvinit", "supervisor"})

# Types that require Docker
_DOCKER_ONLY_TYPES = frozenset({"docker"})

# Map service categories to the tool names they gate
SERVICE_TOOL_MAP: dict[str, list[str]] = {
    "calendar": [
        "get_calendar_events",
        "create_calendar_event",
        "move_calendar_event",
        "delete_calendar_event",
    ],
    "email": [
        "check_email",
        "send_email",
        "read_email",
    ],
    "reminders": [
        "create_reminder",
        "list_reminders",
        "complete_reminder",
    ],
    "messaging": [
        "send_imessage",
        "read_imessages",
    ],
    "contacts": [
        "lookup_contact",
    ],
    "notes": [
        "search_notes",
        "read_note",
        "create_note",
    ],
    "music": [
        "control_music",
    ],
    "shortcuts": [
        "run_shortcut",
        "list_shortcuts",
    ],
    "browser": [
        "control_browser",
    ],
    "system": [
        "system_info",
    ],
    "service_control": [
        "service_control",
    ],
}

# Reverse map: tool name -> category
_TOOL_TO_CATEGORY: dict[str, str] = {}
for _cat, _tools in SERVICE_TOOL_MAP.items():
    for _tool in _tools:
        _TOOL_TO_CATEGORY[_tool] = _cat


@dataclass
class ServiceInstance:
    """A single configured service backend."""

    type: str
    label: str = ""
    config: dict = field(default_factory=dict)
    enabled: bool = True


def _normalize_category(value) -> list[ServiceInstance]:
    """Normalize a category value from YAML into a list of ServiceInstance.

    Handles:
    - false / None -> empty list (disabled)
    - dict with 'type' key -> single-element list (shorthand)
    - list of dicts -> list of ServiceInstance
    """
    if value is None or value is False:
        return []

    if isinstance(value, dict):
        # Shorthand: {type: macosx} -> [{type: macosx}]
        if "type" not in value:
            raise ValueError(f"Service instance missing 'type' field: {value}")
        return [
            ServiceInstance(
                type=value["type"],
                label=value.get("label", ""),
                config=value.get("config", {}),
                enabled=value.get("enabled", True),
            )
        ]

    if isinstance(value, list):
        instances = []
        for item in value:
            if not isinstance(item, dict) or "type" not in item:
                raise ValueError(f"Service instance missing 'type' field: {item}")
            instances.append(
                ServiceInstance(
                    type=item["type"],
                    label=item.get("label", ""),
                    config=item.get("config", {}),
                    enabled=item.get("enabled", True),
                )
            )
        return instances

    raise ValueError(f"Invalid service category value: {value!r}")


def _platform_filter(instances: list[ServiceInstance], category: str) -> list[ServiceInstance]:
    """Remove instances whose type requires a platform we're not on."""
    from platform_detect import IS_MACOS, IS_LINUX

    # Check if running in Docker
    is_docker = False
    if IS_LINUX:
        is_docker = Path("/.dockerenv").exists()

    filtered = []
    for inst in instances:
        if not inst.enabled:
            continue
        if inst.type in _MACOS_ONLY_TYPES and not IS_MACOS:
            logger.warning(
                "Skipping %s service '%s' (type=%s): requires macOS, running on %s",
                category,
                inst.label or inst.type,
                inst.type,
                "Docker" if is_docker else "Linux",
            )
            continue
        if inst.type in _LINUX_ONLY_TYPES and not IS_LINUX:
            logger.warning(
                "Skipping %s service '%s' (type=%s): requires Linux, running on macOS",
                category,
                inst.label or inst.type,
                inst.type,
            )
            continue
        if inst.type in _DOCKER_ONLY_TYPES and not is_docker:
            logger.warning(
                "Skipping %s service '%s' (type=%s): requires Docker",
                category,
                inst.label or inst.type,
                inst.type,
            )
            continue
        filtered.append(inst)
    return filtered


class ServiceRegistry:
    """Loads services.yaml and provides lookup for configured service backends."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "services.yaml"

        self._categories: dict[str, list[ServiceInstance]] = {}
        self._load(config_path)

    def _load(self, config_path: Path) -> None:
        if not config_path.exists():
            logger.info("No services config at %s - all category-gated tools disabled", config_path)
            return

        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        for category, value in raw.items():
            try:
                instances = _normalize_category(value)
                instances = _platform_filter(instances, category)
                if instances:
                    self._categories[category] = instances
            except ValueError as e:
                logger.error("Invalid service config for '%s': %s", category, e)

    def get_instances(self, category: str) -> list[ServiceInstance]:
        """Return all enabled instances for a category, ordered."""
        return self._categories.get(category, [])

    def get_primary(self, category: str) -> Optional[ServiceInstance]:
        """Return the first enabled instance (used for write operations)."""
        instances = self.get_instances(category)
        return instances[0] if instances else None

    def has_category(self, category: str) -> bool:
        """True if at least one enabled instance exists for this category."""
        return bool(self._categories.get(category))

    def get_disabled_tool_names(self) -> set[str]:
        """Return tool names that should be excluded (their category has no backends)."""
        disabled = set()
        for category, tools in SERVICE_TOOL_MAP.items():
            if not self.has_category(category):
                disabled.update(tools)
        return disabled

    def get_enabled_categories(self) -> list[str]:
        """Return list of categories that have at least one active backend."""
        return list(self._categories.keys())


# Module-level singleton, loaded once at import time
_registry: Optional[ServiceRegistry] = None


def get_registry() -> ServiceRegistry:
    """Get the service registry singleton."""
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
    return _registry


def is_service_enabled(category: str) -> bool:
    """Check if a service category has any active backends."""
    return get_registry().has_category(category)
