"""Tool registry for managing available tools."""

import logging
from typing import Any

from ash.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for tool instances.

    Manages tool registration and lookup.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If tool with same name already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Tool name to unregister.
        """
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            Tool instance.

        Raises:
            KeyError: If tool not found.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]

    def has(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name.

        Returns:
            True if tool exists.
        """
        return name in self._tools

    @property
    def tools(self) -> dict[str, Tool]:
        """Get all registered tools."""
        return dict(self._tools)

    @property
    def names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions for LLM.

        Returns:
            List of tool definitions.
        """
        return [tool.to_definition() for tool in self._tools.values()]

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over tools."""
        return iter(self._tools.values())
