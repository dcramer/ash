"""Tool registry for managing available tools."""

import logging

from ash.llm.types import ToolDefinition
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
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def tools(self) -> dict[str, Tool]:
        return dict(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get_definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            for tool in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
