"""Memory tools for explicit memory operations."""

from typing import TYPE_CHECKING, Any

from ash.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from ash.memory.manager import MemoryManager


class RememberTool(Tool):
    """Store facts and preferences in long-term memory.

    Use when:
    - User explicitly asks to remember something
    - User shares important preferences or facts about themselves
    - Information will be relevant to future conversations
    """

    def __init__(self, memory_manager: "MemoryManager"):
        """Initialize remember tool.

        Args:
            memory_manager: Memory manager for storing knowledge.
        """
        self._memory = memory_manager

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return (
            "Store a fact, preference, or piece of information in long-term memory. "
            "Use when the user explicitly asks you to remember something, or when "
            "they share important preferences or facts about themselves."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact or information to remember.",
                },
                "expires_in_days": {
                    "type": "integer",
                    "description": "Optional: number of days until this memory expires.",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Store the fact in the knowledge base.

        Args:
            input_data: Must contain 'content' key.
            context: Execution context.

        Returns:
            Tool result confirming storage.
        """
        content = input_data.get("content")
        if not content:
            return ToolResult.error("Missing required parameter: content")

        expires_in_days = input_data.get("expires_in_days")

        try:
            await self._memory.add_knowledge(
                content=content,
                source="remember_tool",
                expires_in_days=expires_in_days,
            )
            return ToolResult.success(f"Remembered: {content}")
        except Exception as e:
            return ToolResult.error(f"Failed to store memory: {e}")


class RecallTool(Tool):
    """Search memory for relevant information.

    Use when:
    - User asks what you remember or know about something
    - You need to explicitly search past context
    """

    def __init__(self, memory_manager: "MemoryManager"):
        """Initialize recall tool.

        Args:
            memory_manager: Memory manager for searching.
        """
        self._memory = memory_manager

    @property
    def name(self) -> str:
        return "recall"

    @property
    def description(self) -> str:
        return (
            "Search your memory for relevant information. "
            "Use when the user asks what you remember or know about something."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory.",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Search memory for relevant information.

        Args:
            input_data: Must contain 'query' key.
            context: Execution context.

        Returns:
            Tool result with search results.
        """
        query = input_data.get("query")
        if not query:
            return ToolResult.error("Missing required parameter: query")

        try:
            results = await self._memory.search(query, limit=5)

            if not results:
                return ToolResult.success("No relevant memories found.")

            # Format results
            lines = ["Found relevant memories:"]
            for result in results:
                source = result.source_type
                lines.append(f"- [{source}] {result.content}")

            return ToolResult.success("\n".join(lines))
        except Exception as e:
            return ToolResult.error(f"Failed to search memory: {e}")
