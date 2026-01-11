"""Memory tools for explicit memory operations."""

from typing import TYPE_CHECKING, Any

from ash.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from ash.memory.manager import MemoryManager


class RememberTool(Tool):
    """Store facts and preferences in long-term memory.

    Use when:
    - User explicitly asks to remember something
    - User shares important preferences or facts about themselves or others
    - Information will be relevant to future conversations

    Facts should be stored as complete, standalone statements that will
    make sense when retrieved later without context.

    For facts about specific people, specify the subject to enable better
    retrieval later (e.g., "my wife", "Sarah", "my boss").
    """

    def __init__(self, memory_manager: "MemoryManager"):
        """Initialize remember tool.

        Args:
            memory_manager: Memory manager for storing memories.
        """
        self._memory = memory_manager

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return (
            "Store a fact or preference in long-term memory. "
            "IMPORTANT: Always store as a complete, standalone statement. "
            "If the fact is about a specific person (not the user), specify the subject. "
            "Good: 'Sarah's birthday is March 15th' with subject='my wife' or subject='Sarah'. "
            "Good: 'User prefers dark mode' with no subject (general user preference). "
            "Bad: 'March 15th', 'likes it'."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "A complete, standalone statement. "
                        "Examples: 'Sarah likes Italian food', 'User prefers Python', "
                        "'Boss's name is Michael', 'User's birthday is March 15th'."
                    ),
                },
                "subject": {
                    "type": "string",
                    "description": (
                        "Who this fact is about, if not general user info. "
                        "Use relationship terms the user uses: 'my wife', 'my boss', 'Sarah'. "
                        "Leave empty for general facts about the user."
                    ),
                },
                "expires_in_days": {
                    "type": "integer",
                    "description": "Optional: number of days until this memory expires.",
                },
                "shared": {
                    "type": "boolean",
                    "description": (
                        "If true, this memory is shared with everyone in the chat "
                        "(e.g., team facts, group reminders). Default is false (personal memory)."
                    ),
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Store the fact in memory.

        Args:
            input_data: Must contain 'content' key.
            context: Execution context.

        Returns:
            Tool result confirming storage.
        """
        content = input_data.get("content")
        if not content:
            return ToolResult.error("Missing required parameter: content")

        subject_ref = input_data.get("subject")
        expires_in_days = input_data.get("expires_in_days")
        is_shared = input_data.get("shared", False)

        try:
            # Resolve subject to person ID
            subject_person_id = None
            person_created = False
            subject_name = None

            if subject_ref and context.user_id:
                result = await self._memory.resolve_or_create_person(
                    owner_user_id=context.user_id,
                    reference=subject_ref,
                    content_hint=content,
                )
                subject_person_id = result.person_id
                person_created = result.created
                subject_name = result.person_name

            # Shared memories have no owner (visible to everyone)
            # Personal memories are owned by the current user
            owner_user_id = None if is_shared else context.user_id

            await self._memory.add_memory(
                content=content,
                source="remember_tool",
                expires_in_days=expires_in_days,
                owner_user_id=owner_user_id,
                subject_person_id=subject_person_id,
            )

            response = f"Remembered: {content}"
            if is_shared:
                response += " (shared with everyone)"
            if subject_person_id and person_created:
                response += f" (created new person record for '{subject_name}')"
            elif subject_person_id:
                response += f" (about {subject_name})"

            return ToolResult.success(response)
        except Exception as e:
            return ToolResult.error(f"Failed to store memory: {e}")


class RecallTool(Tool):
    """Search memory for relevant information.

    Use when:
    - You need to search for something NOT in the auto-retrieved context
    - User asks about a specific past conversation topic
    - Looking for information with a different query than the user's message

    DO NOT use when:
    - Relevant knowledge is already shown in "Relevant Context from Memory"
    - Answering simple questions about the user (name, preferences, etc.)
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
            "Search memory with a custom query. "
            "Can optionally filter by person (e.g., 'what do I know about my wife?'). "
            "Only use if you need information NOT already in your context. "
            "Check 'Relevant Context from Memory' first."
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
                "about": {
                    "type": "string",
                    "description": (
                        "Optional: filter to memories about a specific person. "
                        "Use same reference as user: 'my wife', 'Sarah', 'boss'."
                    ),
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

        about_ref = input_data.get("about")

        try:
            # If searching about a specific person, filter results
            person_id = None
            if about_ref and context.user_id:
                person = await self._memory.find_person(context.user_id, about_ref)
                if person:
                    person_id = person.id

            results = await self._memory.search(
                query,
                limit=5,
                subject_person_id=person_id,
                owner_user_id=context.user_id,
            )

            if not results:
                if about_ref:
                    return ToolResult.success(f"No memories found about {about_ref}.")
                return ToolResult.success("No relevant memories found.")

            # Format results with subject attribution
            lines = ["Found relevant memories:"]
            for result in results:
                source = result.source_type
                subject_label = ""
                if result.metadata and result.metadata.get("subject_name"):
                    subject_label = f" (about {result.metadata['subject_name']})"
                lines.append(f"- [{source}{subject_label}] {result.content}")

            return ToolResult.success("\n".join(lines))
        except Exception as e:
            return ToolResult.error(f"Failed to search memory: {e}")
