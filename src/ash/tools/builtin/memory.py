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

    DO NOT use for:
    - Speech patterns, verbal tics, or conversation style ("says bud", "uses emojis")
    - Trivial acknowledgments or casual remarks
    - Temporary emotional states or moods
    - Information only relevant to the current conversation
    - Observations about HOW the user communicates rather than WHAT they communicate

    Facts should be stored as complete, standalone statements that will
    make sense when retrieved later without context.

    For facts about specific people, specify the subject to enable better
    retrieval later (e.g., "my wife", "Sarah", "my boss").

    IMPORTANT: When storing multiple facts, use the 'facts' array parameter
    to batch them in a single call instead of calling remember multiple times.
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
            "Store facts or preferences in long-term memory. "
            "Use 'facts' array to batch multiple memories in one call. "
            "Only store facts that will be USEFUL in future conversations. "
            "DO NOT store: speech patterns, verbal tics, conversation quirks, "
            "temporary moods, or anything that won't matter later. "
            "If the fact is about a specific person (not the user), specify the subject. "
            "Good: 'Sarah's birthday is March 15th', 'User prefers dark mode'. "
            "Bad: 'User says bud', 'User uses lowercase', 'User seems happy today'."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "A single fact to store. Use 'facts' array instead for multiple items. "
                        "Examples: 'Sarah likes Italian food', 'User prefers Python'."
                    ),
                },
                "facts": {
                    "type": "array",
                    "maxItems": 20,
                    "description": (
                        "Batch multiple facts in one call. Each fact can have different subjects/shared settings. "
                        "ALWAYS use this instead of calling remember multiple times."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The fact to store.",
                            },
                            "subjects": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 10,
                                "description": (
                                    "Who this fact is about. Array of references like "
                                    "['Sarah'], ['my wife', 'John'], ['Mom', 'Dad']. "
                                    "Omit for general facts about the user."
                                ),
                            },
                            "expires_in_days": {
                                "type": "integer",
                                "description": "Days until expiry (optional).",
                            },
                            "shared": {
                                "type": "boolean",
                                "description": (
                                    "True for group/team facts ('we', 'our team', 'everyone'). "
                                    "False (default) for personal facts."
                                ),
                            },
                        },
                        "required": ["content"],
                    },
                },
                "subjects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 10,
                    "description": (
                        "Who this fact is about. Array of references like "
                        "['Sarah'], ['my wife', 'John'], ['Mom', 'Dad']. "
                        "Omit for general facts about the user."
                    ),
                },
                "expires_in_days": {
                    "type": "integer",
                    "description": "Optional: number of days until this memory expires.",
                },
                "shared": {
                    "type": "boolean",
                    "description": (
                        "Set to true for group/team facts that everyone in the chat should see. "
                        "Use for: 'our team meeting', 'the project deadline', 'everyone should know'. "
                        "Default is false (personal memory only visible to this user)."
                    ),
                },
            },
        }

    async def _store_single_fact(
        self,
        content: str,
        subject_refs: list[str] | None,
        expires_in_days: int | None,
        is_shared: bool,
        context: ToolContext,
    ) -> str:
        """Store a single fact and return a status message."""
        # Resolve subjects to person IDs
        subject_person_ids: list[str] = []
        new_people: list[str] = []
        existing_people: list[str] = []

        if subject_refs and context.user_id:
            # Runtime limit check (defense in depth)
            if len(subject_refs) > 10:
                raise ValueError("Too many subjects: maximum 10 per fact")
            for ref in subject_refs:
                result = await self._memory.resolve_or_create_person(
                    owner_user_id=context.user_id,
                    reference=ref,
                    content_hint=content,
                )
                subject_person_ids.append(result.person_id)
                if result.created:
                    new_people.append(result.person_name)
                else:
                    existing_people.append(result.person_name)

        # Memory scoping:
        # - Personal: owner_user_id set, chat_id NULL - only visible to user
        # - Group: owner_user_id NULL, chat_id set - visible to everyone in chat
        if is_shared:
            owner_user_id = None
            chat_id = context.chat_id
        else:
            owner_user_id = context.user_id
            chat_id = None

        await self._memory.add_memory(
            content=content,
            source="remember_tool",
            expires_in_days=expires_in_days,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids if subject_person_ids else None,
        )

        # Build status message
        status = content
        extras = []
        if is_shared:
            extras.append("shared")
        if new_people:
            extras.append(f"new: {', '.join(new_people)}")
        if existing_people:
            extras.append(f"about: {', '.join(existing_people)}")

        if extras:
            status += f" ({'; '.join(extras)})"
        return status

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Store facts in memory.

        Args:
            input_data: Contains 'content' (single) or 'facts' (batch).
            context: Execution context.

        Returns:
            Tool result confirming storage.
        """
        facts = input_data.get("facts", [])
        single_content = input_data.get("content")

        # Build list of facts to store
        if facts:
            # Runtime limit check (defense in depth)
            if len(facts) > 20:
                return ToolResult.error("Too many facts: maximum 20 per call")
            items_to_store = facts
        elif single_content:
            items_to_store = [input_data]
        else:
            return ToolResult.error("Missing required parameter: content or facts")

        try:
            stored = []
            errors = []

            for item in items_to_store:
                content = item.get("content")
                if not content:
                    errors.append("Skipped item with missing content")
                    continue

                try:
                    status = await self._store_single_fact(
                        content=content,
                        subject_refs=item.get("subjects"),
                        expires_in_days=item.get("expires_in_days"),
                        is_shared=item.get("shared", False),
                        context=context,
                    )
                    stored.append(status)
                except Exception as e:
                    errors.append(f"Failed to store '{content[:30]}...': {e}")

            # Build response
            if not stored and errors:
                return ToolResult.error("\n".join(errors))

            lines = [f"Remembered {len(stored)} fact(s):"]
            for s in stored:
                lines.append(f"  - {s}")
            if errors:
                lines.append(f"\n{len(errors)} error(s):")
                for e in errors:
                    lines.append(f"  - {e}")

            return ToolResult.success("\n".join(lines))
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
                chat_id=context.chat_id,
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
