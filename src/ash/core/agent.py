"""Agent orchestrator with agentic loop."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.core.compaction import CompactionSettings, compact_messages, should_compact
from ash.core.context import ContextGatherer
from ash.core.prompt import (
    PromptContext,
    SystemPromptBuilder,
)
from ash.core.session import SessionState
from ash.core.tokens import estimate_tokens
from ash.core.types import (
    CHECKPOINT_METADATA_KEY,
    AgentComponents,
    AgentConfig,
    AgentResponse,
    CompactionInfo,
    GetSteeringMessagesCallback,
    OnToolStartCallback,
    _MessageSetup,
    _StreamToolAccumulator,
)
from ash.llm import LLMProvider, ToolDefinition
from ash.llm.thinking import resolve_thinking
from ash.llm.types import (
    ContentBlock,
    StreamEventType,
    TextContent,
    ToolUse,
)
from ash.store.types import MemoryType
from ash.tools import ToolContext, ToolExecutor, ToolRegistry

if TYPE_CHECKING:
    from ash.config import AshConfig, Workspace
    from ash.core.prompt import RuntimeInfo
    from ash.db.engine import Database
    from ash.memory import MemoryExtractor, RetrievedContext
    from ash.providers.base import IncomingMessage
    from ash.store.store import Store
    from ash.store.types import PersonEntry

logger = logging.getLogger(__name__)


def _extract_checkpoint(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extract checkpoint from tool calls metadata if present.

    Looks for the most recent use_agent call with checkpoint metadata.
    """
    for call in reversed(tool_calls):
        if call.get("name") == "use_agent":
            metadata = call.get("metadata", {})
            if CHECKPOINT_METADATA_KEY in metadata:
                return metadata[CHECKPOINT_METADATA_KEY]
    return None


def _extract_relationship_term(content: str) -> str | None:
    """Extract a relationship term from fact content.

    Scans for known relationship terms (wife, boss, friend, etc.) to
    attach to person records when a RELATIONSHIP-type fact is extracted.
    Returns the first match found, or None.
    """
    from ash.store.people import RELATIONSHIP_TERMS

    content_lower = content.lower()
    # Check multi-word terms first (e.g., "best friend" before "friend")
    for term in sorted(RELATIONSHIP_TERMS, key=lambda t: len(t), reverse=True):
        if term in content_lower:
            return term
    return None


# Invalid speaker values that indicate assistant attribution
_INVALID_SPEAKERS = frozenset({"agent", "assistant", "bot", "system", "ash"})


def _validate_speaker(speaker: str | None) -> str | None:
    """Validate speaker, filtering out invalid values.

    Returns None for invalid speakers (agent, assistant, etc.) or empty values.
    Preserves original casing — callers already lowercase for comparison.
    """
    if not speaker:
        return None
    if speaker.lower() in _INVALID_SPEAKERS:
        logger.debug("Filtering invalid speaker: %s", speaker)
        return None
    return speaker


def _build_routing_env(
    session: SessionState,
    effective_user_id: str | None,
    timezone: str = "UTC",
) -> dict[str, str]:
    """Build environment variables for routing context in sandbox.

    These env vars allow sandboxed CLI commands (like `ash schedule`) to
    access routing context for operations that need to send responses back.
    Also includes skill env vars set by inline skills.
    """
    env = {
        "ASH_SESSION_ID": session.session_id or "",
        "ASH_USER_ID": effective_user_id or "",
        "ASH_CHAT_ID": session.chat_id or "",
        "ASH_CHAT_TITLE": session.metadata.get("chat_title", ""),
        "ASH_PROVIDER": session.provider or "",
        "ASH_USERNAME": session.metadata.get("username", ""),
        "ASH_DISPLAY_NAME": session.metadata.get("display_name", ""),
        "ASH_TIMEZONE": timezone,
    }

    # Provide chat state paths for sandbox access
    # ASH_CHAT_PATH: always points to chat-level state
    # ASH_THREAD_PATH: points to thread-specific state when in a thread
    if session.provider and session.chat_id:
        env["ASH_CHAT_PATH"] = f"/chats/{session.provider}/{session.chat_id}"
        if thread_id := session.metadata.get("thread_id"):
            env["ASH_THREAD_PATH"] = (
                f"/chats/{session.provider}/{session.chat_id}/threads/{thread_id}"
            )

    return env


class Agent:
    """Main agent orchestrator.

    Handles the agentic loop: receiving messages, calling the LLM,
    executing tools, and returning responses.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_executor: ToolExecutor,
        prompt_builder: SystemPromptBuilder,
        runtime: RuntimeInfo | None = None,
        memory_extractor: MemoryExtractor | None = None,
        config: AgentConfig | None = None,
        graph_store: Store | None = None,
    ):
        """Initialize agent.

        Args:
            llm: LLM provider for completions.
            tool_executor: Tool executor for running tools.
            prompt_builder: System prompt builder with full context.
            runtime: Runtime information for prompt.
            memory_extractor: Optional memory extractor for background extraction.
            config: Agent configuration.
            graph_store: Unified graph store (memory + people).
        """
        self._llm = llm
        self._tools = tool_executor
        self._prompt_builder = prompt_builder
        self._runtime = runtime
        self._graph_store = graph_store
        self._memory: Store | None = graph_store
        self._extractor = memory_extractor
        self._people: Store | None = graph_store
        self._config = config or AgentConfig()
        self._last_extraction_time: float | None = None

    @property
    def system_prompt(self) -> str:
        """Get the base system prompt (without memory context)."""
        runtime = self._refresh_runtime_time()
        return self._prompt_builder.build(PromptContext(runtime=runtime))

    def _refresh_runtime_time(self) -> RuntimeInfo | None:
        """Return runtime with refreshed current time, or None if no runtime."""
        if not self._runtime:
            return None
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(self._timezone)
        local_time = datetime.now(UTC).astimezone(tz)
        return replace(self._runtime, time=local_time.strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def _timezone(self) -> str:
        """Get the configured timezone, defaulting to UTC."""
        return (
            self._runtime.timezone
            if self._runtime and self._runtime.timezone
            else "UTC"
        )

    def _build_system_prompt(
        self,
        context: RetrievedContext | None = None,
        known_people: list[PersonEntry] | None = None,
        conversation_gap_minutes: float | None = None,
        has_reply_context: bool = False,
        sender_username: str | None = None,
        sender_display_name: str | None = None,
        chat_title: str | None = None,
        chat_type: str | None = None,
        chat_state_path: str | None = None,
        thread_state_path: str | None = None,
        is_scheduled_task: bool = False,
        is_passive_engagement: bool = False,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build system prompt with optional memory context."""
        from ash.core.prompt import ChatInfo, SenderInfo

        prompt_context = PromptContext(
            runtime=self._refresh_runtime_time(),
            memory=context,
            known_people=known_people,
            sender=SenderInfo(
                username=sender_username,
                display_name=sender_display_name,
            ),
            chat=ChatInfo(
                title=chat_title,
                chat_type=chat_type,
                state_path=chat_state_path,
                thread_state_path=thread_state_path,
                is_scheduled_task=is_scheduled_task,
                is_passive_engagement=is_passive_engagement,
            ),
            conversation_gap_minutes=conversation_gap_minutes,
            has_reply_context=has_reply_context,
            chat_history=chat_history,
        )
        return self._prompt_builder.build(prompt_context)

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        return self._tools.get_definitions()

    async def _maybe_compact(self, session: SessionState) -> CompactionInfo | None:
        if not self._config.compaction_enabled:
            return None

        token_counts = session._get_token_counts()
        total_tokens = sum(token_counts)

        settings = CompactionSettings(
            enabled=self._config.compaction_enabled,
            reserve_tokens=self._config.compaction_reserve_tokens,
            keep_recent_tokens=self._config.compaction_keep_recent_tokens,
            summary_max_tokens=self._config.compaction_summary_max_tokens,
        )

        if not should_compact(
            total_tokens, self._config.context_token_budget, settings
        ):
            return None

        logger.info(
            f"Context near limit ({total_tokens}/{self._config.context_token_budget} tokens), "
            "running compaction"
        )

        start_time = time.monotonic()
        new_messages, new_token_counts, result = await compact_messages(
            messages=session.messages,
            token_counts=token_counts,
            llm=self._llm,
            settings=settings,
            model=self._config.model,
        )
        duration_ms = int((time.monotonic() - start_time) * 1000)

        if result is None:
            logger.debug("Compaction skipped - not enough messages to summarize")
            return None

        session.messages = new_messages
        session._token_counts = new_token_counts

        logger.info(
            f"Compaction complete: {result.tokens_before} -> {result.tokens_after} tokens "
            f"({result.messages_removed} messages summarized) | {duration_ms}ms"
        )

        return CompactionInfo(
            summary=result.summary,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            messages_removed=result.messages_removed,
        )

    async def _prepare_message_context(
        self,
        user_message: str,
        session: SessionState,
        user_id: str | None,
    ) -> _MessageSetup:
        effective_user_id = user_id or session.user_id

        # Use ContextGatherer to retrieve memory and people context
        context_gatherer = ContextGatherer(self._memory)
        gathered = await context_gatherer.gather(
            user_id=effective_user_id,
            user_message=user_message,
            chat_id=session.chat_id,
            chat_type=session.metadata.get("chat_type"),
            sender_username=session.metadata.get("username"),
        )

        system_prompt = self._build_system_prompt(
            context=gathered.memory,
            known_people=gathered.known_people,
            conversation_gap_minutes=session.metadata.get("conversation_gap_minutes"),
            has_reply_context=session.metadata.get("has_reply_context", False),
            sender_username=session.metadata.get("username"),
            sender_display_name=session.metadata.get("display_name"),
            chat_title=session.metadata.get("chat_title"),
            chat_type=session.metadata.get("chat_type"),
            chat_state_path=(
                f"/chats/{session.provider}/{session.chat_id}"
                if session.provider and session.chat_id
                else None
            ),
            thread_state_path=(
                f"/chats/{session.provider}/{session.chat_id}/threads/{thread_id}"
                if session.provider
                and session.chat_id
                and (thread_id := session.metadata.get("thread_id"))
                else None
            ),
            is_scheduled_task=session.metadata.get("is_scheduled_task", False),
            is_passive_engagement=session.metadata.get("passive_engagement", False),
            chat_history=session.metadata.get("chat_history"),
        )

        system_tokens = estimate_tokens(system_prompt)
        message_budget = (
            self._config.context_token_budget
            - system_tokens
            - self._config.system_prompt_buffer
        )

        return _MessageSetup(
            effective_user_id=effective_user_id,
            system_prompt=system_prompt,
            message_budget=message_budget,
        )

    async def _ensure_self_person(
        self,
        user_id: str,
        username: str,
        display_name: str,
    ) -> str | None:
        """Ensure a self-Person exists for the user with username as alias.

        This enables proper trust determination by linking the username
        (used as source_username) to the display name (used for display).

        Lookup order: display_name first, then username. If a matching person
        exists but lacks a "self" relationship, we claim it (this handles
        the case where another user mentions "David Cramer" before David
        speaks). If no match, create a new self-person.

        Args:
            user_id: The user ID (used as created_by for new records).
            username: The user's handle/username (e.g., "notzeeg").
            display_name: The user's display name (e.g., "David Cramer").

        Returns:
            The person_id for the self-person, or None if no people store.
        """
        if not self._people:
            return None

        async with self._people._self_person_lock:
            try:
                # Try display name first, then username
                existing = await self._people.find_person(display_name)
                if not existing and username:
                    existing = await self._people.find_person(username)

                if existing:
                    is_self = any(
                        rc.relationship == "self" for rc in existing.relationships
                    )
                    if not is_self:
                        await self._people.add_relationship(
                            existing.id, "self", stated_by=username
                        )
                    await self._sync_person_details(
                        existing, display_name, username, user_id
                    )
                    return existing.id

                # No match found -- create new self-person
                # When no username, use numeric user_id as alias to reconnect the graph
                aliases = [username] if username else [user_id]
                new_person = await self._people.create_person(
                    created_by=user_id,
                    name=display_name,
                    relationship="self",
                    aliases=aliases,
                    relationship_stated_by=username or None,
                )
                logger.debug(
                    "Created self-person for user",
                    extra={
                        "user_id": user_id,
                        "person_name": display_name,
                        "username": username,
                    },
                )

                # Dedup: merge new self-person against any existing person with same name.
                # Use exclude_self=False because the new person always has "self" relationship
                # and would be skipped otherwise. _heuristic_match skips pairs where both are
                # self-persons created by *different* users, but allows duplicate self-records
                # from the same user to be detected and merged.
                result_id = new_person.id
                try:
                    candidates = await self._people.find_dedup_candidates(
                        [new_person.id], exclude_self=False
                    )
                    for primary_id, secondary_id in candidates:
                        await self._people.merge_people(primary_id, secondary_id)
                        # If our new person was merged away, track the primary
                        if secondary_id == new_person.id:
                            result_id = primary_id
                except Exception:
                    logger.warning("Self-person dedup failed", exc_info=True)
                return result_id
            except Exception:
                logger.warning("Failed to ensure self-person", exc_info=True)
                return None

    async def _sync_person_details(
        self,
        person: PersonEntry,
        display_name: str,
        username: str,
        user_id: str,
    ) -> None:
        """Update a person's name and ensure username alias exists.

        Args:
            person: Existing person record to sync.
            display_name: Expected display name.
            username: Username to ensure as alias.
            user_id: Who is making the update (for provenance).
        """
        assert self._people is not None

        if display_name and person.name != display_name:
            await self._people.update_person(
                person_id=person.id, name=display_name, updated_by=user_id
            )
        if username:
            aliases_lower = [a.value.lower() for a in person.aliases]
            if username.lower() not in aliases_lower:
                await self._people.add_alias(person.id, username, user_id)

    def _should_extract_memories(self, user_message: str) -> bool:
        if not self._config.extraction_enabled:
            return False

        if not self._extractor or not self._memory:
            return False

        if len(user_message) < self._config.extraction_min_message_length:
            return False

        if self._last_extraction_time is not None:
            elapsed = time.time() - self._last_extraction_time
            if elapsed < self._config.extraction_debounce_seconds:
                return False

        return True

    async def _extract_memories_background(
        self,
        session: SessionState,
        user_id: str,
        chat_id: str | None = None,
    ) -> None:
        from ash.llm.types import Message as LLMMessage
        from ash.llm.types import Role
        from ash.memory.extractor import SpeakerInfo

        if not self._extractor or not self._memory:
            return

        try:
            self._last_extraction_time = time.time()

            existing_memories: list[str] = []
            try:
                recent = await self._memory.list_memories(
                    owner_user_id=user_id,
                    chat_id=chat_id,
                    limit=20,
                )
                existing_memories = [m.content for m in recent]
            except Exception:
                logger.debug(
                    "Failed to get existing memories for extraction", exc_info=True
                )

            llm_messages: list[LLMMessage] = [
                msg
                for msg in session.messages
                if msg.role in (Role.USER, Role.ASSISTANT) and msg.get_text().strip()
            ]

            if not llm_messages:
                return

            # Build speaker info from session metadata for attribution
            speaker_username = session.metadata.get("username")
            speaker_display_name = session.metadata.get("display_name")

            # Collect owner names to avoid treating the user's own name
            # as a third party in extraction
            owner_names: list[str] = []
            if speaker_username:
                owner_names.append(speaker_username)
            if speaker_display_name and speaker_display_name not in owner_names:
                owner_names.append(speaker_display_name)
            speaker_info = SpeakerInfo(
                user_id=user_id,
                username=speaker_username,
                display_name=speaker_display_name,
            )

            # Ensure self-person exists for proper trust determination.
            # Create whenever we have at least one identifier.
            speaker_person_id: str | None = None
            if speaker_username or speaker_display_name:
                effective_display = speaker_display_name or speaker_username
                assert effective_display is not None  # guaranteed by outer if
                speaker_person_id = await self._ensure_self_person(
                    user_id=user_id,
                    username=speaker_username or "",
                    display_name=effective_display,
                )

            facts = await self._extractor.extract_from_conversation(
                messages=llm_messages,
                existing_memories=existing_memories,
                owner_names=owner_names if owner_names else None,
                speaker_info=speaker_info,
                current_datetime=datetime.now(UTC),
            )

            logger.info(
                "Extracted %d facts from conversation (speaker=%s)",
                len(facts),
                speaker_info.username if speaker_info else None,
            )
            for fact in facts:
                logger.debug(
                    "  fact: %s (type=%s, confidence=%.2f, subjects=%s, speaker=%s)",
                    fact.content[:80],
                    fact.memory_type.value,
                    fact.confidence,
                    fact.subjects,
                    fact.speaker,
                )

            # Build owner name matchers for filtering self-references
            from ash.core.filters import build_owner_matchers, is_owner_name

            owner_matchers = build_owner_matchers(owner_names)

            # Track newly created person IDs for post-extraction dedup
            newly_created_person_ids: list[str] = []

            for fact in facts:
                if fact.confidence < self._config.extraction_confidence_threshold:
                    continue

                try:
                    subject_person_ids: list[str] | None = None
                    if fact.subjects and self._people:
                        subject_person_ids = []
                        for subject in fact.subjects:
                            if is_owner_name(subject, owner_matchers):
                                logger.debug("Skipping owner as subject: %s", subject)
                                continue
                            try:
                                result = await self._people.resolve_or_create_person(
                                    created_by=user_id,
                                    reference=subject,
                                    content_hint=fact.content,
                                    relationship_stated_by=speaker_username,
                                )
                                subject_person_ids.append(result.person_id)
                                if result.created:
                                    newly_created_person_ids.append(result.person_id)
                            except Exception:
                                logger.warning("Failed to resolve subject: %s", subject)

                    # For RELATIONSHIP facts, attach the term to the person record
                    # so it shows in the Known People section of the system prompt.
                    if (
                        fact.memory_type == MemoryType.RELATIONSHIP
                        and subject_person_ids
                        and self._people
                    ):
                        rel_term = _extract_relationship_term(fact.content)
                        if rel_term:
                            for pid in subject_person_ids:
                                try:
                                    await self._people.add_relationship(
                                        pid,
                                        rel_term,
                                        stated_by=speaker_username,
                                    )
                                except Exception:
                                    logger.debug(
                                        "Failed to add relationship %s to %s",
                                        rel_term,
                                        pid,
                                    )

                    # Capture whether this is a self-fact (no subjects) before
                    # we inject speaker_person_id for graph traversal, since hearsay
                    # supersession needs to know the original extraction state.
                    is_self_fact = not subject_person_ids

                    # Self-facts (no subjects) should reference the speaker's person
                    # record so they're discoverable via graph traversal.
                    # Skip RELATIONSHIP type to avoid attaching relationship terms to
                    # the speaker instead of the related person.
                    if (
                        is_self_fact
                        and speaker_person_id
                        and fact.memory_type != MemoryType.RELATIONSHIP
                    ):
                        subject_person_ids = [speaker_person_id]

                    # Filter out invalid speaker values that indicate assistant attribution
                    speaker = _validate_speaker(fact.speaker)

                    # Determine source user from extracted speaker or session
                    source_username = speaker or speaker_username or user_id
                    source_display_name = (
                        speaker_display_name
                        if source_username == speaker_username
                        else None
                    )

                    new_memory = await self._memory.add_memory(
                        content=fact.content,
                        source="background_extraction",
                        memory_type=fact.memory_type,
                        owner_user_id=user_id if not fact.shared else None,
                        chat_id=chat_id if fact.shared else None,
                        subject_person_ids=subject_person_ids or None,
                        observed_at=datetime.now(UTC),
                        source_username=source_username,
                        source_display_name=source_display_name,
                        extraction_confidence=fact.confidence,
                        sensitivity=fact.sensitivity,
                        portable=fact.portable,
                    )

                    logger.debug(
                        "Extracted memory: %s (confidence=%.2f, speaker=%s)",
                        fact.content[:50],
                        fact.confidence,
                        source_username,
                    )

                    # Check for hearsay to supersede when this is a self-fact
                    # (user speaking about themselves). Use is_self_fact since
                    # subject_person_ids now has the speaker's person_id injected.
                    if is_self_fact and source_username and self._memory:
                        from ash.store.hearsay import supersede_hearsay_for_fact

                        await supersede_hearsay_for_fact(
                            store=self._memory,
                            new_memory=new_memory,
                            source_username=source_username,
                            owner_user_id=user_id,
                        )
                except Exception:
                    logger.debug(
                        "Failed to store extracted fact: %s",
                        fact.content[:50],
                        exc_info=True,
                    )

            # Post-extraction dedup: merge newly created people that match existing
            if self._people and newly_created_person_ids:
                try:
                    candidates = await self._people.find_dedup_candidates(
                        newly_created_person_ids, exclude_self=True
                    )
                    for primary_id, secondary_id in candidates:
                        await self._people.merge_people(primary_id, secondary_id)
                except Exception:
                    logger.warning("Post-extraction dedup failed", exc_info=True)

        except Exception:
            logger.warning("Background memory extraction failed", exc_info=True)

    def _spawn_memory_extraction(
        self,
        session: SessionState,
        user_id: str,
        chat_id: str | None = None,
    ) -> None:
        import asyncio

        def handle_error(task: asyncio.Task[None]) -> None:
            if not task.cancelled() and (exc := task.exception()):
                logger.warning("Memory extraction task failed: %s", exc)

        task = asyncio.create_task(
            self._extract_memories_background(session, user_id, chat_id),
            name="memory_extraction",
        )
        task.add_done_callback(handle_error)

    def _maybe_spawn_memory_extraction(
        self,
        user_message: str,
        effective_user_id: str,
        session: SessionState,
    ) -> None:
        if self._should_extract_memories(user_message):
            self._spawn_memory_extraction(session, effective_user_id, session.chat_id)

    def _build_tool_context(
        self,
        session: SessionState,
        setup: _MessageSetup,
        session_manager: Any = None,
        tool_overrides: dict[str, Any] | None = None,
    ) -> ToolContext:
        """Build a ToolContext for tool execution, with reply anchor initialized.

        Args:
            session: Current session state.
            setup: Message setup with effective user ID.
            session_manager: Optional session manager for subagent logging.
            tool_overrides: Per-session tool overrides (e.g., progress message tool).

        Returns:
            ToolContext ready for tool execution.
        """
        tool_context = ToolContext(
            session_id=session.session_id,
            user_id=setup.effective_user_id,
            chat_id=session.chat_id,
            thread_id=session.metadata.get("thread_id"),
            provider=session.provider,
            metadata=dict(session.metadata),
            env=_build_routing_env(
                session, setup.effective_user_id, timezone=self._timezone
            ),
            session_manager=session_manager,
            tool_overrides=tool_overrides or {},
        )

        # Initialize reply anchor from incoming message context
        if not tool_context.reply_to_message_id:
            tool_context.reply_to_message_id = session.metadata.get(
                "current_message_id"
            )

        return tool_context

    @staticmethod
    def _sync_reply_anchor(tool_context: ToolContext, session: SessionState) -> None:
        """Sync thread anchor from tool context back to session metadata."""
        if tool_context.reply_to_message_id:
            session.metadata["reply_to_message_id"] = tool_context.reply_to_message_id

    async def _execute_pending_tools(
        self,
        pending_tools: list[ToolUse],
        session: SessionState,
        tool_context: ToolContext,
        on_tool_start: OnToolStartCallback | None,
        get_steering_messages: GetSteeringMessagesCallback | None = None,
    ) -> tuple[list[dict[str, Any]], list[IncomingMessage]]:
        tool_calls: list[dict[str, Any]] = []

        for i, tool_use in enumerate(pending_tools):
            if on_tool_start:
                await on_tool_start(tool_use.name, tool_use.input)

            # Create per-tool context with the tool_use_id for subagent logging
            per_tool_context = replace(tool_context, tool_use_id=tool_use.id)

            result = await self._tools.execute(
                tool_use.name,
                tool_use.input,
                per_tool_context,
            )

            tool_calls.append(
                {
                    "id": tool_use.id,
                    "name": tool_use.name,
                    "input": tool_use.input,
                    "result": result.content,
                    "is_error": result.is_error,
                    "metadata": result.metadata,
                }
            )

            session.add_tool_result(
                tool_use_id=tool_use.id,
                content=result.content,
                is_error=result.is_error,
            )

            if get_steering_messages and i < len(pending_tools) - 1:
                steering = await get_steering_messages()
                if steering:
                    for remaining in pending_tools[i + 1 :]:
                        tool_calls.append(
                            {
                                "id": remaining.id,
                                "name": remaining.name,
                                "input": remaining.input,
                                "result": "Skipped: user sent new message",
                                "is_error": True,
                            }
                        )
                        session.add_tool_result(
                            tool_use_id=remaining.id,
                            content="Skipped: user sent new message",
                            is_error=True,
                        )
                    logger.info(
                        f"Steering received: skipping {len(pending_tools) - i - 1} remaining tools"
                    )
                    return tool_calls, steering

        return tool_calls, []

    async def process_message(
        self,
        user_message: str,
        session: SessionState,
        user_id: str | None = None,
        on_tool_start: OnToolStartCallback | None = None,
        get_steering_messages: GetSteeringMessagesCallback | None = None,
        session_manager: Any = None,  # Type: SessionManager | None
        tool_overrides: dict[str, Any] | None = None,
    ) -> AgentResponse:
        setup = await self._prepare_message_context(user_message, session, user_id)
        session.add_user_message(user_message)
        compaction_info = await self._maybe_compact(session)

        tool_calls: list[dict[str, Any]] = []
        iterations = 0

        while iterations < self._config.max_tool_iterations:
            iterations += 1

            response = await self._llm.complete(
                messages=session.get_messages_for_llm(
                    token_budget=setup.message_budget,
                    recency_window=self._config.recency_window,
                ),
                model=self._config.model,
                tools=self._get_tool_definitions(),
                system=setup.system_prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                thinking=self._config.thinking,
            )

            session.add_assistant_message(response.message.content)

            pending_tools = session.get_pending_tool_uses()
            text_len = len(response.message.get_text() or "")
            tool_names = [t.name for t in pending_tools]
            logger.info(
                f"Main agent iteration {iterations}: text_len={text_len}, "
                f"tools={tool_names}"
            )

            if not pending_tools:
                self._maybe_spawn_memory_extraction(
                    user_message, setup.effective_user_id, session
                )
                return AgentResponse(
                    text=response.message.get_text() or "",
                    tool_calls=tool_calls,
                    iterations=iterations,
                    compaction=compaction_info,
                    checkpoint=_extract_checkpoint(tool_calls),
                )

            tool_context = self._build_tool_context(
                session, setup, session_manager, tool_overrides
            )

            new_calls, steering = await self._execute_pending_tools(
                pending_tools,
                session,
                tool_context,
                on_tool_start,
                get_steering_messages,
            )
            tool_calls.extend(new_calls)

            self._sync_reply_anchor(tool_context, session)

            # Check if any tool returned a checkpoint - stop loop to wait for user input
            checkpoint = _extract_checkpoint(tool_calls)
            if checkpoint:
                self._maybe_spawn_memory_extraction(
                    user_message, setup.effective_user_id, session
                )
                return AgentResponse(
                    text=response.message.get_text() or "",
                    tool_calls=tool_calls,
                    iterations=iterations,
                    compaction=compaction_info,
                    checkpoint=checkpoint,
                )

            if steering:
                for msg in steering:
                    if msg.text:
                        session.add_user_message(msg.text)

        logger.warning(
            f"Max tool iterations ({self._config.max_tool_iterations}) reached"
        )
        self._maybe_spawn_memory_extraction(
            user_message, setup.effective_user_id, session
        )
        return AgentResponse(
            text="I've reached the maximum number of tool calls. Please try again with a simpler request.",
            tool_calls=tool_calls,
            iterations=iterations,
            compaction=compaction_info,
            checkpoint=_extract_checkpoint(tool_calls),
        )

    async def process_message_streaming(
        self,
        user_message: str,
        session: SessionState,
        user_id: str | None = None,
        on_tool_start: OnToolStartCallback | None = None,
        get_steering_messages: GetSteeringMessagesCallback | None = None,
        session_manager: Any = None,  # Type: SessionManager | None
        tool_overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        setup = await self._prepare_message_context(user_message, session, user_id)
        session.add_user_message(user_message)
        await self._maybe_compact(session)

        iterations = 0

        while iterations < self._config.max_tool_iterations:
            iterations += 1

            content_blocks: list[ContentBlock] = []
            current_text = ""
            tool_accumulator = _StreamToolAccumulator()

            async for chunk in self._llm.stream(
                messages=session.get_messages_for_llm(
                    token_budget=setup.message_budget,
                    recency_window=self._config.recency_window,
                ),
                model=self._config.model,
                tools=self._get_tool_definitions(),
                system=setup.system_prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                thinking=self._config.thinking,
            ):
                if chunk.type == StreamEventType.TEXT_DELTA:
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    current_text += text
                    yield text
                elif chunk.type == StreamEventType.TOOL_USE_START:
                    if chunk.tool_use_id and chunk.tool_name:
                        tool_accumulator.start(chunk.tool_use_id, chunk.tool_name)
                elif chunk.type == StreamEventType.TOOL_USE_DELTA:
                    tool_accumulator.add_delta(
                        chunk.content if isinstance(chunk.content, str) else ""
                    )
                elif chunk.type == StreamEventType.TOOL_USE_END:
                    if tool_use := tool_accumulator.finish():
                        content_blocks.append(tool_use)

            if current_text:
                content_blocks.insert(0, TextContent(text=current_text))

            if not content_blocks:
                self._maybe_spawn_memory_extraction(
                    user_message, setup.effective_user_id, session
                )
                return

            session.add_assistant_message(content_blocks)

            pending_tools = [b for b in content_blocks if isinstance(b, ToolUse)]
            if not pending_tools:
                self._maybe_spawn_memory_extraction(
                    user_message, setup.effective_user_id, session
                )
                return

            tool_context = self._build_tool_context(
                session, setup, session_manager, tool_overrides
            )

            _, steering = await self._execute_pending_tools(
                pending_tools,
                session,
                tool_context,
                on_tool_start,
                get_steering_messages,
            )

            self._sync_reply_anchor(tool_context, session)

            if steering:
                for msg in steering:
                    if msg.text:
                        session.add_user_message(msg.text)

        self._maybe_spawn_memory_extraction(
            user_message, setup.effective_user_id, session
        )
        yield "\n\n[Max tool iterations reached]"


async def create_agent(
    config: AshConfig,
    workspace: Workspace,
    db: Database | None = None,
    model_alias: str = "default",
) -> AgentComponents:
    from ash.agents import AgentExecutor, AgentRegistry
    from ash.agents.builtin import register_builtin_agents
    from ash.core.prompt import RuntimeInfo
    from ash.llm import create_llm_provider, create_registry
    from ash.memory import MemoryExtractor
    from ash.sandbox import SandboxExecutor
    from ash.sandbox.packages import build_setup_command, collect_skill_packages
    from ash.skills import SkillRegistry
    from ash.store import create_store
    from ash.tools.base import build_sandbox_manager_config
    from ash.tools.builtin import BashTool, WebFetchTool, WebSearchTool
    from ash.tools.builtin.agents import UseAgentTool
    from ash.tools.builtin.files import ReadFileTool, WriteFileTool
    from ash.tools.builtin.search_cache import SearchCache
    from ash.tools.builtin.skills import UseSkillTool

    model_config = config.get_model(model_alias)
    api_key = config.resolve_api_key(model_alias)

    llm = create_llm_provider(
        model_config.provider,
        api_key=api_key.get_secret_value() if api_key else None,
    )

    tool_registry = ToolRegistry()

    skill_registry = SkillRegistry(skill_config=config.skills)
    skill_registry.discover(config.workspace)
    logger.info(f"Discovered {len(skill_registry)} skills from workspace")

    sandbox_manager_config = build_sandbox_manager_config(
        config.sandbox, config.workspace
    )
    _, python_packages, python_tools = collect_skill_packages(skill_registry)
    setup_command = build_setup_command(
        python_packages=python_packages,
        python_tools=python_tools,
        base_setup_command=config.sandbox.setup_command,
    )
    shared_executor = SandboxExecutor(
        config=sandbox_manager_config,
        setup_command=setup_command,
    )

    tool_registry.register(BashTool(executor=shared_executor))
    tool_registry.register(ReadFileTool(executor=shared_executor))
    tool_registry.register(WriteFileTool(executor=shared_executor))

    # Register interrupt tool for agent checkpointing
    from ash.tools.builtin.interrupt import InterruptTool

    tool_registry.register(InterruptTool())

    if config.brave_search and config.brave_search.api_key:
        search_cache = SearchCache(maxsize=100, ttl=900)
        fetch_cache = SearchCache(maxsize=50, ttl=1800)
        tool_registry.register(
            WebSearchTool(
                api_key=config.brave_search.api_key.get_secret_value(),
                executor=shared_executor,
                cache=search_cache,
            )
        )
        tool_registry.register(
            WebFetchTool(executor=shared_executor, cache=fetch_cache)
        )

    # Create unified graph store (replaces separate memory_manager + person_manager)
    graph_store: Store | None = None
    if not db:
        logger.info("Memory tools disabled: no database")
    elif not config.embeddings:
        logger.info("Memory tools disabled: [embeddings] not configured")
    else:
        try:
            embeddings_key = config.resolve_embeddings_api_key()
            if not embeddings_key:
                logger.info(
                    f"No API key for {config.embeddings.provider} embeddings, "
                    "memory features disabled"
                )
                raise ValueError("Embeddings API key required for memory")

            # Create registry with both embedding provider and Anthropic (for LLM verification)
            # Get Anthropic key from default model if it's anthropic, otherwise from provider config
            default_model = config.get_model("default")
            if default_model.provider == "anthropic":
                anthropic_key = config.resolve_api_key("default")
            else:
                anthropic_key = config._resolve_provider_api_key("anthropic")
            llm_registry = create_registry(
                anthropic_api_key=anthropic_key.get_secret_value()
                if anthropic_key
                else None,
                openai_api_key=embeddings_key.get_secret_value()
                if config.embeddings.provider == "openai"
                else None,
            )
            graph_store = await create_store(
                db=db,
                llm_registry=llm_registry,
                embedding_model=config.embeddings.model,
                embedding_provider=config.embeddings.provider,
                max_entries=config.memory.max_entries,
            )
            logger.debug("Store initialized")
        except ValueError as e:
            logger.debug(f"Memory disabled: {e}")
        except Exception:
            logger.warning("Failed to initialize graph store", exc_info=True)

    memory_extractor: MemoryExtractor | None = None
    if graph_store and config.memory.extraction_enabled:
        extraction_model_alias = config.memory.extraction_model or model_alias
        try:
            extraction_model_config = config.get_model(extraction_model_alias)
            extraction_api_key = config.resolve_api_key(extraction_model_alias)
            extraction_llm = create_llm_provider(
                extraction_model_config.provider,
                api_key=extraction_api_key.get_secret_value()
                if extraction_api_key
                else None,
            )
            memory_extractor = MemoryExtractor(
                llm=extraction_llm,
                model=extraction_model_config.model,
                confidence_threshold=config.memory.extraction_confidence_threshold,
            )
            logger.debug(
                "Memory extractor initialized (model=%s)",
                extraction_model_config.model,
            )
            graph_store.set_llm(extraction_llm, extraction_model_config.model)
        except Exception:
            logger.warning("Failed to initialize memory extractor", exc_info=True)

    tool_executor = ToolExecutor(tool_registry)
    logger.info(f"Registered {len(tool_registry)} tools")

    agent_registry = AgentRegistry()
    register_builtin_agents(agent_registry)
    logger.info(f"Registered {len(agent_registry)} built-in agents")

    agent_executor = AgentExecutor(llm, tool_executor, config)
    tool_registry.register(
        UseAgentTool(
            agent_registry, agent_executor, skill_registry, config, voice=workspace.soul
        )
    )
    tool_registry.register(
        UseSkillTool(skill_registry, agent_executor, config, voice=workspace.soul)
    )

    runtime = RuntimeInfo.from_environment(
        model=model_config.model,
        provider=model_config.provider,
        timezone=config.timezone,
    )

    prompt_builder = SystemPromptBuilder(
        workspace=workspace,
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        config=config,
        agent_registry=agent_registry,
    )

    thinking_config = (
        resolve_thinking(model_config.thinking) if model_config.thinking else None
    )

    agent = Agent(
        llm=llm,
        tool_executor=tool_executor,
        prompt_builder=prompt_builder,
        runtime=runtime,
        memory_extractor=memory_extractor,
        graph_store=graph_store,
        config=AgentConfig(
            model=model_config.model,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            thinking=thinking_config,
            context_token_budget=config.memory.context_token_budget,
            recency_window=config.memory.recency_window,
            system_prompt_buffer=config.memory.system_prompt_buffer,
            compaction_enabled=config.memory.compaction_enabled,
            compaction_reserve_tokens=config.memory.compaction_reserve_tokens,
            compaction_keep_recent_tokens=config.memory.compaction_keep_recent_tokens,
            compaction_summary_max_tokens=config.memory.compaction_summary_max_tokens,
            extraction_enabled=config.memory.extraction_enabled,
            extraction_min_message_length=config.memory.extraction_min_message_length,
            extraction_debounce_seconds=config.memory.extraction_debounce_seconds,
            extraction_confidence_threshold=config.memory.extraction_confidence_threshold,
        ),
    )

    return AgentComponents(
        agent=agent,
        llm=llm,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        prompt_builder=prompt_builder,
        skill_registry=skill_registry,
        memory_manager=graph_store,
        person_manager=graph_store,
        memory_extractor=memory_extractor,
        sandbox_executor=shared_executor,
        agent_registry=agent_registry,
    )
