"""Memory RPC method handlers."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ash.store.visibility import (
    is_private_sourced_outside_current_chat,
)

if TYPE_CHECKING:
    from ash.memory.extractor import MemoryExtractor
    from ash.rpc.server import RPCServer
    from ash.store.store import Store

logger = logging.getLogger(__name__)


def register_memory_methods(
    server: "RPCServer",
    memory_manager: "Store",
    person_manager: "Store | None" = None,
    memory_extractor: "MemoryExtractor | None" = None,
    sessions_path: Path | None = None,
) -> None:
    """Register memory-related RPC methods.

    Args:
        server: RPC server to register methods on.
        memory_manager: Store instance.
        person_manager: Store instance (for subject resolution).
        memory_extractor: Optional extractor for fact classification/extraction.
        sessions_path: Path to sessions directory (for memory.extract).
    """

    async def _build_username_lookup() -> dict[str, str]:
        """Build username → display name lookup from people records."""
        if not person_manager:
            return {}
        try:
            people = await person_manager.list_people()
            lookup: dict[str, str] = {}
            for p in people:
                if p.name:
                    lookup[p.name.lower()] = p.name
                    for alias in p.aliases:
                        lookup[alias.value.lower()] = p.name
            return lookup
        except Exception:
            logger.warning("username_lookup_build_failed", exc_info=True)
            return {}

    def _resolve_source(
        source_username: str | None, lookup: dict[str, str]
    ) -> str | None:
        """Resolve a source_username to a display name."""
        if not source_username:
            return None
        return lookup.get(source_username.lower()) or source_username

    async def _resolve_subject_names(
        person_ids: list[str] | None, people_by_id: dict[str, Any] | None = None
    ) -> list[str]:
        """Resolve subject person IDs to display names."""
        if not person_ids or not person_manager:
            return []
        names: list[str] = []
        for pid in person_ids:
            if people_by_id and pid in people_by_id:
                names.append(people_by_id[pid].name)
            else:
                try:
                    person = await person_manager.get_person(pid)
                    if person:
                        names.append(person.name)
                    else:
                        names.append(pid[:8])
                except Exception:
                    names.append(pid[:8])
        return names

    async def _ensure_speaker(
        user_id: str | None,
        source_username: str | None,
        source_display_name: str | None,
    ) -> tuple[str | None, list[str]]:
        """Ensure self-person exists and build owner_names list.

        Returns:
            (speaker_person_id, owner_names)
        """
        from ash.memory.processing import enrich_owner_names, ensure_self_person

        if not person_manager or not user_id:
            return None, []

        owner_names: list[str] = []
        if source_username:
            owner_names.append(source_username)
        if source_display_name and source_display_name not in owner_names:
            owner_names.append(source_display_name)

        speaker_person_id: str | None = None
        if source_username or source_display_name:
            effective_display = source_display_name or source_username
            assert effective_display is not None
            speaker_person_id = await ensure_self_person(
                person_manager,
                user_id,
                source_username or "",
                effective_display,
            )

        if speaker_person_id:
            await enrich_owner_names(person_manager, owner_names, speaker_person_id)

        return speaker_person_id, owner_names

    def _resolve_chat_type(
        chat_type: str | None, provider: str | None, chat_id: str | None
    ) -> str | None:
        """Resolve chat_type from params or graph lookup."""
        if chat_type:
            return chat_type
        if provider and chat_id:
            chat_entry = memory_manager.graph.find_chat_by_provider(provider, chat_id)
            if chat_entry:
                return chat_entry.chat_type
        return None

    def _resolve_graph_chat_id(provider: str | None, chat_id: str | None) -> str | None:
        """Resolve provider chat ID to graph chat node ID."""
        if not provider or not chat_id:
            return None
        chat_entry = memory_manager.graph.find_chat_by_provider(provider, chat_id)
        return chat_entry.id if chat_entry else None

    def _visible_in_chat_context(
        memory_id: str,
        chat_type: str | None,
        chat_provider_id: str | None,
    ) -> bool:
        """Chat context visibility gate shared by list/search RPCs."""
        graph = memory_manager.graph
        if chat_type in ("group", "supergroup"):
            # In groups, only memories proven to come from non-private chats are visible.
            return not is_private_sourced_outside_current_chat(
                graph,
                memory_id,
                None,
            )
        if chat_type == "private":
            return not is_private_sourced_outside_current_chat(
                graph,
                memory_id,
                chat_provider_id,
            )
        return True

    async def memory_search(params: dict[str, Any]) -> list[dict[str, Any]]:
        """Search memories using semantic search.

        Params:
            query: Search query string (required)
            limit: Maximum results (default 10)
            user_id: Filter to user's personal memories
            chat_id: Include group memories for this chat
            chat_type: Current chat type (for privacy filtering)
            this_chat: If True, only return memories learned in the current chat

        Privacy behavior:
            - Group chats: DM-sourced and unknown-provenance memories are excluded
            - Private chats: DM-sourced memories are only shown when sourced from
              the same DM chat_id; unknown-provenance memories are excluded
        """
        query = params.get("query")
        if not query:
            raise ValueError("query is required")

        limit = params.get("limit", 10)
        user_id = params.get("user_id")
        chat_id = params.get("chat_id")
        chat_type = _resolve_chat_type(
            params.get("chat_type"), params.get("provider"), chat_id
        )

        # Resolve graph_chat_id for --this-chat filtering
        learned_in_chat_id: str | None = None
        if params.get("this_chat"):
            provider = params.get("provider")
            if provider and chat_id:
                chat_entry = memory_manager.graph.find_chat_by_provider(
                    provider, chat_id
                )
                if chat_entry:
                    learned_in_chat_id = chat_entry.id
            if learned_in_chat_id is None:
                return []

        results = await memory_manager.search(
            query=query,
            limit=limit,
            owner_user_id=user_id,
            chat_id=chat_id,
            learned_in_chat_id=learned_in_chat_id,
        )

        # Filter by chat context provenance policy
        results = [
            r for r in results if _visible_in_chat_context(r.id, chat_type, chat_id)
        ]

        lookup = await _build_username_lookup()

        output = []
        for r in results:
            source_username = (r.metadata or {}).get("source_username")
            entry: dict[str, Any] = {
                "id": r.id,
                "content": r.content,
                "similarity": r.similarity,
                "metadata": r.metadata,
            }
            if source_username:
                entry["source"] = _resolve_source(source_username, lookup)
            output.append(entry)

        return output

    async def memory_add(params: dict[str, Any]) -> dict[str, Any]:
        """Add a memory entry with optional LLM classification.

        When a memory extractor is available and subjects are not explicitly
        provided, the fact is classified via LLM for subject linking, type
        classification, sensitivity, and portable flags. The result is then
        routed through the full processing pipeline (hearsay supersession,
        relationship extraction, etc.).

        Params:
            content: Memory content (required)
            source: Source label (default "agent")
            expires_days: Days until expiration (optional)
            user_id: Owner user ID (for personal memories)
            chat_id: Chat ID (for group memories when shared=True)
            shared: If True and chat_id set, creates group memory (default False)
            subjects: List of subject person references
            source_username: Who provided this fact (username/handle)
            source_display_name: Display name of the source user
            assertion_kind: Optional structured assertion kind
            assertion_subject_ids: Optional canonical person IDs for assertion subjects
            speaker_person_id: Optional canonical speaker person ID
            predicates: Optional structured predicates for assertion metadata
        """
        from ash.memory.processing import process_extracted_facts
        from ash.store.types import (
            AssertionEnvelope,
            AssertionKind,
            AssertionPredicate,
            ExtractedFact,
            MemoryType,
        )

        content = params.get("content")
        if not content:
            raise ValueError("content is required")

        source = params.get("source", "agent")
        user_id = params.get("user_id")
        chat_id = params.get("chat_id")
        shared = params.get("shared", False)
        subjects = params.get("subjects", [])
        source_username = params.get("source_username") or params.get("source_user_id")
        source_display_name = params.get("source_display_name") or params.get(
            "source_user_name"
        )

        assertion: AssertionEnvelope | None = None
        assertion_kind_raw = params.get("assertion_kind")
        if assertion_kind_raw is not None:
            try:
                assertion_kind = AssertionKind(assertion_kind_raw)
            except ValueError as exc:
                raise ValueError(
                    "assertion_kind must be one of: "
                    "self_fact, person_fact, relationship_fact, group_fact, context_fact"
                ) from exc

            predicates_raw = params.get("predicates") or []
            if not isinstance(predicates_raw, list):
                raise ValueError("predicates must be a list of predicate objects")

            predicates: list[AssertionPredicate] = []
            for raw in predicates_raw:
                predicates.append(AssertionPredicate.model_validate(raw))

            assertion = AssertionEnvelope(
                assertion_kind=assertion_kind,
                subjects=params.get("assertion_subject_ids") or [],
                speaker_person_id=params.get("speaker_person_id"),
                predicates=predicates,
                confidence=1.0,
            )

        # Ensure self-person and build owner names
        speaker_person_id, owner_names = await _ensure_speaker(
            user_id, source_username, source_display_name
        )

        # Classify the fact via LLM if extractor available and no explicit subjects
        classified = None
        if memory_extractor and not subjects:
            classified = await memory_extractor.classify_fact(content)

        # Build ExtractedFact: explicit params > classified > defaults
        fact = ExtractedFact(
            content=content,
            subjects=subjects
            if subjects
            else (classified.subjects if classified else []),
            shared=shared if shared else (classified.shared if classified else False),
            confidence=1.0,
            memory_type=(
                classified.memory_type if classified else MemoryType.KNOWLEDGE
            ),
            speaker=source_username,
            sensitivity=(classified.sensitivity if classified else None),
            portable=(classified.portable if classified else True),
            assertion=assertion,
        )

        # Resolve graph_chat_id for LEARNED_IN edges
        provider = params.get("provider")
        chat_type = _resolve_chat_type(
            params.get("chat_type"),
            params.get("provider"),
            chat_id,
        )
        graph_chat_id = _resolve_graph_chat_id(provider, chat_id)

        stored_ids = await process_extracted_facts(
            facts=[fact],
            store=memory_manager,
            user_id=user_id or "",
            chat_id=chat_id,
            speaker_username=source_username,
            speaker_display_name=source_display_name,
            speaker_person_id=speaker_person_id,
            owner_names=owner_names,
            source=source,
            confidence_threshold=0.0,  # Always store agent-provided facts
            graph_chat_id=graph_chat_id,
            chat_type=chat_type,
        )

        if stored_ids:
            return {"id": stored_ids[0]}

        # Fallback: store directly if pipeline returned nothing
        memory = await memory_manager.add_memory(
            content=content,
            source=source,
            owner_user_id=user_id if not shared else None,
            chat_id=chat_id if shared else None,
            source_username=source_username,
            source_display_name=source_display_name,
            graph_chat_id=graph_chat_id,
            assertion=assertion,
        )
        return {"id": memory.id}

    async def _extract_and_store_from_messages(
        *,
        llm_messages: list[Any],
        user_id: str | None,
        provider: str | None,
        chat_id: str | None,
        chat_type: str | None,
        shared: bool,
        source_username: str | None,
        source_display_name: str | None,
        source_user_id: str | None,
    ) -> dict[str, Any]:
        """Run extraction and persistence from an explicit message payload."""
        from datetime import UTC, datetime

        from ash.memory.extractor import SpeakerInfo
        from ash.memory.processing import process_extracted_facts

        if not memory_extractor:
            raise ValueError("Memory extractor not available")

        if not llm_messages:
            return {"stored": 0}

        effective_user_id = source_user_id or user_id or ""

        # Ensure self-person and build owner names
        speaker_person_id, owner_names = await _ensure_speaker(
            source_user_id or user_id,
            source_username,
            source_display_name,
        )

        speaker_info = SpeakerInfo(
            user_id=source_user_id or user_id,
            username=source_username,
            display_name=source_display_name,
        )

        facts = await memory_extractor.extract_from_conversation(
            messages=llm_messages,
            owner_names=owner_names if owner_names else None,
            speaker_info=speaker_info,
            current_datetime=datetime.now(UTC),
        )

        if not facts:
            return {"stored": 0}

        if shared:
            for fact in facts:
                fact.shared = True

        graph_chat_id = _resolve_graph_chat_id(provider, chat_id)

        stored_ids = await process_extracted_facts(
            facts=facts,
            store=memory_manager,
            user_id=effective_user_id,
            chat_id=chat_id,
            speaker_username=source_username,
            speaker_display_name=source_display_name,
            speaker_person_id=speaker_person_id,
            owner_names=owner_names,
            source="agent",
            confidence_threshold=0.0,  # Explicit remember request path
            graph_chat_id=graph_chat_id,
            chat_type=chat_type,
        )

        return {"stored": len(stored_ids)}

    async def memory_extract_from_messages(params: dict[str, Any]) -> dict[str, Any]:
        """Extract memories from explicit in-request messages.

        This avoids session-file coupling and provides deterministic harness behavior.

        Params:
            messages: List[{role, content, id?, user_id?, username?, display_name?}]
            provider: Provider name (required)
            user_id: Owner user ID
            chat_id: Chat ID
            chat_type: Current chat type
            message_id: Optional target message id to resolve speaker metadata
            shared: If True, force group memories
            source_username/source_display_name: Optional speaker overrides
        """
        from ash.llm.types import Message, Role

        if not memory_extractor:
            raise ValueError("Memory extractor not available")

        provider = params.get("provider")
        if not provider:
            raise ValueError("provider is required")

        raw_messages = params.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("messages must be a non-empty list")

        target_message_id = params.get("message_id")

        target_raw: dict[str, Any] | None = None
        if target_message_id:
            target_raw = next(
                (
                    m
                    for m in raw_messages
                    if isinstance(m, dict) and m.get("id") == target_message_id
                ),
                None,
            )

        if target_raw is None:
            user_rows = [
                m
                for m in raw_messages
                if isinstance(m, dict) and m.get("role") == "user"
            ]
            if user_rows:
                target_raw = user_rows[-1]
            elif raw_messages and isinstance(raw_messages[-1], dict):
                target_raw = raw_messages[-1]
            else:
                target_raw = {}

        llm_messages: list[Message] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                raise ValueError("each message must be an object")

            role_raw = item.get("role")
            if role_raw not in ("user", "assistant"):
                raise ValueError("message role must be 'user' or 'assistant'")

            content = item.get("content")
            if not isinstance(content, str):
                raise ValueError("message content must be a string")
            if not content.strip():
                continue

            role = Role.USER if role_raw == "user" else Role.ASSISTANT
            llm_messages.append(Message(role=role, content=content))

        source_username = (
            params.get("source_username")
            or params.get("source_user_id")
            or target_raw.get("username")
        )
        source_display_name = (
            params.get("source_display_name")
            or params.get("source_user_name")
            or target_raw.get("display_name")
        )
        source_user_id = target_raw.get("user_id") or params.get("user_id")

        chat_id = params.get("chat_id")
        chat_type = _resolve_chat_type(params.get("chat_type"), provider, chat_id)

        return await _extract_and_store_from_messages(
            llm_messages=llm_messages,
            user_id=params.get("user_id"),
            provider=provider,
            chat_id=chat_id,
            chat_type=chat_type,
            shared=params.get("shared", False),
            source_username=source_username,
            source_display_name=source_display_name,
            source_user_id=source_user_id,
        )

    async def memory_extract(params: dict[str, Any]) -> dict[str, Any]:
        """Extract memories from the triggering message using the full pipeline.

        Reads the message from the session by message_id, runs full LLM extraction,
        and processes through the complete pipeline (subject linking, hearsay
        supersession, relationship extraction, etc.).

        Params (all implicit via env):
            message_id: ID of the triggering message
            provider: Provider name (e.g., "telegram")
            user_id: Owner user ID
            chat_id: Chat ID
            shared: If True, create group memories (default False)
            source_username: Speaker's username
            source_display_name: Speaker's display name
        """
        from ash.sessions.reader import SessionReader
        from ash.sessions.types import MessageEntry, session_key

        if not memory_extractor:
            raise ValueError("Memory extractor not available")

        message_id = params.get("message_id")
        provider = params.get("provider")
        user_id = params.get("user_id")
        chat_id = params.get("chat_id")
        shared = params.get("shared", False)
        source_username = params.get("source_username")
        source_display_name = params.get("source_display_name")

        if not message_id:
            raise ValueError("message_id is required (set via ASH_MESSAGE_ID)")
        if not provider:
            raise ValueError("provider is required")

        # Build session path and reader
        effective_sessions_path = sessions_path
        if not effective_sessions_path:
            from ash.config.paths import get_sessions_path

            effective_sessions_path = get_sessions_path()

        key = session_key(provider, chat_id, user_id)
        session_dir = effective_sessions_path / key
        reader = SessionReader(session_dir)

        # Get the triggering message + surrounding context
        surrounding = await reader.get_messages_around(message_id, window=2)
        if not surrounding:
            return {"stored": 0, "error": "Message not found in session"}

        # Extract author info from the target message
        target_msg = next(
            (m for m in surrounding if m.id == message_id), surrounding[-1]
        )
        msg_username = target_msg.username or source_username
        msg_display_name = target_msg.display_name or source_display_name
        msg_user_id = target_msg.user_id or user_id

        # Convert MessageEntry objects to LLM Message objects
        from ash.llm.types import Message, Role

        llm_messages: list[Message] = []
        for entry in surrounding:
            if not isinstance(entry, MessageEntry):
                continue
            text = entry._extract_text_content()
            if not text.strip():
                continue
            role = Role.USER if entry.role == "user" else Role.ASSISTANT
            llm_messages.append(Message(role=role, content=text))

        if not llm_messages:
            return {"stored": 0}

        return await _extract_and_store_from_messages(
            llm_messages=llm_messages,
            user_id=user_id,
            provider=provider,
            chat_id=chat_id,
            chat_type=_resolve_chat_type(params.get("chat_type"), provider, chat_id),
            shared=shared,
            source_username=msg_username or source_username,
            source_display_name=msg_display_name or source_display_name,
            source_user_id=msg_user_id,
        )

    async def memory_list(params: dict[str, Any]) -> list[dict[str, Any]]:
        """List memory entries.

        Params:
            limit: Maximum results (default 20)
            include_expired: Include expired entries (default False)
            user_id: Filter to user's personal memories
            chat_id: Include group memories for this chat
            chat_type: Current chat type (for privacy filtering)
            this_chat: If True, only return memories learned in the current chat

        Privacy behavior mirrors memory.search.
        """
        limit = params.get("limit", 20)
        include_expired = params.get("include_expired", False)
        user_id = params.get("user_id")
        chat_id = params.get("chat_id")
        chat_type = params.get("chat_type")

        # Resolve graph_chat_id for --this-chat filtering
        learned_in_chat_id: str | None = None
        if params.get("this_chat"):
            provider = params.get("provider")
            if provider and chat_id:
                chat_entry = memory_manager.graph.find_chat_by_provider(
                    provider, chat_id
                )
                if chat_entry:
                    learned_in_chat_id = chat_entry.id
            if learned_in_chat_id is None:
                return []

        memories = await memory_manager.list_memories(
            limit=limit,
            include_expired=include_expired,
            owner_user_id=user_id,
            chat_id=chat_id,
            learned_in_chat_id=learned_in_chat_id,
        )

        memories = [
            m for m in memories if _visible_in_chat_context(m.id, chat_type, chat_id)
        ]

        lookup = await _build_username_lookup()

        # Build people_by_id for subject resolution
        people_by_id: dict[str, Any] = {}
        if person_manager:
            try:
                people = await person_manager.list_people()
                people_by_id = {p.id: p for p in people}
            except Exception:
                logger.warning("people_list_load_failed", exc_info=True)

        result = []
        for m in memories:
            from ash.graph.edges import get_subject_person_ids
            from ash.store.trust import classify_trust

            subject_pids = get_subject_person_ids(memory_manager._graph, m.id)
            about = await _resolve_subject_names(subject_pids, people_by_id)
            trust = classify_trust(memory_manager._graph, m.id)
            entry: dict[str, Any] = {
                "id": m.id,
                "content": m.content,
                "source": _resolve_source(m.source_username, lookup) or m.source,
                "memory_type": m.memory_type.value,
                "subject_person_ids": subject_pids,
                "about": about,
                "trust": trust,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "expires_at": m.expires_at.isoformat() if m.expires_at else None,
            }
            result.append(entry)

        return result

    async def memory_delete(params: dict[str, Any]) -> dict[str, Any]:
        """Delete a memory entry.

        Params:
            memory_id: Memory ID to delete (required)
            user_id: Requester's user ID (for ownership check)
            chat_id: Requester's chat ID (for group memory check)
        """
        memory_id = params.get("memory_id")
        if not memory_id:
            raise ValueError("memory_id is required")

        user_id = params.get("user_id")
        chat_id = params.get("chat_id")

        deleted = await memory_manager.delete_memory(
            memory_id,
            owner_user_id=user_id,
            chat_id=chat_id,
        )
        return {"deleted": deleted}

    async def memory_forget_person(params: dict[str, Any]) -> dict[str, Any]:
        """Archive all memories about a person.

        Params:
            person_id: Person ID to forget (required)
            delete_person_record: Also delete the person record (default False)
        """
        person_id = params.get("person_id")
        if not person_id:
            raise ValueError("person_id is required")

        delete_person_record = params.get("delete_person_record", False)

        archived_count = await memory_manager.forget_person(
            person_id=person_id,
            delete_person_record=delete_person_record,
        )
        return {"archived_count": archived_count}

    # Register handlers
    server.register("memory.search", memory_search)
    server.register("memory.add", memory_add)
    server.register("memory.extract_from_messages", memory_extract_from_messages)
    server.register("memory.extract", memory_extract)
    server.register("memory.list", memory_list)
    server.register("memory.delete", memory_delete)
    server.register("memory.forget_person", memory_forget_person)

    logger.debug("Registered memory RPC methods")
