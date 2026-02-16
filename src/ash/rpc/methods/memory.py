"""Memory RPC method handlers."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.rpc.server import RPCServer
    from ash.store.store import Store

logger = logging.getLogger(__name__)


def register_memory_methods(
    server: "RPCServer",
    memory_manager: "Store",
    person_manager: "Store | None" = None,
) -> None:
    """Register memory-related RPC methods.

    Args:
        server: RPC server to register methods on.
        memory_manager: Store instance.
        person_manager: Store instance (for subject resolution).
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
            logger.warning("Failed to build username lookup", exc_info=True)
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

    async def memory_search(params: dict[str, Any]) -> list[dict[str, Any]]:
        """Search memories using semantic search.

        Params:
            query: Search query string (required)
            limit: Maximum results (default 10)
            user_id: Filter to user's personal memories
            chat_id: Include group memories for this chat
        """
        query = params.get("query")
        if not query:
            raise ValueError("query is required")

        limit = params.get("limit", 10)
        user_id = params.get("user_id")
        chat_id = params.get("chat_id")

        results = await memory_manager.search(
            query=query,
            limit=limit,
            owner_user_id=user_id,
            chat_id=chat_id,
        )

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
        """Add a memory entry.

        Memory scoping:
        - Personal: user_id set, chat_id not set (or shared=False)
        - Group: chat_id set with shared=True (user_id becomes None)

        Params:
            content: Memory content (required)
            source: Source label (default "rpc")
            expires_days: Days until expiration (optional)
            user_id: Owner user ID (for personal memories)
            chat_id: Chat ID (for group memories when shared=True)
            shared: If True and chat_id set, creates group memory (default False)
            subjects: List of subject person references
            source_username: Who provided this fact (username/handle)
            source_display_name: Display name of the source user
        """
        content = params.get("content")
        if not content:
            raise ValueError("content is required")

        source = params.get("source", "rpc")
        expires_days = params.get("expires_days")
        user_id = params.get("user_id")
        chat_id = params.get("chat_id")
        shared = params.get("shared", False)
        subjects = params.get("subjects", [])
        # Accept both old and new param names for API backward compat
        source_username = params.get("source_username") or params.get("source_user_id")
        source_display_name = params.get("source_display_name") or params.get(
            "source_user_name"
        )

        # Apply scoping rules:
        # - Group memory: shared=True with chat_id -> owner_user_id=None, chat_id=chat_id
        # - Personal memory: everything else -> owner_user_id=user_id, chat_id=None
        if shared and chat_id:
            owner_user_id = None
            effective_chat_id = chat_id
        else:
            owner_user_id = user_id
            effective_chat_id = None

        # Resolve subject person IDs if provided
        subject_person_ids: list[str] = []
        if subjects and user_id and person_manager:
            for subject in subjects:
                try:
                    result = await person_manager.resolve_or_create_person(
                        created_by=user_id,
                        reference=subject,
                        content_hint=content,
                        relationship_stated_by=source_username,
                    )
                    subject_person_ids.append(result.person_id)
                except Exception:
                    logger.warning("Failed to resolve person: %s", subject)

        memory = await memory_manager.add_memory(
            content=content,
            source=source,
            expires_in_days=expires_days,
            owner_user_id=owner_user_id,
            chat_id=effective_chat_id,
            subject_person_ids=subject_person_ids if subject_person_ids else None,
            source_username=source_username,
            source_display_name=source_display_name,
        )

        return {"id": memory.id}

    async def memory_list(params: dict[str, Any]) -> list[dict[str, Any]]:
        """List memory entries.

        Params:
            limit: Maximum results (default 20)
            include_expired: Include expired entries (default False)
            user_id: Filter to user's personal memories
            chat_id: Include group memories for this chat
        """
        limit = params.get("limit", 20)
        include_expired = params.get("include_expired", False)
        user_id = params.get("user_id")
        chat_id = params.get("chat_id")

        memories = await memory_manager.list_memories(
            limit=limit,
            include_expired=include_expired,
            owner_user_id=user_id,
            chat_id=chat_id,
        )

        lookup = await _build_username_lookup()

        # Build people_by_id for subject resolution
        people_by_id: dict[str, Any] = {}
        if person_manager:
            try:
                people = await person_manager.list_people()
                people_by_id = {p.id: p for p in people}
            except Exception:
                logger.warning("Failed to load people for list", exc_info=True)

        result = []
        for m in memories:
            about = await _resolve_subject_names(m.subject_person_ids, people_by_id)
            entry: dict[str, Any] = {
                "id": m.id,
                "content": m.content,
                "source": _resolve_source(m.source_username, lookup) or m.source,
                "memory_type": m.memory_type.value,
                "subject_person_ids": m.subject_person_ids,
                "about": about,
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
    server.register("memory.list", memory_list)
    server.register("memory.delete", memory_delete)
    server.register("memory.forget_person", memory_forget_person)

    logger.debug("Registered memory RPC methods")
