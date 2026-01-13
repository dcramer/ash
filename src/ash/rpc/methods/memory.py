"""Memory RPC method handlers."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.memory import MemoryManager
    from ash.rpc.server import RPCServer

logger = logging.getLogger(__name__)


def register_memory_methods(
    server: "RPCServer", memory_manager: "MemoryManager"
) -> None:
    """Register memory-related RPC methods.

    Args:
        server: RPC server to register methods on.
        memory_manager: Memory manager instance.
    """

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

        return [
            {
                "id": r.id,
                "content": r.content,
                "similarity": r.similarity,
                "metadata": r.metadata,
            }
            for r in results
        ]

    async def memory_add(params: dict[str, Any]) -> dict[str, Any]:
        """Add a memory entry.

        Params:
            content: Memory content (required)
            source: Source label (default "rpc")
            expires_days: Days until expiration (optional)
            user_id: Owner user ID
            chat_id: Chat ID for group memories
            subjects: List of subject person references
        """
        content = params.get("content")
        if not content:
            raise ValueError("content is required")

        source = params.get("source", "rpc")
        expires_days = params.get("expires_days")
        user_id = params.get("user_id")
        chat_id = params.get("chat_id")
        subjects = params.get("subjects", [])

        # Resolve subject person IDs if provided
        subject_person_ids: list[str] = []
        if subjects and user_id:
            for subject in subjects:
                try:
                    result = await memory_manager.resolve_or_create_person(
                        owner_user_id=user_id,
                        reference=subject,
                        content_hint=content,
                    )
                    subject_person_ids.append(result.person_id)
                except Exception:
                    logger.warning("Failed to resolve person: %s", subject)

        memory = await memory_manager.add_memory(
            content=content,
            source=source,
            expires_in_days=expires_days,
            owner_user_id=user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids if subject_person_ids else None,
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

        return [
            {
                "id": m.id,
                "content": m.content,
                "source": m.source,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "expires_at": m.expires_at.isoformat() if m.expires_at else None,
            }
            for m in memories
        ]

    # Register handlers
    server.register("memory.search", memory_search)
    server.register("memory.add", memory_add)
    server.register("memory.list", memory_list)

    logger.debug("Registered memory RPC methods")
