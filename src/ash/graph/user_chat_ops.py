"""User and chat CRUD mixin for GraphStore."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ash.graph.types import ChatEntry, EdgeType, UserEntry

if TYPE_CHECKING:
    from ash.graph.index import GraphIndex
    from ash.graph.store import GraphStore

logger = logging.getLogger(__name__)


class UserChatOpsMixin:
    """User and chat node CRUD operations."""

    async def ensure_user(
        self: GraphStore,
        provider: str,
        provider_id: str,
        username: str | None = None,
        display_name: str | None = None,
        person_id: str | None = None,
    ) -> UserEntry:
        """Upsert a user node. Creates if not found, updates if changed."""
        async with self._user_write_lock:
            users = await self._ensure_users_loaded()
            now = datetime.now(UTC)

            for user in users:
                if user.provider == provider and user.provider_id == provider_id:
                    changed = False
                    if username is not None and user.username != username:
                        user.username = username
                        changed = True
                    if display_name is not None and user.display_name != display_name:
                        user.display_name = display_name
                        changed = True
                    if person_id is not None and user.person_id != person_id:
                        user.person_id = person_id
                        changed = True
                    if changed:
                        user.updated_at = now
                        await self._user_jsonl.rewrite(users)
                        self._invalidate_users_cache()
                    return user

            # Create new user
            entry = UserEntry(
                id=str(uuid.uuid4()),
                version=1,
                provider=provider,
                provider_id=provider_id,
                username=username,
                display_name=display_name,
                person_id=person_id,
                created_at=now,
                updated_at=now,
            )
            await self._user_jsonl.append(entry)
            self._invalidate_users_cache()
            logger.debug(
                "user_created",
                extra={
                    "user_id": entry.id,
                    "provider": provider,
                    "provider_id": provider_id,
                },
            )
            return entry

    async def get_user(self: GraphStore, user_id: str) -> UserEntry | None:
        users = await self._ensure_users_loaded()
        for u in users:
            if u.id == user_id:
                return u
        return None

    async def find_user_by_provider(
        self: GraphStore, provider: str, provider_id: str
    ) -> UserEntry | None:
        users = await self._ensure_users_loaded()
        for u in users:
            if u.provider == provider and u.provider_id == provider_id:
                return u
        return None

    async def list_users(self: GraphStore) -> list[UserEntry]:
        return await self._ensure_users_loaded()

    async def find_person_ids_for_username(self: GraphStore, username: str) -> set[str]:
        username_clean = username.lstrip("@").lower()

        # Fast path: use graph to resolve username -> user -> person via IS_PERSON
        graph = await self._ensure_graph_built()
        user_node_id = graph.resolve_user_by_username(username_clean)
        if user_node_id:
            person_ids = graph.neighbors(user_node_id, EdgeType.IS_PERSON, "outgoing")
            if person_ids:
                # Follow merge chains
                result: set[str] = set()
                for pid in person_ids:
                    person = await self.get_person(pid)
                    if person and person.merged_into:
                        primary = await self._follow_merge_chain(person)
                        result.add(primary.id)
                    elif person:
                        result.add(person.id)
                return result

        # Fallback: linear scan over people (handles cases where no user node exists)
        people = await self._ensure_people_loaded()
        matching: set[str] = set()
        for person in people:
            if person.matches_username(username_clean):
                if person.merged_into:
                    primary = await self._follow_merge_chain(person)
                    matching.add(primary.id)
                else:
                    matching.add(person.id)
        return matching

    async def ensure_chat(
        self: GraphStore,
        provider: str,
        provider_id: str,
        chat_type: str | None = None,
        title: str | None = None,
    ) -> ChatEntry:
        """Upsert a chat node. Creates if not found, updates if changed."""
        async with self._chat_write_lock:
            chats = await self._ensure_chats_loaded()
            now = datetime.now(UTC)

            for chat in chats:
                if chat.provider == provider and chat.provider_id == provider_id:
                    changed = False
                    if chat_type is not None and chat.chat_type != chat_type:
                        chat.chat_type = chat_type
                        changed = True
                    if title is not None and chat.title != title:
                        chat.title = title
                        changed = True
                    if changed:
                        chat.updated_at = now
                        await self._chat_jsonl.rewrite(chats)
                        self._invalidate_chats_cache()
                    return chat

            entry = ChatEntry(
                id=str(uuid.uuid4()),
                version=1,
                provider=provider,
                provider_id=provider_id,
                chat_type=chat_type,
                title=title,
                created_at=now,
                updated_at=now,
            )
            await self._chat_jsonl.append(entry)
            self._invalidate_chats_cache()
            logger.debug(
                "chat_created",
                extra={
                    "chat_id": entry.id,
                    "provider": provider,
                    "provider_id": provider_id,
                },
            )
            return entry

    async def get_chat(self: GraphStore, chat_id: str) -> ChatEntry | None:
        chats = await self._ensure_chats_loaded()
        for c in chats:
            if c.id == chat_id:
                return c
        return None

    async def find_chat_by_provider(
        self: GraphStore, provider: str, provider_id: str
    ) -> ChatEntry | None:
        chats = await self._ensure_chats_loaded()
        for c in chats:
            if c.provider == provider and c.provider_id == provider_id:
                return c
        return None

    async def list_chats(self: GraphStore) -> list[ChatEntry]:
        return await self._ensure_chats_loaded()

    async def get_graph(self: GraphStore) -> GraphIndex:
        """Get the graph index, rebuilding if needed."""
        return await self._ensure_graph_built()
