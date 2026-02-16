"""User and chat CRUD mixin for Store (in-memory graph backed)."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ash.graph.edges import IS_PERSON, create_is_person_edge
from ash.store.types import ChatEntry, UserEntry

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class UserChatOpsMixin:
    """User and chat node CRUD operations."""

    async def ensure_user(
        self: Store,
        provider: str,
        provider_id: str,
        username: str | None = None,
        display_name: str | None = None,
        person_id: str | None = None,
    ) -> UserEntry:
        """Upsert a user node. Creates if not found, updates if changed."""
        from ash.graph.edges import get_person_for_user

        now = datetime.now(UTC)

        # Look for existing user by provider+provider_id (O(1) index lookup)
        user = self._graph.find_user_by_provider(provider, provider_id)
        if user:
            changed = False
            if username is not None and user.username != username:
                user.username = username
                changed = True
            if display_name is not None and user.display_name != display_name:
                user.display_name = display_name
                changed = True
            if person_id is not None:
                current_person = get_person_for_user(self._graph, user.id)
                if current_person != person_id:
                    # Remove old IS_PERSON edge if replacing
                    if current_person is not None:
                        old_edges = self._graph.get_outgoing(
                            user.id, edge_type=IS_PERSON
                        )
                        for edge in old_edges:
                            self._graph.remove_edge(edge.id)
                    # Create new IS_PERSON edge
                    self._graph.add_edge(create_is_person_edge(user.id, person_id))
                    changed = True
            if changed:
                user.updated_at = now
                await self._persistence.save_users(self._graph.users)
                await self._persistence.save_edges(self._graph.edges)
            return user

        # Create new user
        user_id = str(uuid.uuid4())
        entry = UserEntry(
            id=user_id,
            version=1,
            provider=provider,
            provider_id=provider_id,
            username=username,
            display_name=display_name,
            created_at=now,
            updated_at=now,
        )
        self._graph.add_user(entry)
        await self._persistence.save_users(self._graph.users)

        # Create IS_PERSON edge for new user
        if person_id is not None:
            self._graph.add_edge(create_is_person_edge(user_id, person_id))
            await self._persistence.save_edges(self._graph.edges)

        logger.debug(
            "user_created",
            extra={
                "user_id": entry.id,
                "provider": provider,
                "provider_id": provider_id,
            },
        )
        return entry

    async def get_user(self: Store, user_id: str) -> UserEntry | None:
        return self._graph.users.get(user_id)

    async def find_user_by_provider(
        self: Store, provider: str, provider_id: str
    ) -> UserEntry | None:
        return self._graph.find_user_by_provider(provider, provider_id)

    async def list_users(self: Store) -> list[UserEntry]:
        return list(self._graph.users.values())

    async def find_person_ids_for_username(self: Store, username: str) -> set[str]:
        from ash.graph.edges import get_merged_into, get_person_for_user

        username_clean = username.lstrip("@").lower()

        # Look up user by username -> person via IS_PERSON edge
        person_ids: set[str] = set()
        for user in self._graph.users.values():
            if user.username and user.username.lower() == username_clean:
                pid = get_person_for_user(self._graph, user.id)
                if pid:
                    person = self._graph.people.get(pid)
                    if person and get_merged_into(self._graph, person.id):
                        primary = await self._follow_merge_chain(person)
                        person_ids.add(primary.id)
                    elif person:
                        person_ids.add(person.id)

        if person_ids:
            return person_ids

        # Fallback: search people by name/alias matching the username
        for person in self._graph.people.values():
            if get_merged_into(self._graph, person.id) is not None:
                continue
            if person.name and person.name.lower() == username_clean:
                person_ids.add(person.id)
            for alias in person.aliases:
                if alias.value.lower() == username_clean:
                    person_ids.add(person.id)

        # Follow merge chains for any found
        resolved: set[str] = set()
        for pid in person_ids:
            person = self._graph.people.get(pid)
            if person and get_merged_into(self._graph, person.id):
                primary = await self._follow_merge_chain(person)
                resolved.add(primary.id)
            elif person:
                resolved.add(person.id)
        return resolved

    async def ensure_chat(
        self: Store,
        provider: str,
        provider_id: str,
        chat_type: str | None = None,
        title: str | None = None,
    ) -> ChatEntry:
        """Upsert a chat node. Creates if not found, updates if changed."""
        now = datetime.now(UTC)

        # Look for existing chat by provider+provider_id (O(1) index lookup)
        chat = self._graph.find_chat_by_provider(provider, provider_id)
        if chat:
            changed = False
            if chat_type is not None and chat.chat_type != chat_type:
                chat.chat_type = chat_type
                changed = True
            if title is not None and chat.title != title:
                chat.title = title
                changed = True
            if changed:
                chat.updated_at = now
                await self._persistence.save_chats(self._graph.chats)
            return chat

        # Create new chat
        chat_id = str(uuid.uuid4())
        entry = ChatEntry(
            id=chat_id,
            version=1,
            provider=provider,
            provider_id=provider_id,
            chat_type=chat_type,
            title=title,
            created_at=now,
            updated_at=now,
        )
        self._graph.add_chat(entry)
        await self._persistence.save_chats(self._graph.chats)

        logger.debug(
            "chat_created",
            extra={
                "chat_id": entry.id,
                "provider": provider,
                "provider_id": provider_id,
            },
        )
        return entry

    async def get_chat(self: Store, chat_id: str) -> ChatEntry | None:
        return self._graph.chats.get(chat_id)

    async def find_chat_by_provider(
        self: Store, provider: str, provider_id: str
    ) -> ChatEntry | None:
        return self._graph.find_chat_by_provider(provider, provider_id)

    async def users_for_person(self: Store, person_id: str) -> list[UserEntry]:
        """Return users linked to a person via IS_PERSON edges."""
        from ash.graph.edges import get_users_for_person

        user_ids = get_users_for_person(self._graph, person_id)
        return [self._graph.users[uid] for uid in user_ids if uid in self._graph.users]

    async def list_chats(self: Store) -> list[ChatEntry]:
        return list(self._graph.chats.values())
