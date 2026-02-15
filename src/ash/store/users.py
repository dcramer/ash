"""User and chat CRUD mixin for Store (SQLite-backed)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import text

from ash.store.types import ChatEntry, UserEntry, _parse_datetime

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


def _row_to_user(row) -> UserEntry:
    """Convert a SQLite row to a UserEntry."""
    return UserEntry(
        id=row.id,
        version=row.version,
        provider=row.provider,
        provider_id=row.provider_id,
        username=row.username,
        display_name=row.display_name,
        person_id=row.person_id,
        created_at=_parse_datetime(row.created_at),
        updated_at=_parse_datetime(row.updated_at),
        metadata=json.loads(row.metadata) if row.metadata else None,
    )


def _row_to_chat(row) -> ChatEntry:
    """Convert a SQLite row to a ChatEntry."""
    return ChatEntry(
        id=row.id,
        version=row.version,
        provider=row.provider,
        provider_id=row.provider_id,
        chat_type=row.chat_type,
        title=row.title,
        created_at=_parse_datetime(row.created_at),
        updated_at=_parse_datetime(row.updated_at),
        metadata=json.loads(row.metadata) if row.metadata else None,
    )


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
        now = datetime.now(UTC)
        now_iso = now.isoformat()

        async with self._db.session() as session:
            result = await session.execute(
                text(
                    "SELECT * FROM users WHERE provider = :provider AND provider_id = :provider_id"
                ),
                {"provider": provider, "provider_id": provider_id},
            )
            row = result.fetchone()

            if row:
                user = _row_to_user(row)
                changed = False
                updates = []
                params: dict = {
                    "provider": provider,
                    "provider_id": provider_id,
                }
                if username is not None and user.username != username:
                    updates.append("username = :username")
                    params["username"] = username
                    user.username = username
                    changed = True
                if display_name is not None and user.display_name != display_name:
                    updates.append("display_name = :display_name")
                    params["display_name"] = display_name
                    user.display_name = display_name
                    changed = True
                if person_id is not None and user.person_id != person_id:
                    updates.append("person_id = :person_id")
                    params["person_id"] = person_id
                    user.person_id = person_id
                    changed = True
                if changed:
                    updates.append("updated_at = :updated_at")
                    params["updated_at"] = now_iso
                    user.updated_at = now
                    await session.execute(
                        text(
                            f"UPDATE users SET {', '.join(updates)} WHERE provider = :provider AND provider_id = :provider_id"
                        ),
                        params,
                    )
                return user

            # Create new user
            user_id = str(uuid.uuid4())
            await session.execute(
                text("""
                    INSERT INTO users (id, version, provider, provider_id, username,
                        display_name, person_id, created_at, updated_at)
                    VALUES (:id, 1, :provider, :provider_id, :username,
                        :display_name, :person_id, :created_at, :updated_at)
                """),
                {
                    "id": user_id,
                    "provider": provider,
                    "provider_id": provider_id,
                    "username": username,
                    "display_name": display_name,
                    "person_id": person_id,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                },
            )

        entry = UserEntry(
            id=user_id,
            version=1,
            provider=provider,
            provider_id=provider_id,
            username=username,
            display_name=display_name,
            person_id=person_id,
            created_at=now,
            updated_at=now,
        )
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
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT * FROM users WHERE id = :id"),
                {"id": user_id},
            )
            row = result.fetchone()
            return _row_to_user(row) if row else None

    async def find_user_by_provider(
        self: Store, provider: str, provider_id: str
    ) -> UserEntry | None:
        async with self._db.session() as session:
            result = await session.execute(
                text(
                    "SELECT * FROM users WHERE provider = :provider AND provider_id = :provider_id"
                ),
                {"provider": provider, "provider_id": provider_id},
            )
            row = result.fetchone()
            return _row_to_user(row) if row else None

    async def list_users(self: Store) -> list[UserEntry]:
        async with self._db.session() as session:
            result = await session.execute(text("SELECT * FROM users"))
            return [_row_to_user(row) for row in result.fetchall()]

    async def find_person_ids_for_username(self: Store, username: str) -> set[str]:
        username_clean = username.lstrip("@").lower()

        async with self._db.session() as session:
            # Look up user by username -> person_id
            result = await session.execute(
                text(
                    "SELECT person_id FROM users WHERE LOWER(username) = :username AND person_id IS NOT NULL"
                ),
                {"username": username_clean},
            )
            person_ids: set[str] = set()
            for row in result.fetchall():
                pid = row[0]
                # Follow merge chains
                person = await self.get_person(pid)
                if person and person.merged_into:
                    primary = await self._follow_merge_chain(person)
                    person_ids.add(primary.id)
                elif person:
                    person_ids.add(person.id)

            if person_ids:
                return person_ids

        # Fallback: search people by name/alias matching the username
        async with self._db.session() as session:
            # Check name match
            result = await session.execute(
                text(
                    "SELECT id FROM people WHERE LOWER(name) = :name AND merged_into IS NULL"
                ),
                {"name": username_clean},
            )
            for row in result.fetchall():
                person_ids.add(row[0])

            # Check alias match
            result = await session.execute(
                text("""
                    SELECT pa.person_id FROM person_aliases pa
                    JOIN people p ON p.id = pa.person_id
                    WHERE LOWER(pa.value) = :val AND p.merged_into IS NULL
                """),
                {"val": username_clean},
            )
            for row in result.fetchall():
                person_ids.add(row[0])

        # Follow merge chains for any found
        resolved: set[str] = set()
        for pid in person_ids:
            person = await self.get_person(pid)
            if person and person.merged_into:
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
        now_iso = now.isoformat()

        async with self._db.session() as session:
            result = await session.execute(
                text(
                    "SELECT * FROM chats WHERE provider = :provider AND provider_id = :provider_id"
                ),
                {"provider": provider, "provider_id": provider_id},
            )
            row = result.fetchone()

            if row:
                chat = _row_to_chat(row)
                changed = False
                updates = []
                params: dict = {
                    "provider": provider,
                    "provider_id": provider_id,
                }
                if chat_type is not None and chat.chat_type != chat_type:
                    updates.append("chat_type = :chat_type")
                    params["chat_type"] = chat_type
                    chat.chat_type = chat_type
                    changed = True
                if title is not None and chat.title != title:
                    updates.append("title = :title")
                    params["title"] = title
                    chat.title = title
                    changed = True
                if changed:
                    updates.append("updated_at = :updated_at")
                    params["updated_at"] = now_iso
                    chat.updated_at = now
                    await session.execute(
                        text(
                            f"UPDATE chats SET {', '.join(updates)} WHERE provider = :provider AND provider_id = :provider_id"
                        ),
                        params,
                    )
                return chat

            # Create new chat
            chat_id = str(uuid.uuid4())
            await session.execute(
                text("""
                    INSERT INTO chats (id, version, provider, provider_id, chat_type,
                        title, created_at, updated_at)
                    VALUES (:id, 1, :provider, :provider_id, :chat_type,
                        :title, :created_at, :updated_at)
                """),
                {
                    "id": chat_id,
                    "provider": provider,
                    "provider_id": provider_id,
                    "chat_type": chat_type,
                    "title": title,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                },
            )

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
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT * FROM chats WHERE id = :id"),
                {"id": chat_id},
            )
            row = result.fetchone()
            return _row_to_chat(row) if row else None

    async def find_chat_by_provider(
        self: Store, provider: str, provider_id: str
    ) -> ChatEntry | None:
        async with self._db.session() as session:
            result = await session.execute(
                text(
                    "SELECT * FROM chats WHERE provider = :provider AND provider_id = :provider_id"
                ),
                {"provider": provider, "provider_id": provider_id},
            )
            row = result.fetchone()
            return _row_to_chat(row) if row else None

    async def users_for_person(self: Store, person_id: str) -> list[UserEntry]:
        """Return users linked to a person via users.person_id."""
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT * FROM users WHERE person_id = :pid"),
                {"pid": person_id},
            )
            return [_row_to_user(row) for row in result.fetchall()]

    async def list_chats(self: Store) -> list[ChatEntry]:
        async with self._db.session() as session:
            result = await session.execute(text("SELECT * FROM chats"))
            return [_row_to_chat(row) for row in result.fetchall()]
