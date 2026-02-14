"""Filesystem and data auto-migration for the graph architecture.

Handles:
1. Filesystem restructure: move files from old layout to new layout
2. Data migration: extract User/Chat nodes from existing memory data
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import UTC, datetime

from ash.config.paths import get_ash_home, get_index_dir, get_users_jsonl_path
from ash.graph.types import ChatEntry, UserEntry
from ash.memory.file_store import FileMemoryStore
from ash.memory.jsonl import TypedJSONL
from ash.people.types import PersonEntry

logger = logging.getLogger(__name__)


def migrate_filesystem() -> bool:
    """Migrate filesystem from old layout to new layout.

    Moves:
    - graph/embeddings.jsonl -> index/embeddings.jsonl
    - data/memory.db -> index/vectors.db
    - data/skills/ -> skills/state/
    - Removes empty data/ directory

    Returns:
        True if any migration was performed.
    """
    ash_home = get_ash_home()
    migrated = False

    # Move embeddings from graph/ to index/
    old_embeddings = ash_home / "graph" / "embeddings.jsonl"
    new_embeddings = get_index_dir() / "embeddings.jsonl"
    if old_embeddings.exists() and not new_embeddings.exists():
        new_embeddings.parent.mkdir(parents=True, exist_ok=True)
        old_embeddings.rename(new_embeddings)
        logger.info("Migrated embeddings.jsonl from graph/ to index/")
        migrated = True

    # Move vector DB from data/ to index/
    old_db = ash_home / "data" / "memory.db"
    new_db = get_index_dir() / "vectors.db"
    if old_db.exists() and not new_db.exists():
        new_db.parent.mkdir(parents=True, exist_ok=True)
        old_db.rename(new_db)
        logger.info("Migrated memory.db from data/ to index/vectors.db")
        migrated = True

    # Move skill state from data/skills/ to skills/state/
    old_skills = ash_home / "data" / "skills"
    new_skills = ash_home / "skills" / "state"
    if old_skills.exists() and old_skills.is_dir() and not new_skills.exists():
        new_skills.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_skills), str(new_skills))
        logger.info("Migrated skill state from data/skills/ to skills/state/")
        migrated = True

    # Remove empty data/ directory
    data_dir = ash_home / "data"
    if data_dir.exists() and data_dir.is_dir():
        try:
            data_dir.rmdir()  # Only removes if empty
            logger.info("Removed empty data/ directory")
            migrated = True
        except OSError:
            # Not empty — other files remain, leave it
            pass

    return migrated


async def migrate_data_to_graph(
    memory_store: FileMemoryStore,
    user_jsonl: TypedJSONL[UserEntry],
    chat_jsonl: TypedJSONL[ChatEntry],
    people_jsonl: TypedJSONL[PersonEntry],
) -> tuple[int, int]:
    """Extract User and Chat nodes from existing memory data.

    On first run (users.jsonl missing), scans existing memories and people
    to populate user and chat JSONL files.

    Strategy:
    - Extract UserEntry from unique owner_user_id + source_username pairs
    - Extract ChatEntry from unique chat_id values
    - Link UserEntry.person_id to self-persons via alias matching

    Args:
        memory_store: Memory store to read existing memories.
        user_jsonl: JSONL store for user entries.
        chat_jsonl: JSONL store for chat entries.
        people_jsonl: JSONL store for person entries (for linking).

    Returns:
        Tuple of (users_created, chats_created).
    """
    # Skip if users.jsonl already exists (idempotent)
    if get_users_jsonl_path().exists():
        return 0, 0

    memories = await memory_store.get_all_memories()
    if not memories:
        return 0, 0

    now = datetime.now(UTC)

    # Collect unique user identities from memories
    # Key: provider_id (owner_user_id), Value: (username, display_name)
    user_identities: dict[str, tuple[str | None, str | None]] = {}
    chat_ids: set[str] = set()

    for memory in memories:
        if memory.archived_at:
            continue

        # Extract user identity from owner_user_id
        if memory.owner_user_id:
            existing = user_identities.get(memory.owner_user_id)
            username = memory.source_username
            display_name = memory.source_display_name
            if existing:
                # Merge: prefer non-None values
                username = username or existing[0]
                display_name = display_name or existing[1]
            user_identities[memory.owner_user_id] = (username, display_name)

        # Extract chat identity
        if memory.chat_id:
            chat_ids.add(memory.chat_id)

    # Load people for self-person linking
    people = await people_jsonl.load_all()
    self_persons: dict[str, str] = {}  # lowercase username/alias -> person_id
    for person in people:
        is_self = any(r.relationship == "self" for r in person.relationships)
        if is_self:
            # Index by name and aliases
            if person.name:
                self_persons[person.name.lower()] = person.id
            for alias in person.aliases:
                self_persons[alias.value.lower()] = person.id

    # Create UserEntry for each unique provider_id
    users_created = 0
    for provider_id, (username, display_name) in user_identities.items():
        # Try to link to self-person
        person_id = None
        if username:
            person_id = self_persons.get(username.lower())
        if not person_id and provider_id:
            person_id = self_persons.get(provider_id.lower())

        user = UserEntry(
            id=str(uuid.uuid4()),
            provider="telegram",  # Default; most common provider
            provider_id=provider_id,
            username=username,
            display_name=display_name,
            person_id=person_id,
            created_at=now,
            updated_at=now,
        )
        await user_jsonl.append(user)
        users_created += 1

    # Create ChatEntry for each unique chat_id
    chats_created = 0
    for chat_id in sorted(chat_ids):
        chat = ChatEntry(
            id=str(uuid.uuid4()),
            provider="telegram",  # Default
            provider_id=chat_id,
            created_at=now,
            updated_at=now,
        )
        await chat_jsonl.append(chat)
        chats_created += 1

    if users_created or chats_created:
        logger.info(
            "Data migration: created %d users, %d chats from existing memories",
            users_created,
            chats_created,
        )

    return users_created, chats_created
