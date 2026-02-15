"""Migration from SQLite to JSONL storage.

Exports existing memories and people from SQLite database to JSONL files,
preserving all data including embeddings.
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ash.config.paths import (
    get_database_path,
    get_embeddings_jsonl_path,
    get_memories_jsonl_path,
    get_people_jsonl_path,
)
from ash.memory.jsonl import EmbeddingJSONL, MemoryJSONL, PersonJSONL
from ash.store.types import EmbeddingRecord, MemoryEntry, MemoryType, PersonEntry

logger = logging.getLogger(__name__)


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    """Parse a datetime from SQLite (may be string or datetime)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    # SQLite stores as ISO format string
    try:
        # Try with timezone
        return datetime.fromisoformat(value)
    except ValueError:
        # Try without timezone, assume UTC
        try:
            return datetime.fromisoformat(value).replace(tzinfo=UTC)
        except ValueError:
            return None


async def migrate_db_to_jsonl(
    session: AsyncSession,
    memories_path: Path | None = None,
    people_path: Path | None = None,
    embeddings_path: Path | None = None,
) -> tuple[int, int]:
    """Migrate memories and people from SQLite to JSONL.

    Args:
        session: Database session with existing data.
        memories_path: Path for memories.jsonl (default: standard location).
        people_path: Path for people.jsonl (default: standard location).
        embeddings_path: Path for embeddings.jsonl (default: standard location).

    Returns:
        Tuple of (memories_migrated, people_migrated).
    """
    if memories_path is None:
        memories_path = get_memories_jsonl_path()
    if people_path is None:
        people_path = get_people_jsonl_path()
    if embeddings_path is None:
        embeddings_path = get_embeddings_jsonl_path()

    # Migrate memories (and extract embeddings)
    memories_count = await _migrate_memories(session, memories_path, embeddings_path)

    # Migrate people
    people_count = await _migrate_people(session, people_path)

    logger.info(
        "migration_complete",
        extra={
            "memories_migrated": memories_count,
            "people_migrated": people_count,
        },
    )

    return (memories_count, people_count)


async def _migrate_memories(
    session: AsyncSession,
    memories_path: Path,
    embeddings_path: Path | None = None,
) -> int:
    """Migrate memories from SQLite to JSONL.

    Embeddings are extracted to a separate embeddings.jsonl file
    rather than being stored inline in the memory entries.

    Args:
        session: Database session.
        memories_path: Path for memories.jsonl.
        embeddings_path: Path for embeddings.jsonl.

    Returns:
        Number of memories migrated.
    """
    # Query all memories with their embeddings
    result = await session.execute(
        text("""
            SELECT
                m.id,
                m.content,
                m.source,
                m.created_at,
                m.expires_at,
                m.metadata,
                m.owner_user_id,
                m.chat_id,
                m.subject_person_ids,
                m.superseded_at,
                m.superseded_by_id,
                e.embedding
            FROM memories m
            LEFT JOIN memory_embeddings e ON m.id = e.memory_id
            ORDER BY m.created_at ASC
        """)
    )

    rows = result.fetchall()
    if not rows:
        return 0

    jsonl = MemoryJSONL(memories_path)
    entries: list[MemoryEntry] = []

    for row in rows:
        # Parse embedding from bytes to base64
        embedding_b64 = ""
        if row[11]:
            embedding_b64 = base64.b64encode(row[11]).decode("ascii")

        # Parse subject_person_ids from JSON
        subject_ids = []
        if row[8]:
            import json

            try:
                subject_ids = json.loads(row[8])
            except json.JSONDecodeError:
                pass

        # Parse metadata from JSON
        metadata = None
        if row[5]:
            import json

            try:
                metadata = json.loads(row[5])
            except json.JSONDecodeError:
                pass

        # Infer memory type from content (defaults to KNOWLEDGE)
        memory_type = _infer_memory_type(row[1])

        # Parse datetime fields (SQLite returns strings)
        created_at = _parse_datetime(row[3])
        if created_at is None:
            # created_at is required, skip this entry if missing
            logger.warning(
                "Skipping memory with missing created_at", extra={"id": row[0]}
            )
            continue

        entry = MemoryEntry(
            id=row[0],
            version=1,
            content=row[1] or "",
            memory_type=memory_type,
            embedding=embedding_b64,
            created_at=created_at,
            observed_at=created_at,  # Set observed_at = created_at for historical entries
            owner_user_id=row[6],
            chat_id=row[7],
            subject_person_ids=subject_ids,
            source=row[2] or "user",
            expires_at=_parse_datetime(row[4]),
            superseded_at=_parse_datetime(row[9]),
            superseded_by_id=row[10],
            metadata=metadata,
        )
        entries.append(entry)

    # Extract embeddings to separate file and clear from entries
    embedding_count = 0
    if embeddings_path:
        embeddings_jsonl = EmbeddingJSONL(embeddings_path)
        for entry in entries:
            if entry.embedding:
                record = EmbeddingRecord(memory_id=entry.id, embedding=entry.embedding)
                await embeddings_jsonl.append(record)
                entry.embedding = ""
                embedding_count += 1

    # Write all entries
    await jsonl.rewrite(entries)

    logger.info(
        "memories_migrated",
        extra={
            "count": len(entries),
            "embeddings": embedding_count,
            "path": str(memories_path),
        },
    )

    return len(entries)


async def _migrate_people(
    session: AsyncSession,
    people_path: Path,
) -> int:
    """Migrate people from SQLite to JSONL.

    Args:
        session: Database session.
        people_path: Path for people.jsonl.

    Returns:
        Number of people migrated.
    """
    result = await session.execute(
        text("""
            SELECT
                id,
                owner_user_id,
                name,
                relation,
                aliases,
                metadata,
                created_at,
                updated_at
            FROM people
            ORDER BY created_at ASC
        """)
    )

    rows = result.fetchall()
    if not rows:
        return 0

    jsonl = PersonJSONL(people_path)
    entries: list[PersonEntry] = []

    for row in rows:
        # Parse aliases from JSON
        aliases = []
        if row[4]:
            import json

            try:
                aliases = json.loads(row[4])
            except json.JSONDecodeError:
                pass

        # Parse metadata from JSON
        metadata = None
        if row[5]:
            import json

            try:
                metadata = json.loads(row[5])
            except json.JSONDecodeError:
                pass

        # Parse datetime fields (SQLite returns strings)
        created_at = _parse_datetime(row[6])
        if created_at is None:
            logger.warning(
                "Skipping person with missing created_at", extra={"id": row[0]}
            )
            continue

        # Build new-format fields from old DB columns
        from ash.store.types import AliasEntry, RelationshipClaim

        alias_entries = [AliasEntry(value=a) for a in aliases]
        relationships = []
        if row[3]:
            relationships = [RelationshipClaim(relationship=row[3])]

        entry = PersonEntry(
            id=row[0],
            version=1,
            created_by=row[1] or "",
            name=row[2] or "",
            relationships=relationships,
            aliases=alias_entries,
            created_at=created_at,
            updated_at=_parse_datetime(row[7]),
            metadata=metadata,
        )
        entries.append(entry)

    # Write all entries
    await jsonl.rewrite(entries)

    logger.info(
        "people_migrated",
        extra={"count": len(entries), "path": str(people_path)},
    )

    return len(entries)


def _infer_memory_type(content: str) -> MemoryType:
    """Infer memory type from content for migration.

    This is a best-effort classification for historical data.

    Args:
        content: Memory content.

    Returns:
        Inferred memory type.
    """
    content_lower = content.lower() if content else ""

    # Preference indicators
    preference_words = [
        "prefer",
        "like",
        "love",
        "hate",
        "dislike",
        "favorite",
        "favourite",
    ]
    if any(word in content_lower for word in preference_words):
        return MemoryType.PREFERENCE

    # Identity indicators
    identity_words = [
        "i am",
        "i'm",
        "my name is",
        "i work",
        "i live",
        "born in",
        "years old",
    ]
    if any(word in content_lower for word in identity_words):
        return MemoryType.IDENTITY

    # Relationship indicators (mentions of other people)
    relationship_words = [
        "wife",
        "husband",
        "partner",
        "mother",
        "father",
        "sister",
        "brother",
        "friend",
        "boss",
        "colleague",
        "coworker",
    ]
    if any(word in content_lower for word in relationship_words):
        return MemoryType.RELATIONSHIP

    # Task indicators
    task_words = [
        "need to",
        "should",
        "must",
        "have to",
        "remind me",
        "don't forget",
        "remember to",
    ]
    if any(word in content_lower for word in task_words):
        return MemoryType.TASK

    # Event indicators
    event_words = [
        "yesterday",
        "last week",
        "last month",
        "happened",
        "went to",
        "attended",
    ]
    if any(word in content_lower for word in event_words):
        return MemoryType.EVENT

    # Default to knowledge
    return MemoryType.KNOWLEDGE


def needs_migration() -> bool:
    """Check if migration is needed.

    Migration is needed if:
    - SQLite database exists with memories
    - JSONL files don't exist

    Returns:
        True if migration is needed.
    """
    db_path = get_database_path()
    memories_path = get_memories_jsonl_path()

    # No migration needed if JSONL already exists
    if memories_path.exists():
        return False

    # Migration needed if database exists
    return db_path.exists()


async def migrate_to_graph_dir() -> bool:
    """Migrate from old scattered paths to ~/.ash/graph/.

    Old layout:
    - ~/.ash/memory/memories.jsonl
    - ~/.ash/memory/archive.jsonl
    - ~/.ash/people.jsonl

    New layout:
    - ~/.ash/graph/memories.jsonl  (active + archived combined, no embeddings)
    - ~/.ash/graph/people.jsonl
    - ~/.ash/graph/embeddings.jsonl (memory_id -> embedding pairs)

    Returns:
        True if migration was performed, False if skipped.
    """
    from ash.config.paths import get_ash_home

    graph_memories = get_memories_jsonl_path()  # Now points to graph/memories.jsonl

    # Idempotent: skip if new path already exists
    if graph_memories.exists():
        return False

    ash_home = get_ash_home()
    old_memories_path = ash_home / "memory" / "memories.jsonl"
    old_archive_path = ash_home / "memory" / "archive.jsonl"
    old_people_path = ash_home / "people.jsonl"

    # Nothing to migrate if old files don't exist
    if not old_memories_path.exists() and not old_people_path.exists():
        return False

    logger.info("Migrating to graph directory layout")

    # Ensure graph dir exists
    graph_memories.parent.mkdir(parents=True, exist_ok=True)

    # Load old memories
    active_entries: list[MemoryEntry] = []
    if old_memories_path.exists():
        old_mem_jsonl = MemoryJSONL(old_memories_path)
        active_entries = await old_mem_jsonl.load_all()

    # Load old archive
    archive_entries: list[MemoryEntry] = []
    if old_archive_path.exists():
        old_archive_jsonl = MemoryJSONL(old_archive_path)
        archive_entries = await old_archive_jsonl.load_all()

    # Extract embeddings and strip from entries
    embeddings_path = get_embeddings_jsonl_path()
    embeddings_jsonl = EmbeddingJSONL(embeddings_path)

    embedding_count = 0
    all_entries = active_entries + archive_entries
    for entry in all_entries:
        if entry.embedding:
            record = EmbeddingRecord(memory_id=entry.id, embedding=entry.embedding)
            await embeddings_jsonl.append(record)
            entry.embedding = ""
            embedding_count += 1

    # Write combined memories (active + archived)
    new_mem_jsonl = MemoryJSONL(graph_memories)
    await new_mem_jsonl.rewrite(all_entries)

    # Copy people
    if old_people_path.exists():
        new_people_path = get_people_jsonl_path()
        old_people_jsonl = PersonJSONL(old_people_path)
        people = await old_people_jsonl.load_all()
        new_people_jsonl = PersonJSONL(new_people_path)
        await new_people_jsonl.rewrite(people)

    logger.info(
        "graph_migration_complete",
        extra={
            "active_memories": len(active_entries),
            "archived_memories": len(archive_entries),
            "embeddings": embedding_count,
        },
    )

    return True


async def check_db_has_memories(session: AsyncSession) -> bool:
    """Check if database has any memories.

    Args:
        session: Database session.

    Returns:
        True if database has memories.
    """
    try:
        result = await session.execute(text("SELECT COUNT(*) FROM memories"))
        count = result.scalar() or 0
        return count > 0
    except Exception:
        return False
