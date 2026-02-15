"""Migrate graph data from JSONL files to SQLite tables.

One-time migration that reads existing JSONL files (memories, people,
users, chats) and inserts them into the new SQLite tables.  JSONL files
are renamed to .jsonl.bak after successful migration.

Idempotent: skips if tables already have data or JSONL files don't exist.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import text

from ash.config.paths import (
    get_chats_jsonl_path,
    get_embeddings_jsonl_path,
    get_memories_jsonl_path,
    get_people_jsonl_path,
    get_users_jsonl_path,
)
from ash.db.engine import Database

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


async def migrate_jsonl_to_sqlite(db: Database) -> bool:
    """Migrate JSONL graph data to SQLite tables.

    Returns True if migration was performed, False if skipped.
    """
    memories_path = get_memories_jsonl_path()
    people_path = get_people_jsonl_path()
    users_path = get_users_jsonl_path()
    chats_path = get_chats_jsonl_path()
    embeddings_path = get_embeddings_jsonl_path()

    # Check if any JSONL files exist with data
    has_jsonl = any(
        p.exists() and p.stat().st_size > 0
        for p in [memories_path, people_path, users_path, chats_path]
    )
    if not has_jsonl:
        return False

    # Check if tables already have data (skip if so)
    async with db.session() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM memories"))
        mem_count = result.scalar() or 0
        result = await session.execute(text("SELECT COUNT(*) FROM people"))
        ppl_count = result.scalar() or 0
        if mem_count > 0 or ppl_count > 0:
            logger.debug("SQLite tables already have data, skipping JSONL migration")
            return False

    # Load all JSONL data
    memories = _load_jsonl(memories_path)
    people = _load_jsonl(people_path)
    users = _load_jsonl(users_path)
    chats = _load_jsonl(chats_path)

    logger.info(
        "Migrating JSONL to SQLite",
        extra={
            "memories": len(memories),
            "people": len(people),
            "users": len(users),
            "chats": len(chats),
        },
    )

    async with db.session() as session:
        # Migrate people first (referenced by memories and users)
        for p in people:
            await session.execute(
                text("""
                    INSERT INTO people (id, version, created_by, name, merged_into,
                                        created_at, updated_at, metadata)
                    VALUES (:id, :version, :created_by, :name, :merged_into,
                            :created_at, :updated_at, :metadata)
                """),
                {
                    "id": p["id"],
                    "version": p.get("version", 1),
                    "created_by": p.get("created_by") or p.get("owner_user_id", ""),
                    "name": p.get("name", ""),
                    "merged_into": p.get("merged_into"),
                    "created_at": p.get("created_at", ""),
                    "updated_at": p.get("updated_at", ""),
                    "metadata": json.dumps(p["metadata"])
                    if p.get("metadata")
                    else None,
                },
            )
            # Migrate aliases
            raw_aliases = p.get("aliases") or []
            for alias in raw_aliases:
                if isinstance(alias, str):
                    value, added_by, created_at = alias, None, None
                else:
                    value = alias["value"]
                    added_by = alias.get("added_by")
                    created_at = alias.get("created_at")
                await session.execute(
                    text("""
                        INSERT INTO person_aliases (person_id, value, added_by, created_at)
                        VALUES (:person_id, :value, :added_by, :created_at)
                    """),
                    {
                        "person_id": p["id"],
                        "value": value,
                        "added_by": added_by,
                        "created_at": created_at,
                    },
                )
            # Migrate relationships
            raw_rels = p.get("relationships")
            if isinstance(raw_rels, list):
                for rel in raw_rels:
                    if isinstance(rel, dict):
                        await session.execute(
                            text("""
                                INSERT INTO person_relationships
                                    (person_id, relationship, stated_by, created_at)
                                VALUES (:person_id, :relationship, :stated_by, :created_at)
                            """),
                            {
                                "person_id": p["id"],
                                "relationship": rel["relationship"],
                                "stated_by": rel.get("stated_by"),
                                "created_at": rel.get("created_at"),
                            },
                        )
            else:
                # Old format: single relationship string
                old_rel = p.get("relationship") or p.get("relation")
                if old_rel:
                    await session.execute(
                        text("""
                            INSERT INTO person_relationships
                                (person_id, relationship, stated_by, created_at)
                            VALUES (:person_id, :relationship, :stated_by, :created_at)
                        """),
                        {
                            "person_id": p["id"],
                            "relationship": old_rel,
                            "stated_by": None,
                            "created_at": p.get("created_at"),
                        },
                    )

        # Migrate users
        for u in users:
            await session.execute(
                text("""
                    INSERT INTO users (id, version, provider, provider_id, username,
                                       display_name, person_id, created_at, updated_at, metadata)
                    VALUES (:id, :version, :provider, :provider_id, :username,
                            :display_name, :person_id, :created_at, :updated_at, :metadata)
                """),
                {
                    "id": u["id"],
                    "version": u.get("version", 1),
                    "provider": u.get("provider", ""),
                    "provider_id": u.get("provider_id", ""),
                    "username": u.get("username"),
                    "display_name": u.get("display_name"),
                    "person_id": u.get("person_id"),
                    "created_at": u.get("created_at", ""),
                    "updated_at": u.get("updated_at", ""),
                    "metadata": json.dumps(u["metadata"])
                    if u.get("metadata")
                    else None,
                },
            )

        # Migrate chats
        for c in chats:
            await session.execute(
                text("""
                    INSERT INTO chats (id, version, provider, provider_id, chat_type,
                                       title, created_at, updated_at, metadata)
                    VALUES (:id, :version, :provider, :provider_id, :chat_type,
                            :title, :created_at, :updated_at, :metadata)
                """),
                {
                    "id": c["id"],
                    "version": c.get("version", 1),
                    "provider": c.get("provider", ""),
                    "provider_id": c.get("provider_id", ""),
                    "chat_type": c.get("chat_type"),
                    "title": c.get("title"),
                    "created_at": c.get("created_at", ""),
                    "updated_at": c.get("updated_at", ""),
                    "metadata": json.dumps(c["metadata"])
                    if c.get("metadata")
                    else None,
                },
            )

        # Migrate memories
        for m in memories:
            await session.execute(
                text("""
                    INSERT INTO memories (id, version, content, memory_type, source,
                        owner_user_id, chat_id, source_username, source_display_name,
                        source_session_id, source_message_id, extraction_confidence,
                        sensitivity, portable, created_at, observed_at, expires_at,
                        superseded_at, superseded_by_id, archived_at, archive_reason,
                        metadata)
                    VALUES (:id, :version, :content, :memory_type, :source,
                        :owner_user_id, :chat_id, :source_username, :source_display_name,
                        :source_session_id, :source_message_id, :extraction_confidence,
                        :sensitivity, :portable, :created_at, :observed_at, :expires_at,
                        :superseded_at, :superseded_by_id, :archived_at, :archive_reason,
                        :metadata)
                """),
                {
                    "id": m["id"],
                    "version": m.get("version", 1),
                    "content": m.get("content", ""),
                    "memory_type": m.get("memory_type", "knowledge"),
                    "source": m.get("source", "user"),
                    "owner_user_id": m.get("owner_user_id"),
                    "chat_id": m.get("chat_id"),
                    "source_username": m.get("source_username")
                    or m.get("source_user_id"),
                    "source_display_name": m.get("source_display_name")
                    or m.get("source_user_name"),
                    "source_session_id": m.get("source_session_id"),
                    "source_message_id": m.get("source_message_id"),
                    "extraction_confidence": m.get("extraction_confidence"),
                    "sensitivity": m.get("sensitivity"),
                    "portable": 0 if m.get("portable") is False else 1,
                    "created_at": m.get("created_at", ""),
                    "observed_at": m.get("observed_at"),
                    "expires_at": m.get("expires_at"),
                    "superseded_at": m.get("superseded_at"),
                    "superseded_by_id": m.get("superseded_by_id"),
                    "archived_at": m.get("archived_at"),
                    "archive_reason": m.get("archive_reason"),
                    "metadata": json.dumps(m["metadata"])
                    if m.get("metadata")
                    else None,
                },
            )
            # Migrate memory_subjects
            for pid in m.get("subject_person_ids") or []:
                await session.execute(
                    text("""
                        INSERT OR IGNORE INTO memory_subjects (memory_id, person_id)
                        VALUES (:memory_id, :person_id)
                    """),
                    {"memory_id": m["id"], "person_id": pid},
                )

    # Rename JSONL files to .bak
    for path in [memories_path, people_path, users_path, chats_path]:
        if path.exists():
            bak = path.with_suffix(".jsonl.bak")
            path.rename(bak)
            logger.info("Renamed %s to %s", path, bak)

    # Also rename embeddings.jsonl since embeddings are in sqlite-vec
    if embeddings_path.exists():
        bak = embeddings_path.with_suffix(".jsonl.bak")
        embeddings_path.rename(bak)
        logger.info("Renamed %s to %s", embeddings_path, bak)

    logger.info(
        "JSONL to SQLite migration complete",
        extra={
            "memories": len(memories),
            "people": len(people),
            "users": len(users),
            "chats": len(chats),
        },
    )
    return True
