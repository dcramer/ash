"""Database management commands.

Provides commands for:
- migrations: manage schema migrations
- export: export data from SQLite to JSONL
- import: import data from JSONL into SQLite
- backup: create SQLite backup using VACUUM INTO
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from sqlalchemy import text

from ash.cli.console import console, dim, error, success, warning
from ash.config.paths import get_ash_home


def register(app: typer.Typer) -> None:
    """Register the db command group."""
    db_app = typer.Typer(help="Database management commands")

    @db_app.command("migrate")
    def db_migrate(
        revision: Annotated[
            str,
            typer.Option(
                "--revision",
                "-r",
                help="Target revision",
            ),
        ] = "head",
    ) -> None:
        """Run database migrations."""
        console.print(f"[bold]Running migrations to {revision}...[/bold]")
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", revision],
            capture_output=False,
        )
        if result.returncode == 0:
            success("Migrations completed successfully")
        else:
            error("Migration failed")
            raise typer.Exit(1)

    @db_app.command("rollback")
    def db_rollback(
        revision: Annotated[
            str,
            typer.Option(
                "--revision",
                "-r",
                help="Target revision",
            ),
        ] = "-1",
    ) -> None:
        """Rollback database migrations."""
        console.print(f"[bold]Rolling back to {revision}...[/bold]")
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "downgrade", revision],
            capture_output=False,
        )
        if result.returncode == 0:
            success("Rollback completed successfully")
        else:
            error("Rollback failed")
            raise typer.Exit(1)

    @db_app.command("status")
    def db_status() -> None:
        """Show migration status."""
        console.print("[bold]Migration status:[/bold]")
        subprocess.run(
            [sys.executable, "-m", "alembic", "current"],
            capture_output=False,
        )
        console.print("\n[bold]Pending migrations:[/bold]")
        subprocess.run(
            [sys.executable, "-m", "alembic", "history", "--indicate-current"],
            capture_output=False,
        )

    @db_app.command("export")
    def db_export(
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                "-o",
                help="Output file path (defaults to stdout)",
            ),
        ] = None,
        format_type: Annotated[
            str,
            typer.Option(
                "--format",
                "-f",
                help="Output format: jsonl",
            ),
        ] = "jsonl",
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
    ) -> None:
        """Export all data from SQLite to JSONL format.

        Exports memories, people, users, and chats to a single JSONL file.
        Each line contains a record with a '_type' field indicating the entity type.

        Examples:
            ash db export > backup.jsonl
            ash db export --output backup.jsonl
        """
        if format_type != "jsonl":
            error(f"Unsupported format: {format_type}. Only 'jsonl' is supported.")
            raise typer.Exit(1)

        asyncio.run(_db_export(output, config_path))

    @db_app.command("import")
    def db_import(
        input_file: Annotated[
            Path,
            typer.Argument(help="Input JSONL file to import"),
        ],
        merge: Annotated[
            bool,
            typer.Option(
                "--merge",
                help="Merge with existing data (skip existing IDs). Default: replace.",
            ),
        ] = False,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                help="Skip confirmation for replace mode",
            ),
        ] = False,
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
    ) -> None:
        """Import data from JSONL into SQLite.

        Imports memories, people, users, and chats from a JSONL file
        previously created with 'ash db export'.

        By default, replaces existing data. Use --merge to skip existing IDs.

        Examples:
            ash db import backup.jsonl
            ash db import backup.jsonl --merge
        """
        if not input_file.exists():
            error(f"File not found: {input_file}")
            raise typer.Exit(1)

        asyncio.run(_db_import(input_file, merge, force, config_path))

    @db_app.command("backup")
    def db_backup(
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                "-o",
                help="Output file path (defaults to timestamped file in backups/)",
            ),
        ] = None,
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
    ) -> None:
        """Create an atomic backup of the SQLite database.

        Uses SQLite's VACUUM INTO for a consistent, atomic backup.
        The backup is a standalone database file that can be copied
        or restored directly.

        Examples:
            ash db backup
            ash db backup --output /tmp/ash-backup.db
        """
        from ash.cli.context import get_config

        config = get_config(config_path)
        db_path = config.memory.database_path

        if not db_path.exists():
            error(f"Database not found: {db_path}")
            raise typer.Exit(1)

        # Determine output path
        if output:
            backup_path = output
        else:
            backups_dir = get_ash_home() / "backups"
            backups_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            backup_path = backups_dir / f"ash_{timestamp}.db"

        async def do_backup() -> None:
            import aiosqlite

            async with aiosqlite.connect(db_path) as db:
                await db.execute(f"VACUUM INTO '{backup_path}'")

        console.print("[bold]Creating backup...[/bold]")
        asyncio.run(do_backup())
        success(f"Backup created: {backup_path}")

        # Show file size
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        console.print(f"  Size: {size_mb:.2f} MB")

    # Register the subcommand group
    app.add_typer(db_app, name="db")


async def _db_export(output: Path | None, config_path: Path | None) -> None:
    """Export all data from SQLite to JSONL."""
    import sys as _sys

    from ash.cli.context import get_config, get_database

    config = get_config(config_path)
    database = await get_database(config)

    try:
        records_exported = 0
        output_file = output.open("w") if output else _sys.stdout

        try:
            async with database.session() as session:
                # Export memories
                result = await session.execute(text("SELECT * FROM memories"))
                for row in result.fetchall():
                    record = _row_to_memory_dict(row._mapping)
                    # Fetch subject_person_ids
                    subjects = await session.execute(
                        text(
                            "SELECT person_id FROM memory_subjects WHERE memory_id = :id"
                        ),
                        {"id": record["id"]},
                    )
                    subject_ids = [r[0] for r in subjects.fetchall()]
                    if subject_ids:
                        record["subject_person_ids"] = subject_ids
                    record["_type"] = "memory"
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_exported += 1

                # Export people (using PersonEntry for proper alias/relationship serialization)
                from ash.store.people.helpers import load_person_full

                ppl_result = await session.execute(text("SELECT id FROM people"))
                for row in ppl_result.fetchall():
                    person = await load_person_full(session, row[0])
                    if person:
                        record = person.to_dict()
                        record["_type"] = "person"
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records_exported += 1

                # Export users
                result = await session.execute(text("SELECT * FROM users"))
                for row in result.fetchall():
                    record = _row_to_entity_dict(row._mapping)
                    record["_type"] = "user"
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_exported += 1

                # Export chats
                result = await session.execute(text("SELECT * FROM chats"))
                for row in result.fetchall():
                    record = _row_to_entity_dict(row._mapping)
                    record["_type"] = "chat"
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_exported += 1

            if output:
                success(f"Exported {records_exported} records to {output}")
            else:
                _sys.stderr.write(f"Exported {records_exported} records\n")
        finally:
            if output and output_file != _sys.stdout:
                output_file.close()
    finally:
        await database.disconnect()


async def _db_import(
    input_file: Path, merge: bool, force: bool, config_path: Path | None
) -> None:
    """Import data from JSONL into SQLite."""
    from ash.cli.context import get_config, get_database

    # Parse input file
    records_by_type: dict[str, list[dict]] = {
        "memory": [],
        "person": [],
        "user": [],
        "chat": [],
    }

    with input_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                record_type = record.pop("_type", None)
                if record_type in records_by_type:
                    records_by_type[record_type].append(record)
            except json.JSONDecodeError:
                continue

    total_records = sum(len(v) for v in records_by_type.values())
    if total_records == 0:
        warning("No records found in input file")
        return

    for record_type, records in records_by_type.items():
        if records:
            dim(f"  {record_type}: {len(records)} records")

    if not merge and not force:
        warning("Replace mode will clear existing data before importing.")
        if not typer.confirm("Proceed?"):
            dim("Cancelled")
            return

    config = get_config(config_path)
    database = await get_database(config)

    try:
        async with database.session() as session:
            if not merge:
                # Clear existing data in dependency order
                await session.execute(text("DELETE FROM memory_subjects"))
                await session.execute(text("DELETE FROM memories"))
                await session.execute(text("DELETE FROM person_aliases"))
                await session.execute(text("DELETE FROM person_relationships"))
                await session.execute(text("DELETE FROM users"))
                await session.execute(text("DELETE FROM chats"))
                await session.execute(text("DELETE FROM people"))

            insert_or = "INSERT OR IGNORE" if merge else "INSERT"
            imported = 0

            # Import people first (referenced by memories and users)
            for p in records_by_type["person"]:
                await session.execute(
                    text(f"""
                        {insert_or} INTO people (id, version, created_by, name, merged_into,
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
                # Import aliases
                for alias in p.get("aliases") or []:
                    if isinstance(alias, str):
                        value, added_by, created_at = alias, None, None
                    else:
                        value = alias["value"]
                        added_by = alias.get("added_by")
                        created_at = alias.get("created_at")
                    await session.execute(
                        text(f"""
                            {insert_or} INTO person_aliases (person_id, value, added_by, created_at)
                            VALUES (:person_id, :value, :added_by, :created_at)
                        """),
                        {
                            "person_id": p["id"],
                            "value": value,
                            "added_by": added_by,
                            "created_at": created_at,
                        },
                    )
                # Import relationships
                raw_rels = p.get("relationships")
                if isinstance(raw_rels, list):
                    for rel in raw_rels:
                        if isinstance(rel, dict):
                            await session.execute(
                                text(f"""
                                    {insert_or} INTO person_relationships
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
                    old_rel = p.get("relationship") or p.get("relation")
                    if old_rel:
                        await session.execute(
                            text(f"""
                                {insert_or} INTO person_relationships
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
                imported += 1

            # Import users
            for u in records_by_type["user"]:
                await session.execute(
                    text(f"""
                        {insert_or} INTO users (id, version, provider, provider_id, username,
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
                imported += 1

            # Import chats
            for c in records_by_type["chat"]:
                await session.execute(
                    text(f"""
                        {insert_or} INTO chats (id, version, provider, provider_id, chat_type,
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
                imported += 1

            # Import memories
            for m in records_by_type["memory"]:
                await session.execute(
                    text(f"""
                        {insert_or} INTO memories (id, version, content, memory_type, source,
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
                # Import memory_subjects
                for pid in m.get("subject_person_ids") or []:
                    await session.execute(
                        text(f"""
                            {insert_or} INTO memory_subjects (memory_id, person_id)
                            VALUES (:memory_id, :person_id)
                        """),
                        {"memory_id": m["id"], "person_id": pid},
                    )
                imported += 1

        success(f"Imported {imported} records from {input_file}")

        console.print(
            "\n[yellow]Note:[/yellow] Run 'ash memory rebuild-index' to rebuild the vector index."
        )
    finally:
        await database.disconnect()


def _row_to_memory_dict(mapping: Any) -> dict[str, Any]:
    """Convert a SQLite memory row to an export dict."""
    d: dict = {}
    for key in (
        "id",
        "version",
        "content",
        "memory_type",
        "source",
        "owner_user_id",
        "chat_id",
        "source_username",
        "source_display_name",
        "source_session_id",
        "source_message_id",
        "extraction_confidence",
        "sensitivity",
        "created_at",
        "observed_at",
        "expires_at",
        "superseded_at",
        "superseded_by_id",
        "archived_at",
        "archive_reason",
    ):
        val = mapping.get(key)
        if val is not None:
            d[key] = val
    # Handle portable (stored as 0/1 in SQLite)
    portable = mapping.get("portable")
    if portable is not None and not portable:
        d["portable"] = False
    # Parse metadata JSON
    meta = mapping.get("metadata")
    if meta:
        try:
            d["metadata"] = json.loads(meta) if isinstance(meta, str) else meta
        except json.JSONDecodeError:
            pass
    return d


def _row_to_entity_dict(mapping: Any) -> dict[str, Any]:
    """Convert a SQLite user/chat row to an export dict."""
    d: dict = {}
    for key, val in mapping.items():
        if val is None:
            continue
        if key == "metadata" and isinstance(val, str):
            try:
                d[key] = json.loads(val)
            except json.JSONDecodeError:
                pass
        else:
            d[key] = val
    return d
