"""Database management commands.

Provides commands for:
- migrations: manage schema migrations
- export: export data to JSONL
- import: import data from JSONL
- backup: create SQLite backup using VACUUM INTO
"""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, error, success
from ash.config.paths import (
    get_ash_home,
    get_chats_jsonl_path,
    get_database_path,
    get_graph_dir,
    get_memories_jsonl_path,
    get_people_jsonl_path,
    get_users_jsonl_path,
)


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
    ) -> None:
        """Export all data to JSONL format.

        Exports memories, people, users, and chats to a single JSONL file.
        Each line contains a record with a 'type' field indicating the entity type.

        Examples:
            ash db export > backup.jsonl
            ash db export --output backup.jsonl
        """
        if format_type != "jsonl":
            error(f"Unsupported format: {format_type}. Only 'jsonl' is supported.")
            raise typer.Exit(1)

        import sys as _sys

        # Collect all data files
        data_files = [
            ("memory", get_memories_jsonl_path()),
            ("person", get_people_jsonl_path()),
            ("user", get_users_jsonl_path()),
            ("chat", get_chats_jsonl_path()),
        ]

        records_exported = 0
        output_file = output.open("w") if output else _sys.stdout

        try:
            for record_type, path in data_files:
                if not path.exists():
                    continue

                with path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            # Add type field for import reconstruction
                            record["_type"] = record_type
                            output_file.write(json.dumps(record) + "\n")
                            records_exported += 1
                        except json.JSONDecodeError:
                            continue

            if output:
                success(f"Exported {records_exported} records to {output}")
            else:
                # If stdout, print summary to stderr
                import sys as sys_module

                sys_module.stderr.write(f"Exported {records_exported} records\n")
        finally:
            if output and output_file != _sys.stdout:
                output_file.close()

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
                help="Merge with existing data (default: replace)",
            ),
        ] = False,
    ) -> None:
        """Import data from JSONL format.

        Imports memories, people, users, and chats from a JSONL file
        previously created with 'ash db export'.

        By default, replaces existing data. Use --merge to add to existing.

        Examples:
            ash db import backup.jsonl
            ash db import backup.jsonl --merge
        """
        if not input_file.exists():
            error(f"File not found: {input_file}")
            raise typer.Exit(1)

        # Ensure graph directory exists
        graph_dir = get_graph_dir()
        graph_dir.mkdir(parents=True, exist_ok=True)

        # Collect records by type
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

        # Output file mapping
        output_files = {
            "memory": get_memories_jsonl_path(),
            "person": get_people_jsonl_path(),
            "user": get_users_jsonl_path(),
            "chat": get_chats_jsonl_path(),
        }

        total_imported = 0

        for record_type, records in records_by_type.items():
            if not records:
                continue

            output_path = output_files[record_type]

            # Load existing records if merging
            existing: dict[str, dict] = {}
            if merge and output_path.exists():
                with output_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if "id" in rec:
                                existing[rec["id"]] = rec
                        except json.JSONDecodeError:
                            continue

            # Merge or replace
            if merge:
                for rec in records:
                    if "id" in rec:
                        existing[rec["id"]] = rec
                final_records = list(existing.values())
            else:
                final_records = records

            # Write atomically
            tmp_path = output_path.with_suffix(".tmp")
            with tmp_path.open("w") as f:
                for rec in final_records:
                    f.write(json.dumps(rec) + "\n")
            tmp_path.rename(output_path)

            total_imported += len(records)
            console.print(f"  {record_type}: {len(records)} records")

        success(f"Imported {total_imported} records from {input_file}")

        # Remind about rebuilding index
        console.print(
            "\n[yellow]Note:[/yellow] Run 'ash memory rebuild-index' to rebuild the vector index."
        )

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
    ) -> None:
        """Create an atomic backup of the SQLite database.

        Uses SQLite's VACUUM INTO for a consistent, atomic backup.
        The backup is a standalone database file that can be copied
        or restored directly.

        Examples:
            ash db backup
            ash db backup --output /tmp/ash-backup.db
        """
        import asyncio

        import aiosqlite

        db_path = get_database_path()
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
