"""Graph inspection commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import click
import typer
from sqlalchemy import text

from ash.cli.console import console, create_table, dim, error

if TYPE_CHECKING:
    from ash.store.store import Store

_EDGE_LABELS = {
    "about": "Memory -> Person",
    "owned_by": "Memory -> User",
    "in_chat": "Memory -> Chat",
    "supersedes": "Memory -> Memory",
    "is_person": "User -> Person",
    "merged_into": "Person -> Person",
}


def register(app: typer.Typer) -> None:
    """Register the graph command."""

    @app.command()
    def graph(
        action: Annotated[
            str | None,
            typer.Argument(
                help="Action: users, chats, edges, stats",
            ),
        ] = None,
        target: Annotated[
            str | None,
            typer.Argument(
                help="Node ID for edges command",
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
        """Inspect graph structure (users, chats, edges).

        Examples:
            ash graph users         # List user nodes
            ash graph chats         # List chat nodes
            ash graph edges <id>    # Show edges for a node
            ash graph stats         # Node/edge counts
        """
        if action is None:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            raise typer.Exit(0)

        try:
            asyncio.run(
                _run_graph_action(
                    action=action,
                    target=target,
                    config_path=config_path,
                )
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled[/dim]")


async def _run_graph_action(
    action: str,
    target: str | None,
    config_path: Path | None,
) -> None:
    """Run graph action asynchronously."""
    from ash.cli.commands.memory._helpers import get_store
    from ash.cli.context import get_config, get_database

    config = get_config(config_path)
    database = await get_database(config)

    try:
        store = await get_store(config, database)
        if not store:
            error("Graph commands require [embeddings] configuration")
            raise typer.Exit(1)

        if action == "users":
            await _graph_users(store)
        elif action == "chats":
            await _graph_chats(store)
        elif action == "edges":
            if not target:
                error("Usage: ash graph edges <node-id>")
                raise typer.Exit(1)
            await _graph_edges(store, target)
        elif action == "stats":
            await _graph_stats(store)
        else:
            error(f"Unknown action: {action}")
            console.print("Valid actions: users, chats, edges, stats")
            raise typer.Exit(1)
    finally:
        await database.disconnect()


async def _graph_users(store: Store) -> None:
    """List all user nodes."""
    users = await store.list_users()

    if not users:
        dim("No user nodes found.")
        return

    table = create_table(
        f"Users ({len(users)})",
        [
            ("ID", {"style": "dim", "max_width": 12}),
            ("Provider", "cyan"),
            ("Provider ID", ""),
            ("Username", "bold"),
            ("Display Name", ""),
            ("Person", "green"),
            ("Updated", "dim"),
        ],
    )

    for user in users:
        person_link = user.person_id[:12] if user.person_id else "-"
        updated = user.updated_at.strftime("%Y-%m-%d") if user.updated_at else "-"
        table.add_row(
            user.id[:12],
            user.provider,
            user.provider_id,
            user.username or "-",
            user.display_name or "-",
            person_link,
            updated,
        )

    console.print(table)


async def _graph_chats(store: Store) -> None:
    """List all chat nodes."""
    chats = await store.list_chats()

    if not chats:
        dim("No chat nodes found.")
        return

    table = create_table(
        f"Chats ({len(chats)})",
        [
            ("ID", {"style": "dim", "max_width": 12}),
            ("Provider", "cyan"),
            ("Provider ID", ""),
            ("Type", ""),
            ("Title", "bold"),
            ("Updated", "dim"),
        ],
    )

    for chat in chats:
        updated = chat.updated_at.strftime("%Y-%m-%d") if chat.updated_at else "-"
        table.add_row(
            chat.id[:12],
            chat.provider,
            chat.provider_id,
            chat.chat_type or "-",
            chat.title or "-",
            updated,
        )

    console.print(table)


async def _resolve_node_id(store: Store, prefix: str) -> tuple[str, str] | None:
    """Resolve a node ID prefix to (full_id, table_name).

    Searches across all 4 node tables using exact match first,
    then prefix match via LIKE.
    """
    tables = ["memories", "people", "users", "chats"]
    async with store._db.session() as session:
        # Try exact match first
        for table in tables:
            result = await session.execute(
                text(f"SELECT id FROM {table} WHERE id = :id"),  # noqa: S608
                {"id": prefix},
            )
            row = result.fetchone()
            if row:
                return (row[0], table)

        # Try prefix match
        for table in tables:
            result = await session.execute(
                text(f"SELECT id FROM {table} WHERE id LIKE :prefix"),  # noqa: S608
                {"prefix": f"{prefix}%"},
            )
            rows = result.fetchall()
            if len(rows) == 1:
                return (rows[0][0], table)

    return None


async def _graph_edges(store: Store, node_id: str) -> None:
    """Show all edges connected to a node via SQL queries."""
    # Resolve prefix to full ID
    resolved = await _resolve_node_id(store, node_id)
    if resolved:
        full_id, table_name = resolved
        if full_id != node_id:
            dim(f"Resolved prefix to: {full_id} ({table_name})")
        node_id = full_id

    edges: list[tuple[str, str, str, str]] = []

    async with store._db.session() as session:
        # Memory -> Person (about)
        result = await session.execute(
            text(
                "SELECT memory_id, person_id FROM memory_subjects WHERE memory_id = :id OR person_id = :id"
            ),
            {"id": node_id},
        )
        for row in result.fetchall():
            if row[0] == node_id:
                edges.append(("about", "->", row[0], row[1]))
            else:
                edges.append(("about", "<-", row[0], row[1]))

        # Memory -> User (owned_by)
        result = await session.execute(
            text(
                "SELECT id, owner_user_id FROM memories WHERE (id = :id OR owner_user_id = :id) AND owner_user_id IS NOT NULL AND archived_at IS NULL"
            ),
            {"id": node_id},
        )
        for row in result.fetchall():
            if row[0] == node_id:
                edges.append(("owned_by", "->", row[0], row[1]))
            else:
                edges.append(("owned_by", "<-", row[0], row[1]))

        # Memory -> Chat (in_chat)
        result = await session.execute(
            text(
                "SELECT id, chat_id FROM memories WHERE (id = :id OR chat_id = :id) AND chat_id IS NOT NULL AND archived_at IS NULL"
            ),
            {"id": node_id},
        )
        for row in result.fetchall():
            if row[0] == node_id:
                edges.append(("in_chat", "->", row[0], row[1]))
            else:
                edges.append(("in_chat", "<-", row[0], row[1]))

        # User -> Person (is_person)
        result = await session.execute(
            text(
                "SELECT id, person_id FROM users WHERE (id = :id OR person_id = :id) AND person_id IS NOT NULL"
            ),
            {"id": node_id},
        )
        for row in result.fetchall():
            if row[0] == node_id:
                edges.append(("is_person", "->", row[0], row[1]))
            else:
                edges.append(("is_person", "<-", row[0], row[1]))

        # Person -> Person (merged_into)
        result = await session.execute(
            text(
                "SELECT id, merged_into FROM people WHERE (id = :id OR merged_into = :id) AND merged_into IS NOT NULL"
            ),
            {"id": node_id},
        )
        for row in result.fetchall():
            if row[0] == node_id:
                edges.append(("merged_into", "->", row[0], row[1]))
            else:
                edges.append(("merged_into", "<-", row[0], row[1]))

        # Memory -> Memory (supersedes)
        result = await session.execute(
            text(
                "SELECT id, superseded_by_id FROM memories WHERE (id = :id OR superseded_by_id = :id) AND superseded_by_id IS NOT NULL"
            ),
            {"id": node_id},
        )
        for row in result.fetchall():
            if row[0] == node_id:
                edges.append(("supersedes", "->", row[0], row[1]))
            else:
                edges.append(("supersedes", "<-", row[0], row[1]))

    if not edges:
        dim(f"No edges found for node: {node_id}")
        dim(
            "Tip: use a full node ID or prefix from 'ash graph users' / 'ash graph chats'"
        )
        return

    table = create_table(
        f"Edges for {node_id[:12]} ({len(edges)})",
        [
            ("Type", "cyan"),
            ("Direction", "dim"),
            ("From", {"style": "bold", "max_width": 14}),
            ("To", {"style": "bold", "max_width": 14}),
            ("Meaning", "dim"),
        ],
    )

    for edge_type, direction, source, target in edges:
        meaning = _EDGE_LABELS.get(edge_type, "")
        table.add_row(
            edge_type,
            direction,
            source[:14],
            target[:14],
            meaning,
        )

    console.print(table)


async def _graph_stats(store: Store) -> None:
    """Show graph node/edge counts and health summary."""
    users = await store.list_users()
    chats = await store.list_chats()
    people = await store.list_people()

    # Count nodes and edges via SQL
    node_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    async with store._db.session() as session:
        result = await session.execute(
            text(
                "SELECT COUNT(*) FROM memories WHERE archived_at IS NULL AND superseded_at IS NULL"
            )
        )
        node_counts["memories"] = result.scalar() or 0

        result = await session.execute(text("SELECT COUNT(*) FROM memory_subjects"))
        edge_counts["about"] = result.scalar() or 0

        result = await session.execute(
            text(
                "SELECT COUNT(*) FROM memories WHERE owner_user_id IS NOT NULL AND archived_at IS NULL"
            )
        )
        edge_counts["owned_by"] = result.scalar() or 0

        result = await session.execute(
            text(
                "SELECT COUNT(*) FROM memories WHERE chat_id IS NOT NULL AND archived_at IS NULL"
            )
        )
        edge_counts["in_chat"] = result.scalar() or 0

        result = await session.execute(
            text("SELECT COUNT(*) FROM users WHERE person_id IS NOT NULL")
        )
        edge_counts["is_person"] = result.scalar() or 0

        result = await session.execute(
            text("SELECT COUNT(*) FROM people WHERE merged_into IS NOT NULL")
        )
        edge_counts["merged_into"] = result.scalar() or 0

        result = await session.execute(
            text(
                "SELECT COUNT(*) FROM memories WHERE superseded_by_id IS NOT NULL AND archived_at IS NULL"
            )
        )
        edge_counts["supersedes"] = result.scalar() or 0

    # Filter to non-zero
    edge_counts = {k: v for k, v in edge_counts.items() if v > 0}

    # Node summary
    console.print("[bold]Node Counts[/bold]")
    console.print(f"  Users:    {len(users)}")
    console.print(f"  Chats:    {len(chats)}")
    console.print(f"  People:   {len(people)}")
    console.print(f"  Memories: {node_counts['memories']}")
    console.print()

    if edge_counts:
        table = create_table(
            "Edge Counts",
            [
                ("Type", "cyan"),
                ("Count", "bold"),
                ("Meaning", "dim"),
            ],
        )
        total = 0
        for edge_type, count in sorted(edge_counts.items()):
            meaning = _EDGE_LABELS.get(edge_type, "")
            table.add_row(edge_type, str(count), meaning)
            total += count

        console.print(table)
        dim(f"\nTotal edges: {total}")
    else:
        dim("No edges in graph")

    # Health checks
    console.print()
    console.print("[bold]Health[/bold]")

    # Users without person links
    unlinked_users = [u for u in users if not u.person_id]
    if unlinked_users:
        console.print(
            f"  [yellow]Users without person link: {len(unlinked_users)}[/yellow]"
        )
    else:
        console.print("  [green]All users linked to person records[/green]")

    # Orphaned people (no memories about them)
    orphan_count = 0
    for p in people:
        memory_ids = await store.memories_about_person(p.id)
        if not memory_ids:
            orphan_count += 1
    if orphan_count:
        console.print(f"  [yellow]People with no memories: {orphan_count}[/yellow]")
    else:
        console.print("  [green]All people have associated memories[/green]")
