"""Graph inspection commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import click
import typer

from ash.cli.console import console, create_table, dim, error

if TYPE_CHECKING:
    from ash.graph.store import GraphStore

_EDGE_LABELS = {
    "about": "Memory -> Person",
    "owned_by": "Memory -> User",
    "in_chat": "Memory -> Chat",
    "stated_by": "Memory -> User",
    "supersedes": "Memory -> Memory",
    "knows": "User -> Person",
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
    from ash.cli.context import get_config, get_database

    config = get_config(config_path)
    database = await get_database(config)

    try:
        async with database.session() as session:
            from ash.cli.commands.memory._helpers import get_graph_store

            graph_store = await get_graph_store(config, session)
            if not graph_store:
                error("Graph commands require [embeddings] configuration")
                raise typer.Exit(1)

            if action == "users":
                await _graph_users(graph_store)
            elif action == "chats":
                await _graph_chats(graph_store)
            elif action == "edges":
                if not target:
                    error("Usage: ash graph edges <node-id>")
                    raise typer.Exit(1)
                await _graph_edges(graph_store, target)
            elif action == "stats":
                await _graph_stats(graph_store)
            else:
                error(f"Unknown action: {action}")
                console.print("Valid actions: users, chats, edges, stats")
                raise typer.Exit(1)
    finally:
        await database.disconnect()


async def _graph_users(graph_store: GraphStore) -> None:
    """List all user nodes."""
    users = await graph_store.list_users()

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


async def _graph_chats(graph_store: GraphStore) -> None:
    """List all chat nodes."""
    chats = await graph_store.list_chats()

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


async def _graph_edges(graph_store: GraphStore, node_id: str) -> None:
    """Show all edges connected to a node."""
    from ash.graph.types import EdgeType

    graph = await graph_store.get_graph()

    # Collect all edges (outgoing and incoming) for this node
    # Each tuple: (edge_type, direction, source, target)
    edges: list[tuple[str, str, str, str]] = []

    for et in EdgeType:
        outgoing = graph.neighbors(node_id, et, "outgoing")
        for target in outgoing:
            edges.append((et.value, "->", node_id, target))

        incoming = graph.neighbors(node_id, et, "incoming")
        for source in incoming:
            edges.append((et.value, "<-", source, node_id))

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


async def _graph_stats(graph_store: GraphStore) -> None:
    """Show graph node/edge counts and health summary."""

    graph = await graph_store.get_graph()

    users = await graph_store.list_users()
    chats = await graph_store.list_chats()
    people = await graph_store.list_people()
    memories = await graph_store.list_memories(limit=100000)

    # Count edges by type
    edge_counts: dict[str, int] = {
        et.value: count for et, count in graph.edge_counts().items() if count > 0
    }

    # Node summary
    console.print("[bold]Node Counts[/bold]")
    console.print(f"  Users:    {len(users)}")
    console.print(f"  Chats:    {len(chats)}")
    console.print(f"  People:   {len(people)}")
    console.print(f"  Memories: {len(memories)}")
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
    orphaned = [p for p in people if not graph.memories_about(p.id)]
    if orphaned:
        console.print(f"  [yellow]People with no memories: {len(orphaned)}[/yellow]")
    else:
        console.print("  [green]All people have associated memories[/green]")
