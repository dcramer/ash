"""Graph inspection commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import click
import typer

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
    from ash.cli.context import get_config

    config = get_config(config_path)

    store = await get_store(config)
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

    from ash.graph.edges import get_person_for_user

    for user in users:
        person_id = get_person_for_user(store._graph, user.id)
        person_link = person_id[:12] if person_id else "-"
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
    """Resolve a node ID prefix to (full_id, type_name).

    Searches across all 4 node types using exact match first,
    then prefix match.
    """
    graph = store._graph
    collections = [
        ("memories", graph.memories),
        ("people", graph.people),
        ("users", graph.users),
        ("chats", graph.chats),
    ]

    # Try exact match first
    for type_name, collection in collections:
        if prefix in collection:
            return (prefix, type_name)

    # Try prefix match
    for type_name, collection in collections:
        matches = [nid for nid in collection if nid.startswith(prefix)]
        if len(matches) == 1:
            return (matches[0], type_name)

    return None


async def _graph_edges(store: Store, node_id: str) -> None:
    """Show all edges connected to a node."""
    # Resolve prefix to full ID
    resolved = await _resolve_node_id(store, node_id)
    if resolved:
        full_id, table_name = resolved
        if full_id != node_id:
            dim(f"Resolved prefix to: {full_id} ({table_name})")
        node_id = full_id

    from ash.graph.edges import get_subject_person_ids

    edges: list[tuple[str, str, str, str]] = []
    graph = store._graph

    # Memory -> Person (about) via subject_person_ids
    for mem in graph.memories.values():
        if mem.archived_at:
            continue
        subject_ids = get_subject_person_ids(graph, mem.id)
        if mem.id == node_id:
            for pid in subject_ids:
                edges.append(("about", "->", mem.id, pid))
        elif node_id in subject_ids:
            edges.append(("about", "<-", mem.id, node_id))

    # Memory -> User (owned_by) via owner_user_id
    for mem in graph.memories.values():
        if mem.archived_at or not mem.owner_user_id:
            continue
        if mem.id == node_id:
            edges.append(("owned_by", "->", mem.id, mem.owner_user_id))
        elif mem.owner_user_id == node_id:
            edges.append(("owned_by", "<-", mem.id, mem.owner_user_id))

    # Memory -> Chat (in_chat) via chat_id
    for mem in graph.memories.values():
        if mem.archived_at or not mem.chat_id:
            continue
        if mem.id == node_id:
            edges.append(("in_chat", "->", mem.id, mem.chat_id))
        elif mem.chat_id == node_id:
            edges.append(("in_chat", "<-", mem.id, mem.chat_id))

    from ash.graph.edges import get_person_for_user

    # User -> Person (is_person) via person_id
    for user in graph.users.values():
        person_id = get_person_for_user(graph, user.id)
        if not person_id:
            continue
        if user.id == node_id:
            edges.append(("is_person", "->", user.id, person_id))
        elif person_id == node_id:
            edges.append(("is_person", "<-", user.id, person_id))

    from ash.graph.edges import get_merged_into

    # Person -> Person (merged_into)
    for person in graph.people.values():
        merged_into_id = get_merged_into(graph, person.id)
        if not merged_into_id:
            continue
        if person.id == node_id:
            edges.append(("merged_into", "->", person.id, merged_into_id))
        elif merged_into_id == node_id:
            edges.append(("merged_into", "<-", person.id, merged_into_id))

    from ash.graph.edges import get_superseded_by

    # Memory -> Memory (supersedes) via superseded_by_id
    for mem in graph.memories.values():
        superseded_by_id = get_superseded_by(graph, mem.id)
        if not superseded_by_id:
            continue
        if mem.id == node_id:
            edges.append(("supersedes", "->", mem.id, superseded_by_id))
        elif superseded_by_id == node_id:
            edges.append(("supersedes", "<-", mem.id, superseded_by_id))

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
    graph = store._graph

    users = await store.list_users()
    chats = await store.list_chats()
    people = await store.list_people()

    # Count active memories
    active_memories = sum(
        1 for m in graph.memories.values() if not m.archived_at and not m.superseded_at
    )

    from ash.graph.edges import get_subject_person_ids

    # Count edges by type from in-memory data
    edge_counts: dict[str, int] = {}

    # about: memories with subject_person_ids
    about_count = sum(
        len(get_subject_person_ids(graph, m.id))
        for m in graph.memories.values()
        if not m.archived_at
    )
    if about_count:
        edge_counts["about"] = about_count

    # owned_by: memories with owner_user_id
    owned_by = sum(
        1 for m in graph.memories.values() if m.owner_user_id and not m.archived_at
    )
    if owned_by:
        edge_counts["owned_by"] = owned_by

    # in_chat: memories with chat_id
    in_chat = sum(1 for m in graph.memories.values() if m.chat_id and not m.archived_at)
    if in_chat:
        edge_counts["in_chat"] = in_chat

    from ash.graph.edges import get_person_for_user

    # is_person: users with person_id
    is_person = sum(1 for u in graph.users.values() if get_person_for_user(graph, u.id))
    if is_person:
        edge_counts["is_person"] = is_person

    from ash.graph.edges import get_merged_into

    # merged_into: people with merged_into
    merged = sum(1 for p in graph.people.values() if get_merged_into(graph, p.id))
    if merged:
        edge_counts["merged_into"] = merged

    from ash.graph.edges import get_superseded_by

    # supersedes: memories with superseded_by_id
    supersedes = sum(
        1
        for m in graph.memories.values()
        if get_superseded_by(graph, m.id) and not m.archived_at
    )
    if supersedes:
        edge_counts["supersedes"] = supersedes

    # Node summary
    console.print("[bold]Node Counts[/bold]")
    console.print(f"  Users:    {len(users)}")
    console.print(f"  Chats:    {len(chats)}")
    console.print(f"  People:   {len(people)}")
    console.print(f"  Memories: {active_memories}")
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

    from ash.graph.edges import get_person_for_user

    # Users without person links
    unlinked_users = [u for u in users if not get_person_for_user(graph, u.id)]
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
