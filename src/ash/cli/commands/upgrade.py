"""Upgrade command for running setup tasks."""

import typer

from ash.cli.console import (
    console,
    dim,
    error,
    info,
    success,
    warning,
)


def register(app: typer.Typer) -> None:
    """Register the upgrade command."""

    @app.command()
    def upgrade() -> None:
        """Upgrade Ash (build sandbox)."""
        console.print("[bold]Upgrading Ash...[/bold]\n")

        _migrate_schedule_into_graph()
        _migrate_todo_scope_ids()
        _migrate_schedule_scope_edges()

        # Build sandbox
        info("Building sandbox...")

        from ash.cli.commands.sandbox import _get_dockerfile_path, _sandbox_build

        dockerfile_path = _get_dockerfile_path()
        if not dockerfile_path:
            error("Dockerfile.sandbox not found")
            dim("Sandbox build skipped")
        elif not _sandbox_build(dockerfile_path):
            warning("Sandbox build failed (retry with 'ash sandbox build')")

        console.print("\n[bold green]Upgrade complete![/bold green]")


def _migrate_todo_scope_ids() -> None:
    """Resolve TodoEntry provider IDs to graph node UUIDs and ensure scope edges."""
    import asyncio

    from ash.config.paths import get_graph_dir
    from ash.graph.edges import (
        TODO_OWNED_BY,
        TODO_REMINDER_SCHEDULED_AS,
        TODO_SHARED_IN,
        create_todo_owned_by_edge,
        create_todo_shared_in_edge,
        resolve_chat_node_id,
        resolve_user_node_id,
    )
    from ash.graph.persistence import GraphPersistence, hydrate_graph
    from ash.todos.types import TodoEntry, register_todo_graph_schema

    graph_dir = get_graph_dir()
    todos_file = graph_dir / "todos.jsonl"
    if not todos_file.exists():
        dim("No todo data found (skipping todo scope migration)")
        return

    register_todo_graph_schema()

    async def _run() -> None:
        persistence = GraphPersistence(graph_dir)
        raw_data = await persistence.load_raw()
        graph = hydrate_graph(raw_data)

        todos = {
            k: v
            for k, v in graph.get_node_collection("todo").items()
            if isinstance(v, TodoEntry)
        }
        if not todos:
            return

        edges_dirty = False
        nodes_dirty = False
        for todo in todos.values():
            # Migrate TodoEntry fields: resolve provider IDs to graph node UUIDs.
            if todo.owner_user_id and todo.owner_user_id not in graph.users:
                resolved = resolve_user_node_id(graph, todo.owner_user_id)
                if resolved:
                    todo.owner_user_id = resolved
                    nodes_dirty = True
            if todo.chat_id and todo.chat_id not in graph.chats:
                resolved = resolve_chat_node_id(graph, todo.chat_id)
                if resolved:
                    todo.chat_id = resolved
                    nodes_dirty = True

            # Ensure scope edges exist and point at graph node UUIDs.
            owner_edges = graph.get_outgoing(todo.id, edge_type=TODO_OWNED_BY)
            chat_edges = graph.get_outgoing(todo.id, edge_type=TODO_SHARED_IN)
            reminder_edges = graph.get_outgoing(
                todo.id, edge_type=TODO_REMINDER_SCHEDULED_AS
            )
            if todo.owner_user_id and not owner_edges:
                graph.add_edge(create_todo_owned_by_edge(todo.id, todo.owner_user_id))
                edges_dirty = True
            elif owner_edges:
                for edge in owner_edges:
                    if edge.target_id not in graph.users:
                        resolved = resolve_user_node_id(graph, edge.target_id)
                        if resolved and resolved != edge.target_id:
                            graph.remove_edge(edge.id)
                            graph.add_edge(create_todo_owned_by_edge(todo.id, resolved))
                            edges_dirty = True
            if todo.chat_id and not chat_edges:
                graph.add_edge(create_todo_shared_in_edge(todo.id, todo.chat_id))
                edges_dirty = True
            elif chat_edges:
                for edge in chat_edges:
                    if edge.target_id not in graph.chats:
                        resolved = resolve_chat_node_id(graph, edge.target_id)
                        if resolved and resolved != edge.target_id:
                            graph.remove_edge(edge.id)
                            graph.add_edge(
                                create_todo_shared_in_edge(todo.id, resolved)
                            )
                            edges_dirty = True
            if todo.linked_schedule_entry_id and not reminder_edges:
                from ash.graph.edges import create_todo_reminder_scheduled_as_edge

                graph.add_edge(
                    create_todo_reminder_scheduled_as_edge(
                        todo.id, todo.linked_schedule_entry_id
                    )
                )
                edges_dirty = True

        if not edges_dirty and not nodes_dirty:
            dim("Todo scope IDs already migrated")
            return

        if nodes_dirty:
            persistence.mark_dirty("todos")
        if edges_dirty:
            persistence.mark_dirty("edges")
        await persistence.flush(graph)
        success("Todo scope IDs migrated to graph node UUIDs")

    asyncio.run(_run())


def _migrate_schedule_scope_edges() -> None:
    """Resolve schedule edge targets from provider IDs to graph node UUIDs."""
    from ash.config.paths import get_graph_dir
    from ash.graph.edges import (
        SCHEDULE_FOR_CHAT,
        SCHEDULE_FOR_USER,
        create_schedule_for_chat_edge,
        create_schedule_for_user_edge,
        resolve_chat_node_id,
        resolve_user_node_id,
    )
    from ash.graph.persistence import GraphPersistence, hydrate_graph
    from ash.scheduling.types import ScheduleEntry, register_schedule_graph_schema

    graph_dir = get_graph_dir()
    schedules_file = graph_dir / "schedules.jsonl"
    if not schedules_file.exists():
        dim("No schedule data found (skipping schedule edge migration)")
        return

    register_schedule_graph_schema()
    persistence = GraphPersistence(graph_dir)
    raw_data = persistence.load_raw_sync()
    graph = hydrate_graph(raw_data)

    entries = {
        k: v
        for k, v in graph.get_node_collection("schedule_entry").items()
        if isinstance(v, ScheduleEntry)
    }
    if not entries:
        return

    migrated = False
    for entry in entries.values():
        if not entry.id:
            continue
        for edge in list(graph.get_outgoing(entry.id, edge_type=SCHEDULE_FOR_USER)):
            if edge.target_id not in graph.users:
                resolved = resolve_user_node_id(graph, edge.target_id)
                if resolved and resolved != edge.target_id:
                    graph.remove_edge(edge.id)
                    graph.add_edge(create_schedule_for_user_edge(entry.id, resolved))
                    migrated = True
        for edge in list(graph.get_outgoing(entry.id, edge_type=SCHEDULE_FOR_CHAT)):
            if edge.target_id not in graph.chats:
                resolved = resolve_chat_node_id(graph, edge.target_id)
                if resolved and resolved != edge.target_id:
                    graph.remove_edge(edge.id)
                    graph.add_edge(create_schedule_for_chat_edge(entry.id, resolved))
                    migrated = True

    if not migrated:
        dim("Schedule scope edges already migrated")
        return

    persistence.mark_dirty("edges")
    persistence.flush_sync(graph)
    success("Schedule scope edges migrated to graph node UUIDs")


def _migrate_schedule_into_graph() -> None:
    """Migrate legacy schedule.jsonl entries into graph schedule nodes."""
    from ash.config.paths import get_ash_home, get_graph_dir
    from ash.scheduling import ScheduleStore
    from ash.scheduling.types import ScheduleEntry

    schedule_file = get_ash_home() / "schedule.jsonl"
    if not schedule_file.exists():
        dim("No legacy schedule file found (skipping schedule migration)")
        return

    graph_schedules = get_graph_dir() / "schedules.jsonl"
    if graph_schedules.exists() and graph_schedules.stat().st_size > 0:
        dim("Graph schedule storage already initialized (skipping legacy migration)")
        return

    info("Migrating schedule entries into ash.graph...")
    try:
        store = ScheduleStore(get_graph_dir())
        imported = 0
        for line_number, line in enumerate(schedule_file.read_text().splitlines()):
            entry = ScheduleEntry.from_line(line, line_number)
            if entry is None:
                continue
            store.add_entry(entry)
            imported += 1

        schedule_file.unlink(missing_ok=True)

        success(
            f"Schedule graph ready ({imported} entr{'y' if imported == 1 else 'ies'} migrated)"
        )
    except Exception as exc:
        warning(f"Schedule migration failed ({exc})")
        dim("Retry by running `ash schedule list` after fixing data issues")
