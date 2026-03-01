"""Todo management commands."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated

import typer

from ash.cli.console import console, error, success, warning

if TYPE_CHECKING:
    from ash.todos import TodoManager

app = typer.Typer(
    name="todo",
    help="Manage todos.",
    invoke_without_command=True,
)


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="todo")


def _run(coro) -> None:
    """Run an async todo operation with ValueError handling."""
    try:
        asyncio.run(coro)
    except ValueError as e:
        error(str(e))
        raise typer.Exit(1) from None


async def _create_manager() -> TodoManager:
    from ash.config.paths import get_graph_dir
    from ash.todos import create_todo_manager

    return await create_todo_manager(get_graph_dir())


async def _get_todo_scope(
    manager: TodoManager, todo_id: str
) -> tuple[str | None, str | None]:
    """Read a todo's owner_user_id/chat_id for passing to manager methods."""
    todo = await manager.get(todo_id)
    if todo is None:
        raise ValueError(f"todo {todo_id} not found")
    return todo.owner_user_id, todo.chat_id


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.callback()
def _default(ctx: typer.Context) -> None:
    """Manage todos. Run without a subcommand to list open todos."""
    if ctx.invoked_subcommand is None:
        _run(_todo_list())


@app.command("list")
def list_cmd(
    show_all: Annotated[
        bool, typer.Option("--all", "-a", help="Include done and deleted")
    ] = False,
    show_done: Annotated[
        bool, typer.Option("--done", help="Include completed todos")
    ] = False,
    show_deleted: Annotated[
        bool, typer.Option("--deleted", help="Include deleted todos")
    ] = False,
    user: Annotated[
        str | None, typer.Option("--user", "-u", help="Filter by user ID")
    ] = None,
    chat: Annotated[
        str | None, typer.Option("--chat", "-c", help="Filter by chat ID")
    ] = None,
) -> None:
    """List todos."""
    _run(
        _todo_list(
            user_id=user,
            chat_id=chat,
            show_all=show_all,
            show_done=show_done,
            show_deleted=show_deleted,
        )
    )


@app.command("add")
def add_cmd(
    content: Annotated[str, typer.Argument(help="Todo text")],
    user: Annotated[
        str | None,
        typer.Option("--user", "-u", help="User ID (required for personal todos)"),
    ] = None,
    chat: Annotated[
        str | None,
        typer.Option("--chat", "-c", help="Chat ID (for shared todos)"),
    ] = None,
    shared: Annotated[
        bool, typer.Option("--shared", help="Create as shared todo (requires --chat)")
    ] = False,
    due: Annotated[
        str | None,
        typer.Option("--due", help="Due date (ISO 8601 or natural language)"),
    ] = None,
) -> None:
    """Add a new todo."""
    _run(_todo_add(content=content, user_id=user, chat_id=chat, shared=shared, due=due))


@app.command("done")
def done_cmd(
    todo_id: Annotated[str, typer.Option("--id", "-i", help="Todo ID")],
) -> None:
    """Mark a todo as complete."""
    _run(_todo_done(todo_id))


@app.command("undone")
def undone_cmd(
    todo_id: Annotated[str, typer.Option("--id", "-i", help="Todo ID")],
) -> None:
    """Reopen a completed todo."""
    _run(_todo_undone(todo_id))


@app.command("delete")
def delete_cmd(
    todo_id: Annotated[str, typer.Option("--id", "-i", help="Todo ID")],
) -> None:
    """Soft-delete a todo."""
    _run(_todo_delete(todo_id))


@app.command("edit")
def edit_cmd(
    todo_id: Annotated[str, typer.Option("--id", "-i", help="Todo ID")],
    content: Annotated[str | None, typer.Argument(help="New todo text")] = None,
    due: Annotated[
        str | None,
        typer.Option("--due", help="Due date (ISO 8601 or natural language)"),
    ] = None,
    clear_due: Annotated[
        bool, typer.Option("--clear-due", help="Remove due date")
    ] = False,
) -> None:
    """Update a todo's content or due date."""
    if content is None and due is None and not clear_due:
        error("at least one of content, --due, or --clear-due is required")
        raise typer.Exit(1)
    _run(_todo_edit(todo_id=todo_id, content=content, due=due, clear_due=clear_due))


# ---------------------------------------------------------------------------
# Async implementations
# ---------------------------------------------------------------------------


async def _todo_list(
    *,
    user_id: str | None = None,
    chat_id: str | None = None,
    show_all: bool = False,
    show_done: bool = False,
    show_deleted: bool = False,
) -> None:
    from ash.cli.console import create_table
    from ash.todos.types import TodoStatus

    manager = await _create_manager()
    all_todos = list(manager._todos().values())

    if user_id is not None:
        all_todos = [t for t in all_todos if t.owner_user_id == user_id]
    if chat_id is not None:
        all_todos = [t for t in all_todos if t.chat_id == chat_id]

    if not (show_all or show_deleted):
        all_todos = [t for t in all_todos if t.deleted_at is None]
    if not (show_all or show_done):
        all_todos = [t for t in all_todos if t.status == TodoStatus.OPEN]

    all_todos.sort(
        key=lambda t: (
            0 if t.status == TodoStatus.OPEN else 1,
            -t.created_at.timestamp(),
        )
    )

    if not all_todos:
        warning("No todos found")
        return

    table = create_table(
        "Todos",
        [
            ("ID", "dim"),
            ("Status", ""),
            ("Task", ""),
            ("Due", ""),
            ("Owner", "dim"),
        ],
    )

    for todo in all_todos:
        if todo.deleted_at is not None:
            status = "[red]deleted[/red]"
        elif todo.status == TodoStatus.DONE:
            status = "[green]done[/green]"
        else:
            status = "[cyan]open[/cyan]"

        task = todo.content[:50] + "..." if len(todo.content) > 50 else todo.content

        if todo.due_at is not None:
            due = _format_due(todo.due_at)
        else:
            due = "[dim]-[/dim]"

        if todo.owner_user_id:
            owner = todo.owner_user_id
        elif todo.chat_id:
            owner = f"chat:{todo.chat_id[:10]}"
        else:
            owner = "[dim]-[/dim]"

        table.add_row(todo.id, status, task, due, owner)

    console.print(table)
    console.print(f"\n[dim]Total: {len(all_todos)} todo(s)[/dim]")


async def _todo_add(
    *,
    content: str,
    user_id: str | None,
    chat_id: str | None,
    shared: bool,
    due: str | None,
) -> None:
    if shared:
        if not chat_id:
            error("--chat is required for shared todos")
            raise typer.Exit(1)
    else:
        if not user_id:
            error("--user is required for personal todos")
            raise typer.Exit(1)

    due_at: datetime | None = None
    if due is not None:
        due_at = _parse_time(due)
        if due_at is None:
            error(f"Could not parse due date: {due}")
            raise typer.Exit(1)

    manager = await _create_manager()
    todo, replayed = await manager.create(
        content=content,
        user_id=user_id,
        chat_id=chat_id,
        shared=shared,
        due_at=due_at,
    )

    if replayed:
        warning(f"Todo already exists: {todo.id}")
    else:
        success(f"Created todo {todo.id}: {todo.content[:50]}")


async def _todo_done(todo_id: str) -> None:
    manager = await _create_manager()
    user_id, chat_id = await _get_todo_scope(manager, todo_id)
    todo, replayed = await manager.complete(
        todo_id=todo_id, user_id=user_id, chat_id=chat_id
    )
    if replayed:
        warning(f"Todo {todo_id} already completed")
    else:
        success(f"Completed: {todo.content[:50]}")


async def _todo_undone(todo_id: str) -> None:
    manager = await _create_manager()
    user_id, chat_id = await _get_todo_scope(manager, todo_id)
    todo, replayed = await manager.uncomplete(
        todo_id=todo_id, user_id=user_id, chat_id=chat_id
    )
    if replayed:
        warning(f"Todo {todo_id} already open")
    else:
        success(f"Reopened: {todo.content[:50]}")


async def _todo_delete(todo_id: str) -> None:
    manager = await _create_manager()
    user_id, chat_id = await _get_todo_scope(manager, todo_id)
    todo, replayed = await manager.delete(
        todo_id=todo_id, user_id=user_id, chat_id=chat_id
    )
    if replayed:
        warning(f"Todo {todo_id} already deleted")
    else:
        success(f"Deleted: {todo.content[:50]}")


async def _todo_edit(
    *,
    todo_id: str,
    content: str | None,
    due: str | None,
    clear_due: bool,
) -> None:
    due_at: datetime | None = None
    if due is not None:
        due_at = _parse_time(due)
        if due_at is None:
            error(f"Could not parse due date: {due}")
            raise typer.Exit(1)

    manager = await _create_manager()
    user_id, chat_id = await _get_todo_scope(manager, todo_id)
    todo, replayed = await manager.update(
        todo_id=todo_id,
        user_id=user_id,
        chat_id=chat_id,
        content=content,
        due_at=due_at,
        clear_due_at=clear_due,
    )
    if replayed:
        warning(f"Todo {todo_id} already updated")
    else:
        success(f"Updated: {todo.content[:50]}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_due(due_at: datetime) -> str:
    now = datetime.now(UTC)
    label = due_at.strftime("%Y-%m-%d %H:%M")
    if due_at < now:
        return f"[red]{label}[/red]"
    return label


def _parse_time(time_str: str) -> datetime | None:
    from ash.config import get_default_config, load_config

    try:
        config = load_config()
    except FileNotFoundError:
        config = get_default_config()
    timezone = config.timezone

    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    import dateparser

    settings: dict = {
        "TIMEZONE": timezone,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
    }
    parsed = dateparser.parse(time_str, settings=settings)
    if parsed:
        return parsed.astimezone(UTC)
    return None
