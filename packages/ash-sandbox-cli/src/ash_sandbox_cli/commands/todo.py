"""Todo management commands for sandboxed CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

import typer

from ash_sandbox_cli.rpc import RPCError, get_context_params, rpc_call

app = typer.Typer(
    name="todo", help="Manage canonical todo items.", no_args_is_help=True
)


def _get_context() -> dict[str, str]:
    context = get_context_params()
    return {
        "user_id": context.get("user_id") or "",
        "chat_id": context.get("chat_id") or "",
        "chat_title": context.get("chat_title") or "",
        "provider": context.get("provider") or "",
        "username": context.get("username") or "",
        "timezone": context.get("timezone") or "UTC",
    }


def _parse_time(time_str: str, timezone: str) -> datetime | None:
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    import dateparser

    parsed = dateparser.parse(
        time_str,
        settings={
            "TIMEZONE": timezone,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
        },
    )
    if parsed:
        return parsed.astimezone(UTC)
    return None


def _call(method: str, params: dict[str, Any]) -> Any:
    try:
        return rpc_call(method, params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None


def _todo_status(todo: dict[str, Any]) -> str:
    return str(todo.get("status", "open")).lower()


def _todo_marker(todo: dict[str, Any]) -> str:
    return "[x]" if _todo_status(todo) == "done" else "[ ]"


def _echo_todo_block(
    todo: dict[str, Any],
    *,
    prefix: str | None = None,
    include_reminder: bool = True,
) -> None:
    if prefix:
        typer.echo(prefix)
    typer.echo(f"  ID: {todo.get('id', '?')}")
    typer.echo(f"  Status: {_todo_status(todo)} {_todo_marker(todo)}")
    typer.echo(f"  Task: {todo.get('content', '')}")
    if todo.get("due_at"):
        typer.echo(f"  Due: {todo['due_at']}")
    if include_reminder and todo.get("linked_schedule_entry_id"):
        typer.echo(f"  Reminder: {todo['linked_schedule_entry_id']}")


@app.command("add")
def add_todo(
    content: Annotated[str, typer.Argument(help="Todo text")],
    due: Annotated[
        str | None,
        typer.Option("--due", help="Optional due time (ISO or natural language)"),
    ] = None,
    shared: Annotated[
        bool,
        typer.Option("--shared", help="Create shared todo in current chat"),
    ] = False,
) -> None:
    ctx = _get_context()
    due_at: str | None = None
    if due:
        parsed = _parse_time(due, ctx["timezone"])
        if parsed is None:
            typer.echo(f"Error: Could not parse due time: {due}", err=True)
            raise typer.Exit(1)
        due_at = parsed.isoformat()

    params: dict[str, Any] = {
        "content": content,
        "shared": shared,
        "user_id": ctx["user_id"] or None,
        "chat_id": ctx["chat_id"] or None,
    }
    if due_at:
        params["due_at"] = due_at

    result = _call("todo.create", params)
    todo = result["todo"]
    _echo_todo_block(todo, prefix="Created todo:")


@app.command("list")
def list_todos(
    all_items: Annotated[
        bool,
        typer.Option("--all", help="Include done and deleted todos"),
    ] = False,
    include_done: Annotated[
        bool,
        typer.Option("--include-done", help="Include completed todos"),
    ] = False,
    include_deleted: Annotated[
        bool,
        typer.Option("--include-deleted", help="Include deleted todos"),
    ] = False,
) -> None:
    if all_items:
        include_done = True
        include_deleted = True
    ctx = _get_context()
    params = {
        "user_id": ctx["user_id"] or None,
        "chat_id": ctx["chat_id"] or None,
        "include_done": include_done,
        "include_deleted": include_deleted,
    }
    todos = _call("todo.list", params)
    if not todos:
        typer.echo("No todos found.")
        return

    typer.echo("Todos:")
    for todo in todos:
        _echo_todo_block(todo, prefix="-")
    typer.echo(f"Total: {len(todos)} todo(s)")


@app.command("edit")
def edit_todo(
    todo_id: Annotated[str, typer.Option("--id", "-i", help="Todo ID")],
    text: Annotated[
        str | None,
        typer.Option("--text", help="Updated todo text"),
    ] = None,
    due: Annotated[
        str | None,
        typer.Option("--due", help="Updated due time (ISO or natural language)"),
    ] = None,
    clear_due: Annotated[
        bool,
        typer.Option("--clear-due", help="Clear due date"),
    ] = False,
) -> None:
    ctx = _get_context()
    due_at: str | None = None
    if due is not None:
        parsed = _parse_time(due, ctx["timezone"])
        if parsed is None:
            typer.echo(f"Error: Could not parse due time: {due}", err=True)
            raise typer.Exit(1)
        due_at = parsed.isoformat()

    params: dict[str, Any] = {
        "todo_id": todo_id,
        "user_id": ctx["user_id"] or None,
        "chat_id": ctx["chat_id"] or None,
        "clear_due_at": clear_due,
    }
    if text is not None:
        params["content"] = text
    if due_at is not None:
        params["due_at"] = due_at

    result = _call("todo.update", params)
    todo = result["todo"]
    _echo_todo_block(todo, prefix="Updated todo:")


@app.command("done")
def complete_todo(
    todo_id: Annotated[str, typer.Option("--id", "-i", help="Todo ID")],
) -> None:
    ctx = _get_context()
    result = _call(
        "todo.complete",
        {
            "todo_id": todo_id,
            "user_id": ctx["user_id"] or None,
            "chat_id": ctx["chat_id"] or None,
        },
    )
    _echo_todo_block(result["todo"], prefix="Completed todo:")


@app.command("undone")
def uncomplete_todo(
    todo_id: Annotated[str, typer.Option("--id", "-i", help="Todo ID")],
) -> None:
    ctx = _get_context()
    result = _call(
        "todo.uncomplete",
        {
            "todo_id": todo_id,
            "user_id": ctx["user_id"] or None,
            "chat_id": ctx["chat_id"] or None,
        },
    )
    _echo_todo_block(result["todo"], prefix="Reopened todo:")


@app.command("delete")
def delete_todo(
    todo_id: Annotated[str, typer.Option("--id", "-i", help="Todo ID")],
) -> None:
    ctx = _get_context()
    result = _call(
        "todo.delete",
        {
            "todo_id": todo_id,
            "user_id": ctx["user_id"] or None,
            "chat_id": ctx["chat_id"] or None,
        },
    )
    _echo_todo_block(result["todo"], prefix="Deleted todo:")


@app.command("remind")
def remind_todo(
    todo_id: Annotated[str, typer.Option("--id", "-i", help="Todo ID")],
    at: Annotated[
        str | None,
        typer.Option("--at", help="One-time reminder time"),
    ] = None,
    cron: Annotated[
        str | None,
        typer.Option("--cron", help="Recurring reminder cron"),
    ] = None,
    timezone: Annotated[
        str | None,
        typer.Option("--tz", help="Timezone override"),
    ] = None,
) -> None:
    if (at is None and cron is None) or (at is not None and cron is not None):
        typer.echo("Error: Must specify exactly one of --at or --cron", err=True)
        raise typer.Exit(1)

    ctx = _get_context()
    tz = timezone or ctx["timezone"]

    trigger_at: str | None = None
    if at is not None:
        parsed = _parse_time(at, tz)
        if parsed is None:
            typer.echo(f"Error: Could not parse reminder time: {at}", err=True)
            raise typer.Exit(1)
        trigger_at = parsed.isoformat()

    params: dict[str, Any] = {
        "todo_id": todo_id,
        "user_id": ctx["user_id"] or None,
        "chat_id": ctx["chat_id"] or None,
        "chat_title": ctx["chat_title"] or None,
        "provider": ctx["provider"] or None,
        "username": ctx["username"] or None,
        "timezone": tz,
    }
    if trigger_at:
        params["reminder_at"] = trigger_at
    if cron:
        params["reminder_cron"] = cron

    result = _call("todo.update", params)
    linked = result.get("todo", {}).get("linked_schedule_entry_id", "?")
    typer.echo(f"Linked reminder for todo {todo_id}: schedule_id={linked}")


@app.command("unremind")
def unremind_todo(
    todo_id: Annotated[str, typer.Option("--id", "-i", help="Todo ID")],
) -> None:
    ctx = _get_context()
    _call(
        "todo.update",
        {
            "todo_id": todo_id,
            "user_id": ctx["user_id"] or None,
            "chat_id": ctx["chat_id"] or None,
            "clear_reminder": True,
        },
    )
    typer.echo(f"Removed reminder for todo {todo_id}")
