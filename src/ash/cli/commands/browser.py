"""Browser session inspection and lifecycle commands."""

from __future__ import annotations

import json

import typer

from ash.browser import create_browser_manager
from ash.cli.console import console, create_table, error, success
from ash.config import load_config
from ash.config.paths import get_config_path

app = typer.Typer(
    name="browser",
    help="Inspect and manage browser sessions.",
    no_args_is_help=True,
)


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="browser")


@app.command("list")
def list_sessions(
    user_id: str = typer.Option("unknown", help="Effective user id scope."),
    include_archived: bool = typer.Option(
        False, "--archived", help="Include archived sessions."
    ),
) -> None:
    config = load_config(get_config_path())
    manager = create_browser_manager(config)
    sessions = manager.store.list_sessions(
        effective_user_id=user_id,
        include_archived=include_archived,
    )

    table = create_table(
        "Browser Sessions",
        [
            ("ID", "cyan"),
            ("Name", "white"),
            ("Provider", "magenta"),
            ("Status", "white"),
            ("Profile", "green"),
            ("URL", "white"),
            ("Updated", "white"),
        ],
    )
    for row in sessions:
        table.add_row(
            row.id[:8],
            row.name,
            row.provider,
            row.status,
            row.profile_name or "-",
            row.current_url or "-",
            row.updated_at.isoformat(timespec="seconds"),
        )
    console.print(table)


@app.command("show")
def show_session(
    session: str = typer.Argument(..., help="Session id prefix or exact session name."),
    user_id: str = typer.Option("unknown", help="Effective user id scope."),
    provider: str = typer.Option("sandbox", help="Provider for name lookups."),
) -> None:
    config = load_config(get_config_path())
    manager = create_browser_manager(config)

    target = manager.store.get_session(session)
    if target is None:
        target = manager.store.get_session_by_name(
            name=session,
            effective_user_id=user_id,
            provider=provider,
            include_archived=True,
        )
    if target is None:
        # Prefix lookup fallback
        matches = [
            s
            for s in manager.store.list_sessions(
                effective_user_id=user_id,
                include_archived=True,
            )
            if s.id.startswith(session)
        ]
        target = matches[0] if len(matches) == 1 else None

    if target is None:
        error("Browser session not found")
        raise typer.Exit(1)

    console.print(
        json.dumps(
            {
                "id": target.id,
                "name": target.name,
                "provider": target.provider,
                "status": target.status,
                "profile_name": target.profile_name,
                "current_url": target.current_url,
                "created_at": target.created_at.isoformat(),
                "updated_at": target.updated_at.isoformat(),
                "metadata": target.metadata,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


@app.command("archive")
def archive_session(
    session: str = typer.Argument(..., help="Session id prefix or exact session name."),
    user_id: str = typer.Option("unknown", help="Effective user id scope."),
    provider: str = typer.Option("sandbox", help="Provider for name lookups."),
) -> None:
    config = load_config(get_config_path())
    manager = create_browser_manager(config)

    resolved_id: str | None = None
    if manager.store.get_session(session):
        resolved_id = session
    else:
        target = manager.store.get_session_by_name(
            name=session,
            effective_user_id=user_id,
            provider=provider,
            include_archived=True,
        )
        if target is None:
            matches = [
                s
                for s in manager.store.list_sessions(
                    effective_user_id=user_id,
                    include_archived=True,
                )
                if s.id.startswith(session)
            ]
            if len(matches) == 1:
                target = matches[0]
        if target is not None:
            resolved_id = target.id

    if not resolved_id:
        error("Browser session not found")
        raise typer.Exit(1)

    # archive via manager action path for consistent state transitions
    import asyncio

    payload = asyncio.run(
        manager.execute_action(
            action="session.archive",
            effective_user_id=user_id,
            provider_name=provider,
            session_id=resolved_id,
        )
    )
    if not payload.ok:
        error(payload.error_message or "Failed to archive session")
        raise typer.Exit(1)
    success(f"Archived browser session {resolved_id[:8]}")
