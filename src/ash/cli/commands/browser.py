"""Browser session inspection and lifecycle commands."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import typer

from ash.browser import create_browser_manager
from ash.browser.types import BrowserActionResult
from ash.cli.console import console, create_table, error, success
from ash.config import load_config
from ash.config.paths import get_config_path

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="browser",
    help="Inspect and manage browser sessions.",
    no_args_is_help=True,
)


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="browser")


async def _browser_action_or_raise(
    manager,
    *,
    action: str,
    user_id: str,
    provider: str,
    session_id: str | None = None,
    session_name: str | None = None,
    params: dict | None = None,
) -> BrowserActionResult:
    result = await manager.execute_action(
        action=action,
        effective_user_id=user_id,
        provider_name=provider,
        session_id=session_id,
        session_name=session_name,
        params=params or {},
    )
    if result.ok:
        return result
    raise RuntimeError(result.error_message or f"{action} failed")


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


@app.command("smoke")
def smoke_browser(
    url: str = typer.Argument(..., help="URL to validate in browser runtime."),
    user_id: str = typer.Option("unknown", help="Effective user id scope."),
    provider: str = typer.Option("sandbox", help="Browser provider."),
    cleanup: bool = typer.Option(
        True, help="Archive smoke-test session after completion."
    ),
) -> None:
    """Run end-to-end browser smoke (start -> goto -> extract -> screenshot)."""
    config = load_config(get_config_path())
    manager = create_browser_manager(config)

    session_name = f"smoke-{uuid.uuid4().hex[:8]}"
    session_id: str | None = None
    archived = False

    try:
        started = asyncio.run(
            _browser_action_or_raise(
                manager,
                action="session.start",
                user_id=user_id,
                provider=provider,
                session_name=session_name,
            )
        )
        session_id = started.session_id
        if not session_id:
            raise RuntimeError("session.start did not return a session id")

        goto = asyncio.run(
            _browser_action_or_raise(
                manager,
                action="page.goto",
                user_id=user_id,
                provider=provider,
                session_id=session_id,
                params={"url": url},
            )
        )
        title = asyncio.run(
            _browser_action_or_raise(
                manager,
                action="page.extract",
                user_id=user_id,
                provider=provider,
                session_id=session_id,
                params={"mode": "title"},
            )
        )
        text = asyncio.run(
            _browser_action_or_raise(
                manager,
                action="page.extract",
                user_id=user_id,
                provider=provider,
                session_id=session_id,
                params={"mode": "text", "max_chars": 280},
            )
        )
        screenshot = asyncio.run(
            _browser_action_or_raise(
                manager,
                action="page.screenshot",
                user_id=user_id,
                provider=provider,
                session_id=session_id,
            )
        )

        if cleanup:
            asyncio.run(
                _browser_action_or_raise(
                    manager,
                    action="session.archive",
                    user_id=user_id,
                    provider=provider,
                    session_id=session_id,
                )
            )
            archived = True

        table = create_table(
            "Browser Smoke",
            [
                ("Field", "cyan"),
                ("Value", "white"),
            ],
        )
        table.add_row("session_id", session_id)
        table.add_row("provider", str(goto.provider or provider))
        table.add_row("url", str(goto.page_url or url))
        table.add_row("title", str(title.data.get("title", "")))
        table.add_row(
            "text_preview",
            str(text.data.get("text", ""))[:120].replace("\n", " ").strip() or "-",
        )
        table.add_row(
            "screenshot",
            screenshot.artifact_refs[0] if screenshot.artifact_refs else "-",
        )
        table.add_row("archived", "yes" if archived else "no")
        console.print(table)
        success("Browser smoke passed")
    except Exception as e:
        if cleanup and session_id and not archived:
            try:
                asyncio.run(
                    manager.execute_action(
                        action="session.archive",
                        effective_user_id=user_id,
                        provider_name=provider,
                        session_id=session_id,
                    )
                )
            except Exception:
                logger.warning("browser_smoke_cleanup_failed", exc_info=True)
        error(str(e))
        raise typer.Exit(1) from e
