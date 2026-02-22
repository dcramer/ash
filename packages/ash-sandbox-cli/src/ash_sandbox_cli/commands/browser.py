"""Browser commands for sandboxed CLI."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from ash_sandbox_cli.rpc import RPCError, get_context_params, rpc_call

app = typer.Typer(
    name="browser",
    help="Control browser sessions and page actions.",
    no_args_is_help=True,
)


def _base_params(provider: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    context = get_context_params()
    if context.get("user_id"):
        params["user_id"] = context["user_id"]
    if provider:
        params["provider"] = provider
    return params


def _merge_session_ref(
    params: dict[str, Any],
    *,
    session_id: str | None,
    session_name: str | None,
) -> dict[str, Any]:
    if session_id:
        params["session_id"] = session_id
    if session_name:
        params["session_name"] = session_name
    return params


def _print_result(result: dict[str, Any]) -> None:
    typer.echo(json.dumps(result, indent=2, ensure_ascii=True))
    if not bool(result.get("ok")):
        raise typer.Exit(1)


def _call(method: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        return rpc_call(method, params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None


@app.command("start")
def start_session(
    name: Annotated[
        str | None,
        typer.Option("--name", help="Optional session name."),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Optional profile name."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider override: sandbox|kernel."),
    ] = None,
) -> None:
    """Start a browser session."""
    params = _base_params(provider)
    if name:
        params["session_name"] = name
    if profile:
        params["profile_name"] = profile
    _print_result(_call("browser.session.start", params))


@app.command("list")
def list_sessions(
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider filter override."),
    ] = None,
) -> None:
    """List browser sessions."""
    params = _base_params(provider)
    _print_result(_call("browser.session.list", params))


@app.command("show")
def show_session(
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Session id."),
    ] = None,
    session_name: Annotated[
        str | None,
        typer.Option("--session-name", help="Session name."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider override: sandbox|kernel."),
    ] = None,
) -> None:
    """Show browser session details."""
    params = _merge_session_ref(
        _base_params(provider),
        session_id=session_id,
        session_name=session_name,
    )
    _print_result(_call("browser.session.show", params))


@app.command("close")
def close_session(
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Session id."),
    ] = None,
    session_name: Annotated[
        str | None,
        typer.Option("--session-name", help="Session name."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider override: sandbox|kernel."),
    ] = None,
) -> None:
    """Close a browser session."""
    params = _merge_session_ref(
        _base_params(provider),
        session_id=session_id,
        session_name=session_name,
    )
    _print_result(_call("browser.session.close", params))


@app.command("archive")
def archive_session(
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Session id."),
    ] = None,
    session_name: Annotated[
        str | None,
        typer.Option("--session-name", help="Session name."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider override: sandbox|kernel."),
    ] = None,
) -> None:
    """Archive a browser session."""
    params = _merge_session_ref(
        _base_params(provider),
        session_id=session_id,
        session_name=session_name,
    )
    _print_result(_call("browser.session.archive", params))


@app.command("goto")
def page_goto(
    url: Annotated[str, typer.Argument(help="URL to open.")],
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Session id."),
    ] = None,
    session_name: Annotated[
        str | None,
        typer.Option("--session-name", help="Session name."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider override: sandbox|kernel."),
    ] = None,
) -> None:
    """Navigate to a URL."""
    params = _merge_session_ref(
        _base_params(provider),
        session_id=session_id,
        session_name=session_name,
    )
    params["url"] = url
    _print_result(_call("browser.page.goto", params))


@app.command("extract")
def page_extract(
    mode: Annotated[
        str,
        typer.Option("--mode", help="Extract mode: text|title."),
    ] = "text",
    selector: Annotated[
        str | None,
        typer.Option("--selector", help="Optional CSS selector."),
    ] = None,
    max_chars: Annotated[
        int,
        typer.Option("--max-chars", help="Maximum characters."),
    ] = 3000,
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Session id."),
    ] = None,
    session_name: Annotated[
        str | None,
        typer.Option("--session-name", help="Session name."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider override: sandbox|kernel."),
    ] = None,
) -> None:
    """Extract content from the current page."""
    params = _merge_session_ref(
        _base_params(provider),
        session_id=session_id,
        session_name=session_name,
    )
    params["mode"] = mode
    params["max_chars"] = max_chars
    if selector:
        params["selector"] = selector
    _print_result(_call("browser.page.extract", params))


@app.command("click")
def page_click(
    selector: Annotated[str, typer.Argument(help="CSS selector to click.")],
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Session id."),
    ] = None,
    session_name: Annotated[
        str | None,
        typer.Option("--session-name", help="Session name."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider override: sandbox|kernel."),
    ] = None,
) -> None:
    """Click an element on the current page."""
    params = _merge_session_ref(
        _base_params(provider),
        session_id=session_id,
        session_name=session_name,
    )
    params["selector"] = selector
    _print_result(_call("browser.page.click", params))


@app.command("type")
def page_type(
    selector: Annotated[str, typer.Argument(help="CSS selector to type into.")],
    text: Annotated[str, typer.Argument(help="Text value.")],
    clear_first: Annotated[
        bool,
        typer.Option("--clear-first/--no-clear-first", help="Clear field first."),
    ] = True,
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Session id."),
    ] = None,
    session_name: Annotated[
        str | None,
        typer.Option("--session-name", help="Session name."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider override: sandbox|kernel."),
    ] = None,
) -> None:
    """Type text into an element on the current page."""
    params = _merge_session_ref(
        _base_params(provider),
        session_id=session_id,
        session_name=session_name,
    )
    params["selector"] = selector
    params["text"] = text
    params["clear_first"] = clear_first
    _print_result(_call("browser.page.type", params))


@app.command("wait")
def page_wait(
    selector: Annotated[str, typer.Argument(help="CSS selector to wait for.")],
    timeout_seconds: Annotated[
        float | None,
        typer.Option("--timeout-seconds", help="Optional timeout."),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Session id."),
    ] = None,
    session_name: Annotated[
        str | None,
        typer.Option("--session-name", help="Session name."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider override: sandbox|kernel."),
    ] = None,
) -> None:
    """Wait for an element on the current page."""
    params = _merge_session_ref(
        _base_params(provider),
        session_id=session_id,
        session_name=session_name,
    )
    params["selector"] = selector
    if timeout_seconds is not None:
        params["timeout_seconds"] = timeout_seconds
    _print_result(_call("browser.page.wait_for", params))


@app.command("screenshot")
def page_screenshot(
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Session id."),
    ] = None,
    session_name: Annotated[
        str | None,
        typer.Option("--session-name", help="Session name."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider override: sandbox|kernel."),
    ] = None,
) -> None:
    """Capture a screenshot."""
    params = _merge_session_ref(
        _base_params(provider),
        session_id=session_id,
        session_name=session_name,
    )
    _print_result(_call("browser.page.screenshot", params))
