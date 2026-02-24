"""Capability management commands for sandboxed CLI."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from ash_sandbox_cli.rpc import RPCError, rpc_call

app = typer.Typer(
    name="capability",
    help="List and invoke host-managed capabilities.",
    no_args_is_help=True,
)
auth_app = typer.Typer(
    name="auth",
    help="Capability authentication flows.",
    no_args_is_help=True,
)


def _call(method: str, params: dict[str, Any]) -> Any:
    try:
        return rpc_call(method, params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None


def _parse_input_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON input: {e}") from e
    if not isinstance(value, dict):
        raise ValueError("input JSON must decode to an object")
    return value


@app.command("list")
def list_capabilities(
    include_unavailable: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Include unavailable capabilities (e.g. blocked by current chat policy).",
        ),
    ] = False,
) -> None:
    """List capabilities visible to the current caller scope."""
    result = _call(
        "capability.list",
        {"include_unavailable": include_unavailable},
    )
    capabilities = result.get("capabilities", [])
    if not capabilities:
        typer.echo("No capabilities available.")
        return

    typer.echo("Capabilities:")
    for capability in capabilities:
        capability_id = capability.get("id", "?")
        description = capability.get("description", "")
        available = "yes" if capability.get("available") else "no"
        authenticated = "yes" if capability.get("authenticated") else "no"
        typer.echo(f"- {capability_id}: {description}")
        typer.echo(f"  Available: {available}")
        typer.echo(f"  Authenticated: {authenticated}")
        operations = capability.get("operations") or []
        if isinstance(operations, list) and operations:
            typer.echo(f"  Operations: {', '.join(str(item) for item in operations)}")
    typer.echo(f"Total: {len(capabilities)} capability(ies)")


@app.command("invoke")
def invoke_capability(
    capability: Annotated[
        str,
        typer.Option("--capability", "-c", help="Namespaced capability id"),
    ],
    operation: Annotated[
        str,
        typer.Option("--operation", "-o", help="Operation name"),
    ],
    input_json: Annotated[
        str,
        typer.Option(
            "--input-json",
            help="JSON object for operation input",
        ),
    ] = "{}",
    idempotency_key: Annotated[
        str | None,
        typer.Option("--idempotency-key", help="Optional idempotency key"),
    ] = None,
) -> None:
    """Invoke one capability operation."""
    try:
        operation_input = _parse_input_json(input_json)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    params: dict[str, Any] = {
        "capability": capability,
        "operation": operation,
        "input": operation_input,
    }
    if idempotency_key:
        params["idempotency_key"] = idempotency_key

    result = _call("capability.invoke", params)
    request_id = result.get("request_id", "?")
    output = result.get("output", {})
    typer.echo(f"Capability invocation succeeded (request_id={request_id})")
    typer.echo(f"  Capability: {capability}")
    typer.echo(f"  Operation: {operation}")
    typer.echo(f"  Output: {json.dumps(output, ensure_ascii=True, sort_keys=True)}")


@auth_app.command("begin")
def auth_begin(
    capability: Annotated[
        str,
        typer.Option("--capability", "-c", help="Namespaced capability id"),
    ],
    account_hint: Annotated[
        str | None,
        typer.Option("--account", help="Optional account reference hint"),
    ] = None,
) -> None:
    """Start capability auth flow."""
    params: dict[str, Any] = {"capability": capability}
    if account_hint:
        params["account_hint"] = account_hint
    result = _call("capability.auth.begin", params)
    typer.echo(f"Started capability auth flow (flow_id={result.get('flow_id', '?')})")
    typer.echo(f"  Capability: {capability}")
    typer.echo(f"  Auth URL: {result.get('auth_url', '')}")
    typer.echo(f"  Expires: {result.get('expires_at', '')}")


@auth_app.command("complete")
def auth_complete(
    flow_id: Annotated[
        str,
        typer.Option("--flow-id", help="Auth flow id from auth-begin"),
    ],
    callback_url: Annotated[
        str | None,
        typer.Option("--callback-url", help="OAuth callback URL"),
    ] = None,
    code: Annotated[
        str | None,
        typer.Option("--code", help="Authorization code"),
    ] = None,
) -> None:
    """Complete capability auth flow."""
    if not callback_url and not code:
        typer.echo("Error: Must specify either --callback-url or --code", err=True)
        raise typer.Exit(1)

    params: dict[str, Any] = {"flow_id": flow_id}
    if callback_url:
        params["callback_url"] = callback_url
    if code:
        params["code"] = code

    result = _call("capability.auth.complete", params)
    if not result.get("ok"):
        typer.echo("Error: capability auth completion failed", err=True)
        raise typer.Exit(1)
    typer.echo(
        "Capability auth completed "
        f"(flow_id={flow_id}, account_ref={result.get('account_ref', '')})"
    )


app.add_typer(auth_app, name="auth")
