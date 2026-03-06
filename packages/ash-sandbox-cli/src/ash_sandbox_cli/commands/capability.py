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
        linked_accounts = capability.get("linked_accounts") or []
        if isinstance(linked_accounts, list) and linked_accounts:
            labels: list[str] = []
            for item in linked_accounts:
                if not isinstance(item, dict):
                    continue
                account_ref = str(item.get("account_ref", "")).strip()
                raw_email = item.get("account_email")
                account_email = str(raw_email).strip() if raw_email is not None else ""
                if not account_ref:
                    continue
                if account_email:
                    labels.append(f"{account_ref} ({account_email})")
                else:
                    labels.append(account_ref)
            if labels:
                typer.echo(f"  Accounts: {', '.join(labels)}")
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
    account: Annotated[
        str | None,
        typer.Option("--account", help="Optional linked account alias"),
    ] = None,
    mutation_plan_id: Annotated[
        str | None,
        typer.Option(
            "--plan-id",
            help="Optional mutation plan id for host confirmation proof",
        ),
    ] = None,
    target_fingerprint: Annotated[
        str | None,
        typer.Option(
            "--target-fingerprint",
            help="Optional target fingerprint for host confirmation proof",
        ),
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
    if account:
        params["account_ref"] = account
    if mutation_plan_id:
        params["mutation_plan_id"] = mutation_plan_id
    if target_fingerprint:
        params["target_fingerprint"] = target_fingerprint

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
    flow_type = result.get("flow_type", "authorization_code")
    typer.echo(f"  Flow type: {flow_type}")
    if result.get("user_code"):
        typer.echo(f"  User code: {result['user_code']}")
    if result.get("poll_interval_seconds") is not None:
        typer.echo(f"  Poll interval: {result['poll_interval_seconds']}s")
    typer.echo(f"  Expires: {result.get('expires_at', '')}")


@auth_app.command("list")
def auth_list(
    capability: Annotated[
        str | None,
        typer.Option("--capability", "-c", help="Optional namespaced capability id"),
    ] = None,
    account_hint: Annotated[
        str | None,
        typer.Option("--account", help="Optional account reference hint"),
    ] = None,
) -> None:
    """List pending capability auth flows for the current caller."""
    params: dict[str, Any] = {}
    if capability:
        params["capability"] = capability
    if account_hint:
        params["account_hint"] = account_hint
    result = _call("capability.auth.list", params)
    flows = result.get("flows") or []
    if not isinstance(flows, list) or not flows:
        typer.echo("No pending capability auth flows.")
        return

    typer.echo("Pending capability auth flows:")
    for flow in flows:
        if not isinstance(flow, dict):
            continue
        flow_id = str(flow.get("flow_id", "")).strip() or "?"
        capability_id = str(flow.get("capability", "")).strip()
        account = str(flow.get("account_hint", "")).strip()
        if capability_id and account:
            typer.echo(f"- {flow_id} ({capability_id}, account={account})")
        elif capability_id:
            typer.echo(f"- {flow_id} ({capability_id})")
        else:
            typer.echo(f"- {flow_id}")
        typer.echo(f"  Auth URL: {flow.get('auth_url', '')}")
        typer.echo(f"  Flow type: {flow.get('flow_type', 'authorization_code')}")
        if flow.get("user_code"):
            typer.echo(f"  User code: {flow['user_code']}")
        if flow.get("poll_interval_seconds") is not None:
            typer.echo(f"  Poll interval: {flow['poll_interval_seconds']}s")
        typer.echo(f"  Expires: {flow.get('expires_at', '')}")


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


@auth_app.command("complete-callback")
def auth_complete_callback(
    callback_url: Annotated[
        str | None,
        typer.Option("--callback-url", help="OAuth callback URL"),
    ] = None,
    code: Annotated[
        str | None,
        typer.Option("--code", help="Authorization code"),
    ] = None,
    capability: Annotated[
        str | None,
        typer.Option("--capability", "-c", help="Optional namespaced capability id"),
    ] = None,
    account_hint: Annotated[
        str | None,
        typer.Option("--account", help="Optional account reference hint"),
    ] = None,
) -> None:
    """Complete capability auth by callback/code with host-side flow resolution."""
    if not callback_url and not code:
        typer.echo("Error: Must specify either --callback-url or --code", err=True)
        raise typer.Exit(1)

    params: dict[str, Any] = {}
    if callback_url:
        params["callback_url"] = callback_url
    if code:
        params["code"] = code
    if capability:
        params["capability"] = capability
    if account_hint:
        params["account_hint"] = account_hint

    result = _call("capability.auth.complete_callback", params)
    if not result.get("ok"):
        typer.echo("Error: capability auth completion failed", err=True)
        raise typer.Exit(1)
    typer.echo(
        "Capability auth completed "
        f"(flow_id={result.get('flow_id', '')}, "
        f"capability={result.get('capability', '')}, "
        f"account_ref={result.get('account_ref', '')})"
    )


@auth_app.command("poll")
def auth_poll(
    flow_id: Annotated[
        str,
        typer.Option("--flow-id", help="Auth flow id from auth-begin"),
    ],
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout", help="Block and poll until complete or timeout (seconds)"
        ),
    ] = None,
    interval: Annotated[
        int | None,
        typer.Option("--interval", help="Override poll interval (seconds)"),
    ] = None,
) -> None:
    """Poll a device code auth flow for completion."""
    import time

    params: dict[str, Any] = {"flow_id": flow_id}
    result = _call("capability.auth.poll", params)

    if timeout is None:
        # Single poll
        if result.get("ok"):
            typer.echo(
                f"Capability auth completed "
                f"(flow_id={flow_id}, account_ref={result.get('account_ref', '')})"
            )
        else:
            retry = result.get("retry_after_seconds", 5)
            typer.echo(f"Auth flow pending (flow_id={flow_id}, retry_after={retry}s)")
        return

    # Blocking poll loop
    deadline = time.monotonic() + max(1, timeout)
    while True:
        if result.get("ok"):
            typer.echo(
                f"Capability auth completed "
                f"(flow_id={flow_id}, account_ref={result.get('account_ref', '')})"
            )
            return

        retry_after = result.get("retry_after_seconds") or 5
        sleep_seconds = interval if interval is not None else retry_after
        sleep_seconds = max(1, sleep_seconds)

        if time.monotonic() + sleep_seconds > deadline:
            typer.echo(f"Auth flow timed out (flow_id={flow_id})", err=True)
            raise typer.Exit(1)

        time.sleep(sleep_seconds)
        result = _call("capability.auth.poll", params)


app.add_typer(auth_app, name="auth")
