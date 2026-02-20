"""Authentication management commands."""

import asyncio
import time
from typing import Annotated

import typer

from ash.cli.console import console, error, success


def register(app: typer.Typer) -> None:
    """Register the auth command group."""

    auth_app = typer.Typer(name="auth", help="Manage OAuth authentication")

    @auth_app.command()
    def login(
        provider: Annotated[
            str,
            typer.Argument(help="Provider to authenticate with"),
        ] = "openai-oauth",
    ) -> None:
        """Log in to an OAuth provider.

        Currently supported: openai-oauth (ChatGPT Plus/Pro)

        Examples:
            ash auth login                  # Login to OpenAI OAuth (default)
            ash auth login openai-oauth     # Explicit provider
        """
        if provider != "openai-oauth":
            error(f"Unknown provider: {provider}. Supported: openai-oauth")
            raise typer.Exit(1)

        try:
            asyncio.run(_login_openai_oauth())
        except KeyboardInterrupt:
            console.print("\n[dim]Login cancelled.[/dim]")
            raise typer.Exit(1) from None
        except RuntimeError as e:
            error(str(e))
            raise typer.Exit(1) from None

    @auth_app.command()
    def status() -> None:
        """Show authentication status for all providers."""
        from ash.auth.storage import AuthStorage

        storage = AuthStorage()
        providers = storage.list_providers()

        if not providers:
            console.print("[dim]No authenticated providers.[/dim]")
            console.print("Run [bold]ash auth login[/bold] to authenticate.")
            return

        for provider_id in providers:
            creds = storage.load(provider_id)
            if not creds:
                console.print(f"  [bold]{provider_id}[/bold]: [red]invalid[/red]")
                continue

            now = time.time()
            if now < creds.expires:
                remaining = int(creds.expires - now)
                hours, remainder = divmod(remaining, 3600)
                minutes = remainder // 60
                expiry_text = f"[green]valid[/green] (expires in {hours}h {minutes}m)"
            else:
                expiry_text = "[yellow]expired[/yellow] (will refresh on next use)"

            console.print(f"  [bold]{provider_id}[/bold]: {expiry_text}")
            console.print(f"    Account ID: {creds.account_id[:12]}...")

    @auth_app.command()
    def logout(
        provider: Annotated[
            str,
            typer.Argument(help="Provider to log out from"),
        ] = "openai-oauth",
    ) -> None:
        """Remove stored credentials for a provider.

        Examples:
            ash auth logout                 # Logout from OpenAI OAuth (default)
            ash auth logout openai-oauth    # Explicit provider
        """
        from ash.auth.storage import AuthStorage

        storage = AuthStorage()
        if storage.remove(provider):
            success(f"Logged out from {provider}")
        else:
            console.print(f"[dim]No credentials found for {provider}.[/dim]")

    app.add_typer(auth_app)


async def _login_openai_oauth() -> None:
    """Run the OpenAI OAuth login flow and save credentials."""
    from ash.auth.oauth import login_openai_oauth
    from ash.auth.storage import AuthStorage, OAuthCredentials

    result = await login_openai_oauth()

    creds = OAuthCredentials(
        access=str(result["access"]),
        refresh=str(result["refresh"]),
        expires=float(result["expires"]),
        account_id=str(result["account_id"]),
    )

    storage = AuthStorage()
    storage.save("openai-oauth", creds)

    success("Authenticated with OpenAI OAuth")
    console.print(f"  Account ID: {creds.account_id[:12]}...")
    console.print("  Credentials saved to ~/.ash/auth.json")
