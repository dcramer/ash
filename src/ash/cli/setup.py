"""Interactive setup wizard for Ash configuration."""

import os
import tomllib
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Model options by provider
ANTHROPIC_MODELS = [
    ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5 (Recommended - balanced)"),
    ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku (Fast, lower cost)"),
    ("claude-opus-4-5-20251101", "Claude Opus 4.5 (Most capable)"),
]

OPENAI_MODELS = [
    ("gpt-4o", "GPT-4o (Recommended - balanced)"),
    ("gpt-4o-mini", "GPT-4o Mini (Fast, lower cost)"),
    ("o1", "o1 (Reasoning model)"),
]

# Environment variables to check
ENV_VARS = [
    ("ANTHROPIC_API_KEY", "Anthropic"),
    ("OPENAI_API_KEY", "OpenAI"),
    ("TELEGRAM_BOT_TOKEN", "Telegram"),
    ("BRAVE_SEARCH_API_KEY", "Brave Search"),
]

# Available sections
SECTIONS = [
    ("models", "Models", "LLM provider and model selection", True),
    ("telegram", "Telegram", "Bot integration for messaging", False),
    ("search", "Web Search", "Brave Search API for web queries", False),
    ("advanced", "Advanced", "Sandbox, server, memory settings", False),
]


class SetupWizard:
    """Interactive setup wizard for Ash configuration."""

    def __init__(self, config_path: Path):
        """Initialize the setup wizard.

        Args:
            config_path: Path to the config file to create/modify.
        """
        self.config_path = config_path
        self.console = Console()
        self.config: dict = {}
        self.existing_config: dict = self._load_existing_config()

    def _load_existing_config(self) -> dict:
        """Load existing config file if it exists.

        Returns:
            Dictionary of existing config, or empty dict if no file exists.
        """
        if not self.config_path.exists():
            return {}

        try:
            with self.config_path.open("rb") as f:
                return tomllib.load(f)
        except Exception:
            return {}

    def _has_config_api_key(self, provider: str) -> bool:
        """Check if API key is set in existing config file.

        Args:
            provider: Provider name ('anthropic', 'openai').

        Returns:
            True if API key exists in config file.
        """
        # Check provider-level config
        if provider in self.existing_config:
            if self.existing_config[provider].get("api_key"):
                return True

        # Check legacy default_llm config
        default_llm = self.existing_config.get("default_llm", {})
        if default_llm.get("provider") == provider and default_llm.get("api_key"):
            return True

        return False

    def _has_config_telegram_token(self) -> bool:
        """Check if Telegram bot token is set in existing config file."""
        telegram = self.existing_config.get("telegram", {})
        return bool(telegram.get("bot_token"))

    def run(self, sections: list[str] | None = None) -> bool:
        """Run the setup wizard.

        Args:
            sections: Specific sections to configure, or None for interactive selection.

        Returns:
            True if setup completed successfully.
        """
        try:
            self._show_welcome()
            self._show_env_status()

            if sections is None:
                # Interactive mode: configure each section inline after selection
                self._configure_sections_interactive()
            else:
                # Explicit sections provided: configure them in order
                # Always include models if not explicitly provided
                if "models" not in sections:
                    sections = ["models"] + sections

                for section in sections:
                    method = getattr(self, f"_configure_{section}", None)
                    if method:
                        self.console.print()
                        method()
                    else:
                        self.console.print(f"[yellow]Unknown section: {section}[/yellow]")

            # Check if any configuration was added
            if not self.config:
                self.console.print("\n[yellow]No configuration changes made.[/yellow]")
                return False

            self._write_config()
            self._show_summary()
            return True

        except KeyboardInterrupt:
            self.console.print("\n\n[yellow]Setup cancelled.[/yellow]")
            return False

    def _show_welcome(self) -> None:
        """Show welcome message and config file location."""
        self.console.print()
        self.console.print(
            Panel.fit(
                "[bold]Welcome to Ash Setup[/bold]\n\n"
                "This wizard will help you configure Ash.\n"
                f"Config file: [cyan]{self.config_path}[/cyan]",
                border_style="blue",
            )
        )

    def _show_env_status(self) -> None:
        """Show status of API keys (both environment and config file)."""
        self.console.print("\n[bold]API Key Status:[/bold]")

        # Map env vars to config file check methods
        config_checks = {
            "ANTHROPIC_API_KEY": lambda: self._has_config_api_key("anthropic"),
            "OPENAI_API_KEY": lambda: self._has_config_api_key("openai"),
            "TELEGRAM_BOT_TOKEN": self._has_config_telegram_token,
            "BRAVE_SEARCH_API_KEY": lambda: False,  # Not stored in config
        }

        for var, name in ENV_VARS:
            has_env = bool(os.environ.get(var))
            has_config = config_checks.get(var, lambda: False)()

            if has_env and has_config:
                self.console.print(f"  [green]✓[/green] {name}: set [dim](env + config)[/dim]")
            elif has_env:
                self.console.print(f"  [green]✓[/green] {name}: set [dim](env)[/dim]")
            elif has_config:
                self.console.print(f"  [green]✓[/green] {name}: set [dim](config)[/dim]")
            else:
                self.console.print(f"  [dim]✗[/dim] {name}: [dim]not set[/dim]")

    def _configure_sections_interactive(self) -> None:
        """Interactively configure sections one at a time.

        Each section is configured immediately after the user selects it.
        """
        self.console.print("\n[bold]Configuration Sections:[/bold]")
        self.console.print("[dim]Models is required and always included.[/dim]\n")

        for key, name, description, required in SECTIONS:
            if required:
                self.console.print(f"  [green]✓[/green] {name} - {description} [dim](required)[/dim]")
                # Configure required sections immediately
                method = getattr(self, f"_configure_{key}", None)
                if method:
                    self.console.print()
                    method()
            else:
                if Confirm.ask(f"  Configure [cyan]{name}[/cyan]? ({description})", default=False):
                    # Configure immediately after user says yes
                    method = getattr(self, f"_configure_{key}", None)
                    if method:
                        self.console.print()
                        method()

    def _configure_models(self) -> None:
        """Configure model settings."""
        self.console.print(
            Panel.fit(
                "[bold]Model Configuration[/bold]",
                border_style="cyan",
            )
        )

        # Select provider
        self.console.print("\nSelect your primary LLM provider:")
        self.console.print("  [cyan]1[/cyan]. Anthropic (Claude models)")
        self.console.print("  [cyan]2[/cyan]. OpenAI (GPT models)")

        provider_choice = Prompt.ask(
            "Provider",
            choices=["1", "2"],
            default="1",
        )
        provider = "anthropic" if provider_choice == "1" else "openai"

        # Select model
        models = ANTHROPIC_MODELS if provider == "anthropic" else OPENAI_MODELS
        self.console.print(f"\nSelect {provider.title()} model:")
        for i, (model_id, description) in enumerate(models, 1):
            self.console.print(f"  [cyan]{i}[/cyan]. {description}")
            self.console.print(f"      [dim]{model_id}[/dim]")

        model_choice = Prompt.ask(
            "Model",
            choices=[str(i) for i in range(1, len(models) + 1)],
            default="1",
        )
        model = models[int(model_choice) - 1][0]

        # Check for API key in both environment and config file
        env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        has_env_key = bool(os.environ.get(env_var))
        has_config_key = self._has_config_api_key(provider)

        if has_env_key and has_config_key:
            self.console.print(f"\n[green]✓[/green] {env_var} found in environment and config file")
            self.console.print("[dim]Environment variable will be used if present.[/dim]")
        elif has_env_key:
            self.console.print(f"\n[green]✓[/green] {env_var} found in environment")
        elif has_config_key:
            self.console.print(f"\n[green]✓[/green] {env_var} found in config file")
            # Preserve the existing config key
            existing_key = self.existing_config.get(provider, {}).get("api_key")
            if existing_key:
                self.config.setdefault(provider, {})["api_key"] = existing_key
        else:
            self.console.print(f"\n[yellow]![/yellow] {env_var} not set")
            self.console.print("[dim]You can set it in your shell or enter it here.[/dim]")

            if Confirm.ask("Enter API key now?", default=False):
                api_key = Prompt.ask("Enter API key", password=True)
                if api_key:
                    self.config.setdefault(provider, {})["api_key"] = api_key
            else:
                self.console.print(
                    f"[dim]Remember to set {env_var} before using Ash.[/dim]"
                )

        # Store model config
        self.config.setdefault("models", {})["default"] = {
            "provider": provider,
            "model": model,
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        # Ask about additional model aliases
        if Confirm.ask("\nAdd another model alias (e.g., 'fast' for quick queries)?", default=False):
            self._add_model_alias(provider)

    def _add_model_alias(self, default_provider: str) -> None:
        """Add an additional model alias."""
        alias = Prompt.ask("Alias name (e.g., 'fast', 'smart')").strip().lower()
        if not alias or alias == "default":
            self.console.print("[yellow]Invalid alias name.[/yellow]")
            return

        # Quick selection for common aliases
        if alias == "fast":
            if default_provider == "anthropic":
                model = "claude-3-5-haiku-20241022"
            else:
                model = "gpt-4o-mini"
            self.config["models"][alias] = {
                "provider": default_provider,
                "model": model,
                "temperature": 0.5,
                "max_tokens": 2048,
            }
            self.console.print(f"[green]✓[/green] Added '{alias}' alias using {model}")
        else:
            # Manual configuration
            provider = Prompt.ask("Provider", choices=["anthropic", "openai"], default=default_provider)
            models = ANTHROPIC_MODELS if provider == "anthropic" else OPENAI_MODELS

            self.console.print("Select model:")
            for i, (model_id, _description) in enumerate(models, 1):
                self.console.print(f"  [cyan]{i}[/cyan]. {model_id}")

            model_choice = Prompt.ask("Model", choices=[str(i) for i in range(1, len(models) + 1)])
            model = models[int(model_choice) - 1][0]

            self.config["models"][alias] = {
                "provider": provider,
                "model": model,
                "temperature": 0.7,
                "max_tokens": 4096,
            }
            self.console.print(f"[green]✓[/green] Added '{alias}' alias")

    def _configure_telegram(self) -> None:
        """Configure Telegram integration."""
        self.console.print(
            Panel.fit(
                "[bold]Telegram Configuration[/bold]",
                border_style="cyan",
            )
        )

        self.console.print("\nTo use Telegram, you need a bot token from @BotFather.")
        self.console.print("[dim]See: https://core.telegram.org/bots#creating-a-new-bot[/dim]")

        has_env_token = bool(os.environ.get("TELEGRAM_BOT_TOKEN"))
        has_config_token = self._has_config_telegram_token()

        if has_env_token and has_config_token:
            self.console.print("\n[green]✓[/green] TELEGRAM_BOT_TOKEN found in environment and config file")
            self.console.print("[dim]Environment variable will be used if present.[/dim]")
        elif has_env_token:
            self.console.print("\n[green]✓[/green] TELEGRAM_BOT_TOKEN found in environment")
        elif has_config_token:
            self.console.print("\n[green]✓[/green] TELEGRAM_BOT_TOKEN found in config file")
            # Preserve the existing config token
            existing_token = self.existing_config.get("telegram", {}).get("bot_token")
            if existing_token:
                self.config.setdefault("telegram", {})["bot_token"] = existing_token
        else:
            self.console.print("\n[yellow]![/yellow] TELEGRAM_BOT_TOKEN not set")
            if Confirm.ask("Enter bot token now?", default=False):
                token = Prompt.ask("Enter bot token", password=True)
                if token:
                    self.config.setdefault("telegram", {})["bot_token"] = token
            else:
                self.console.print(
                    "[dim]Set TELEGRAM_BOT_TOKEN environment variable before using Telegram.[/dim]"
                )

        # Allowed users - required, numerical IDs only
        self.console.print("\n[bold]Allowed Users[/bold]")
        self.console.print("You must specify which Telegram user IDs can interact with this bot.")
        self.console.print("[dim]Find your user ID by messaging @userinfobot on Telegram.[/dim]")

        allowed_users: list[str] = []
        while not allowed_users:
            users_input = Prompt.ask("User IDs (comma-separated numbers)")

            if not users_input.strip():
                self.console.print("[yellow]At least one user ID is required.[/yellow]")
                continue

            # Parse and validate user IDs
            valid = True
            parsed_ids = []
            for part in users_input.split(","):
                part = part.strip()
                if not part:
                    continue
                # Remove @ prefix if accidentally included
                if part.startswith("@"):
                    self.console.print(f"[yellow]'{part}' looks like a username. Use numerical IDs only.[/yellow]")
                    valid = False
                    break
                # Validate it's a number
                try:
                    int(part)
                    parsed_ids.append(part)
                except ValueError:
                    self.console.print(f"[yellow]'{part}' is not a valid user ID. Use numerical IDs only.[/yellow]")
                    valid = False
                    break

            if valid and parsed_ids:
                allowed_users = parsed_ids
            elif valid:
                self.console.print("[yellow]At least one user ID is required.[/yellow]")

        self.config.setdefault("telegram", {})["allowed_users"] = allowed_users

        # Group mode
        self.console.print("\n[bold]Group Chat Mode[/bold]")
        self.console.print("  [cyan]1[/cyan]. mention - Only respond when @mentioned (recommended)")
        self.console.print("  [cyan]2[/cyan]. always - Respond to all messages")

        mode_choice = Prompt.ask("Mode", choices=["1", "2"], default="1")
        group_mode = "mention" if mode_choice == "1" else "always"

        self.config.setdefault("telegram", {})["group_mode"] = group_mode

        # Preserve existing allowed_groups if any
        existing_groups = self.existing_config.get("telegram", {}).get("allowed_groups", [])
        self.config["telegram"]["allowed_groups"] = existing_groups

    def _configure_search(self) -> None:
        """Configure web search."""
        self.console.print(
            Panel.fit(
                "[bold]Web Search Configuration[/bold]",
                border_style="cyan",
            )
        )

        self.console.print("\nBrave Search enables web queries for current information.")
        self.console.print("[dim]Get an API key at: https://brave.com/search/api/[/dim]")

        has_env_key = bool(os.environ.get("BRAVE_SEARCH_API_KEY"))

        if has_env_key:
            self.console.print("\n[green]✓[/green] BRAVE_SEARCH_API_KEY found in environment")
            self.console.print("[dim]Web search will be enabled automatically.[/dim]")
        else:
            self.console.print("\n[yellow]![/yellow] BRAVE_SEARCH_API_KEY not set")
            self.console.print(
                "[dim]Set BRAVE_SEARCH_API_KEY environment variable to enable web search.[/dim]"
            )

        # We don't store the key in config - just inform the user
        self.config["brave_search"] = {}

    def _configure_advanced(self) -> None:
        """Configure advanced settings."""
        self.console.print(
            Panel.fit(
                "[bold]Advanced Configuration[/bold]",
                border_style="cyan",
            )
        )

        # Workspace
        from ash.config.paths import get_workspace_path

        default_workspace = get_workspace_path()
        self.console.print("\n[bold]Workspace[/bold]")
        self.console.print(f"[dim]Default: {default_workspace}[/dim]")

        if Confirm.ask("Use custom workspace path?", default=False):
            workspace = Prompt.ask("Workspace path", default=str(default_workspace))
            self.config["workspace"] = workspace

        # Server settings
        self.console.print("\n[bold]Server Settings[/bold]")
        if Confirm.ask("Configure server (host/port)?", default=False):
            host = Prompt.ask("Host", default="127.0.0.1")
            port = Prompt.ask("Port", default="8080")
            self.config["server"] = {
                "host": host,
                "port": int(port),
            }

        # Sandbox settings
        self.console.print("\n[bold]Sandbox Settings[/bold]")
        self.console.print("[dim]The sandbox runs bash commands in isolated Docker containers.[/dim]")

        if Confirm.ask("Configure sandbox settings?", default=False):
            self.console.print("\nNetwork mode:")
            self.console.print("  [cyan]1[/cyan]. bridge - Has network access (default)")
            self.console.print("  [cyan]2[/cyan]. none - Fully isolated (more secure)")

            network_choice = Prompt.ask("Network", choices=["1", "2"], default="1")
            network_mode = "bridge" if network_choice == "1" else "none"

            timeout = Prompt.ask("Command timeout (seconds)", default="60")
            memory = Prompt.ask("Memory limit", default="512m")

            self.config["sandbox"] = {
                "network_mode": network_mode,
                "timeout": int(timeout),
                "memory_limit": memory,
            }

        # Embeddings for semantic search
        self.console.print("\n[bold]Semantic Search (Embeddings)[/bold]")
        self.console.print("[dim]Enables semantic memory search using OpenAI embeddings.[/dim]")

        has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
        if has_openai_key:
            if Confirm.ask("Enable semantic search? (requires OpenAI API)", default=True):
                self.config["embeddings"] = {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                }
        else:
            self.console.print(
                "[dim]Set OPENAI_API_KEY to enable semantic search.[/dim]"
            )

    def _write_config(self) -> None:
        """Write configuration to TOML file."""
        # Ensure parent directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge new config with existing config (new values override)
        merged = dict(self.existing_config)
        for key, value in self.config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

        # Use merged config for writing
        config_to_write = merged

        # Build TOML content with comments
        lines = ["# Ash Configuration", "# Generated by ash setup", ""]

        # Workspace (if custom)
        if "workspace" in config_to_write:
            lines.append(f'workspace = "{config_to_write["workspace"]}"')
            lines.append("")

        # Models
        if "models" in config_to_write:
            for alias, model_config in config_to_write["models"].items():
                lines.append(f"[models.{alias}]")
                lines.append(f'provider = "{model_config["provider"]}"')
                lines.append(f'model = "{model_config["model"]}"')
                if model_config.get("temperature") is not None:
                    lines.append(f'temperature = {model_config["temperature"]}')
                lines.append(f'max_tokens = {model_config["max_tokens"]}')
                lines.append("")

        # Provider API keys (if configured in file)
        for provider in ["anthropic", "openai"]:
            if provider in config_to_write and "api_key" in config_to_write[provider]:
                lines.append(f"[{provider}]")
                lines.append(f'api_key = "{config_to_write[provider]["api_key"]}"')
                lines.append("")

        # Add comment about env vars for API keys
        if "anthropic" not in config_to_write and "openai" not in config_to_write:
            provider = config_to_write.get("models", {}).get("default", {}).get("provider", "anthropic")
            env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            lines.append(f"# API key loaded from {env_var} environment variable")
            lines.append("")

        # Telegram
        if "telegram" in config_to_write:
            lines.append("[telegram]")
            if "bot_token" in config_to_write["telegram"]:
                lines.append(f'bot_token = "{config_to_write["telegram"]["bot_token"]}"')
            else:
                lines.append("# bot_token loaded from TELEGRAM_BOT_TOKEN env var")

            users = config_to_write["telegram"].get("allowed_users", [])
            users_str = ", ".join(f'"{u}"' for u in users)
            lines.append(f"allowed_users = [{users_str}]")

            groups = config_to_write["telegram"].get("allowed_groups", [])
            groups_str = ", ".join(f'"{g}"' for g in groups)
            lines.append(f"allowed_groups = [{groups_str}]")

            lines.append(f'group_mode = "{config_to_write["telegram"].get("group_mode", "mention")}"')
            lines.append("")

        # Sandbox
        if "sandbox" in config_to_write:
            lines.append("[sandbox]")
            for key, value in config_to_write["sandbox"].items():
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                else:
                    lines.append(f"{key} = {value}")
            lines.append("")

        # Server
        if "server" in config_to_write:
            lines.append("[server]")
            lines.append(f'host = "{config_to_write["server"]["host"]}"')
            lines.append(f'port = {config_to_write["server"]["port"]}')
            lines.append("")

        # Embeddings
        if "embeddings" in config_to_write:
            lines.append("[embeddings]")
            lines.append(f'provider = "{config_to_write["embeddings"]["provider"]}"')
            lines.append(f'model = "{config_to_write["embeddings"]["model"]}"')
            lines.append("")

        # Brave search (just a placeholder comment)
        if "brave_search" in config_to_write:
            lines.append("[brave_search]")
            lines.append("# api_key loaded from BRAVE_SEARCH_API_KEY env var")
            lines.append("")

        # Write file
        content = "\n".join(lines)
        self.config_path.write_text(content)

    def _show_summary(self) -> None:
        """Show configuration summary."""
        self.console.print()
        self.console.print(
            Panel.fit(
                "[bold green]Setup Complete![/bold green]",
                border_style="green",
            )
        )

        self.console.print(f"\nConfiguration saved to: [cyan]{self.config_path}[/cyan]")

        # Show what was configured
        table = Table(title="Configuration Summary", show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")

        if "models" in self.config:
            default_model = self.config["models"].get("default", {})
            table.add_row(
                "Default Model",
                f"{default_model.get('provider')}/{default_model.get('model')}",
            )
            for alias in self.config["models"]:
                if alias != "default":
                    model = self.config["models"][alias]
                    table.add_row(f"  {alias}", f"{model.get('provider')}/{model.get('model')}")

        if "telegram" in self.config:
            users = self.config["telegram"].get("allowed_users", [])
            user_str = ", ".join(users) if users else "all users"
            table.add_row("Telegram", f"enabled ({user_str})")

        if "embeddings" in self.config:
            table.add_row("Semantic Search", "enabled")

        if "sandbox" in self.config:
            network = self.config["sandbox"].get("network_mode", "bridge")
            table.add_row("Sandbox Network", network)

        self.console.print(table)

        # Next steps
        self.console.print("\n[bold]Next Steps:[/bold]")
        self.console.print("  1. Build the sandbox: [cyan]ash sandbox build[/cyan]")
        self.console.print("  2. Start chatting: [cyan]ash chat[/cyan]")
        self.console.print("  3. Or start the server: [cyan]ash serve[/cyan]")
