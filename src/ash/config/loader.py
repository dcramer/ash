"""Configuration loading from TOML files and environment variables."""

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from ash.config.models import AshConfig

DEFAULT_CONFIG_PATHS = [
    Path("config.toml"),
    Path.home() / ".ash" / "config.toml",
    Path("/etc/ash/config.toml"),
]


def _resolve_env_secrets(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve API keys from environment variables where not set in config."""
    env_mappings = {
        ("default_llm", "api_key"): {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        },
        ("fallback_llm", "api_key"): {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        },
        ("telegram", "bot_token"): "TELEGRAM_BOT_TOKEN",
        ("brave_search", "api_key"): "BRAVE_SEARCH_API_KEY",
    }

    for path, env_var in env_mappings.items():
        section = config
        for key in path[:-1]:
            if key not in section or section[key] is None:
                break
            section = section[key]
        else:
            final_key = path[-1]
            if section.get(final_key) is None:
                if isinstance(env_var, dict):
                    # Provider-specific env var
                    provider = section.get("provider")
                    if provider and provider in env_var:
                        value = os.environ.get(env_var[provider])
                        if value:
                            section[final_key] = SecretStr(value)
                else:
                    # Simple env var
                    value = os.environ.get(env_var)
                    if value:
                        section[final_key] = SecretStr(value)

    return config


def load_config(path: Path | None = None) -> AshConfig:
    """Load configuration from TOML file.

    Args:
        path: Explicit path to config file. If None, searches default locations.

    Returns:
        Validated AshConfig instance.

    Raises:
        FileNotFoundError: If no config file is found.
        ValueError: If config file is invalid.
    """
    config_path: Path | None = None

    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        for default_path in DEFAULT_CONFIG_PATHS:
            expanded = default_path.expanduser()
            if expanded.exists():
                config_path = expanded
                break

    if config_path is None:
        raise FileNotFoundError(
            f"No config file found. Searched: {', '.join(str(p) for p in DEFAULT_CONFIG_PATHS)}"
        )

    with config_path.open("rb") as f:
        raw_config = tomllib.load(f)

    # Resolve secrets from environment
    raw_config = _resolve_env_secrets(raw_config)

    return AshConfig.model_validate(raw_config)


def get_default_config() -> AshConfig:
    """Get a default configuration for development/testing."""
    return AshConfig(
        default_llm={
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
        }
    )
