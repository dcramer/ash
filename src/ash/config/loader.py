"""Configuration loading from TOML files and environment variables."""

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from ash.config.models import AshConfig
from ash.config.paths import get_config_path


def _get_default_config_paths() -> list[Path]:
    """Get ordered list of default config file locations."""
    return [
        Path("config.toml"),  # Current directory
        get_config_path(),  # ~/.ash/config.toml (or ASH_HOME)
        Path("/etc/ash/config.toml"),  # System-wide
    ]


def _get_nested(config: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    """Get nested dict by keys, returning None if any key is missing."""
    section = config
    for key in keys:
        if key not in section or section[key] is None:
            return None
        section = section[key]
    return section


def _set_secret_from_env(section: dict[str, Any], key: str, env_var: str) -> None:
    """Set a secret value from environment if not already set."""
    if section.get(key) is None:
        value = os.environ.get(env_var)
        if value:
            section[key] = SecretStr(value)


def _resolve_env_secrets(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve API keys from environment variables where not set in config."""
    provider_env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    # Provider-level API keys
    for provider, env_var in provider_env_vars.items():
        if provider in config:
            _set_secret_from_env(config[provider], "api_key", env_var)

    # Legacy LLM config API keys (backward compatibility)
    for llm_key in ("default_llm", "fallback_llm"):
        if section := _get_nested(config, llm_key):
            provider = section.get("provider")
            if provider in provider_env_vars:
                _set_secret_from_env(section, "api_key", provider_env_vars[provider])

    # Other secrets (telegram, brave_search, sentry)
    simple_mappings = [
        ("telegram", "bot_token", "TELEGRAM_BOT_TOKEN"),
        ("brave_search", "api_key", "BRAVE_SEARCH_API_KEY"),
        ("sentry", "dsn", "SENTRY_DSN"),
    ]
    for parent_key, secret_key, env_var in simple_mappings:
        if (section := _get_nested(config, parent_key)) is not None:
            _set_secret_from_env(section, secret_key, env_var)

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

    default_paths = _get_default_config_paths()

    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        for default_path in default_paths:
            expanded = default_path.expanduser()
            if expanded.exists():
                config_path = expanded
                break

    if config_path is None:
        raise FileNotFoundError(
            f"No config file found. Searched: {', '.join(str(p) for p in default_paths)}"
        )

    with config_path.open("rb") as f:
        raw_config = tomllib.load(f)

    # Resolve secrets from environment
    raw_config = _resolve_env_secrets(raw_config)

    return AshConfig.model_validate(raw_config)


def get_default_config() -> AshConfig:
    """Get a default configuration for development/testing."""
    from ash.config.models import ModelConfig

    return AshConfig(
        models={
            "default": ModelConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
            )
        }
    )
