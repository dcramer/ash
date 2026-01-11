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


def _resolve_env_secrets(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve API keys from environment variables where not set in config."""
    # Provider-level API keys
    provider_env_mappings = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    for provider, env_var in provider_env_mappings.items():
        if provider in config:
            if config[provider].get("api_key") is None:
                value = os.environ.get(env_var)
                if value:
                    config[provider]["api_key"] = SecretStr(value)

    # Legacy LLM config API keys (backward compatibility)
    llm_env_mappings = {
        ("default_llm", "api_key"): {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        },
        ("fallback_llm", "api_key"): {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        },
    }

    for path, env_var_map in llm_env_mappings.items():
        section = config
        for key in path[:-1]:
            if key not in section or section[key] is None:
                break
            section = section[key]
        else:
            final_key = path[-1]
            if section.get(final_key) is None:
                provider = section.get("provider")
                if provider and provider in env_var_map:
                    value = os.environ.get(env_var_map[provider])
                    if value:
                        section[final_key] = SecretStr(value)

    # Other secrets (telegram, brave_search, sentry)
    simple_mappings = {
        ("telegram", "bot_token"): "TELEGRAM_BOT_TOKEN",
        ("brave_search", "api_key"): "BRAVE_SEARCH_API_KEY",
        ("sentry", "dsn"): "SENTRY_DSN",
    }

    for path, env_var in simple_mappings.items():
        section = config
        for key in path[:-1]:
            if key not in section or section[key] is None:
                break
            section = section[key]
        else:
            final_key = path[-1]
            if section.get(final_key) is None:
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
