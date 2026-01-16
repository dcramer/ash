"""Configuration loading from TOML files and environment variables."""

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from ash.config.models import AshConfig
from ash.config.paths import get_config_path

DEFAULT_CONFIG_PATHS = [
    Path("config.toml"),
    get_config_path(),
    Path("/etc/ash/config.toml"),
]

ENV_VAR_MAPPINGS = {
    "anthropic": ("api_key", "ANTHROPIC_API_KEY"),
    "openai": ("api_key", "OPENAI_API_KEY"),
    "telegram": ("bot_token", "TELEGRAM_BOT_TOKEN"),
    "brave_search": ("api_key", "BRAVE_SEARCH_API_KEY"),
    "sentry": ("dsn", "SENTRY_DSN"),
}

PROVIDER_ENV_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def _resolve_env_secrets(config: dict[str, Any]) -> dict[str, Any]:
    for section_name, (key, env_var) in ENV_VAR_MAPPINGS.items():
        if (section := config.get(section_name)) is not None:
            if section.get(key) is None and (value := os.environ.get(env_var)):
                section[key] = SecretStr(value)

    for llm_key in ("default_llm", "fallback_llm"):
        if (section := config.get(llm_key)) and (provider := section.get("provider")):
            if env_var := PROVIDER_ENV_VARS.get(provider):
                if section.get("api_key") is None and (
                    value := os.environ.get(env_var)
                ):
                    section["api_key"] = SecretStr(value)

    return config


def load_config(path: Path | None = None) -> AshConfig:
    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        config_path = next(
            (p.expanduser() for p in DEFAULT_CONFIG_PATHS if p.expanduser().exists()),
            None,
        )
        if config_path is None:
            raise FileNotFoundError(
                f"No config file found. Searched: {', '.join(str(p) for p in DEFAULT_CONFIG_PATHS)}"
            )

    with config_path.open("rb") as f:
        raw_config = tomllib.load(f)

    return AshConfig.model_validate(_resolve_env_secrets(raw_config))


def get_default_config() -> AshConfig:
    from ash.config.models import ModelConfig

    return AshConfig(
        models={"default": ModelConfig(provider="anthropic", model="claude-haiku-4-5")}
    )
