"""Configuration loading from TOML files and environment variables."""

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from ash.config.models import AshConfig
from ash.config.paths import get_config_path

ENV_VAR_MAPPINGS = {
    "anthropic": ("api_key", "ANTHROPIC_API_KEY"),
    "openai": ("api_key", "OPENAI_API_KEY"),
    "telegram": ("bot_token", "TELEGRAM_BOT_TOKEN"),
    "brave_search": ("api_key", "BRAVE_SEARCH_API_KEY"),
    "sentry": ("dsn", "SENTRY_DSN"),
    "browser.kernel": ("api_key", "KERNEL_API_KEY"),
}


def _resolve_env_secrets(config: dict[str, Any]) -> dict[str, Any]:
    for section_name, (key, env_var) in ENV_VAR_MAPPINGS.items():
        section: dict[str, Any] | None = config
        for part in section_name.split("."):
            if not isinstance(section, dict):
                section = None
                break
            section = section.get(part)
        if section is not None and isinstance(section, dict):
            if section.get(key) is None and (value := os.environ.get(env_var)):
                section[key] = SecretStr(value)

    return config


def load_config(path: Path | None = None) -> AshConfig:
    config_path = Path(path).expanduser() if path is not None else get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as f:
        raw_config = tomllib.load(f)

    return AshConfig.model_validate(_resolve_env_secrets(raw_config))


def get_default_config() -> AshConfig:
    from ash.config.models import ModelConfig

    return AshConfig(
        models={"default": ModelConfig(provider="openai", model="gpt-5.2")}
    )
