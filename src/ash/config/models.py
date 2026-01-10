"""Configuration models using Pydantic."""

import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator

from ash.config.paths import get_database_path, get_workspace_path

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a named model.

    Temperature is optional - if None, the provider's default is used.
    Omit temperature for reasoning models that don't support it.
    """

    provider: Literal["anthropic", "openai"]
    model: str
    temperature: float | None = None  # None = use provider default
    max_tokens: int = 4096


class ProviderConfig(BaseModel):
    """Provider-level configuration."""

    api_key: SecretStr | None = None


class LLMConfig(BaseModel):
    """Configuration for an LLM provider (backward compatibility)."""

    provider: Literal["anthropic", "openai"]
    model: str
    api_key: SecretStr | None = None
    temperature: float = 0.7
    max_tokens: int = 4096


class TelegramConfig(BaseModel):
    """Configuration for Telegram provider."""

    bot_token: SecretStr | None = None
    allowed_users: list[str] = []
    webhook_url: str | None = None


class SandboxConfig(BaseModel):
    """Configuration for Docker sandbox.

    The sandbox is mandatory - all bash commands run in an isolated container
    with security hardening including read-only root filesystem, dropped
    capabilities, process limits, and more.
    """

    image: str = "ash-sandbox:latest"
    timeout: int = 60
    memory_limit: str = "512m"
    cpu_limit: float = 1.0

    # Container runtime: "runc" (default) or "runsc" (gVisor for enhanced security)
    runtime: Literal["runc", "runsc"] = "runc"

    # Network: "none" = isolated, "bridge" = has network access
    network_mode: Literal["none", "bridge"] = "bridge"
    # Optional DNS servers for filtering (e.g., Pi-hole, NextDNS)
    dns_servers: list[str] = []
    # Optional HTTP proxy for monitoring/filtering traffic
    http_proxy: str | None = None

    # Workspace mounting into sandbox
    # Access: "none" = not mounted, "ro" = read-only, "rw" = read-write
    workspace_access: Literal["none", "ro", "rw"] = "rw"


class ServerConfig(BaseModel):
    """Configuration for HTTP server."""

    host: str = "127.0.0.1"
    port: int = 8080
    webhook_path: str = "/webhook"


class EmbeddingsConfig(BaseModel):
    """Configuration for embedding model.

    Embeddings are used for semantic search in memory.
    Currently only OpenAI embeddings are supported.
    """

    provider: Literal["openai"] = "openai"
    model: str = "text-embedding-3-small"


class MemoryConfig(BaseModel):
    """Configuration for memory system."""

    database_path: Path = Field(default_factory=get_database_path)
    max_context_messages: int = 20


class BraveSearchConfig(BaseModel):
    """Configuration for Brave Search API."""

    api_key: SecretStr | None = None


class ConfigError(Exception):
    """Configuration error."""

    pass


class AshConfig(BaseModel):
    """Root configuration model."""

    workspace: Path = Field(default_factory=get_workspace_path)
    # Named model configurations (new style)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    # Provider-level API keys
    anthropic: ProviderConfig | None = None
    openai: ProviderConfig | None = None
    # Backward compatibility - deprecated, use models.default instead
    default_llm: LLMConfig | None = None
    fallback_llm: LLMConfig | None = None
    telegram: TelegramConfig | None = None
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    embeddings: EmbeddingsConfig | None = None
    brave_search: BraveSearchConfig | None = None

    @model_validator(mode="after")
    def _migrate_default_llm(self) -> "AshConfig":
        """Migrate [default_llm] to models.default for backward compatibility."""
        if self.default_llm is not None:
            if "default" in self.models:
                logger.warning(
                    "Both [default_llm] and [models.default] present. "
                    "Using [models.default], ignoring [default_llm]."
                )
            else:
                # Migrate default_llm to models.default
                self.models["default"] = ModelConfig(
                    provider=self.default_llm.provider,
                    model=self.default_llm.model,
                    temperature=self.default_llm.temperature,
                    max_tokens=self.default_llm.max_tokens,
                )
                # Store api_key in provider config if present
                if self.default_llm.api_key is not None:
                    if self.default_llm.provider == "anthropic":
                        if self.anthropic is None:
                            self.anthropic = ProviderConfig(
                                api_key=self.default_llm.api_key
                            )
                        elif self.anthropic.api_key is None:
                            self.anthropic.api_key = self.default_llm.api_key
                    elif self.default_llm.provider == "openai":
                        if self.openai is None:
                            self.openai = ProviderConfig(api_key=self.default_llm.api_key)
                        elif self.openai.api_key is None:
                            self.openai.api_key = self.default_llm.api_key
        return self

    @model_validator(mode="after")
    def _validate_default_model(self) -> "AshConfig":
        """Validate that a default model is configured."""
        if "default" not in self.models and self.default_llm is None:
            raise ValueError(
                "No default model configured. Add [models.default] or [default_llm]"
            )
        return self

    def get_model(self, alias: str) -> ModelConfig:
        """Get model config by alias.

        Args:
            alias: The model alias to look up.

        Returns:
            The ModelConfig for the alias.

        Raises:
            ConfigError: If the alias is not found.
        """
        if alias not in self.models:
            available = ", ".join(sorted(self.models.keys()))
            raise ConfigError(
                f"Unknown model alias '{alias}'. Available: {available}"
            )
        return self.models[alias]

    def list_models(self) -> list[str]:
        """List available model aliases.

        Returns:
            Sorted list of model alias names.
        """
        return sorted(self.models.keys())

    @property
    def default_model(self) -> ModelConfig:
        """Get the default model (alias 'default').

        Returns:
            The default ModelConfig.

        Raises:
            ConfigError: If no default model is configured.
        """
        return self.get_model("default")

    def resolve_api_key(self, alias: str) -> SecretStr | None:
        """Resolve API key for a model alias.

        Resolution order:
        1. Provider-level config api_key
        2. Environment variable (ANTHROPIC_API_KEY or OPENAI_API_KEY)

        Args:
            alias: The model alias to resolve API key for.

        Returns:
            The resolved API key, or None if not found.
        """
        model = self.get_model(alias)
        provider = model.provider

        # Check provider-level config
        if provider == "anthropic" and self.anthropic and self.anthropic.api_key:
            return self.anthropic.api_key
        if provider == "openai" and self.openai and self.openai.api_key:
            return self.openai.api_key

        # Check environment variable
        env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        env_value = os.environ.get(env_var)
        if env_value:
            return SecretStr(env_value)

        return None

    def resolve_embeddings_api_key(self) -> SecretStr | None:
        """Resolve API key for embeddings.

        Resolution order:
        1. Provider-level config api_key (based on embeddings.provider)
        2. Environment variable (OPENAI_API_KEY for openai provider)

        Returns:
            The resolved API key, or None if not found.
        """
        if self.embeddings is None:
            return None

        provider = self.embeddings.provider

        # Check provider-level config
        if provider == "openai" and self.openai and self.openai.api_key:
            return self.openai.api_key

        # Check environment variable
        env_var = "OPENAI_API_KEY"  # Currently only openai supported
        env_value = os.environ.get(env_var)
        if env_value:
            return SecretStr(env_value)

        return None
