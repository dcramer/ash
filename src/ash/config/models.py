"""Configuration models using Pydantic."""

import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from ash.config.paths import get_database_path, get_workspace_path

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a named model.

    Temperature is optional - if None, the provider's default is used.
    Omit temperature for reasoning models that don't support it.

    Thinking is optional - levels: off, minimal, low, medium, high.
    Only supported by Anthropic Claude models.
    """

    provider: Literal["anthropic", "openai"]
    model: str
    temperature: float | None = None  # None = use provider default
    max_tokens: int = 4096
    thinking: Literal["off", "minimal", "low", "medium", "high"] | None = None


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
    # Group chat settings
    allowed_groups: list[
        str
    ] = []  # Group IDs (empty = all groups; authorized groups imply user auth)
    group_mode: Literal["mention", "always"] = "mention"  # How to respond in groups


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

    # Sessions mounting into sandbox (for agent to read chat history)
    # Mounted at /sessions in the container
    sessions_access: Literal["none", "ro"] = "ro"

    # Build-time packages (requires `ash sandbox build` to take effect)
    apt_packages: list[str] = []
    python_packages: list[str] = []

    # Runtime setup command (runs once per container creation)
    # Use for packages that don't need to be baked into the image
    # Example: "uv pip install --user some-package"
    setup_command: str | None = None


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
    # Smart pruning configuration
    context_token_budget: int = 100000  # Target context window size in tokens
    recency_window: int = 10  # Always keep last N messages
    system_prompt_buffer: int = 8000  # Reserve tokens for system prompt
    # Compaction configuration (summarizes old messages instead of dropping)
    compaction_enabled: bool = True
    compaction_reserve_tokens: int = 16384  # Buffer to trigger compaction
    compaction_keep_recent_tokens: int = 20000  # Always keep recent context
    compaction_summary_max_tokens: int = 2000  # Max tokens for summary
    # Retention configuration
    auto_gc: bool = True  # Run gc on server startup
    max_entries: int | None = None  # Cap on active memories (None = unlimited)
    # Background extraction configuration
    extraction_enabled: bool = True  # Enable automatic memory extraction
    extraction_model: str | None = (
        None  # Model alias for extraction (None = use default)
    )
    extraction_min_message_length: int = 20  # Skip extraction for short messages
    extraction_debounce_seconds: int = 30  # Minimum seconds between extractions
    extraction_confidence_threshold: float = 0.7  # Minimum confidence to store


class ConversationConfig(BaseModel):
    """Configuration for conversation context management."""

    recency_window: int = 10  # Always include last N messages
    gap_threshold_minutes: int = 15  # Signal gap if longer than this
    reply_context_window: int = 3  # Messages before/after reply target


class SessionsConfig(BaseModel):
    """Configuration for session management."""

    mode: Literal["persistent", "fresh"] = "persistent"
    max_concurrent: int = 2  # Parallel session processing limit


class BraveSearchConfig(BaseModel):
    """Configuration for Brave Search API."""

    api_key: SecretStr | None = None


class SentryConfig(BaseModel):
    """Configuration for Sentry error tracking and observability.

    Sentry is optional - if this section is not configured or DSN is not set,
    error tracking is disabled.
    """

    dsn: SecretStr | None = None
    environment: str | None = None
    release: str | None = None
    traces_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    profiles_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    send_default_pii: bool = False
    debug: bool = False


class AgentOverrideConfig(BaseModel):
    """Configuration overrides for a built-in agent.

    Used to customize agent behavior via [agents.<name>] sections.
    Example:
        [agents.research]
        model = "sonnet"
    """

    model: str | None = None  # Model alias to use (None = agent default)
    max_iterations: int | None = None  # Override max iterations


class SkillConfig(BaseModel):
    """Per-skill configuration.

    Used to configure skill behavior via [skills.<name>] sections.
    Environment variables are stored as extra fields with UPPER_CASE names.

    Example:
        [skills.research]
        PERPLEXITY_API_KEY = "pplx-..."
        model = "haiku"
        enabled = true
    """

    model_config = ConfigDict(extra="allow")  # Allow UPPER_CASE env var fields

    model: str | None = None  # Model alias override (None = skill default)
    enabled: bool = True  # Can disable without removing file

    def get_env_vars(self) -> dict[str, str]:
        """Get env vars (extra fields with UPPER_CASE names).

        Returns:
            Dict of env var name to value.
        """
        # Get all extra fields (not model or enabled)
        known_fields = {"model", "enabled"}
        return {
            k: str(v)
            for k, v in self.model_dump().items()
            if k not in known_fields and k.isupper()
        }


class ConfigError(Exception):
    """Configuration error."""


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
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    sessions: SessionsConfig = Field(default_factory=SessionsConfig)
    embeddings: EmbeddingsConfig | None = None
    brave_search: BraveSearchConfig | None = None
    sentry: SentryConfig | None = None
    # Environment variables from [env] section
    # Loaded into session environment for skills and bash commands
    env: dict[str, str] = Field(default_factory=dict)
    # Agent-specific configuration: [agents.<name>] sections
    # Allows overriding model, max_iterations per agent
    agents: dict[str, AgentOverrideConfig] = Field(default_factory=dict)
    # Skill-specific configuration: [skills.<name>] sections
    # Allows setting API keys, model override, and enabled flag per skill
    skills: dict[str, SkillConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _migrate_default_llm(self) -> "AshConfig":
        """Migrate [default_llm] to models.default for backward compatibility."""
        if self.default_llm is None:
            return self

        if "default" in self.models:
            logger.warning(
                "Both [default_llm] and [models.default] present. "
                "Using [models.default], ignoring [default_llm]."
            )
            return self

        logger.warning(
            "[default_llm] is deprecated and will be removed in a future version. "
            "Please migrate to [models.default] format. See docs for details."
        )

        self.models["default"] = ModelConfig(
            provider=self.default_llm.provider,
            model=self.default_llm.model,
            temperature=self.default_llm.temperature,
            max_tokens=self.default_llm.max_tokens,
        )

        # Store api_key in provider config if present
        if self.default_llm.api_key is None:
            return self

        self._set_provider_api_key(self.default_llm.provider, self.default_llm.api_key)
        return self

    def _set_provider_api_key(self, provider: str, api_key: SecretStr) -> None:
        """Set API key on provider config, creating it if needed."""
        if provider == "anthropic":
            if self.anthropic is None:
                self.anthropic = ProviderConfig(api_key=api_key)
            elif self.anthropic.api_key is None:
                self.anthropic.api_key = api_key
        elif provider == "openai":
            if self.openai is None:
                self.openai = ProviderConfig(api_key=api_key)
            elif self.openai.api_key is None:
                self.openai.api_key = api_key

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
            raise ConfigError(f"Unknown model alias '{alias}'. Available: {available}")
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

    def _resolve_provider_api_key(
        self, provider: Literal["anthropic", "openai"]
    ) -> SecretStr | None:
        """Resolve API key for a provider.

        Resolution order:
        1. Provider-level config api_key
        2. Environment variable (ANTHROPIC_API_KEY or OPENAI_API_KEY)
        """
        # Check provider-level config
        provider_config = self.anthropic if provider == "anthropic" else self.openai
        if provider_config and provider_config.api_key:
            return provider_config.api_key

        # Check environment variable
        env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        env_value = os.environ.get(env_var)
        if env_value:
            return SecretStr(env_value)

        return None

    def resolve_api_key(self, alias: str) -> SecretStr | None:
        """Resolve API key for a model alias."""
        model = self.get_model(alias)
        return self._resolve_provider_api_key(model.provider)

    def resolve_embeddings_api_key(self) -> SecretStr | None:
        """Resolve API key for embeddings."""
        if self.embeddings is None:
            return None
        return self._resolve_provider_api_key(self.embeddings.provider)

    def get_resolved_env(self) -> dict[str, str]:
        """Get env vars from [env] section with $VAR references resolved.

        Values starting with $ are resolved from environment variables.

        Returns:
            Dict of env var name to resolved value.
        """
        resolved = {}
        for name, value in self.env.items():
            if value.startswith("$"):
                # Resolve from environment
                env_var = value[1:]
                resolved[name] = os.environ.get(env_var, "")
            else:
                resolved[name] = value
        return resolved
