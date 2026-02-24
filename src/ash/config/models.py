"""Configuration models using Pydantic."""

import logging
import os
import re
import shlex
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from ash.config.paths import (
    get_system_timezone,
    get_workspace_path,
)

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a named model.

    Temperature is optional - if None, the provider's default is used.
    Omit temperature for reasoning models that don't support it.

    Thinking is optional - levels: off, minimal, low, medium, high.
    Only supported by Anthropic Claude models.
    """

    provider: Literal["anthropic", "openai", "openai-oauth"]
    model: str
    temperature: float | None = None  # None = use provider default
    max_tokens: int = 4096
    thinking: Literal["off", "minimal", "low", "medium", "high"] | None = None
    reasoning: Literal["low", "medium", "high"] | None = None


class ProviderConfig(BaseModel):
    """Provider-level configuration."""

    api_key: SecretStr | None = None


class PassiveListeningConfig(BaseModel):
    """Configuration for passive listening in group chats.

    When enabled, the bot observes all group messages (even when not mentioned),
    extracts memories in the background, and decides whether to engage.
    """

    enabled: bool = False
    model: str | None = (
        None  # Model for engagement decisions (default: uses cheap model)
    )

    # Throttling
    chat_cooldown_minutes: int = (
        1  # Minimum minutes between passive engagements per chat
    )
    max_engagements_per_hour: int = 30  # Global rate limit across all chats
    skip_after_active_messages: int = 6  # Skip passive if N active messages recently
    direct_followup_window_seconds: int = (
        180  # Bypass passive throttle for direct follow-ups within this window
    )

    # Extraction
    extraction_enabled: bool = True  # Run memory extraction on passive messages
    context_messages: int = 5  # Number of recent messages to include for context

    # Memory lookup for engagement decisions
    memory_lookup_enabled: bool = True
    memory_lookup_timeout: float = 2.0  # seconds
    memory_similarity_threshold: float = 0.4


class TelegramConfig(BaseModel):
    """Configuration for Telegram provider."""

    bot_token: SecretStr | None = None
    allowed_users: list[str] = []
    # Group chat settings
    allowed_groups: list[
        str
    ] = []  # Group IDs (empty = all groups; authorized groups imply user auth)
    group_mode: Literal["mention", "always"] = "mention"  # How to respond in groups
    # Passive listening (observe group messages without @mention)
    passive: PassiveListeningConfig = Field(default_factory=PassiveListeningConfig)


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

    # Chats mounting into sandbox (for agent to read chat state/participants)
    # Mounted at /chats in the container
    chats_access: Literal["none", "ro"] = "ro"

    # Build-time packages (requires `ash sandbox build` to take effect)
    apt_packages: list[str] = []
    python_packages: list[str] = []

    # Runtime setup command (runs once per container creation)
    # Use for packages that don't need to be baked into the image
    # Example: "uv pip install --user some-package"
    setup_command: str | None = None

    # Source code mounting (for debug-myself skill)
    # Mounted read-only at /source in container
    source_access: Literal["none", "ro"] = "none"

    # Mount prefix for sandbox paths (sessions, chats, logs, etc.)
    # All sandbox bind mounts (except /workspace and user cache) use this prefix
    mount_prefix: str = "/ash"


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
    extraction_context_messages: int = (
        8  # Number of recent messages to include in extraction context
    )
    extraction_confidence_threshold: float = 0.7  # Minimum confidence to store
    extraction_verification_enabled: bool = (
        True  # Run second-pass LLM verification/rewriting of extracted facts
    )
    extraction_verification_model: str | None = (
        None  # Verification model alias or provider model name (None = use default)
    )
    query_planning_enabled: bool = (
        True  # Enable a fast LLM planner that rewrites one retrieval query
    )
    query_planning_model_alias: str | None = (
        None  # Planner model alias from [models.*] (None = use [models.default])
    )
    query_planning_fetch_memories: int = (
        25  # Initial retrieval count before pruning to final context budget
    )
    context_injection_limit: int = 10  # Final memory count injected into prompt


class ImageConfig(BaseModel):
    """Configuration for image understanding on inbound provider messages."""

    enabled: bool = True
    provider: Literal["openai"] = "openai"
    model: str | None = None  # Model alias or provider model name (None = use default)
    max_images_per_message: int = 1
    max_image_bytes: int = 8_000_000
    request_timeout_seconds: float = 12.0
    include_ocr_text: bool = True
    inject_position: Literal["prepend", "append"] = "prepend"
    no_caption_auto_respond: bool = True


class TodoConfig(BaseModel):
    """Configuration for todo subsystem integration."""

    enabled: bool = True


class CapabilityProviderConfig(BaseModel):
    """Configuration for one capability provider plugin."""

    enabled: bool = True
    namespace: str | None = None
    command: list[str]
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)

    @model_validator(mode="before")
    @classmethod
    def _coerce_command(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        raw_command = normalized.get("command")
        if isinstance(raw_command, str):
            normalized["command"] = shlex.split(raw_command)
        return normalized

    @field_validator("namespace")
    @classmethod
    def _validate_namespace(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        if not re.match(r"^[a-z0-9][a-z0-9_-]*$", text):
            raise ValueError("namespace must match [a-z0-9][a-z0-9_-]*")
        return text

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item and item.strip()]
        if not normalized:
            raise ValueError("command is required")
        return normalized


class CapabilitiesConfig(BaseModel):
    """Configuration for capability provider plugins."""

    providers: dict[str, CapabilityProviderConfig] = Field(default_factory=dict)


class ToolOutputTrustConfig(BaseModel):
    """Configuration for tool-output trust boundary behavior."""

    mode: Literal["warn_sanitize", "block", "log_only"] = "warn_sanitize"
    max_chars: int = Field(default=12_000, ge=256, le=200_000)
    include_provenance_header: bool = True


class BrowserSandboxConfig(BaseModel):
    """Sandbox provider config for browser integration."""

    headless: bool = True
    browser_channel: Literal["chromium"] = "chromium"
    # Deprecated compatibility option; "legacy" is coerced to "dedicated".
    runtime_mode: Literal["legacy", "dedicated"] = "dedicated"
    container_image: str = "ash-sandbox-browser:latest"
    container_name_prefix: str = "ash-browser-"
    # Deprecated compatibility flag; sandbox-only execution is always enforced.
    runtime_required: bool = True
    runtime_warmup_on_start: bool = True
    runtime_restart_attempts: int = 1

    @field_validator("runtime_mode", mode="before")
    @classmethod
    def _coerce_legacy_runtime_mode(cls, value: object) -> str:
        if isinstance(value, str) and value.strip().lower() == "legacy":
            return "dedicated"
        return str(value) if isinstance(value, str) else "dedicated"


class BrowserKernelConfig(BaseModel):
    """Kernel provider config for browser integration."""

    api_key: SecretStr | None = None
    base_url: str = "https://api.kernel.sh"
    project_id: str | None = None


class BrowserConfig(BaseModel):
    """Configuration for browser integration."""

    enabled: bool = True
    provider: Literal["sandbox", "kernel"] = "sandbox"
    timeout_seconds: float = 20.0
    max_session_minutes: int = 20
    artifacts_retention_days: int = 7
    state_dir: Path | None = None
    default_viewport_width: int = 1280
    default_viewport_height: int = 720
    sandbox: BrowserSandboxConfig = Field(default_factory=BrowserSandboxConfig)
    kernel: BrowserKernelConfig = Field(default_factory=BrowserKernelConfig)


class ConversationConfig(BaseModel):
    """Configuration for conversation context management."""

    recency_window: int = 10  # Always include last N messages
    gap_threshold_minutes: int = 15  # Signal gap if longer than this
    reply_context_window: int = 3  # Messages before/after reply target
    chat_history_limit: int = 5  # Recent chat messages to include in LLM context


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
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    enabled: bool = True
    allow_chat_ids: list[str] | None = None

    @field_validator("allow_chat_ids", mode="before")
    @classmethod
    def _normalize_allow_chat_ids(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            normalized: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    normalized.append(text)
            return normalized
        raise ValueError("allow_chat_ids must be a string or list of strings")

    def get_env_vars(self) -> dict[str, str]:
        """Get environment variables from config, auto-uppercasing keys.

        Config like `ha_token = "xyz"` becomes env var `HA_TOKEN=xyz`.
        """
        return {
            k.upper(): str(v)
            for k, v in self.model_dump().items()
            if k.lower() not in {"model", "enabled", "allow_chat_ids"}
        }


class SkillDefaultsConfig(BaseModel):
    """Global skill defaults from [skills.defaults]."""

    allow_chat_ids: list[str] = Field(default_factory=list)

    @field_validator("allow_chat_ids", mode="before")
    @classmethod
    def _normalize_allow_chat_ids(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            normalized: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    normalized.append(text)
            return normalized
        raise ValueError("allow_chat_ids must be a string or list of strings")


class SkillSource(BaseModel):
    """A skill source - either GitHub repo or local path.

    Examples:
        [[skills.sources]]
        repo = "owner/repo"          # GitHub repo

        [[skills.sources]]
        repo = "owner/other"
        ref = "v2.0"                 # Pin to version

        [[skills.sources]]
        path = "~/my-skills"         # Local path (symlinked)
    """

    repo: str | None = None  # GitHub repo (owner/repo format)
    path: str | None = None  # Local path (~/... or /...)
    ref: str | None = None  # Git ref (branch/tag/commit)

    @model_validator(mode="after")
    def _validate_source(self) -> "SkillSource":
        if not self.repo and not self.path:
            raise ValueError("Must specify either 'repo' or 'path'")
        if self.repo and self.path:
            raise ValueError("Cannot specify both 'repo' and 'path'")
        if self.ref and not self.repo:
            raise ValueError("'ref' only applies to repo sources")
        return self


class ConfigError(Exception):
    """Configuration error."""


class AshConfig(BaseModel):
    """Root configuration model."""

    workspace: Path = Field(default_factory=get_workspace_path)
    # User's timezone (IANA timezone name, e.g., "America/New_York")
    # Used for displaying times and evaluating cron schedules
    # Default: detect from system (TZ env, /etc/timezone, /etc/localtime)
    timezone: str = Field(default_factory=get_system_timezone)
    # Named model configurations (new style)
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: str) -> str:
        """Validate that timezone is a valid IANA timezone name."""
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(v)
            return v
        except ZoneInfoNotFoundError:
            raise ValueError(
                f"Invalid timezone '{v}'. Use an IANA timezone name like "
                "'America/New_York', 'Europe/London', or 'UTC'."
            ) from None

    # Provider-level API keys
    anthropic: ProviderConfig | None = None
    openai: ProviderConfig | None = None
    telegram: TelegramConfig | None = None
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    todo: TodoConfig = Field(default_factory=TodoConfig)
    capabilities: CapabilitiesConfig = Field(default_factory=CapabilitiesConfig)
    tool_output_trust: ToolOutputTrustConfig = Field(
        default_factory=ToolOutputTrustConfig
    )
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
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
    # Skill defaults from [skills.defaults]
    skill_defaults: SkillDefaultsConfig = Field(default_factory=SkillDefaultsConfig)

    # External skill sources: [[skills.sources]] array
    skill_sources: list[SkillSource] = Field(default_factory=list)
    # Auto-sync configuration from [skills] section
    skill_auto_sync: bool = False
    skill_update_interval_minutes: int = Field(default=5, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _extract_skills_config(cls, data: dict) -> dict:
        """Extract [skills] settings and [[skills.sources]] from skills dict.

        TOML structure:
            [skills]
            auto_sync = true
            update_interval_minutes = 5

            [skills.defaults]
            allow_chat_ids = ["dm-1", "dm-2"]

            [[skills.sources]]
            repo = "owner/repo"

            [skills.research]
            PERPLEXITY_API_KEY = "..."

        Becomes:
            skill_auto_sync = True
            skill_update_interval_minutes = 5
            skill_sources = [SkillSource(repo="owner/repo")]
            skills = {"research": SkillConfig(...)}
        """
        if not isinstance(data, dict):
            return data

        skills_data = data.get("skills")
        if not isinstance(skills_data, dict):
            return data

        # Make a copy to avoid mutating the original
        skills_data = dict(skills_data)

        # Extract sources array and global settings
        sources = skills_data.pop("sources", [])
        defaults = skills_data.pop("defaults", {})
        auto_sync = skills_data.pop("auto_sync", False)
        update_interval_minutes = skills_data.pop("update_interval_minutes", None)
        # Backward-compat: legacy `update_interval` was in hours.
        update_interval_hours = skills_data.pop("update_interval", None)
        if update_interval_minutes is None:
            if update_interval_hours is not None:
                update_interval_minutes = int(update_interval_hours) * 60
            else:
                update_interval_minutes = 5

        data["skill_sources"] = sources
        data["skill_defaults"] = defaults
        data["skill_auto_sync"] = auto_sync
        data["skill_update_interval_minutes"] = update_interval_minutes
        # Remaining entries are per-skill configs
        data["skills"] = skills_data
        return data

    @model_validator(mode="after")
    def _validate_default_model(self) -> "AshConfig":
        if "default" not in self.models:
            raise ValueError("No default model configured. Add [models.default]")
        return self

    def get_model(self, alias: str) -> ModelConfig:
        if alias not in self.models:
            raise ConfigError(
                f"Unknown model alias '{alias}'. Available: {', '.join(sorted(self.models.keys()))}"
            )
        return self.models[alias]

    def list_models(self) -> list[str]:
        return sorted(self.models.keys())

    @property
    def default_model(self) -> ModelConfig:
        return self.get_model("default")

    def _resolve_provider_api_key(
        self, provider: Literal["anthropic", "openai", "openai-oauth"]
    ) -> SecretStr | None:
        if provider == "openai-oauth":
            # OAuth-based provider — API key comes from auth.json, not config
            return self._resolve_oauth_api_key(provider)
        provider_config = getattr(self, provider, None)
        if provider_config and provider_config.api_key:
            return provider_config.api_key
        env_var = f"{provider.upper()}_API_KEY"
        if env_value := os.environ.get(env_var):
            return SecretStr(env_value)
        return None

    def _resolve_oauth_api_key(self, provider: str) -> SecretStr | None:
        """Resolve an API key from OAuth credentials in auth.json."""
        from ash.auth.storage import AuthStorage

        storage = AuthStorage()
        creds = storage.load(provider)
        if creds:
            return SecretStr(creds.access)
        return None

    def resolve_oauth_credentials(self, provider: str):  # noqa: ANN201
        """Load OAuth credentials for a provider from auth.json.

        Returns:
            OAuthCredentials | None
        """
        from ash.auth.storage import AuthStorage

        return AuthStorage().load(provider)

    def resolve_api_key(self, alias: str) -> SecretStr | None:
        return self._resolve_provider_api_key(self.get_model(alias).provider)

    def resolve_provider_api_key(
        self, provider: Literal["anthropic", "openai", "openai-oauth"]
    ) -> SecretStr | None:
        """Resolve API key for a provider from config/env/oauth storage."""
        return self._resolve_provider_api_key(provider)

    def create_llm_provider_for_model(self, alias: str):  # noqa: ANN201
        """Create an LLM provider instance for a model alias.

        Handles both API key-based providers and OAuth-based providers
        (openai-oauth). This is the preferred way to create providers —
        callers should use this instead of manually resolving credentials.

        Returns:
            LLMProvider instance.

        Raises:
            ValueError: If credentials are missing.
        """
        from ash.llm.registry import create_llm_provider

        model_config = self.get_model(alias)

        if model_config.provider == "openai-oauth":
            from ash.auth.storage import AuthStorage

            oauth_creds = self.resolve_oauth_credentials("openai-oauth")
            if not oauth_creds:
                raise ValueError(
                    "No OAuth credentials for openai-oauth. Run 'ash auth login' first."
                )
            return create_llm_provider(
                "openai-oauth",
                access_token=oauth_creds.access,
                account_id=oauth_creds.account_id,
                auth_storage=AuthStorage(),
            )

        api_key = self._resolve_provider_api_key(model_config.provider)
        return create_llm_provider(
            model_config.provider,
            api_key=api_key.get_secret_value() if api_key else None,
        )

    def create_llm_provider_for_provider(
        self, provider: Literal["anthropic", "openai", "openai-oauth"]
    ):
        """Create an LLM provider instance directly from a provider id."""
        from ash.llm.registry import create_llm_provider

        if provider == "openai-oauth":
            from ash.auth.storage import AuthStorage

            oauth_creds = self.resolve_oauth_credentials("openai-oauth")
            if not oauth_creds:
                raise ValueError(
                    "No OAuth credentials for openai-oauth. Run 'ash auth login' first."
                )
            return create_llm_provider(
                "openai-oauth",
                access_token=oauth_creds.access,
                account_id=oauth_creds.account_id,
                auth_storage=AuthStorage(),
            )

        api_key = self._resolve_provider_api_key(provider)
        return create_llm_provider(
            provider,
            api_key=api_key.get_secret_value() if api_key else None,
        )

    def resolve_embeddings_api_key(self) -> SecretStr | None:
        return (
            self._resolve_provider_api_key(self.embeddings.provider)
            if self.embeddings
            else None
        )

    def get_resolved_env(self) -> dict[str, str]:
        return {
            name: os.environ.get(value[1:], "") if value.startswith("$") else value
            for name, value in self.env.items()
        }
