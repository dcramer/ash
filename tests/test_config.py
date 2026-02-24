"""Tests for configuration loading and models."""

import pytest
from pydantic import SecretStr, ValidationError

from ash.config.loader import _resolve_env_secrets, get_default_config, load_config
from ash.config.models import (
    AshConfig,
    BrowserConfig,
    CapabilitiesConfig,
    CapabilityProviderConfig,
    ConfigError,
    EmbeddingsConfig,
    MemoryConfig,
    ModelConfig,
    ProviderConfig,
    SandboxConfig,
    ServerConfig,
    SkillConfig,
    TelegramConfig,
    TodoConfig,
)


class TestTelegramConfig:
    """Tests for TelegramConfig model."""

    def test_defaults(self):
        config = TelegramConfig()
        assert config.bot_token is None
        assert config.allowed_users == []

    def test_with_values(self):
        config = TelegramConfig(
            allowed_users=["@user1", "123456"],
        )
        assert config.allowed_users == ["@user1", "123456"]


class TestSandboxConfig:
    """Tests for SandboxConfig model."""

    def test_defaults(self):
        config = SandboxConfig()
        assert config.image == "ash-sandbox:latest"
        assert config.timeout == 60
        assert config.memory_limit == "512m"
        assert config.cpu_limit == 1.0
        assert config.runtime == "runc"
        assert config.network_mode == "bridge"
        assert config.dns_servers == []
        assert config.http_proxy is None
        assert config.workspace_access == "rw"

    def test_gvisor_runtime(self):
        config = SandboxConfig(runtime="runsc")
        assert config.runtime == "runsc"

    def test_network_none(self):
        config = SandboxConfig(network_mode="none")
        assert config.network_mode == "none"

    def test_with_proxy(self):
        config = SandboxConfig(
            http_proxy="http://localhost:8888",
            dns_servers=["1.1.1.1", "8.8.8.8"],
        )
        assert config.http_proxy == "http://localhost:8888"
        assert config.dns_servers == ["1.1.1.1", "8.8.8.8"]

    def test_workspace_readonly(self):
        config = SandboxConfig(workspace_access="ro")
        assert config.workspace_access == "ro"

    def test_workspace_none(self):
        config = SandboxConfig(workspace_access="none")
        assert config.workspace_access == "none"


class TestServerConfig:
    """Tests for ServerConfig model."""

    def test_defaults(self):
        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.webhook_path == "/webhook"


class TestEmbeddingsConfig:
    """Tests for EmbeddingsConfig model."""

    def test_defaults(self):
        config = EmbeddingsConfig()
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"

    def test_custom_model(self):
        config = EmbeddingsConfig(model="text-embedding-3-large")
        assert config.model == "text-embedding-3-large"


class TestMemoryConfig:
    """Tests for MemoryConfig model."""

    def test_defaults(self):
        config = MemoryConfig()
        assert config.max_context_messages == 20
        assert config.query_planning_enabled is True
        assert config.query_planning_model_alias is None
        assert config.query_planning_fetch_memories == 25
        assert config.context_injection_limit == 10


class TestBrowserConfig:
    """Tests for BrowserConfig model."""

    def test_defaults(self):
        config = BrowserConfig()
        assert config.enabled is True
        assert config.provider == "sandbox"
        assert config.timeout_seconds == 20.0


class TestCapabilitiesConfig:
    def test_defaults(self):
        config = CapabilitiesConfig()
        assert config.providers == {}

    def test_provider_config_values(self):
        config = CapabilitiesConfig(
            providers={
                "gog": CapabilityProviderConfig(
                    enabled=True,
                    namespace="gog",
                    command=["gogcli", "bridge"],
                    timeout_seconds=45.0,
                )
            }
        )
        assert config.providers["gog"].enabled is True
        assert config.providers["gog"].namespace == "gog"
        assert config.providers["gog"].command == ["gogcli", "bridge"]
        assert config.providers["gog"].timeout_seconds == 45.0

    def test_provider_config_parses_command_string(self):
        config = CapabilityProviderConfig.model_validate(
            {
                "namespace": "gog",
                "command": "gogcli bridge --mode mock",
            }
        )
        assert config.command == ["gogcli", "bridge", "--mode", "mock"]


class TestAshConfig:
    """Tests for root AshConfig model."""

    def test_minimal_config(self, minimal_config):
        assert minimal_config.default_model.provider == "openai"
        assert minimal_config.telegram is None

    def test_full_config(self, full_config):
        assert full_config.default_model.provider == "openai"
        assert "fast" in full_config.models
        assert isinstance(full_config.todo, TodoConfig)
        assert full_config.todo.enabled is True

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AshConfig()


class TestLoadConfig:
    """Tests for config file loading."""

    def test_load_from_file(self, config_file):
        config = load_config(config_file)
        assert config.default_model.provider == "openai"
        assert config.default_model.model == "gpt-5.2"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.toml")

    def test_invalid_toml(self, tmp_path):
        import tomllib

        invalid_file = tmp_path / "invalid.toml"
        invalid_file.write_text("this is not valid toml [[[")
        with pytest.raises(tomllib.TOMLDecodeError):
            load_config(invalid_file)

    def test_invalid_config_values(self, tmp_path):
        invalid_config = tmp_path / "invalid_config.toml"
        invalid_config.write_text("""
[models.default]
provider = "invalid_provider"
model = "test"
""")
        with pytest.raises(ValidationError):
            load_config(invalid_config)


class TestGetDefaultConfig:
    """Tests for default configuration."""

    def test_returns_valid_config(self):
        config = get_default_config()
        assert isinstance(config, AshConfig)
        assert config.default_model.provider == "openai"
        assert "default" in config.list_models()


class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_minimal_config(self):
        config = ModelConfig(provider="openai", model="gpt-5.2")
        assert config.provider == "openai"
        assert config.model == "gpt-5.2"
        assert config.temperature is None  # default: use API default
        assert config.max_tokens == 4096  # default

    def test_full_config(self):
        config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
        )
        assert config.provider == "openai"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_temperature_omitted_for_reasoning_models(self):
        """Test that temperature can be None (for reasoning models)."""
        config = ModelConfig(
            provider="openai",
            model="gpt-5.2-codex",
            temperature=None,  # Explicitly None for reasoning models
        )
        assert config.temperature is None

    def test_reasoning_field(self):
        """Test reasoning field accepts valid values."""
        config = ModelConfig(
            provider="openai",
            model="gpt-5.2-pro",
            reasoning="high",
        )
        assert config.reasoning == "high"

    def test_reasoning_default_none(self):
        """Test reasoning defaults to None."""
        config = ModelConfig(provider="openai", model="gpt-5.2")
        assert config.reasoning is None

    def test_reasoning_invalid_value(self):
        """Test reasoning rejects invalid values."""
        with pytest.raises(ValidationError):
            ModelConfig(
                provider="openai",
                model="gpt-5.2",
                reasoning="xhigh",  # type: ignore[arg-type]
            )

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            ModelConfig(provider="invalid", model="test")  # type: ignore[arg-type]


class TestNamedModelConfigs:
    """Tests for named model configurations."""

    def test_models_dict_config(self):
        """Test [models.<alias>] configuration."""
        config = AshConfig(
            models={
                "default": ModelConfig(provider="openai", model="gpt-5.2"),
                "fast": ModelConfig(provider="openai", model="gpt-5-mini"),
            }
        )
        assert "default" in config.models
        assert "fast" in config.models
        assert config.models["default"].model == "gpt-5.2"
        assert config.models["fast"].model == "gpt-5-mini"

    def test_get_model(self):
        """Test get_model() lookup."""
        config = AshConfig(
            models={
                "default": ModelConfig(provider="openai", model="gpt-5.2"),
                "fast": ModelConfig(provider="openai", model="gpt-5-mini"),
            }
        )
        model = config.get_model("fast")
        assert model.provider == "openai"
        assert model.model == "gpt-5-mini"

    def test_get_model_unknown_alias(self):
        """Test get_model() with unknown alias raises ConfigError."""
        config = AshConfig(
            models={
                "default": ModelConfig(provider="openai", model="gpt-5.2"),
            }
        )
        with pytest.raises(ConfigError) as exc_info:
            config.get_model("unknown")
        assert "Unknown model alias 'unknown'" in str(exc_info.value)
        assert "default" in str(exc_info.value)  # Should list available

    def test_list_models(self):
        """Test list_models() returns sorted aliases."""
        config = AshConfig(
            models={
                "default": ModelConfig(provider="openai", model="gpt-5.2"),
                "fast": ModelConfig(provider="openai", model="gpt-5-mini"),
                "capable": ModelConfig(provider="openai", model="gpt-4o"),
            }
        )
        aliases = config.list_models()
        assert aliases == ["capable", "default", "fast"]

    def test_default_model_property(self):
        """Test default_model property returns 'default' alias."""
        config = AshConfig(
            models={
                "default": ModelConfig(provider="openai", model="gpt-5.2"),
            }
        )
        assert config.default_model.provider == "openai"
        assert config.default_model.model == "gpt-5.2"

    def test_resolve_api_key_from_provider_config(self):
        """Test API key resolution from provider-level config."""
        config = AshConfig(
            models={
                "default": ModelConfig(provider="openai", model="gpt-5.2"),
            },
            openai=ProviderConfig(api_key=SecretStr("test-key")),
        )
        api_key = config.resolve_api_key("default")
        assert api_key is not None
        assert api_key.get_secret_value() == "test-key"

    def test_resolve_api_key_from_env(self, monkeypatch):
        """Test API key resolution from environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        config = AshConfig(
            models={
                "default": ModelConfig(provider="openai", model="gpt-5.2"),
            }
        )
        api_key = config.resolve_api_key("default")
        assert api_key is not None
        assert api_key.get_secret_value() == "env-key"

    def test_resolve_api_key_provider_takes_precedence(self, monkeypatch):
        """Test provider-level config takes precedence over env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        config = AshConfig(
            models={
                "default": ModelConfig(provider="openai", model="gpt-5.2"),
            },
            openai=ProviderConfig(api_key=SecretStr("config-key")),
        )
        api_key = config.resolve_api_key("default")
        assert api_key is not None
        assert api_key.get_secret_value() == "config-key"

    def test_resolve_api_key_none_if_missing(self, monkeypatch):
        """Test API key resolution returns None if not found."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = AshConfig(
            models={
                "default": ModelConfig(provider="openai", model="gpt-5.2"),
            }
        )
        api_key = config.resolve_api_key("default")
        assert api_key is None


class TestConfigValidation:
    def test_no_default_model_raises_error(self):
        with pytest.raises(ValueError) as exc_info:
            AshConfig(models={})
        assert "No default model configured" in str(exc_info.value)


class TestLoadConfigWithModels:
    """Tests for loading config with [models.*] sections."""

    def test_load_models_from_toml(self, tmp_path):
        """Test loading [models.*] sections from TOML."""
        config_content = """
[models.default]
provider = "openai"
model = "gpt-5.2"
temperature = 0.7

[models.fast]
provider = "openai"
model = "gpt-5-mini"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content)
        config = load_config(config_file)

        assert "default" in config.models
        assert "fast" in config.models
        assert config.models["default"].model == "gpt-5.2"
        assert config.models["fast"].model == "gpt-5-mini"

    def test_load_provider_api_key_from_toml(self, tmp_path):
        """Test loading provider API keys from TOML."""
        config_content = """
[models.default]
provider = "openai"
model = "gpt-5.2"

[openai]
api_key = "test-api-key"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content)
        config = load_config(config_file)

        assert config.openai is not None
        assert config.openai.api_key is not None
        assert config.openai.api_key.get_secret_value() == "test-api-key"


class TestResolveEnvSecrets:
    """Tests for environment variable resolution."""

    def test_resolves_anthropic_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        config = {"anthropic": {}}
        result = _resolve_env_secrets(config)
        assert result["anthropic"]["api_key"].get_secret_value() == "test-key"

    def test_resolves_openai_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        config = {"openai": {}}
        result = _resolve_env_secrets(config)
        assert result["openai"]["api_key"].get_secret_value() == "test-openai-key"

    def test_resolves_kernel_api_key(self, monkeypatch):
        monkeypatch.setenv("KERNEL_API_KEY", "kernel-test-key")
        config = {"browser": {"kernel": {"api_key": None}}}
        result = _resolve_env_secrets(config)
        assert (
            result["browser"]["kernel"]["api_key"].get_secret_value()
            == "kernel-test-key"
        )

    def test_resolves_telegram_token(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        config = {
            "models": {"default": {"provider": "anthropic", "model": "test"}},
            "telegram": {},
        }
        result = _resolve_env_secrets(config)
        assert result["telegram"]["bot_token"].get_secret_value() == "test-token"

    def test_resolves_brave_search_key(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
        config = {
            "models": {"default": {"provider": "anthropic", "model": "test"}},
            "brave_search": {},
        }
        result = _resolve_env_secrets(config)
        assert result["brave_search"]["api_key"].get_secret_value() == "brave-key"

    def test_does_not_override_existing_value(self, monkeypatch):
        from pydantic import SecretStr

        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        config = {"anthropic": {"api_key": SecretStr("file-key")}}
        result = _resolve_env_secrets(config)
        # Should keep file-key, not override with env-key
        assert result["anthropic"]["api_key"].get_secret_value() == "file-key"

    def test_missing_env_var_leaves_none(self, monkeypatch):
        # Ensure env var is not set
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = {"anthropic": {}}
        result = _resolve_env_secrets(config)
        assert result["anthropic"].get("api_key") is None


class TestSystemTimezone:
    """Tests for system timezone detection."""

    def test_get_system_timezone_returns_string(self):
        """Test that get_system_timezone returns a valid timezone string."""
        from ash.config.paths import get_system_timezone

        tz = get_system_timezone()
        assert isinstance(tz, str)
        assert len(tz) > 0

    def test_get_system_timezone_respects_tz_env(self, monkeypatch):
        """Test that TZ environment variable takes precedence."""
        from ash.config.paths import get_system_timezone

        monkeypatch.setenv("TZ", "America/New_York")
        tz = get_system_timezone()
        assert tz == "America/New_York"

    def test_get_system_timezone_is_valid_iana(self):
        """Test that returned timezone is valid IANA name."""
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        from ash.config.paths import get_system_timezone

        tz = get_system_timezone()
        try:
            ZoneInfo(tz)
        except ZoneInfoNotFoundError:
            pytest.fail(f"get_system_timezone returned invalid IANA name: {tz}")


class TestAshConfigTimezoneDefault:
    """Tests for AshConfig timezone default behavior."""

    def test_timezone_defaults_to_system(self, monkeypatch):
        """Test that AshConfig timezone defaults to system timezone."""
        from ash.config.models import AshConfig, ModelConfig
        from ash.config.paths import get_system_timezone

        # Set a known timezone
        monkeypatch.setenv("TZ", "Europe/London")

        config = AshConfig(
            models={"default": ModelConfig(provider="openai", model="test")}
        )

        # Should default to system timezone
        assert config.timezone == get_system_timezone()
        assert config.timezone == "Europe/London"

    def test_timezone_can_be_overridden(self):
        """Test that timezone can be explicitly set."""
        from ash.config.models import AshConfig, ModelConfig

        config = AshConfig(
            models={"default": ModelConfig(provider="openai", model="test")},
            timezone="America/Los_Angeles",
        )
        assert config.timezone == "America/Los_Angeles"

    def test_invalid_timezone_raises_error(self):
        """Test that invalid timezone raises validation error."""
        from ash.config.models import AshConfig, ModelConfig

        with pytest.raises(ValidationError) as exc_info:
            AshConfig(
                models={"default": ModelConfig(provider="openai", model="test")},
                timezone="Invalid/Timezone",
            )
        assert "Invalid timezone" in str(exc_info.value)


class TestSkillAutoSyncConfig:
    def test_parses_skill_auto_sync_minutes(self):
        from ash.config.models import AshConfig

        config = AshConfig.model_validate(
            {
                "models": {"default": {"provider": "openai", "model": "test"}},
                "skills": {
                    "auto_sync": True,
                    "update_interval_minutes": 5,
                    "sources": [{"repo": "owner/repo"}],
                },
            }
        )
        assert config.skill_auto_sync is True
        assert config.skill_update_interval_minutes == 5
        assert len(config.skill_sources) == 1

    def test_legacy_hours_interval_converts_to_minutes(self):
        from ash.config.models import AshConfig

        config = AshConfig.model_validate(
            {
                "models": {"default": {"provider": "openai", "model": "test"}},
                "skills": {"auto_sync": True, "update_interval": 2},
            }
        )
        assert config.skill_update_interval_minutes == 120

    def test_parses_skill_defaults_allow_chat_ids(self):
        config = AshConfig.model_validate(
            {
                "models": {"default": {"provider": "openai", "model": "test"}},
                "skills": {
                    "defaults": {"allow_chat_ids": ["dm-1", "dm-2"]},
                    "research": {"enabled": True},
                },
            }
        )
        assert config.skill_defaults.allow_chat_ids == ["dm-1", "dm-2"]
        assert config.skills["research"].allow_chat_ids is None

    def test_per_skill_allow_chat_ids_override(self):
        config = AshConfig.model_validate(
            {
                "models": {"default": {"provider": "openai", "model": "test"}},
                "skills": {
                    "defaults": {"allow_chat_ids": ["dm-default"]},
                    "research": {"allow_chat_ids": ["dm-team"]},
                },
            }
        )
        assert config.skill_defaults.allow_chat_ids == ["dm-default"]
        assert config.skills["research"].allow_chat_ids == ["dm-team"]

    def test_skill_config_get_env_vars_ignores_allow_chat_ids(self):
        cfg = SkillConfig.model_validate(
            {
                "API_KEY": "secret",
                "allow_chat_ids": ["dm-1"],
            }
        )
        assert cfg.get_env_vars() == {"API_KEY": "secret"}

    def test_bundle_gog_enable_applies_skill_and_provider_defaults(self):
        config = AshConfig.model_validate(
            {
                "models": {"default": {"provider": "openai", "model": "test"}},
                "bundles": {"gog": {"enabled": True}},
            }
        )
        assert config.bundles.gog.enabled is True
        assert config.skills["gog"].enabled is True
        provider = config.capabilities.providers["gog"]
        assert provider.enabled is True
        assert provider.namespace == "gog"
        assert provider.command == ["gogcli", "bridge"]
        assert provider.timeout_seconds == 30.0

    def test_bundle_gog_enable_does_not_override_explicit_config(self):
        config = AshConfig.model_validate(
            {
                "models": {"default": {"provider": "openai", "model": "test"}},
                "bundles": {"gog": {"enabled": True}},
                "skills": {"gog": {"enabled": False}},
                "capabilities": {
                    "providers": {
                        "gog": {
                            "enabled": False,
                            "namespace": "gog",
                            "command": ["custom-gogcli", "bridge"],
                            "timeout_seconds": 45.0,
                        }
                    }
                },
            }
        )
        assert config.skills["gog"].enabled is False
        provider = config.capabilities.providers["gog"]
        assert provider.enabled is False
        assert provider.command == ["custom-gogcli", "bridge"]
        assert provider.timeout_seconds == 45.0
