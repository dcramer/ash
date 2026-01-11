"""Tests for configuration loading and models."""

import pytest
from pydantic import SecretStr, ValidationError

from ash.config.loader import _resolve_env_secrets, get_default_config, load_config
from ash.config.models import (
    AshConfig,
    ConfigError,
    EmbeddingsConfig,
    LLMConfig,
    MemoryConfig,
    ModelConfig,
    ProviderConfig,
    SandboxConfig,
    ServerConfig,
    TelegramConfig,
)


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_minimal_config(self):
        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929")
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-5-20250929"
        assert config.temperature == 0.7  # default
        assert config.max_tokens == 4096  # default

    def test_full_config(self):
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
        )
        assert config.provider == "openai"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider="invalid", model="test")


class TestTelegramConfig:
    """Tests for TelegramConfig model."""

    def test_defaults(self):
        config = TelegramConfig()
        assert config.bot_token is None
        assert config.allowed_users == []
        assert config.webhook_url is None

    def test_with_values(self):
        config = TelegramConfig(
            allowed_users=["@user1", "123456"],
            webhook_url="https://example.com/webhook",
        )
        assert config.allowed_users == ["@user1", "123456"]
        assert config.webhook_url == "https://example.com/webhook"


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
        from ash.config.paths import get_database_path

        config = MemoryConfig()
        assert config.database_path == get_database_path()
        assert config.max_context_messages == 20


class TestAshConfig:
    """Tests for root AshConfig model."""

    def test_minimal_config(self, minimal_config):
        assert minimal_config.default_llm.provider == "anthropic"
        assert minimal_config.fallback_llm is None
        assert minimal_config.telegram is None

    def test_full_config(self, full_config):
        assert full_config.default_llm.provider == "anthropic"
        assert full_config.fallback_llm is not None
        assert full_config.fallback_llm.provider == "openai"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AshConfig()  # missing default_llm


class TestLoadConfig:
    """Tests for config file loading."""

    def test_load_from_file(self, config_file):
        config = load_config(config_file)
        assert config.default_llm.provider == "anthropic"
        assert config.default_llm.model == "claude-sonnet-4-5-20250929"

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
[default_llm]
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
        assert config.default_model.provider == "anthropic"
        assert "default" in config.list_models()


class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_minimal_config(self):
        config = ModelConfig(provider="anthropic", model="claude-sonnet-4-5-20250929")
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-5-20250929"
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
            provider="anthropic",
            model="claude-3-5-opus-20241219",
            temperature=None,  # Explicitly None for reasoning models
        )
        assert config.temperature is None

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            ModelConfig(provider="invalid", model="test")


class TestNamedModelConfigs:
    """Tests for named model configurations."""

    def test_models_dict_config(self):
        """Test [models.<alias>] configuration."""
        config = AshConfig(
            models={
                "default": ModelConfig(
                    provider="anthropic", model="claude-sonnet-4-5-20250929"
                ),
                "fast": ModelConfig(
                    provider="anthropic", model="claude-3-5-haiku-20241022"
                ),
            }
        )
        assert "default" in config.models
        assert "fast" in config.models
        assert config.models["default"].model == "claude-sonnet-4-5-20250929"
        assert config.models["fast"].model == "claude-3-5-haiku-20241022"

    def test_get_model(self):
        """Test get_model() lookup."""
        config = AshConfig(
            models={
                "default": ModelConfig(
                    provider="anthropic", model="claude-sonnet-4-5-20250929"
                ),
                "fast": ModelConfig(
                    provider="anthropic", model="claude-3-5-haiku-20241022"
                ),
            }
        )
        model = config.get_model("fast")
        assert model.provider == "anthropic"
        assert model.model == "claude-3-5-haiku-20241022"

    def test_get_model_unknown_alias(self):
        """Test get_model() with unknown alias raises ConfigError."""
        config = AshConfig(
            models={
                "default": ModelConfig(
                    provider="anthropic", model="claude-sonnet-4-5-20250929"
                ),
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
                "default": ModelConfig(
                    provider="anthropic", model="claude-sonnet-4-5-20250929"
                ),
                "fast": ModelConfig(
                    provider="anthropic", model="claude-3-5-haiku-20241022"
                ),
                "capable": ModelConfig(provider="openai", model="gpt-4o"),
            }
        )
        aliases = config.list_models()
        assert aliases == ["capable", "default", "fast"]

    def test_default_model_property(self):
        """Test default_model property returns 'default' alias."""
        config = AshConfig(
            models={
                "default": ModelConfig(
                    provider="anthropic", model="claude-sonnet-4-5-20250929"
                ),
            }
        )
        assert config.default_model.provider == "anthropic"
        assert config.default_model.model == "claude-sonnet-4-5-20250929"

    def test_resolve_api_key_from_provider_config(self):
        """Test API key resolution from provider-level config."""
        config = AshConfig(
            models={
                "default": ModelConfig(
                    provider="anthropic", model="claude-sonnet-4-5-20250929"
                ),
            },
            anthropic=ProviderConfig(api_key=SecretStr("test-key")),
        )
        api_key = config.resolve_api_key("default")
        assert api_key is not None
        assert api_key.get_secret_value() == "test-key"

    def test_resolve_api_key_from_env(self, monkeypatch):
        """Test API key resolution from environment variable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        config = AshConfig(
            models={
                "default": ModelConfig(
                    provider="anthropic", model="claude-sonnet-4-5-20250929"
                ),
            }
        )
        api_key = config.resolve_api_key("default")
        assert api_key is not None
        assert api_key.get_secret_value() == "env-key"

    def test_resolve_api_key_provider_takes_precedence(self, monkeypatch):
        """Test provider-level config takes precedence over env var."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        config = AshConfig(
            models={
                "default": ModelConfig(
                    provider="anthropic", model="claude-sonnet-4-5-20250929"
                ),
            },
            anthropic=ProviderConfig(api_key=SecretStr("config-key")),
        )
        api_key = config.resolve_api_key("default")
        assert api_key.get_secret_value() == "config-key"

    def test_resolve_api_key_none_if_missing(self, monkeypatch):
        """Test API key resolution returns None if not found."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = AshConfig(
            models={
                "default": ModelConfig(
                    provider="anthropic", model="claude-sonnet-4-5-20250929"
                ),
            }
        )
        api_key = config.resolve_api_key("default")
        assert api_key is None


class TestBackwardCompatibility:
    """Tests for backward compatibility with [default_llm]."""

    def test_default_llm_migrates_to_models(self):
        """Test [default_llm] is migrated to models.default."""
        config = AshConfig(
            default_llm=LLMConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                temperature=0.5,
                max_tokens=2048,
            )
        )
        assert "default" in config.models
        assert config.models["default"].provider == "anthropic"
        assert config.models["default"].model == "claude-sonnet-4-5-20250929"
        assert config.models["default"].temperature == 0.5
        assert config.models["default"].max_tokens == 2048

    def test_default_llm_api_key_migrates_to_provider(self):
        """Test default_llm api_key is migrated to provider config."""
        config = AshConfig(
            default_llm=LLMConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                api_key=SecretStr("test-key"),
            )
        )
        assert config.anthropic is not None
        assert config.anthropic.api_key is not None
        assert config.anthropic.api_key.get_secret_value() == "test-key"

    def test_models_default_takes_precedence_over_default_llm(self, caplog):
        """Test [models.default] takes precedence over [default_llm]."""
        import logging

        with caplog.at_level(logging.WARNING):
            config = AshConfig(
                models={
                    "default": ModelConfig(provider="openai", model="gpt-4o"),
                },
                default_llm=LLMConfig(
                    provider="anthropic",
                    model="claude-sonnet-4-5-20250929",
                ),
            )
        # models.default should win
        assert config.models["default"].provider == "openai"
        assert config.models["default"].model == "gpt-4o"
        # Warning should be logged
        assert "Both [default_llm] and [models.default] present" in caplog.text

    def test_no_default_model_raises_error(self):
        """Test error when no default model is configured."""
        with pytest.raises(ValueError) as exc_info:
            AshConfig(models={})
        assert "No default model configured" in str(exc_info.value)


class TestLoadConfigWithModels:
    """Tests for loading config with [models.*] sections."""

    def test_load_models_from_toml(self, tmp_path):
        """Test loading [models.*] sections from TOML."""
        config_content = """
[models.default]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
temperature = 0.7

[models.fast]
provider = "anthropic"
model = "claude-3-5-haiku-20241022"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content)
        config = load_config(config_file)

        assert "default" in config.models
        assert "fast" in config.models
        assert config.models["default"].model == "claude-sonnet-4-5-20250929"
        assert config.models["fast"].model == "claude-3-5-haiku-20241022"

    def test_load_provider_api_key_from_toml(self, tmp_path):
        """Test loading provider API keys from TOML."""
        config_content = """
[models.default]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"

[anthropic]
api_key = "test-api-key"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content)
        config = load_config(config_file)

        assert config.anthropic is not None
        assert config.anthropic.api_key is not None
        assert config.anthropic.api_key.get_secret_value() == "test-api-key"


class TestResolveEnvSecrets:
    """Tests for environment variable resolution."""

    def test_resolves_anthropic_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        config = {
            "default_llm": {
                "provider": "anthropic",
                "model": "test",
            }
        }
        result = _resolve_env_secrets(config)
        assert result["default_llm"]["api_key"].get_secret_value() == "test-key"

    def test_resolves_openai_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        config = {
            "default_llm": {
                "provider": "openai",
                "model": "test",
            }
        }
        result = _resolve_env_secrets(config)
        assert result["default_llm"]["api_key"].get_secret_value() == "test-openai-key"

    def test_resolves_telegram_token(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        config = {
            "default_llm": {"provider": "anthropic", "model": "test"},
            "telegram": {},
        }
        result = _resolve_env_secrets(config)
        assert result["telegram"]["bot_token"].get_secret_value() == "test-token"

    def test_resolves_brave_search_key(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
        config = {
            "default_llm": {"provider": "anthropic", "model": "test"},
            "brave_search": {},
        }
        result = _resolve_env_secrets(config)
        assert result["brave_search"]["api_key"].get_secret_value() == "brave-key"

    def test_does_not_override_existing_value(self, monkeypatch):
        from pydantic import SecretStr

        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        config = {
            "default_llm": {
                "provider": "anthropic",
                "model": "test",
                "api_key": SecretStr("file-key"),
            }
        }
        result = _resolve_env_secrets(config)
        # Should keep file-key, not override with env-key
        assert result["default_llm"]["api_key"].get_secret_value() == "file-key"

    def test_missing_env_var_leaves_none(self, monkeypatch):
        # Ensure env var is not set
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = {
            "default_llm": {
                "provider": "anthropic",
                "model": "test",
            }
        }
        result = _resolve_env_secrets(config)
        assert result["default_llm"].get("api_key") is None
