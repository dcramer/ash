"""LLM provider registry."""

from typing import Literal

from pydantic import SecretStr

from ash.llm.anthropic import AnthropicProvider
from ash.llm.base import LLMProvider
from ash.llm.openai import OpenAIProvider

ProviderName = Literal["anthropic", "openai"]


def create_llm_provider(
    provider: ProviderName,
    api_key: str | SecretStr | None = None,
) -> LLMProvider:
    """Create a single LLM provider instance.

    Args:
        provider: Provider name ("anthropic" or "openai").
        api_key: API key (or uses env var if not provided).

    Returns:
        LLM provider instance.

    Raises:
        ValueError: If provider name is unknown.
    """
    # Extract string from SecretStr if needed
    key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key

    if provider == "anthropic":
        return AnthropicProvider(api_key=key)
    elif provider == "openai":
        return OpenAIProvider(api_key=key)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


class LLMRegistry:
    """Registry for LLM providers."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        """Register a provider instance."""
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider:
        """Get a provider by name.

        Raises:
            KeyError: If provider not found.
        """
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        return self._providers[name]

    def has(self, name: str) -> bool:
        """Check if a provider is registered."""
        return name in self._providers

    @property
    def providers(self) -> dict[str, LLMProvider]:
        """Get all registered providers."""
        return dict(self._providers)


def create_registry(
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> LLMRegistry:
    """Create a registry with default providers.

    Args:
        anthropic_api_key: Anthropic API key (or uses env var).
        openai_api_key: OpenAI API key (or uses env var).

    Returns:
        Registry with Anthropic and OpenAI providers.
    """
    registry = LLMRegistry()
    registry.register(AnthropicProvider(api_key=anthropic_api_key))
    registry.register(OpenAIProvider(api_key=openai_api_key))
    return registry
