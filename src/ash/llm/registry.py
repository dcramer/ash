"""LLM provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import SecretStr

from ash.llm.anthropic import AnthropicProvider
from ash.llm.base import LLMProvider
from ash.llm.openai import OpenAIProvider
from ash.llm.openai_oauth import OpenAIOAuthProvider

if TYPE_CHECKING:
    from ash.auth.storage import AuthStorage

ProviderName = Literal["anthropic", "openai", "openai-oauth"]


def create_llm_provider(
    provider: ProviderName,
    api_key: str | SecretStr | None = None,
    *,
    access_token: str | None = None,
    account_id: str | None = None,
    auth_storage: AuthStorage | None = None,
) -> LLMProvider:
    """Create a single LLM provider instance.

    Args:
        provider: Provider name.
        api_key: API key (for anthropic/openai providers).
        access_token: OAuth access token (for openai-oauth).
        account_id: ChatGPT account ID (for openai-oauth).
        auth_storage: Auth storage for token refresh (for openai-oauth).

    Returns:
        LLM provider instance.

    Raises:
        ValueError: If provider name is unknown or required args missing.
    """
    key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key

    if provider == "anthropic":
        return AnthropicProvider(api_key=key)
    if provider == "openai":
        return OpenAIProvider(api_key=key)
    if provider == "openai-oauth":
        if not access_token or not account_id:
            raise ValueError(
                "openai-oauth provider requires access_token and account_id. "
                "Run 'ash auth login' first."
            )
        return OpenAIOAuthProvider(
            access_token=access_token,
            account_id=account_id,
            auth_storage=auth_storage,
        )

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
