"""Provider registry for managing communication providers."""

import logging

from ash.providers.base import Provider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry for communication provider instances."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        """Register a provider.

        Args:
            provider: Provider instance to register.

        Raises:
            ValueError: If provider with same name already registered.
        """
        if provider.name in self._providers:
            raise ValueError(f"Provider '{provider.name}' already registered")
        self._providers[provider.name] = provider
        logger.debug(f"Registered provider: {provider.name}")

    def unregister(self, name: str) -> None:
        """Unregister a provider by name.

        Args:
            name: Provider name to unregister.
        """
        self._providers.pop(name, None)

    def get(self, name: str) -> Provider:
        """Get a provider by name.

        Args:
            name: Provider name.

        Returns:
            Provider instance.

        Raises:
            KeyError: If provider not found.
        """
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not found")
        return self._providers[name]

    def has(self, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name.

        Returns:
            True if provider exists.
        """
        return name in self._providers

    @property
    def providers(self) -> dict[str, Provider]:
        """Get all registered providers."""
        return dict(self._providers)

    @property
    def names(self) -> list[str]:
        """Get list of registered provider names."""
        return list(self._providers.keys())

    def __len__(self) -> int:
        """Get number of registered providers."""
        return len(self._providers)

    def __contains__(self, name: str) -> bool:
        """Check if provider is registered."""
        return name in self._providers

    def __iter__(self):
        """Iterate over providers."""
        return iter(self._providers.values())
