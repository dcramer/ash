"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ash.llm.types import (
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)

if TYPE_CHECKING:
    from ash.llm.thinking import ThinkingConfig


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'anthropic', 'openai')."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this provider."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        thinking: "ThinkingConfig | None" = None,
    ) -> CompletionResponse:
        """Generate a completion (non-streaming).

        Args:
            messages: Conversation history.
            model: Model to use (defaults to provider's default).
            tools: Available tools for the model.
            system: System prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature. None = use API default (omit for reasoning models).
            thinking: Extended thinking configuration (Anthropic only).

        Returns:
            Complete response with message and metadata.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        thinking: "ThinkingConfig | None" = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion.

        Args:
            messages: Conversation history.
            model: Model to use (defaults to provider's default).
            tools: Available tools for the model.
            system: System prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature. None = use API default (omit for reasoning models).
            thinking: Extended thinking configuration (Anthropic only).

        Yields:
            Stream chunks as they arrive.
        """
        ...

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: Texts to embed.
            model: Embedding model to use.

        Returns:
            List of embedding vectors.
        """
        ...
