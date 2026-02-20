"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
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
        reasoning: str | None = None,
    ) -> CompletionResponse:
        """Generate a completion (non-streaming)."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        thinking: "ThinkingConfig | None" = None,
        reasoning: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming completion."""
        ...

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...
