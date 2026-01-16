"""Anthropic Claude LLM provider."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import anthropic

from ash.llm.base import LLMProvider
from ash.llm.retry import RetryConfig, with_retry
from ash.llm.types import (
    CompletionResponse,
    ContentBlock,
    Message,
    Role,
    StreamChunk,
    StreamEventType,
    TextContent,
    ToolDefinition,
    ToolResult,
    ToolUse,
    Usage,
)

if TYPE_CHECKING:
    from ash.llm.thinking import ThinkingConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    _semaphore: asyncio.Semaphore | None = None
    _max_concurrent: int = 2

    def __init__(self, api_key: str | None = None, max_concurrent: int | None = None):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        if max_concurrent is not None:
            AnthropicProvider._max_concurrent = max_concurrent
        if AnthropicProvider._semaphore is None:
            AnthropicProvider._semaphore = asyncio.Semaphore(
                AnthropicProvider._max_concurrent
            )

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return DEFAULT_MODEL

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue

            content: str | list[dict[str, Any]]
            if isinstance(msg.content, str):
                content = msg.content
            else:
                content = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        content.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolUse):
                        content.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )
                    elif isinstance(block, ToolResult):
                        content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id,
                                "content": block.content,
                                "is_error": block.is_error,
                            }
                        )

            result.append({"role": msg.role.value, "content": content})
        return result

    def _convert_tools(
        self, tools: list[ToolDefinition] | None
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    def _build_request_kwargs(
        self,
        messages: list[Message],
        model: str | None,
        tools: list[ToolDefinition] | None,
        system: str | None,
        max_tokens: int,
        temperature: float | None,
        thinking: "ThinkingConfig | None",
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        if system:
            kwargs["system"] = system

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools

        if thinking and thinking.enabled:
            thinking_params = thinking.to_api_params()
            if thinking_params:
                kwargs.update(thinking_params)
                logger.debug(
                    f"Extended thinking enabled with budget={thinking.effective_budget}"
                )

        return kwargs

    def _parse_response(self, response: anthropic.types.Message) -> CompletionResponse:
        content: list[ContentBlock] = []

        for block in response.content:
            if block.type == "text":
                content.append(TextContent(text=block.text))
            elif block.type == "tool_use":
                content.append(
                    ToolUse(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input),
                    )
                )

        return CompletionResponse(
            message=Message(role=Role.ASSISTANT, content=content),
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason,
            model=response.model,
            raw=response.model_dump(),
        )

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
        kwargs = self._build_request_kwargs(
            messages, model, tools, system, max_tokens, temperature, thinking
        )
        model_name = kwargs["model"]

        assert self._semaphore is not None
        semaphore = self._semaphore
        logger.debug(f"Waiting for API slot (model={model_name})")

        async def _make_request() -> anthropic.types.Message:
            async with semaphore:
                logger.debug(f"Acquired API slot, calling {model_name}")
                response = await self._client.messages.create(**kwargs)
                logger.debug(
                    f"API call complete: {response.usage.input_tokens}in/"
                    f"{response.usage.output_tokens}out tokens"
                )
                return response

        response = await with_retry(
            _make_request,
            config=RetryConfig(enabled=True, max_retries=3),
            operation_name=f"Anthropic {model_name}",
        )
        return self._parse_response(response)

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
    ) -> AsyncGenerator[StreamChunk, None]:
        kwargs = self._build_request_kwargs(
            messages, model, tools, system, max_tokens, temperature, thinking
        )
        model_name = kwargs["model"]
        current_tool_id: str | None = None

        assert self._semaphore is not None
        logger.debug(f"Waiting for API slot (stream, model={model_name})")
        async with self._semaphore:
            logger.debug(f"Acquired API slot, streaming {model_name}")
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        yield StreamChunk(type=StreamEventType.MESSAGE_START)

                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool_id = event.content_block.id
                            yield StreamChunk(
                                type=StreamEventType.TOOL_USE_START,
                                tool_use_id=current_tool_id,
                                tool_name=event.content_block.name,
                            )

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield StreamChunk(
                                type=StreamEventType.TEXT_DELTA,
                                content=event.delta.text,
                            )
                        elif event.delta.type == "input_json_delta":
                            yield StreamChunk(
                                type=StreamEventType.TOOL_USE_DELTA,
                                content=event.delta.partial_json,
                                tool_use_id=current_tool_id,
                            )

                    elif event.type == "content_block_stop":
                        if current_tool_id:
                            yield StreamChunk(
                                type=StreamEventType.TOOL_USE_END,
                                tool_use_id=current_tool_id,
                            )
                            current_tool_id = None

                    elif event.type == "message_stop":
                        yield StreamChunk(type=StreamEventType.MESSAGE_END)
            logger.debug("Stream complete")

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. Use OpenAI for embeddings."
        )
