"""OpenAI LLM provider (Responses API)."""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import openai

from ash.llm.base import LLMProvider
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

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIProvider(LLMProvider):
    """OpenAI provider using the Responses API."""

    def __init__(self, api_key: str | None = None):
        self._client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return DEFAULT_MODEL

    def _convert_input(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert messages to Responses API input format.

        Returns:
            Tuple of (instructions, input_items) where instructions is extracted
            from system messages and input_items is the conversation history.
        """
        instructions: str | None = None
        result: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                instructions = msg.get_text()
                continue

            if isinstance(msg.content, str):
                result.append({"role": msg.role.value, "content": msg.content})
                continue

            tool_results = []
            text_parts = []
            tool_calls = []

            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUse):
                    tool_calls.append(
                        {
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        }
                    )
                elif isinstance(block, ToolResult):
                    tool_results.append(block)

            if msg.role == Role.ASSISTANT:
                if text_parts:
                    result.append(
                        {"role": "assistant", "content": "\n".join(text_parts)}
                    )
                # Append function_call items directly as output items
                for tc in tool_calls:
                    result.append(tc)

            for tool_result in tool_results:
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_result.tool_use_id,
                        "output": tool_result.content,
                    }
                )

            if msg.role == Role.USER and text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})

        return instructions, result

    def _convert_tools(
        self, tools: list[ToolDefinition] | None
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
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
        reasoning: str | None = None,
    ) -> dict[str, Any]:
        msg_instructions, input_items = self._convert_input(messages)
        # Prefer explicit system param, fall back to system message from conversation
        instructions = system or msg_instructions

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "input": input_items,
            "max_output_tokens": max_tokens,
        }

        if instructions:
            kwargs["instructions"] = instructions

        if temperature is not None:
            kwargs["temperature"] = temperature

        if reasoning:
            kwargs["reasoning"] = {"effort": reasoning}

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools
            kwargs["tool_choice"] = "auto"

        return kwargs

    def _parse_response(self, response: Any) -> CompletionResponse:
        content: list[ContentBlock] = []

        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        content.append(TextContent(text=part.text))
            elif item.type == "function_call":
                content.append(
                    ToolUse(
                        id=item.call_id,
                        name=item.name,
                        input=json.loads(item.arguments),
                    )
                )

        usage = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        stop_reason = (
            "end_turn"
            if not any(item.type == "function_call" for item in response.output)
            else "tool_use"
        )

        return CompletionResponse(
            message=Message(
                role=Role.ASSISTANT,
                content=content if content else "",
            ),
            usage=usage,
            stop_reason=stop_reason,
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
        reasoning: str | None = None,
    ) -> CompletionResponse:
        kwargs = self._build_request_kwargs(
            messages, model, tools, system, max_tokens, temperature, reasoning
        )
        model_name = kwargs["model"]

        start_time = time.monotonic()
        response = await self._client.responses.create(**kwargs)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        usage = response.usage
        extra: dict[str, object] = {
            "provider": "openai",
            "model": model_name,
            "duration_ms": duration_ms,
        }
        if usage:
            extra["tokens_in"] = usage.input_tokens
            extra["tokens_out"] = usage.output_tokens
        logger.info("llm_complete", extra=extra)

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
        reasoning: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        kwargs = self._build_request_kwargs(
            messages, model, tools, system, max_tokens, temperature, reasoning
        )
        kwargs["stream"] = True

        current_tool_calls: dict[str, str] = {}  # call_id -> accumulated arguments
        response_stream = await self._client.responses.create(**kwargs)

        yield StreamChunk(type=StreamEventType.MESSAGE_START)

        async for event in response_stream:
            event_type = event.type

            if event_type == "response.output_text.delta":
                yield StreamChunk(type=StreamEventType.TEXT_DELTA, content=event.delta)

            elif event_type == "response.output_item.added":
                if event.item.type == "function_call":
                    call_id = event.item.call_id
                    current_tool_calls[call_id] = ""
                    yield StreamChunk(
                        type=StreamEventType.TOOL_USE_START,
                        tool_use_id=call_id,
                        tool_name=event.item.name,
                    )

            elif event_type == "response.function_call_arguments.delta":
                call_id = event.call_id
                if call_id in current_tool_calls:
                    current_tool_calls[call_id] += event.delta
                    yield StreamChunk(
                        type=StreamEventType.TOOL_USE_DELTA,
                        content=event.delta,
                        tool_use_id=call_id,
                    )

            elif event_type == "response.function_call_arguments.done":
                call_id = event.call_id
                if call_id in current_tool_calls:
                    yield StreamChunk(
                        type=StreamEventType.TOOL_USE_END,
                        tool_use_id=call_id,
                        content=current_tool_calls[call_id],
                    )

            elif event_type == "response.completed":
                yield StreamChunk(type=StreamEventType.MESSAGE_END)

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        embed_model = model or DEFAULT_EMBEDDING_MODEL
        logger.debug("Embedding %d texts with model %s", len(texts), embed_model)
        response = await self._client.embeddings.create(
            model=embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
