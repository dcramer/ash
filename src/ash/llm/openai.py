"""OpenAI LLM provider."""

import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import openai
import openai.types.chat

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

DEFAULT_MODEL = "gpt-4o"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""

    def __init__(self, api_key: str | None = None):
        self._client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return DEFAULT_MODEL

    def _convert_messages(
        self, messages: list[Message], system: str | None = None
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == Role.SYSTEM:
                result.append({"role": "system", "content": msg.get_text()})
                continue

            if isinstance(msg.content, str):
                result.append({"role": msg.role.value, "content": msg.content})
                continue

            tool_calls = []
            tool_results = []
            text_parts = []

            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUse):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        }
                    )
                elif isinstance(block, ToolResult):
                    tool_results.append(block)

            if msg.role == Role.ASSISTANT:
                msg_dict: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    msg_dict["content"] = "\n".join(text_parts)
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                result.append(msg_dict)

            for tool_result in tool_results:
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.tool_use_id,
                        "content": tool_result.content,
                    }
                )

            if msg.role == Role.USER and text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})

        return result

    def _convert_tools(
        self, tools: list[ToolDefinition] | None
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
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
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._convert_messages(messages, system),
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools

        return kwargs

    def _parse_response(
        self, response: openai.types.chat.ChatCompletion
    ) -> CompletionResponse:
        choice = response.choices[0]
        msg = choice.message
        content: list[ContentBlock] = []

        if msg.content:
            content.append(TextContent(text=msg.content))

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if isinstance(
                    tool_call,
                    openai.types.chat.ChatCompletionMessageFunctionToolCall,
                ):
                    content.append(
                        ToolUse(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            input=json.loads(tool_call.function.arguments),
                        )
                    )

        usage = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        return CompletionResponse(
            message=Message(
                role=Role.ASSISTANT,
                content=content if content else "",
            ),
            usage=usage,
            stop_reason=choice.finish_reason,
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
            messages, model, tools, system, max_tokens, temperature
        )
        response = await self._client.chat.completions.create(**kwargs)
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
            messages, model, tools, system, max_tokens, temperature
        )
        kwargs["stream"] = True

        current_tool_calls: dict[int, dict[str, Any]] = {}
        response_stream = await self._client.chat.completions.create(**kwargs)

        yield StreamChunk(type=StreamEventType.MESSAGE_START)

        async for chunk in response_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                yield StreamChunk(
                    type=StreamEventType.TEXT_DELTA, content=delta.content
                )

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index

                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tool_call.id,
                            "name": tool_call.function.name
                            if tool_call.function
                            else "",
                            "arguments": "",
                        }
                        yield StreamChunk(
                            type=StreamEventType.TOOL_USE_START,
                            tool_use_id=tool_call.id,
                            tool_name=tool_call.function.name
                            if tool_call.function
                            else None,
                        )

                    if tool_call.function and tool_call.function.arguments:
                        current_tool_calls[idx]["arguments"] += (
                            tool_call.function.arguments
                        )
                        yield StreamChunk(
                            type=StreamEventType.TOOL_USE_DELTA,
                            content=tool_call.function.arguments,
                            tool_use_id=current_tool_calls[idx]["id"],
                        )

            if chunk.choices[0].finish_reason:
                for tc in current_tool_calls.values():
                    yield StreamChunk(
                        type=StreamEventType.TOOL_USE_END,
                        tool_use_id=tc["id"],
                        content=tc["arguments"],
                    )
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
