"""OpenAI OAuth LLM provider (ChatGPT OAuth, Codex Responses API)."""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import openai

from ash.llm.openai import OpenAIProvider
from ash.llm.types import (
    CompletionResponse,
    ContentBlock,
    Message,
    Role,
    StreamChunk,
    StreamEventType,
    TextContent,
    ToolDefinition,
    ToolUse,
)

if TYPE_CHECKING:
    from ash.auth.storage import AuthStorage
    from ash.llm.thinking import ThinkingConfig

logger = logging.getLogger(__name__)

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Refresh tokens 5 minutes before expiry
TOKEN_REFRESH_BUFFER_SECONDS = 300


class OpenAIOAuthProvider(OpenAIProvider):
    """OpenAI provider using the Codex Responses API with OAuth credentials.

    Uses the same Responses API format as OpenAIProvider but targets the
    Codex endpoint at chatgpt.com with OAuth-based authentication.
    Embeddings are not supported via this provider.
    """

    def __init__(
        self,
        access_token: str,
        account_id: str,
        auth_storage: "AuthStorage | None" = None,
    ):
        # Skip OpenAIProvider.__init__ — we configure the client ourselves
        self._account_id = account_id
        self._auth_storage = auth_storage
        self._client = openai.AsyncOpenAI(
            api_key=access_token,
            base_url=CODEX_BASE_URL,
            default_headers={
                "chatgpt-account-id": account_id,
                "OpenAI-Beta": "responses=experimental",
                "originator": "ash",
            },
        )

    @property
    def name(self) -> str:
        return "openai-oauth"

    def _build_request_kwargs(
        self, *args: object, **kwargs: object
    ) -> dict[str, object]:
        result = super()._build_request_kwargs(*args, **kwargs)  # type: ignore[arg-type]
        # Codex endpoint requires store=false
        result["store"] = False
        return result

    async def _maybe_refresh_token(self) -> None:
        """Check token expiry and refresh if needed."""
        if not self._auth_storage:
            return

        creds = self._auth_storage.load("openai-oauth")
        if not creds:
            return

        if time.time() < creds.expires - TOKEN_REFRESH_BUFFER_SECONDS:
            return

        logger.info("Refreshing OpenAI OAuth token (expires %.0f)", creds.expires)

        from ash.auth.oauth import extract_account_id, refresh_access_token

        try:
            tokens = await refresh_access_token(creds.refresh)
            new_access = str(tokens["access"])
            new_account_id = extract_account_id(new_access) or creds.account_id

            from ash.auth.storage import OAuthCredentials

            new_creds = OAuthCredentials(
                access=new_access,
                refresh=str(tokens["refresh"]),
                expires=float(tokens["expires"]),
                account_id=new_account_id,
            )
            self._auth_storage.save("openai-oauth", new_creds)

            # Update the client with new token
            self._client = openai.AsyncOpenAI(
                api_key=new_access,
                base_url=CODEX_BASE_URL,
                default_headers={
                    "chatgpt-account-id": new_account_id,
                    "OpenAI-Beta": "responses=experimental",
                    "originator": "ash",
                },
            )
            self._account_id = new_account_id
            logger.info("OpenAI OAuth token refreshed successfully")
        except Exception:
            logger.warning("Failed to refresh OpenAI OAuth token", exc_info=True)

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
        # Codex endpoint requires stream=true for all requests.
        # Collect the stream into a CompletionResponse.
        await self._maybe_refresh_token()

        text_parts: list[str] = []
        tool_calls: dict[str, tuple[str, str]] = {}  # call_id -> (name, args)
        current_tool_args: dict[str, str] = {}  # call_id -> accumulated args

        async for chunk in self.stream(
            messages,
            model=model,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
            reasoning=reasoning,
        ):
            if chunk.type == StreamEventType.TEXT_DELTA and chunk.content:
                text_parts.append(str(chunk.content))
            elif chunk.type == StreamEventType.TOOL_USE_START and chunk.tool_use_id:
                current_tool_args[chunk.tool_use_id] = ""
                tool_calls[chunk.tool_use_id] = (chunk.tool_name or "", "")
            elif chunk.type == StreamEventType.TOOL_USE_DELTA and chunk.tool_use_id:
                current_tool_args[chunk.tool_use_id] += str(chunk.content or "")
            elif chunk.type == StreamEventType.TOOL_USE_END and chunk.tool_use_id:
                name = tool_calls[chunk.tool_use_id][0]
                args = current_tool_args.get(chunk.tool_use_id, "")
                tool_calls[chunk.tool_use_id] = (name, args)

        content: list[ContentBlock] = []
        if text_parts:
            content.append(TextContent(text="".join(text_parts)))
        for call_id, (name, args) in tool_calls.items():
            content.append(
                ToolUse(id=call_id, name=name, input=json.loads(args) if args else {})
            )

        stop_reason = "tool_use" if tool_calls else "end_turn"

        return CompletionResponse(
            message=Message(role=Role.ASSISTANT, content=content if content else ""),
            usage=None,
            stop_reason=stop_reason,
            model=model,
        )

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
        await self._maybe_refresh_token()
        async for chunk in super().stream(
            messages,
            model=model,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
            reasoning=reasoning,
        ):
            yield chunk

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        raise NotImplementedError(
            "Embeddings are not supported via OpenAI OAuth. "
            "Use the standard OpenAI provider with an API key for embeddings."
        )
