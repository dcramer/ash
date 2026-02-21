"""Tests for OpenAI OAuth LLM provider."""

import pytest

from ash.llm.openai_oauth import (
    CODEX_BASE_URL,
    DEFAULT_CODEX_INSTRUCTIONS,
    OpenAIOAuthProvider,
)
from ash.llm.types import Message, Role


class TestOpenAIOAuthProvider:
    def test_name(self):
        provider = OpenAIOAuthProvider(
            access_token="test-token",
            account_id="acct_123",
        )
        assert provider.name == "openai-oauth"

    def test_client_configuration(self):
        provider = OpenAIOAuthProvider(
            access_token="test-token",
            account_id="acct_123",
        )
        assert str(provider._client.base_url).rstrip("/") == CODEX_BASE_URL
        assert provider._client.api_key == "test-token"

    def test_default_headers(self):
        provider = OpenAIOAuthProvider(
            access_token="test-token",
            account_id="acct_123",
        )
        headers = provider._client._custom_headers
        assert headers["chatgpt-account-id"] == "acct_123"
        assert headers["OpenAI-Beta"] == "responses=experimental"
        assert headers["originator"] == "ash"

    async def test_embed_raises(self):
        provider = OpenAIOAuthProvider(
            access_token="test-token",
            account_id="acct_123",
        )
        with pytest.raises(NotImplementedError, match="not supported"):
            await provider.embed(["test"])

    def test_inherits_openai_provider(self):
        from ash.llm.openai import OpenAIProvider

        assert issubclass(OpenAIOAuthProvider, OpenAIProvider)

    def test_build_request_kwargs_injects_default_instructions_when_missing(self):
        provider = OpenAIOAuthProvider(
            access_token="test-token",
            account_id="acct_123",
        )
        kwargs = provider._build_request_kwargs(
            messages=[Message(role=Role.USER, content="hello")],
            model="gpt-5",
            tools=None,
            system=None,
            max_tokens=32,
            temperature=None,
            reasoning=None,
        )
        assert kwargs["instructions"] == DEFAULT_CODEX_INSTRUCTIONS

    def test_build_request_kwargs_preserves_explicit_instructions(self):
        provider = OpenAIOAuthProvider(
            access_token="test-token",
            account_id="acct_123",
        )
        kwargs = provider._build_request_kwargs(
            messages=[Message(role=Role.USER, content="hello")],
            model="gpt-5",
            tools=None,
            system="custom system",
            max_tokens=32,
            temperature=None,
            reasoning=None,
        )
        assert kwargs["instructions"] == "custom system"
