"""Tests for OpenAI OAuth LLM provider."""

import pytest

from ash.llm.openai_oauth import CODEX_BASE_URL, OpenAIOAuthProvider


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
