"""Tests for OpenAI Codex LLM provider."""

import pytest

from ash.llm.openai_codex import CODEX_BASE_URL, OpenAICodexProvider


class TestOpenAICodexProvider:
    def test_name(self):
        provider = OpenAICodexProvider(
            access_token="test-token",
            account_id="acct_123",
        )
        assert provider.name == "openai-codex"

    def test_client_configuration(self):
        provider = OpenAICodexProvider(
            access_token="test-token",
            account_id="acct_123",
        )
        assert str(provider._client.base_url).rstrip("/") == CODEX_BASE_URL
        assert provider._client.api_key == "test-token"

    def test_default_headers(self):
        provider = OpenAICodexProvider(
            access_token="test-token",
            account_id="acct_123",
        )
        headers = provider._client._custom_headers
        assert headers["chatgpt-account-id"] == "acct_123"
        assert headers["OpenAI-Beta"] == "responses=experimental"
        assert headers["originator"] == "ash"

    async def test_embed_raises(self):
        provider = OpenAICodexProvider(
            access_token="test-token",
            account_id="acct_123",
        )
        with pytest.raises(NotImplementedError, match="not supported"):
            await provider.embed(["test"])

    def test_inherits_openai_provider(self):
        from ash.llm.openai import OpenAIProvider

        assert issubclass(OpenAICodexProvider, OpenAIProvider)
