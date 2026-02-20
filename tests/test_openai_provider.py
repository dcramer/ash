"""Tests for OpenAI LLM provider."""

from ash.llm.openai import OpenAIProvider
from ash.llm.types import Message, Role


class TestOpenAIBuildRequestKwargs:
    """Tests for OpenAI provider request building."""

    def setup_method(self):
        self.provider = OpenAIProvider(api_key="test-key")

    def test_reasoning_included_when_set(self):
        """Test that reasoning effort is passed to API kwargs."""
        messages = [Message(role=Role.USER, content="Hello")]
        kwargs = self.provider._build_request_kwargs(
            messages=messages,
            model="gpt-5.2-pro",
            tools=None,
            system=None,
            max_tokens=4096,
            temperature=None,
            reasoning="high",
        )
        assert kwargs["reasoning"] == {"effort": "high"}

    def test_reasoning_not_included_when_none(self):
        """Test that reasoning is omitted when not set."""
        messages = [Message(role=Role.USER, content="Hello")]
        kwargs = self.provider._build_request_kwargs(
            messages=messages,
            model="gpt-5.2",
            tools=None,
            system=None,
            max_tokens=4096,
            temperature=0.7,
        )
        assert "reasoning" not in kwargs

    def test_reasoning_medium(self):
        """Test medium reasoning effort value."""
        messages = [Message(role=Role.USER, content="Hello")]
        kwargs = self.provider._build_request_kwargs(
            messages=messages,
            model="gpt-5.2-pro",
            tools=None,
            system=None,
            max_tokens=4096,
            temperature=None,
            reasoning="medium",
        )
        assert kwargs["reasoning"] == {"effort": "medium"}

    def test_reasoning_low(self):
        """Test low reasoning effort value."""
        messages = [Message(role=Role.USER, content="Hello")]
        kwargs = self.provider._build_request_kwargs(
            messages=messages,
            model="gpt-5.2",
            tools=None,
            system=None,
            max_tokens=4096,
            temperature=None,
            reasoning="low",
        )
        assert kwargs["reasoning"] == {"effort": "low"}
