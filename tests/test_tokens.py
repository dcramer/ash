"""Tests for token estimation utilities."""

import pytest

from ash.core.tokens import estimate_message_tokens, estimate_tokens


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        # "Hi" is 2 chars, should be at least 1 token
        result = estimate_tokens("Hi")
        assert result >= 1

    def test_typical_message(self):
        # ~100 chars should be ~25 tokens
        text = "Hello, how are you doing today? I hope everything is going well!"
        result = estimate_tokens(text)
        # Should be roughly len(text) / 4 + 1
        expected = len(text) // 4 + 1
        assert result == expected

    def test_long_text(self):
        # Longer text should scale linearly
        text = "a" * 1000
        result = estimate_tokens(text)
        # Should be around 250 tokens
        assert 200 < result < 300


class TestEstimateMessageTokens:
    """Tests for estimate_message_tokens function."""

    def test_simple_text_content(self):
        result = estimate_message_tokens("user", "Hello, world!")
        # Overhead + text estimate
        assert result > estimate_tokens("Hello, world!")

    def test_empty_content(self):
        result = estimate_message_tokens("user", "")
        # Should just be overhead
        assert result == 4  # Base overhead

    def test_content_blocks_text(self):
        blocks = [{"type": "text", "text": "Hello there!"}]
        result = estimate_message_tokens("assistant", blocks)
        assert result > 0

    def test_content_blocks_tool_use(self):
        blocks = [
            {
                "type": "tool_use",
                "name": "bash",
                "input": {"command": "ls -la"},
            }
        ]
        result = estimate_message_tokens("assistant", blocks)
        # Should include name + JSON serialized input
        assert result > 10

    def test_content_blocks_tool_result(self):
        blocks = [
            {
                "type": "tool_result",
                "content": "file1.txt\nfile2.txt\nfile3.txt",
            }
        ]
        result = estimate_message_tokens("user", blocks)
        assert result > 0

    def test_mixed_content_blocks(self):
        blocks = [
            {"type": "text", "text": "Let me run that command."},
            {"type": "tool_use", "name": "bash", "input": {"command": "pwd"}},
        ]
        result = estimate_message_tokens("assistant", blocks)
        # Should be sum of both
        assert result > estimate_tokens("Let me run that command.")

    def test_dataclass_content_blocks(self):
        from ash.llm.types import TextContent, ToolResult, ToolUse

        blocks = [
            TextContent(text="Here's the result:"),
            ToolUse(id="t1", name="bash", input={"cmd": "ls"}),
        ]
        result = estimate_message_tokens("assistant", blocks)
        assert result > 0

    def test_tool_result_dataclass(self):
        from ash.llm.types import ToolResult

        blocks = [ToolResult(tool_use_id="t1", content="Success!", is_error=False)]
        result = estimate_message_tokens("user", blocks)
        assert result > 0
