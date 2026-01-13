"""Tests for memory extraction from conversations."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.llm.types import CompletionResponse, Message, Role, TextContent, Usage
from ash.memory import ExtractedFact, MemoryExtractor


class TestMemoryExtractor:
    """Tests for MemoryExtractor."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        return MagicMock()

    @pytest.fixture
    def extractor(self, mock_llm):
        """Create a MemoryExtractor with mocked LLM."""
        return MemoryExtractor(
            llm=mock_llm,
            model="test-model",
            confidence_threshold=0.7,
        )

    def test_format_conversation_basic(self, extractor):
        """Test formatting a simple conversation."""
        messages = [
            Message(role=Role.USER, content="Hello, I'm David"),
            Message(role=Role.ASSISTANT, content="Hi David, nice to meet you!"),
        ]

        result = extractor._format_conversation(messages)

        assert "User: Hello, I'm David" in result
        assert "Assistant: Hi David, nice to meet you!" in result

    def test_format_conversation_skips_system(self, extractor):
        """Test that system messages are skipped."""
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant"),
            Message(role=Role.USER, content="Hello"),
        ]

        result = extractor._format_conversation(messages)

        assert "You are a helpful assistant" not in result
        assert "User: Hello" in result

    def test_format_conversation_truncates_long_messages(self, extractor):
        """Test that very long messages are truncated."""
        long_content = "x" * 3000
        messages = [
            Message(role=Role.USER, content=long_content),
        ]

        result = extractor._format_conversation(messages)

        assert len(result) < 3000
        assert "..." in result

    def test_format_conversation_handles_content_blocks(self, extractor):
        """Test formatting messages with content blocks."""
        messages = [
            Message(
                role=Role.ASSISTANT,
                content=[TextContent(text="Hello from assistant")],
            ),
        ]

        result = extractor._format_conversation(messages)

        assert "Assistant: Hello from assistant" in result

    def test_parse_extraction_response_valid_json(self, extractor):
        """Test parsing a valid JSON response."""
        response = """[
            {"content": "User prefers dark mode", "subjects": [], "shared": false, "confidence": 0.9},
            {"content": "Sarah is user's wife", "subjects": ["Sarah"], "shared": false, "confidence": 0.85}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 2
        assert facts[0].content == "User prefers dark mode"
        assert facts[0].confidence == 0.9
        assert facts[0].subjects == []
        assert facts[1].content == "Sarah is user's wife"
        assert facts[1].subjects == ["Sarah"]

    def test_parse_extraction_response_filters_low_confidence(self, extractor):
        """Test that low confidence facts are filtered out."""
        response = """[
            {"content": "High confidence fact", "subjects": [], "shared": false, "confidence": 0.9},
            {"content": "Low confidence fact", "subjects": [], "shared": false, "confidence": 0.5}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].content == "High confidence fact"

    def test_parse_extraction_response_handles_code_block(self, extractor):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = """```json
[
    {"content": "User likes Python", "subjects": [], "shared": false, "confidence": 0.8}
]
```"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].content == "User likes Python"

    def test_parse_extraction_response_empty_array(self, extractor):
        """Test parsing an empty array response."""
        response = "[]"

        facts = extractor._parse_extraction_response(response)

        assert facts == []

    def test_parse_extraction_response_invalid_json(self, extractor):
        """Test handling invalid JSON gracefully."""
        response = "This is not valid JSON"

        facts = extractor._parse_extraction_response(response)

        assert facts == []

    def test_parse_extraction_response_missing_content(self, extractor):
        """Test that items without content are skipped."""
        response = """[
            {"subjects": [], "shared": false, "confidence": 0.9},
            {"content": "Valid fact", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].content == "Valid fact"

    async def test_extract_from_conversation_calls_llm(self, extractor, mock_llm):
        """Test that extraction calls the LLM correctly."""
        # Mock the LLM response
        mock_llm.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content='[{"content": "User is David", "subjects": [], "shared": false, "confidence": 0.9}]',
                ),
                usage=Usage(input_tokens=100, output_tokens=50),
            )
        )

        messages = [
            Message(role=Role.USER, content="My name is David"),
            Message(role=Role.ASSISTANT, content="Nice to meet you, David!"),
        ]

        facts = await extractor.extract_from_conversation(messages)

        assert len(facts) == 1
        assert facts[0].content == "User is David"
        mock_llm.complete.assert_called_once()

    async def test_extract_from_conversation_includes_existing_memories(
        self, extractor, mock_llm
    ):
        """Test that existing memories are included in the prompt."""
        mock_llm.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(role=Role.ASSISTANT, content="[]"),
                usage=Usage(input_tokens=100, output_tokens=10),
            )
        )

        messages = [Message(role=Role.USER, content="I still like dark mode")]
        existing = ["User prefers dark mode"]

        await extractor.extract_from_conversation(messages, existing_memories=existing)

        # Check that existing memories were in the prompt
        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert "User prefers dark mode" in prompt

    async def test_extract_from_conversation_empty_messages(self, extractor, mock_llm):
        """Test extraction with empty message list."""
        facts = await extractor.extract_from_conversation([])

        assert facts == []
        mock_llm.complete.assert_not_called()

    async def test_extract_from_conversation_handles_llm_error(
        self, extractor, mock_llm
    ):
        """Test graceful handling of LLM errors."""
        mock_llm.complete = AsyncMock(side_effect=Exception("API Error"))

        messages = [Message(role=Role.USER, content="Hello")]

        facts = await extractor.extract_from_conversation(messages)

        assert facts == []


class TestExtractedFact:
    """Tests for ExtractedFact dataclass."""

    def test_extracted_fact_creation(self):
        """Test creating an ExtractedFact."""
        fact = ExtractedFact(
            content="User likes coffee",
            subjects=[],
            shared=False,
            confidence=0.85,
        )

        assert fact.content == "User likes coffee"
        assert fact.subjects == []
        assert fact.shared is False
        assert fact.confidence == 0.85

    def test_extracted_fact_with_subjects(self):
        """Test ExtractedFact with subjects."""
        fact = ExtractedFact(
            content="Sarah likes tea",
            subjects=["Sarah"],
            shared=False,
            confidence=0.9,
        )

        assert fact.subjects == ["Sarah"]

    def test_extracted_fact_shared(self):
        """Test ExtractedFact with shared flag."""
        fact = ExtractedFact(
            content="Team meeting is at 10am",
            subjects=[],
            shared=True,
            confidence=0.95,
        )

        assert fact.shared is True
