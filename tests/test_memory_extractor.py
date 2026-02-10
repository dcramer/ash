"""Tests for memory extraction from conversations.

Tests focus on:
- JSON parsing edge cases (real parsing logic)
- Error handling (LLM failures)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.llm.types import CompletionResponse, Message, Role, Usage
from ash.memory import MemoryExtractor
from ash.memory.extractor import SpeakerInfo


class TestSpeakerInfo:
    """Tests for SpeakerInfo class."""

    def test_format_label_with_username_and_display_name(self):
        """Test format_label with both username and display name."""
        speaker = SpeakerInfo(username="david", display_name="David Cramer")
        assert speaker.format_label() == "@david (David Cramer)"

    def test_format_label_with_username_only(self):
        """Test format_label with just username."""
        speaker = SpeakerInfo(username="david")
        assert speaker.format_label() == "@david"

    def test_format_label_with_display_name_only(self):
        """Test format_label with just display name."""
        speaker = SpeakerInfo(display_name="David Cramer")
        assert speaker.format_label() == "David Cramer"

    def test_format_label_empty(self):
        """Test format_label with no info."""
        speaker = SpeakerInfo()
        assert speaker.format_label() == "User"

    def test_get_identifier_prefers_username(self):
        """Test get_identifier returns username over user_id."""
        speaker = SpeakerInfo(user_id="12345", username="david")
        assert speaker.get_identifier() == "david"

    def test_get_identifier_falls_back_to_user_id(self):
        """Test get_identifier falls back to user_id."""
        speaker = SpeakerInfo(user_id="12345")
        assert speaker.get_identifier() == "12345"


class TestExtractionParsing:
    """Tests for extraction response parsing."""

    @pytest.fixture
    def extractor(self):
        """Create a MemoryExtractor with mocked LLM."""
        return MemoryExtractor(
            llm=MagicMock(),
            model="test-model",
            confidence_threshold=0.7,
        )

    def test_parses_valid_json(self, extractor):
        """Test parsing a valid JSON response."""
        response = """[
            {"content": "User prefers dark mode", "subjects": [], "shared": false, "confidence": 0.9},
            {"content": "Sarah is user's wife", "subjects": ["Sarah"], "shared": false, "confidence": 0.85}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 2
        assert facts[0].content == "User prefers dark mode"
        assert facts[0].confidence == 0.9
        assert facts[1].subjects == ["Sarah"]

    def test_parses_speaker_field(self, extractor):
        """Test parsing the speaker field for multi-user attribution."""
        response = """[
            {"content": "Likes pizza", "speaker": "david", "subjects": [], "shared": false, "confidence": 0.9},
            {"content": "Bob likes pasta", "speaker": "@bob", "subjects": ["Bob"], "shared": false, "confidence": 0.85}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 2
        assert facts[0].speaker == "david"
        assert facts[1].speaker == "bob"  # @ prefix removed

    def test_speaker_field_can_be_null(self, extractor):
        """Test that speaker field can be null/missing."""
        response = """[
            {"content": "Some fact", "speaker": null, "subjects": [], "shared": false, "confidence": 0.9},
            {"content": "Another fact", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 2
        assert facts[0].speaker is None
        assert facts[1].speaker is None

    def test_filters_low_confidence(self, extractor):
        """Test that low confidence facts are filtered out."""
        response = """[
            {"content": "High confidence", "subjects": [], "shared": false, "confidence": 0.9},
            {"content": "Low confidence", "subjects": [], "shared": false, "confidence": 0.5}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].content == "High confidence"

    def test_handles_markdown_code_block(self, extractor):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = """```json
[
    {"content": "User likes Python", "subjects": [], "shared": false, "confidence": 0.8}
]
```"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].content == "User likes Python"

    def test_handles_empty_array(self, extractor):
        """Test parsing an empty array response."""
        facts = extractor._parse_extraction_response("[]")
        assert facts == []

    def test_handles_invalid_json(self, extractor):
        """Test graceful handling of invalid JSON."""
        facts = extractor._parse_extraction_response("This is not valid JSON")
        assert facts == []

    def test_skips_items_without_content(self, extractor):
        """Test that items without content are skipped."""
        response = """[
            {"subjects": [], "shared": false, "confidence": 0.9},
            {"content": "Valid fact", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].content == "Valid fact"


class TestExtractionErrors:
    """Tests for extraction error handling."""

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

    async def test_returns_empty_for_empty_messages(self, extractor, mock_llm):
        """Test extraction with empty message list."""
        facts = await extractor.extract_from_conversation([])

        assert facts == []
        mock_llm.complete.assert_not_called()

    async def test_handles_llm_error_gracefully(self, extractor, mock_llm):
        """Test graceful handling of LLM errors."""
        mock_llm.complete = AsyncMock(side_effect=Exception("API Error"))

        facts = await extractor.extract_from_conversation(
            [Message(role=Role.USER, content="Hello")]
        )

        assert facts == []

    async def test_extracts_facts_successfully(self, extractor, mock_llm):
        """Test successful fact extraction."""
        mock_llm.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content='[{"content": "User is David", "subjects": [], "shared": false, "confidence": 0.9}]',
                ),
                usage=Usage(input_tokens=100, output_tokens=50),
            )
        )

        facts = await extractor.extract_from_conversation(
            [
                Message(role=Role.USER, content="My name is David"),
                Message(role=Role.ASSISTANT, content="Nice to meet you, David!"),
            ]
        )

        assert len(facts) == 1
        assert facts[0].content == "User is David"
