"""Tests for memory extraction from conversations.

Tests focus on:
- JSON parsing edge cases (real parsing logic)
- Error handling (LLM failures)
"""

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.llm.types import CompletionResponse, Message, Role, Usage
from ash.memory.extractor import MemoryExtractor, SpeakerInfo
from ash.store.types import MemoryType


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


class TestSensitivityParsing:
    """Tests for sensitivity parsing in extraction."""

    @pytest.fixture
    def extractor(self):
        """Create a MemoryExtractor with mocked LLM."""
        return MemoryExtractor(
            llm=MagicMock(),
            model="test-model",
            confidence_threshold=0.7,
        )

    def test_parses_public_sensitivity(self, extractor):
        """Test parsing public sensitivity."""
        from ash.store.types import Sensitivity

        response = """[
            {"content": "Likes pizza", "subjects": [], "shared": false, "confidence": 0.9, "sensitivity": "public"}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].sensitivity == Sensitivity.PUBLIC

    def test_parses_personal_sensitivity(self, extractor):
        """Test parsing personal sensitivity."""
        from ash.store.types import Sensitivity

        response = """[
            {"content": "Looking for new job", "subjects": [], "shared": false, "confidence": 0.9, "sensitivity": "personal"}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].sensitivity == Sensitivity.PERSONAL

    def test_parses_sensitive_sensitivity(self, extractor):
        """Test parsing sensitive sensitivity."""
        from ash.store.types import Sensitivity

        response = """[
            {"content": "Has anxiety disorder", "subjects": [], "shared": false, "confidence": 0.9, "sensitivity": "sensitive"}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].sensitivity == Sensitivity.SENSITIVE

    def test_missing_sensitivity_defaults_to_none(self, extractor):
        """Test that missing sensitivity defaults to None (treated as public)."""
        response = """[
            {"content": "Works at Acme", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].sensitivity is None

    def test_invalid_sensitivity_defaults_to_none(self, extractor):
        """Test that invalid sensitivity values default to None."""
        response = """[
            {"content": "Some fact", "subjects": [], "shared": false, "confidence": 0.9, "sensitivity": "invalid_value"}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].sensitivity is None


class TestSecretsFiltering:
    """Tests for secrets filtering in extraction."""

    @pytest.fixture
    def extractor(self):
        """Create a MemoryExtractor with mocked LLM."""
        return MemoryExtractor(
            llm=MagicMock(),
            model="test-model",
            confidence_threshold=0.7,
        )

    def test_filters_openai_api_key(self, extractor):
        """Test that OpenAI API keys are filtered."""
        response = """[
            {"content": "My API key is sk-abc123def456ghi789abcdef", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 0

    def test_filters_github_token(self, extractor):
        """Test that GitHub tokens are filtered."""
        response = """[
            {"content": "Use this token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 0

    def test_filters_aws_access_key(self, extractor):
        """Test that AWS access keys are filtered."""
        response = """[
            {"content": "AWS key: AKIAIOSFODNN7EXAMPLE", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 0

    def test_filters_credit_card_number(self, extractor):
        """Test that credit card numbers are filtered."""
        response = """[
            {"content": "Card number: 4111-1111-1111-1111", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 0

    def test_filters_ssn(self, extractor):
        """Test that Social Security Numbers are filtered."""
        response = """[
            {"content": "SSN is 123-45-6789", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 0

    def test_filters_password_pattern(self, extractor):
        """Test that password patterns are filtered."""
        response = """[
            {"content": "Password is hunter2", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 0

    def test_filters_password_colon_format(self, extractor):
        """Test that password: format is filtered."""
        response = """[
            {"content": "Remember my pwd: secretpass123", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 0

    def test_filters_private_key(self, extractor):
        """Test that private keys are filtered."""
        response = """[
            {"content": "-----BEGIN PRIVATE KEY-----\\nMIIEvg...", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 0

    def test_allows_safe_content(self, extractor):
        """Test that normal content is not filtered."""
        response = """[
            {"content": "User prefers dark mode", "subjects": [], "shared": false, "confidence": 0.9},
            {"content": "User's favorite number is 1111", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 2
        assert facts[0].content == "User prefers dark mode"
        assert facts[1].content == "User's favorite number is 1111"


class TestTemporalContext:
    """Tests for temporal context in extraction."""

    @pytest.fixture
    def extractor(self):
        """Create a MemoryExtractor with mocked LLM."""
        return MemoryExtractor(
            llm=MagicMock(),
            model="test-model",
            confidence_threshold=0.7,
        )

    async def test_datetime_included_in_prompt(self, extractor):
        """Test that datetime is included in extraction prompt when provided."""
        from datetime import datetime

        mock_llm = AsyncMock(
            return_value=CompletionResponse(
                message=Message(role=Role.ASSISTANT, content="[]"),
                usage=Usage(input_tokens=100, output_tokens=10),
            )
        )
        extractor._llm.complete = mock_llm

        test_datetime = datetime(2026, 2, 15, 14, 30, 0, tzinfo=UTC)

        await extractor.extract_from_conversation(
            messages=[Message(role=Role.USER, content="I have tasks for this weekend")],
            current_datetime=test_datetime,
        )

        # Check that the prompt includes the datetime
        call_args = mock_llm.call_args
        # Access keyword args - messages is passed as a keyword argument
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        prompt = messages[0].content
        assert "February 15, 2026" in prompt
        assert "Sunday" in prompt
        assert "Convert ALL relative time references to absolute dates" in prompt


class TestCoherenceFiltering:
    """Tests for coherence filtering in extraction."""

    @pytest.fixture
    def extractor(self):
        """Create a MemoryExtractor with mocked LLM."""
        return MemoryExtractor(
            llm=MagicMock(),
            model="test-model",
            confidence_threshold=0.7,
        )

    async def test_coherence_guidance_in_prompt(self, extractor):
        """Test that coherence guidance is included in extraction prompt."""
        mock_llm = AsyncMock(
            return_value=CompletionResponse(
                message=Message(role=Role.ASSISTANT, content="[]"),
                usage=Usage(input_tokens=100, output_tokens=10),
            )
        )
        extractor._llm.complete = mock_llm

        await extractor.extract_from_conversation(
            messages=[Message(role=Role.USER, content="I spent $100 on something")],
        )

        # Check that the prompt includes coherence guidance
        call_args = mock_llm.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        prompt = messages[0].content
        assert "CRITICAL: Require coherence" in prompt
        assert "something" in prompt
        assert "Spent money on something" in prompt
        assert "If you cannot identify WHAT, WHO, or WHERE specifically" in prompt

    def test_vague_fact_filtered_via_low_confidence(self, extractor):
        """Test that vague facts with low confidence are filtered out.

        When the LLM follows our coherence guidance, it should return
        confidence 0.0 for vague facts like "Spent money on something".
        """
        # Simulate LLM returning low confidence for a vague fact
        response = """[
            {"content": "Spent money on something", "subjects": [], "shared": false, "confidence": 0.0},
            {"content": "Owns a Grand Seiko watch", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        # Only the concrete fact should be extracted
        assert len(facts) == 1
        assert facts[0].content == "Owns a Grand Seiko watch"

    def test_concrete_fact_extracted(self, extractor):
        """Test that concrete, specific facts are properly extracted."""
        response = """[
            {"content": "Spent $100 on a Grand Seiko watch", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].content == "Spent $100 on a Grand Seiko watch"


class TestExtractionPromptContent:
    """Tests verifying extraction prompt contains key guidance."""

    @pytest.fixture
    def prompt(self):
        """Load the extraction prompt (lowercased for case-insensitive checks)."""
        from ash.memory.extractor import EXTRACTION_PROMPT

        return EXTRACTION_PROMPT.lower()

    def test_prompt_rejects_negative_knowledge(self, prompt):
        """Extraction prompt should reject negative knowledge."""
        assert "negative knowledge" in prompt
        assert "blood type is unknown" in prompt
        assert "only store what is known" in prompt

    def test_prompt_rejects_meta_knowledge(self, prompt):
        """Extraction prompt should reject meta-knowledge about the system."""
        assert "meta-knowledge" in prompt
        assert "memory system" in prompt

    def test_prompt_rejects_vague_relationships(self, prompt):
        """Extraction prompt should reject vague relationships."""
        assert "knows someone named" in prompt

    def test_prompt_rejects_actions_without_specifics(self, prompt):
        """Extraction prompt should reject actions without specifics."""
        assert "just arrived at a location" in prompt
        assert "fixed some issues" in prompt

    def test_prompt_includes_long_term_utility_gate(self, prompt):
        """Extraction prompt should enforce long-term utility over chat noise."""
        assert "primary objective: long-term utility" in prompt
        assert "30+ days later" in prompt
        assert "personalization, planning, or relationship context" in prompt

    def test_prompt_rejects_system_operational_noise(self, prompt):
        """Extraction prompt should explicitly reject system/dev operational details."""
        assert "refactored the bot" in prompt
        assert "wiped session history" in prompt
        assert "assistant/harness/eval internals" in prompt

    def test_prompt_enforces_ephemeral_quality_gate(self, prompt):
        """Extraction prompt should gate low-value ephemeral facts."""
        assert "ephemeral quality gate" in prompt
        assert "trivial, stale, or purely situational status" in prompt
        assert '"going in may"' in prompt


class TestClassifyFact:
    """Tests for classify_fact() method."""

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

    async def test_classify_returns_extracted_fact(self, extractor, mock_llm):
        """Test that classify_fact returns a properly parsed ExtractedFact."""
        from ash.store.types import Sensitivity

        mock_llm.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content='{"subjects": ["Sarah"], "type": "relationship", "sensitivity": "public", "portable": true, "shared": false}',
                ),
                usage=Usage(input_tokens=50, output_tokens=30),
            )
        )

        result = await extractor.classify_fact("Sarah is my sister")

        assert result is not None
        assert result.content == "Sarah is my sister"
        assert result.subjects == ["Sarah"]
        assert result.memory_type == MemoryType.RELATIONSHIP
        assert result.confidence == 1.0
        assert result.sensitivity == Sensitivity.PUBLIC
        assert result.portable is True

    async def test_classify_preserves_original_content(self, extractor, mock_llm):
        """Test that the original content is preserved, not the LLM output."""
        mock_llm.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content='{"subjects": [], "type": "preference", "sensitivity": "public", "portable": true, "shared": false}',
                ),
                usage=Usage(input_tokens=50, output_tokens=30),
            )
        )

        original = "I prefer dark mode in all apps"
        result = await extractor.classify_fact(original)

        assert result is not None
        assert result.content == original

    async def test_classify_returns_none_on_failure(self, extractor, mock_llm):
        """Test that classify_fact returns None on LLM failure."""
        mock_llm.complete = AsyncMock(side_effect=Exception("API Error"))

        result = await extractor.classify_fact("Some fact")

        assert result is None

    async def test_classify_returns_none_on_invalid_json(self, extractor, mock_llm):
        """Test that classify_fact returns None when LLM returns invalid JSON."""
        mock_llm.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content="I cannot classify this fact.",
                ),
                usage=Usage(input_tokens=50, output_tokens=30),
            )
        )

        result = await extractor.classify_fact("Some fact")

        assert result is None

    async def test_classify_parses_aliases(self, extractor, mock_llm):
        """Test that classify_fact populates aliases from LLM output."""
        mock_llm.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content='{"subjects": ["Sukhpreet"], "type": "identity", "sensitivity": "public", "portable": true, "shared": false, "aliases": {"Sukhpreet": ["SK"]}}',
                ),
                usage=Usage(input_tokens=50, output_tokens=30),
            )
        )

        result = await extractor.classify_fact("Sukhpreet goes by SK")

        assert result is not None
        assert result.aliases == {"Sukhpreet": ["SK"]}


class TestGroundingPass:
    """Tests for optional second-pass grounding/rewriting."""

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    @pytest.fixture
    def extractor(self, mock_llm):
        return MemoryExtractor(
            llm=mock_llm,
            model="test-model",
            confidence_threshold=0.7,
            grounding_enabled=True,
        )

    async def test_grounding_rewrites_incomplete_fact(self, extractor, mock_llm):
        mock_llm.complete = AsyncMock(
            side_effect=[
                CompletionResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        content='[{"content":"Randolf is going in May","subjects":["Randolf"],"shared":false,"confidence":0.9}]',
                    ),
                    usage=Usage(input_tokens=100, output_tokens=40),
                ),
                CompletionResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        content='[{"index":0,"grounded":true,"content":"Randolf is going to Tokyo in May"}]',
                    ),
                    usage=Usage(input_tokens=120, output_tokens=25),
                ),
            ]
        )

        facts = await extractor.extract_from_conversation(
            [
                Message(role=Role.USER, content="Randolf is going to Tokyo in May"),
                Message(role=Role.USER, content="he's still going in May"),
            ]
        )

        assert len(facts) == 1
        assert facts[0].content == "Randolf is going to Tokyo in May"

    async def test_grounding_can_drop_ungrounded_fact(self, extractor, mock_llm):
        mock_llm.complete = AsyncMock(
            side_effect=[
                CompletionResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        content='[{"content":"Spent money on something","subjects":[],"shared":false,"confidence":0.9}]',
                    ),
                    usage=Usage(input_tokens=80, output_tokens=25),
                ),
                CompletionResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        content='[{"index":0,"grounded":false}]',
                    ),
                    usage=Usage(input_tokens=100, output_tokens=15),
                ),
            ]
        )

        facts = await extractor.extract_from_conversation(
            [
                Message(role=Role.USER, content="I spent money on something"),
                Message(role=Role.ASSISTANT, content="what did you buy?"),
            ]
        )

        assert facts == []

    async def test_grounding_partial_decisions_keep_undecided_facts(
        self, extractor, mock_llm
    ):
        mock_llm.complete = AsyncMock(
            side_effect=[
                CompletionResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        content='[{"content":"Randolf is going in May","subjects":["Randolf"],"shared":false,"confidence":0.9},{"content":"User prefers dark mode","subjects":[],"shared":false,"confidence":0.9}]',
                    ),
                    usage=Usage(input_tokens=140, output_tokens=50),
                ),
                CompletionResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        content='[{"index":0,"grounded":true,"content":"Randolf is going to Tokyo in May"}]',
                    ),
                    usage=Usage(input_tokens=110, output_tokens=20),
                ),
            ]
        )

        facts = await extractor.extract_from_conversation(
            [
                Message(role=Role.USER, content="Randolf is going to Tokyo in May"),
                Message(role=Role.USER, content="I prefer dark mode"),
            ]
        )

        assert len(facts) == 2
        assert facts[0].content == "Randolf is going to Tokyo in May"
        assert facts[1].content == "User prefers dark mode"

    async def test_grounding_duplicate_decisions_last_one_wins(
        self, extractor, mock_llm
    ):
        mock_llm.complete = AsyncMock(
            side_effect=[
                CompletionResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        content='[{"content":"Randolf is going in May","subjects":["Randolf"],"shared":false,"confidence":0.9}]',
                    ),
                    usage=Usage(input_tokens=90, output_tokens=35),
                ),
                CompletionResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        content='[{"index":0,"grounded":false},{"index":0,"grounded":true,"content":"Randolf is going to Tokyo in May"}]',
                    ),
                    usage=Usage(input_tokens=90, output_tokens=25),
                ),
            ]
        )

        facts = await extractor.extract_from_conversation(
            [
                Message(role=Role.USER, content="Randolf is going to Tokyo in May"),
                Message(role=Role.USER, content="he's still going in May"),
            ]
        )

        assert len(facts) == 1
        assert facts[0].content == "Randolf is going to Tokyo in May"

    async def test_grounding_can_use_dedicated_provider_and_model(self):
        extraction_llm = MagicMock()
        grounding_llm = MagicMock()
        extractor = MemoryExtractor(
            llm=extraction_llm,
            model="extract-model",
            confidence_threshold=0.7,
            grounding_enabled=True,
            grounding_llm=grounding_llm,
            grounding_model="ground-model",
        )

        extraction_llm.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content='[{"content":"Randolf is going in May","subjects":["Randolf"],"shared":false,"confidence":0.9}]',
                ),
                usage=Usage(input_tokens=100, output_tokens=40),
            )
        )
        grounding_llm.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content='[{"index":0,"grounded":true,"content":"Randolf is going to Tokyo in May"}]',
                ),
                usage=Usage(input_tokens=120, output_tokens=25),
            )
        )

        facts = await extractor.extract_from_conversation(
            [
                Message(role=Role.USER, content="Randolf is going to Tokyo in May"),
                Message(role=Role.USER, content="he's still going in May"),
            ]
        )

        assert len(facts) == 1
        assert facts[0].content == "Randolf is going to Tokyo in May"
        assert extraction_llm.complete.await_count == 1
        assert grounding_llm.complete.await_count == 1
        assert extraction_llm.complete.call_args.kwargs["model"] == "extract-model"
        assert grounding_llm.complete.call_args.kwargs["model"] == "ground-model"


class TestGroundingPromptContent:
    """Tests verifying grounding prompt contains key guardrails."""

    @pytest.fixture
    def prompt(self):
        from ash.memory.extractor import GROUNDING_PROMPT

        return GROUNDING_PROMPT.lower()

    def test_prompt_requires_decision_for_every_index(self, prompt):
        assert "process every input index exactly once" in prompt

    def test_prompt_rejects_missing_slots(self, prompt):
        assert "key slots are missing" in prompt
        assert '"user is going in may"' in prompt


class TestAliasParsing:
    """Tests for alias parsing in extraction."""

    @pytest.fixture
    def extractor(self):
        """Create a MemoryExtractor with mocked LLM."""
        return MemoryExtractor(
            llm=MagicMock(),
            model="test-model",
            confidence_threshold=0.7,
        )

    def test_parses_aliases_field(self, extractor):
        """Test that _parse_fact_item parses aliases dict."""
        response = """[
            {"content": "Sukhpreet goes by SK", "subjects": ["Sukhpreet"], "shared": false, "confidence": 0.9, "aliases": {"Sukhpreet": ["SK"]}}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].aliases == {"Sukhpreet": ["SK"]}

    def test_aliases_defaults_to_empty_dict(self, extractor):
        """Test that missing aliases field defaults to empty dict."""
        response = """[
            {"content": "Some fact", "subjects": [], "shared": false, "confidence": 0.9}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].aliases == {}

    def test_aliases_handles_invalid_types(self, extractor):
        """Test that non-dict aliases values produce empty dict."""
        response = """[
            {"content": "Some fact", "subjects": [], "shared": false, "confidence": 0.9, "aliases": "not a dict"}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].aliases == {}

    def test_aliases_multiple_values(self, extractor):
        """Test multiple aliases for one subject."""
        response = """[
            {"content": "Bob is also known as Bobby or Robert", "subjects": ["Bob"], "shared": false, "confidence": 0.9, "aliases": {"Bob": ["Bobby", "Robert"]}}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].aliases == {"Bob": ["Bobby", "Robert"]}

    def test_aliases_strips_whitespace(self, extractor):
        """Test that alias names and values are stripped."""
        response = """[
            {"content": "Bob goes by Bobby", "subjects": ["Bob"], "shared": false, "confidence": 0.9, "aliases": {" Bob ": [" Bobby "]}}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].aliases == {"Bob": ["Bobby"]}

    def test_aliases_skips_empty_values(self, extractor):
        """Test that empty alias values are filtered out."""
        response = """[
            {"content": "Bob goes by Bobby", "subjects": ["Bob"], "shared": false, "confidence": 0.9, "aliases": {"Bob": ["Bobby", "", "  "]}}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].aliases == {"Bob": ["Bobby"]}

    def test_aliases_skips_invalid_inner_types(self, extractor):
        """Test that non-list alias values are skipped."""
        response = """[
            {"content": "Bob goes by Bobby", "subjects": ["Bob"], "shared": false, "confidence": 0.9, "aliases": {"Bob": "Bobby"}}
        ]"""

        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].aliases == {}
