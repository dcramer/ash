"""Background memory extraction from conversations.

Extracts memorable facts from conversations using a secondary LLM call,
running asynchronously after each exchange.
"""

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ash.llm.types import Message, Role

if TYPE_CHECKING:
    from ash.llm import LLMProvider

logger = logging.getLogger(__name__)

# Default extraction prompt template
EXTRACTION_PROMPT = """You are a memory extraction system. Analyze this conversation and identify facts worth remembering about the user(s).

## What to extract:
- User preferences (likes, dislikes, habits)
- Facts about people in their life (names, relationships, details)
- Important dates or events
- Explicit requests to remember something
- Corrections to previously known information

## What NOT to extract:
- Actions the assistant took
- Temporary task context ("working on X project")
- Generic conversation flow
- Credentials or sensitive data
- Things already in memory (avoid duplicates)

## CRITICAL: Resolve references
Convert pronouns and references to concrete facts:
- "I liked that restaurant" -> Find which restaurant from context, store "User liked [restaurant name]"
- "She's visiting next week" -> Find who "she" is, store "[Person name] is visiting [date]"
- "Yes, that one" -> Don't extract - too ambiguous

{existing_memories_section}

## Conversation to analyze:
{conversation}

## Output format:
Return a JSON array of facts. Each fact has:
- content: The fact (MUST be standalone, no unresolved pronouns)
- subjects: Names of people this is about (empty array if about user themselves)
- shared: true if this is group/team knowledge, false if personal
- confidence: 0.0-1.0 how confident this should be stored

Only include facts with confidence >= 0.7. If you cannot resolve a reference, do not extract it.

Return ONLY valid JSON, no other text. Example:
[
  {{"content": "User prefers dark mode", "subjects": [], "shared": false, "confidence": 0.9}},
  {{"content": "Sarah's birthday is March 15", "subjects": ["Sarah"], "shared": false, "confidence": 0.85}}
]

If there are no facts worth extracting, return an empty array: []"""


@dataclass
class ExtractedFact:
    """A fact extracted from conversation."""

    content: str
    subjects: list[str]
    shared: bool
    confidence: float


class MemoryExtractor:
    """Extracts memorable facts from conversations using a secondary LLM.

    Designed to run in the background after each agent response,
    using a cheap/fast model to identify facts worth remembering.
    """

    def __init__(
        self,
        llm: "LLMProvider",
        model: str | None = None,
        max_tokens: int = 1024,
        confidence_threshold: float = 0.7,
    ):
        """Initialize memory extractor.

        Args:
            llm: LLM provider for extraction calls.
            model: Model to use (defaults to provider default).
            max_tokens: Maximum tokens for extraction response.
            confidence_threshold: Minimum confidence to include a fact.
        """
        self._llm = llm
        self._model = model
        self._max_tokens = max_tokens
        self._confidence_threshold = confidence_threshold

    async def extract_from_conversation(
        self,
        messages: list[Message],
        existing_memories: list[str] | None = None,
    ) -> list[ExtractedFact]:
        """Analyze conversation and extract facts worth remembering.

        Args:
            messages: Conversation messages to analyze.
            existing_memories: Facts already in memory (to avoid duplicates).

        Returns:
            List of extracted facts with confidence scores.
        """
        if not messages:
            return []

        # Format conversation for the prompt
        conversation_text = self._format_conversation(messages)
        if not conversation_text.strip():
            return []

        # Build existing memories section
        existing_section = ""
        if existing_memories:
            memory_list = "\n".join(f"- {m}" for m in existing_memories[:20])
            existing_section = f"""## Already in memory (don't re-extract):
{memory_list}
"""

        # Build the extraction prompt
        prompt = EXTRACTION_PROMPT.format(
            existing_memories_section=existing_section,
            conversation=conversation_text,
        )

        try:
            response = await self._llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=0.1,  # Low temperature for consistent extraction
            )

            # Parse the response
            return self._parse_extraction_response(response.message.get_text())

        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return []

    def _format_conversation(self, messages: list[Message]) -> str:
        """Format messages into a readable conversation transcript.

        Args:
            messages: Messages to format.

        Returns:
            Formatted conversation text.
        """
        lines = []
        for msg in messages:
            # Skip tool results and system messages
            if msg.role == Role.SYSTEM:
                continue

            text = msg.get_text()
            if not text.strip():
                continue

            # Truncate very long messages
            if len(text) > 2000:
                text = text[:2000] + "..."

            role_label = "User" if msg.role == Role.USER else "Assistant"
            lines.append(f"{role_label}: {text}")

        return "\n\n".join(lines)

    def _parse_extraction_response(self, response_text: str) -> list[ExtractedFact]:
        """Parse the LLM's JSON response into ExtractedFact objects.

        Args:
            response_text: Raw response from LLM.

        Returns:
            List of parsed facts.
        """
        # Try to extract JSON from the response
        text = response_text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            # Find the JSON content between code blocks
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Failed to parse extraction response as JSON: %s", text[:200])
            return []

        if not isinstance(data, list):
            logger.debug("Extraction response is not a list: %s", type(data))
            return []

        facts = []
        for item in data:
            if not isinstance(item, dict):
                continue

            content = item.get("content", "").strip()
            if not content:
                continue

            confidence = float(item.get("confidence", 0.0))
            if confidence < self._confidence_threshold:
                continue

            subjects = item.get("subjects", [])
            if not isinstance(subjects, list):
                subjects = []
            subjects = [str(s) for s in subjects if s]

            shared = bool(item.get("shared", False))

            facts.append(
                ExtractedFact(
                    content=content,
                    subjects=subjects,
                    shared=shared,
                    confidence=confidence,
                )
            )

        return facts
