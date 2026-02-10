"""Background memory extraction from conversations.

Extracts memorable facts from conversations using a secondary LLM call,
running asynchronously after each exchange.
"""

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ash.llm.types import Message, Role
from ash.memory.types import ExtractedFact, MemoryType

if TYPE_CHECKING:
    from ash.llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class SpeakerInfo:
    """Information about a message speaker for attribution."""

    user_id: str | None = None
    username: str | None = None
    display_name: str | None = None

    def format_label(self) -> str:
        """Format speaker label for conversation transcript.

        Returns a label like "@david (David Cramer)" or just "User" if no info.
        """
        if self.username and self.display_name:
            return f"@{self.username} ({self.display_name})"
        elif self.username:
            return f"@{self.username}"
        elif self.display_name:
            return self.display_name
        return "User"

    def get_identifier(self) -> str | None:
        """Get the best identifier for this speaker (username preferred)."""
        return self.username or self.user_id


# Default extraction prompt template
EXTRACTION_PROMPT = """You are a memory extraction system. Analyze this conversation and identify facts worth remembering about the user(s).
{owner_section}
## What to extract:
- User preferences (likes, dislikes, habits)
- Facts about people in their life (names, relationships, details)
- Important dates or events
- Explicit requests to remember something
- Corrections to previously known information

## What NOT to extract:
- Facts stated BY the assistant (only extract facts from user messages)
- The assistant's summaries, clarifications, or restatements of user facts
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

## CRITICAL: Ownership and perspective
Facts should be attributed to who they're ABOUT, not who performed an action:
- "My wife got me a watch" -> The SPEAKER owns the watch (subjects: []), wife gave it (mentioned in relationship context)
- "I bought Sarah a gift" -> SARAH received the gift - if relevant, fact is about Sarah (subjects: ["Sarah"])
- "The watch my wife gave me" -> SPEAKER owns the watch, wife is the giver
- Be precise: "gave", "got for", "bought for" means the OTHER person receives/owns it

## Speaker Attribution
Each message in the conversation shows who said it. Track WHO provided each fact:
- If @david says "I like pizza" -> speaker is "david"
- If @bob says "David likes pasta" -> speaker is "bob", subjects is ["David"]
- The speaker field should contain the username (without @) of who stated the fact

## Subjects field
The subjects array should contain people the fact is PRIMARILY ABOUT:
- "I own a Grand Seiko" -> subjects: [] (about the speaker)
- "My wife gave me a watch" -> subjects: [] (fact is about speaker owning watch; wife is just context)
- "The watch my wife got me" -> subjects: [] (speaker OWNS the watch, wife GAVE it)
- "My wife loves watches" -> subjects: ["wife's name"] (fact is about the wife)
- "Sarah and I went to dinner" -> subjects: ["Sarah"] if fact is about Sarah, [] if about speaker's experience

WRONG: "My wife got me a Grand Seiko" -> extracting that wife owns a Grand Seiko (she GAVE it, speaker owns it)
RIGHT: "My wife got me a Grand Seiko" -> speaker owns Grand Seiko, subjects: []

{existing_memories_section}

## Conversation to analyze:
{conversation}

## Output format:
Return a JSON array of facts. Each fact has:
- content: The fact (MUST be standalone, no unresolved pronouns)
- speaker: Username of who stated this fact (without @), or null if unknown
- subjects: Names of people this is about (empty array if about the speaker themselves)
- shared: true if this is group/team knowledge, false if personal
- confidence: 0.0-1.0 how confident this should be stored
- type: One of: "preference", "identity", "relationship", "knowledge", "context", "event", "task", "observation"

## Memory Types:
Long-lived (no automatic expiration):
- preference: likes, dislikes, habits (e.g., "prefers dark mode", "hates olives")
- identity: facts about user (e.g., "works as engineer", "lives in SF")
- relationship: people in user's life (e.g., "Sarah is my wife", "boss is John")
- knowledge: factual info (e.g., "project uses Python", "company uses Slack")

Ephemeral (decay over time):
- context: current situation (e.g., "working on project X", "feeling stressed")
- event: past occurrences (e.g., "had dinner with Sarah Tuesday")
- task: things to do (e.g., "needs to call dentist")
- observation: fleeting observations (e.g., "seemed tired today")

Only include facts with confidence >= 0.7. If you cannot resolve a reference, do not extract it.

Return ONLY valid JSON, no other text. Example:
[
  {{"content": "Prefers dark mode", "speaker": "david", "subjects": [], "shared": false, "confidence": 0.9, "type": "preference"}},
  {{"content": "Sarah's birthday is March 15", "speaker": "david", "subjects": ["Sarah"], "shared": false, "confidence": 0.85, "type": "relationship"}}
]

If there are no facts worth extracting, return an empty array: []"""


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
        owner_names: list[str] | None = None,
        speaker_info: SpeakerInfo | None = None,
    ) -> list[ExtractedFact]:
        """Analyze conversation and extract facts worth remembering.

        Args:
            messages: Conversation messages to analyze.
            existing_memories: Facts already in memory (to avoid duplicates).
            owner_names: Names/handles that refer to the user themselves
                (e.g., their username, display name). These should not appear
                in subjects - they're the owner, not a third party.
            speaker_info: Information about the speaker for attribution.
                Used to label user messages with their identity.

        Returns:
            List of extracted facts with confidence scores.
        """
        if not messages:
            return []

        # Format conversation for the prompt with speaker identity
        conversation_text = self._format_conversation(messages, speaker_info)
        if not conversation_text.strip():
            return []

        # Build owner section
        owner_section = ""
        if owner_names:
            names_str = ", ".join(f'"{n}"' for n in owner_names)
            owner_section = f"""
## The user (owner)
The following names refer to the user themselves: {names_str}
Facts about these names are facts about the user - use subjects: [] (empty array).
Do NOT put the user's own name in subjects - subjects is for OTHER people in their life.
"""

        # Build existing memories section
        existing_section = ""
        if existing_memories:
            memory_list = "\n".join(f"- {m}" for m in existing_memories[:20])
            existing_section = f"""## Already in memory (don't re-extract):
{memory_list}
"""

        # Build the extraction prompt
        prompt = EXTRACTION_PROMPT.format(
            owner_section=owner_section,
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

    def _format_conversation(
        self,
        messages: list[Message],
        speaker_info: SpeakerInfo | None = None,
    ) -> str:
        """Format messages into a readable conversation transcript.

        Args:
            messages: Messages to format.
            speaker_info: Optional speaker info for user messages.

        Returns:
            Formatted conversation text with speaker attribution.
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

            # Format speaker label with identity if available
            if msg.role == Role.USER and speaker_info:
                role_label = speaker_info.format_label()
            elif msg.role == Role.USER:
                role_label = "User"
            else:
                role_label = "Assistant"

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

            # Parse memory type - fall back to KNOWLEDGE for invalid values
            type_str = item.get("type", "knowledge")
            try:
                memory_type = MemoryType(type_str)
            except ValueError:
                memory_type = MemoryType.KNOWLEDGE

            # Parse speaker (who stated this fact)
            speaker = item.get("speaker")
            if speaker:
                speaker = str(speaker).strip()
                # Remove @ prefix if present
                if speaker.startswith("@"):
                    speaker = speaker[1:]
            else:
                speaker = None

            facts.append(
                ExtractedFact(
                    content=content,
                    subjects=subjects,
                    shared=shared,
                    confidence=confidence,
                    memory_type=memory_type,
                    speaker=speaker,
                )
            )

        return facts
