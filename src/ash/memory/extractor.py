"""Background memory extraction from conversations.

Extracts memorable facts from conversations using a secondary LLM call,
running asynchronously after each exchange.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ash.llm.types import Message, Role
from ash.memory.secrets import contains_secret
from ash.memory.types import ExtractedFact, MemoryType, Sensitivity

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
## Conversation format
The conversation uses XML tags to clearly separate speakers:
- <user> contains what the USER said (may include "@username (Name):" prefix)
- <assistant> contains what the ASSISTANT said

## CRITICAL: Only extract from <user> tags
- ONLY extract facts from content inside <user> tags
- NEVER extract facts from <assistant> tags
- The assistant may summarize or restate user info - ignore this completely
- If the assistant says "You mentioned you like pizza", do NOT extract that
- Only the user's own words inside <user> tags are valid sources

## What to extract:
- User preferences (likes, dislikes, habits)
- Facts about people in their life (names, relationships, details)
- Important dates or events
- Explicit requests to remember something
- Corrections to previously known information

## What NOT to extract:
- Anything inside <assistant> tags
- Information restated or summarized by the assistant
- Actions the assistant took
- Temporary task context ("working on X project")
- Generic conversation flow
- Credentials or sensitive data (see CRITICAL section below)
- Things already in memory (avoid duplicates)
- Vague or incoherent facts (see "Require coherence" section below)
- Meta-knowledge about the system itself ("the memory system stores X", "the assistant won't store SSN", "made improvements to the system")
- Negative knowledge — only store what IS known, not what is unknown ("blood type is unknown", "hasn't explored X yet", "doesn't know their schedule")
- Vague relationships without context ("knows someone named David" — who is David and why does it matter?)
- Actions without specifics ("just arrived at a location" — WHERE?, "fixed some issues" — WHAT issues?)

## CRITICAL: Never store secrets or credentials
NEVER extract the following - reject with confidence 0.0:
- Passwords or passphrases (e.g., "my password is hunter2")
- API keys or tokens (e.g., "sk-abc123...", "ghp_...", "AKIA...")
- Social Security Numbers (e.g., "123-45-6789")
- Credit card numbers (16-digit numbers)
- Bank account or routing numbers
- Private keys (SSH, PGP, crypto wallet)
- Authentication secrets (MFA codes, recovery codes)
- Connection strings with credentials

If the user says "remember my password is X" - DO NOT store it. Return an empty array for such requests.

## CRITICAL: Resolve references
Convert pronouns and references to concrete facts:
- "I liked that restaurant" -> Find which restaurant from context, store "User liked [restaurant name]"
- "She's visiting next week" -> Find who "she" is, store "[Person name] is visiting [date]"
- "Yes, that one" -> Don't extract - too ambiguous

## CRITICAL: Require coherence
Only extract facts that are COMPLETE and USEFUL on their own:
- Reject if it contains unresolved words: "something", "somewhere", "someone", "that thing", "the thing", "it"
- Reject if recalling this fact months later would be meaningless without conversation context
- Facts must be actionable - "Spent $100 on something" is useless; "Spent $100 on a watch" is useful

Examples of INCOHERENT facts to REJECT (confidence 0.0):
- "Spent money on something" - what was bought?
- "Is uncertain about the outcome" - outcome of what?
- "Had a good experience with that" - with what?
- "Wants to do the thing we discussed" - what thing?
- "Blood type is unknown" - negative knowledge, only store what IS known
- "Hasn't explored the area yet" - absence of action is not a fact
- "Knows someone named David" - who is David? why does it matter?
- "Just arrived at a location" - WHERE specifically?
- "Fixed some issues with the code" - WHAT issues? WHAT code?
- "The memory system stores personal facts" - meta-knowledge about the system itself
- "The assistant can help with scheduling" - system capability, not a user fact

If you cannot identify WHAT, WHO, or WHERE specifically, do not extract the fact.

## CRITICAL: Ownership and perspective
Facts should be attributed to who they're ABOUT, not who performed an action:
- "My wife got me a watch" -> The SPEAKER owns the watch (subjects: []), wife gave it (mentioned in relationship context)
- "I bought Sarah a gift" -> SARAH received the gift - if relevant, fact is about Sarah (subjects: ["Sarah"])
- "The watch my wife gave me" -> SPEAKER owns the watch, wife is the giver
- Be precise: "gave", "got for", "bought for" means the OTHER person receives/owns it

## Speaker Attribution
Look for the @username prefix in <user> content to determine the speaker:
- "@david (David Cramer): I like pizza" -> speaker is "david"
- "@bob: David likes pasta" -> speaker is "bob", subjects is ["David"]
- Never use "agent", "assistant", "bot", or "system" as speaker
- If no @username prefix in <user> content, set speaker to null

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
{datetime_section}
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
- sensitivity: One of: "public", "personal", "sensitive" (see Sensitivity Classification)
- portable: true if this is an enduring fact about a person (crosses chat boundaries), false if chat-operational/ephemeral (default true)

## Portable vs Non-Portable:
When a fact has subjects (is about a person), decide whether it's portable:
- portable=true: Enduring traits and facts ("Bob loves pizza", "Sarah's birthday is March 15")
- portable=false: Chat-operational, time-bound, or context-specific ("Bob is presenting next", "Sarah will send the report by EOD", "Alice is on mute")
Default to true unless the fact is clearly ephemeral or only meaningful in this chat.

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

## Sensitivity Classification:
Classify each fact's privacy level for sharing decisions:
- "public": Can be shared anywhere. General facts, preferences, work info.
  Examples: "Prefers dark mode", "Works at Acme Corp", "Has a dog named Max"
- "personal": Share only with the subject or owner. Personal details not for group disclosure.
  Examples: "Just went through a breakup", "Looking for a new job", "Having relationship troubles"
- "sensitive": High privacy - medical, financial, mental health. Only share in private with the subject.
  Examples: "Has anxiety", "Taking medication for depression", "Salary is $X", "Has diabetes", "Seeing a therapist"

Default to "public" unless the content clearly involves private matters.

Only include facts with confidence >= 0.7. If you cannot resolve a reference, do not extract it.

Return ONLY valid JSON, no other text. Example:
[
  {{"content": "Prefers dark mode", "speaker": "david", "subjects": [], "shared": false, "confidence": 0.9, "type": "preference", "sensitivity": "public", "portable": true}},
  {{"content": "Sarah's birthday is March 15", "speaker": "david", "subjects": ["Sarah"], "shared": false, "confidence": 0.85, "type": "relationship", "sensitivity": "public", "portable": true}},
  {{"content": "Has been dealing with anxiety", "speaker": "david", "subjects": [], "shared": false, "confidence": 0.9, "type": "identity", "sensitivity": "sensitive", "portable": true}}
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
        current_datetime: datetime | None = None,
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
            current_datetime: Current datetime for resolving relative time
                references (e.g., "this weekend" -> "Feb 15-16, 2026").

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

        # Build datetime section for temporal context
        datetime_section = ""
        if current_datetime:
            formatted_dt = current_datetime.strftime("%B %d, %Y at %H:%M")
            weekday = current_datetime.strftime("%A")
            datetime_section = f"""
## Current date and time
The current datetime is: {weekday}, {formatted_dt}

CRITICAL: Convert ALL relative time references to absolute dates in extracted facts:
- "this weekend" → "the weekend of [actual date]"
- "next Tuesday" → "[actual weekday, Month Day, Year]"
- "tomorrow" → "[actual Month Day, Year]"
- "last week" → "the week of [actual date range]"
- "in 2 days" → "[actual Month Day, Year]"

This ensures memories remain meaningful when recalled later.
"""

        # Build the extraction prompt
        prompt = EXTRACTION_PROMPT.format(
            owner_section=owner_section,
            existing_memories_section=existing_section,
            datetime_section=datetime_section,
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

        Uses XML tags to clearly separate user and assistant messages,
        making it unambiguous which content comes from which speaker.

        Args:
            messages: Messages to format.
            speaker_info: Optional speaker info for user messages.

        Returns:
            Formatted conversation text with XML tags for speaker separation.
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

            # Use XML tags to clearly separate speakers
            if msg.role == Role.USER:
                # Include speaker info in the content for attribution
                if speaker_info:
                    label = speaker_info.format_label()
                    lines.append(f"<user>\n{label}: {text}\n</user>")
                else:
                    lines.append(f"<user>\n{text}\n</user>")
            else:
                lines.append(f"<assistant>\n{text}\n</assistant>")

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

            # Filter out potential secrets (defense in depth)
            if contains_secret(content):
                logger.warning(
                    "Filtered potential secret from extraction: %s...", content[:30]
                )
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

            # Parse sensitivity - default to None (treated as PUBLIC)
            sensitivity_str = item.get("sensitivity")
            sensitivity = None
            if sensitivity_str:
                try:
                    sensitivity = Sensitivity(sensitivity_str)
                except ValueError:
                    pass  # Leave as None (default PUBLIC)

            # Parse portable - default to True
            portable = item.get("portable", True)
            if not isinstance(portable, bool):
                portable = True

            facts.append(
                ExtractedFact(
                    content=content,
                    subjects=subjects,
                    shared=shared,
                    confidence=confidence,
                    memory_type=memory_type,
                    speaker=speaker,
                    sensitivity=sensitivity,
                    portable=portable,
                )
            )

        return facts
