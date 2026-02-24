"""Background memory extraction from conversations.

Extracts memorable facts from conversations using a secondary LLM call,
running asynchronously after each exchange.
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ash.llm.types import Message, Role
from ash.memory.secrets import contains_secret
from ash.store.types import ExtractedFact, MemoryType, Sensitivity

if TYPE_CHECKING:
    from ash.llm import LLMProvider

logger = logging.getLogger(__name__)
_VERIFICATION_DROP_REASONS = {
    "unsupported",
    "ambiguous",
    "meta_system",
    "stale_status",
    "low_utility",
    "unsafe",
}


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

## PRIMARY OBJECTIVE: Long-term utility
Store only facts that will likely improve future assistance. Prefer durable,
high-signal personal information over low-value chat noise.

Utility check before storing:
- Will this still be useful or unambiguous 30+ days later?
- Does it materially help personalization, planning, or relationship context?
- Is it specific enough to act on without the original conversation?
If any answer is no, do not extract.

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
- Engineering/system operations details ("refactored the bot", "wiped session history", "fixed extraction bug")
- Assistant/harness/eval internals ("scheduler hooks", "memory interface", "prompt updated")
- Short-lived status snapshots ("is not in Tokyo right now", "can't type well today") unless they are explicitly important to future planning

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
- Facts must pass a "30-day recall" test: if future recall would confuse the assistant, reject it

## CRITICAL: Content must be self-contained
The "content" field must be understandable on its own, without the subjects array:
- WRONG: "birthday is August 12" (whose birthday?)
- RIGHT: "David's birthday is August 12"
- WRONG: "Located in San Francisco" (who?)
- RIGHT: "User is located in San Francisco" (or use their name)
- WRONG: "Prefers dark mode" (who prefers it?)
- RIGHT: "User prefers dark mode" (or use their name)

If the fact is about the speaker and there's no name available, prefix with "User".
If the fact is about a third party, their name MUST appear in the content.
- WRONG: "pregnant with David's baby, due date August 19" (who is pregnant?)
- RIGHT: "Sarah (David's wife) is pregnant, due August 19, 2026"

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
- "Expecting a child and planning to prepare" - too vague; WHAT preparations?
- "Made improvements to the system" - meta-knowledge AND too vague

If you cannot identify WHAT, WHO, or WHERE specifically, do not extract the fact.

## CRITICAL: Ownership and perspective
Facts should be attributed to who they're ABOUT, not who performed an action:
- "My wife got me a watch" -> The SPEAKER owns the watch (subjects: []), wife gave it (mentioned in relationship context)
- "I bought Sarah a gift" -> SARAH received the gift - if relevant, fact is about Sarah (subjects: ["Sarah"])
- "The watch my wife gave me" -> SPEAKER owns the watch, wife is the giver
- Be precise: "gave", "got for", "bought for" means the OTHER person receives/owns it

Medical conditions, pregnancy, achievements, and personal states belong to the person
experiencing them, NOT the person reporting them:
- "My wife is pregnant" -> The WIFE is pregnant (subjects: ["wife's name"]), not the speaker
- "My dad has diabetes" -> The DAD has the condition (subjects: ["dad's name"])
- "My son got into Stanford" -> The SON was admitted (subjects: ["son's name"])
- "My brother broke his leg" -> The BROTHER is injured (subjects: ["brother's name"])

HOWEVER, when the speaker is an EQUAL PARTICIPANT in a joint activity or life event
with another person, include BOTH in subjects. Use the speaker's name (from the owner names above):
- "Alice and I are starting a company" -> BOTH are co-founders (subjects: [speaker_name, "Alice"])
- "Bob and I went to Tokyo together" -> BOTH traveled (subjects: [speaker_name, "Bob"])
- "My wife and I are expecting a baby" -> BOTH are expecting (subjects: [speaker_name, "wife's name"])
- "Sarah and I bought a house" -> BOTH bought it (subjects: [speaker_name, "Sarah"])

The key distinction: REPORTING about someone else's state = only that person is a subject.
PARTICIPATING together in something = both speaker and other person are subjects.

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

When a fact involves BOTH the speaker AND another person as equal participants,
include both in subjects. Use the speaker's name (from the owner names above):
- "Alice and I are starting a company" -> subjects: [speaker_name, "Alice"] (joint venture)
- "Bob and I went to Tokyo together" -> subjects: [speaker_name, "Bob"] (joint trip)
- "My wife and I are expecting" -> subjects: [speaker_name, "wife's name"] (joint life event)

But if the speaker is just REPORTING about someone else's state:
- "My dad has diabetes" -> subjects: ["dad's name"] (only dad has it)
- "My coworker got promoted" -> subjects: ["coworker's name"] (only they were promoted)

WRONG: "My wife got me a Grand Seiko" -> extracting that wife owns a Grand Seiko (she GAVE it, speaker owns it)
RIGHT: "My wife got me a Grand Seiko" -> speaker owns Grand Seiko, subjects: []

## Aliases
If the user explicitly states that a person has a nickname, alias, or alternate name,
extract that mapping. Only extract when there is a CLEAR declaration of equivalence.

Patterns: "X is also known as Y", "X goes by Y", "X's nickname is Y",
"Everyone calls X 'Y'", "Y is short for X"

Do NOT infer aliases from casual usage — only from explicit statements.
The aliases dict maps the person's name (matching a subjects entry) to alias strings.
If no aliases stated, use an empty dict {{}}.

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
- aliases: Dict mapping subject name to list of alias strings (empty dict {{}} if none)

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

Ephemeral quality gate:
- Do not store ephemeral facts that are trivial, stale, or purely situational status.
- For events/tasks/context/observations, require concrete details (who/what/where/when).
- Reject ambiguous time/location statements like "going in May" unless destination/subject is explicit.

## Sensitivity Classification:
Classify each fact's privacy level for sharing decisions:
- "public": Can be shared anywhere. General facts, preferences, work info.
  Examples: "Prefers dark mode", "Works at Acme Corp", "Has a dog named Max"
- "personal": Share only with the subject or owner. Personal details not for group disclosure.
  Examples: "Just went through a breakup", "Looking for a new job", "Having relationship troubles"
- "sensitive": High privacy - medical, financial, mental health. Only share in private with the subject.
  Examples: "Has anxiety", "Taking medication for depression", "Salary is $X", "Has diabetes", "Seeing a therapist", "Is pregnant", "Undergoing fertility treatment"

Default to "public" unless the content clearly involves private matters.

Only include facts with confidence >= 0.7. If you cannot resolve a reference, do not extract it.

Return ONLY valid JSON, no other text. Example:
[
  {{"content": "David prefers dark mode", "speaker": "david", "subjects": [], "shared": false, "confidence": 0.9, "type": "preference", "sensitivity": "public", "portable": true, "aliases": {{}}}},
  {{"content": "Sarah's birthday is March 15", "speaker": "david", "subjects": ["Sarah"], "shared": false, "confidence": 0.85, "type": "relationship", "sensitivity": "public", "portable": true, "aliases": {{}}}},
  {{"content": "Sukhpreet goes by SK", "speaker": "david", "subjects": ["Sukhpreet"], "shared": false, "confidence": 0.9, "type": "identity", "sensitivity": "public", "portable": true, "aliases": {{"Sukhpreet": ["SK"]}}}}
]

If there are no facts worth extracting, return an empty array: []"""


CLASSIFICATION_PROMPT = """You are a memory classification system. Given a single fact, classify it.

## Fact to classify:
{content}

## Output format:
Return a JSON object with:
- subjects: Names of people this is about (empty array if about the speaker themselves)
- type: One of: "preference", "identity", "relationship", "knowledge", "context", "event", "task", "observation"
- sensitivity: One of: "public", "personal", "sensitive"
- portable: true if this is an enduring fact about a person, false if ephemeral
- shared: true if this is group/team knowledge, false if personal
- aliases: Dict mapping subject name to list of alias strings (empty dict {{}} if none). Only extract when fact explicitly declares a nickname or alternate name.

## Guidelines:
- subjects should contain people the fact is PRIMARILY ABOUT (not the speaker)
- Default sensitivity to "public" unless clearly private/medical/financial. Pregnancy, reproductive health, and medical conditions are always "sensitive"
- Default portable to true unless clearly ephemeral
- Default shared to false unless clearly group knowledge

Return ONLY valid JSON, no other text. Example:
{{"subjects": ["Sarah"], "type": "relationship", "sensitivity": "public", "portable": true, "shared": false, "aliases": {{}}}}"""


VERIFICATION_PROMPT = """You are a memory verification system.

You are given:
1) A conversation transcript
2) Candidate extracted facts

Primary objective:
- Keep only facts that are supported by the transcript and worth storing long-term.
- Prioritize precision and future utility over recall.

For each fact:
- verified=true if supported, self-contained, and useful
- verified=false if unsupported, ambiguous, stale, low-utility, or policy-disallowed
- content should be a rewritten self-contained version when needed
- drop_reason is required when verified=false

CRITICAL:
- Keep meaning faithful to the transcript. Do not invent new facts.
- If a fact says "going in May" but transcript clearly says "going to Tokyo in May",
  rewrite content to include "Tokyo".
- If a fact cannot be verified precisely, mark verified=false.
- If key slots are missing (subject/action/object/time/location where relevant), mark verified=false unless transcript supports a precise rewrite.
- Example: "User is going in May" is NOT verifiable unless destination/context can be resolved from transcript.
- Drop meta/system facts even if true ("refactored bot", "updated prompt", "wiped session history").
- Drop low-value transient status snapshots unless clearly useful for future planning.

Return ONLY valid JSON as an array:
[
  {{"index": 0, "verified": true, "content": "Rewritten fact"}},
  {{"index": 1, "verified": false, "drop_reason": "ambiguous"}}
]

Rules:
- Process every input index exactly once.
- verified must be a JSON boolean (true/false), not a string.
- If verified=true and content is omitted, the original content will be kept.
- Do not output extra keys beyond: index, verified, content, drop_reason.
- Allowed drop_reason values:
  - "unsupported"
  - "ambiguous"
  - "meta_system"
  - "stale_status"
  - "low_utility"
  - "unsafe"

## Conversation
{conversation}

## Candidate facts
{facts}
"""


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
        verification_enabled: bool = True,
        verification_llm: "LLMProvider | None" = None,
        verification_model: str | None = None,
    ):
        """Initialize memory extractor.

        Args:
            llm: LLM provider for extraction calls.
            model: Model to use (defaults to provider default).
            max_tokens: Maximum tokens for extraction response.
            confidence_threshold: Minimum confidence to include a fact.
            verification_enabled: Whether to run second-pass verification/rewriting.
            verification_llm: Optional LLM provider for verification calls.
            verification_model: Optional model name for verification calls.
        """
        self._llm = llm
        self._model = model
        self._max_tokens = max_tokens
        self._confidence_threshold = confidence_threshold
        self._verification_enabled = verification_enabled
        self._verification_llm = verification_llm or llm
        self._verification_model = verification_model or model

    async def classify_fact(self, content: str) -> ExtractedFact | None:
        """Classify a pre-formed fact using LLM.

        Used by the enriched `memory add` RPC path to get subject linking,
        type classification, sensitivity, and portable classification for
        facts provided directly by the agent.

        Args:
            content: The fact content string to classify.

        Returns:
            ExtractedFact with classification fields, or None on failure.
        """
        prompt = CLASSIFICATION_PROMPT.format(content=content)

        try:
            response = await self._llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._model,
                max_tokens=256,
                temperature=0.1,
            )

            text = (response.message.get_text() or "").strip()

            # Strip markdown code fences if present
            match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

            data = json.loads(text)
            if not isinstance(data, dict):
                return None

            # Build an ExtractedFact using _parse_fact_item logic for field parsing,
            # but override content and confidence
            data["content"] = content
            data["confidence"] = 1.0
            return self._parse_fact_item(data)

        except Exception:
            logger.warning(
                "fact_classification_failed",
                extra={"content_preview": content[:80]},
                exc_info=True,
            )
            return None

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

        owner_section = self._build_owner_section(owner_names)
        existing_section = self._build_existing_memories_section(existing_memories)
        datetime_section = self._build_datetime_section(current_datetime)

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

            drop_counts: Counter[str] = Counter()
            # Parse the response
            facts = self._parse_extraction_response(
                response.message.get_text(),
                drop_counts=drop_counts,
            )
            total_candidates = len(facts) + sum(drop_counts.values())
            stats_level = (
                logger.debug
                if total_candidates == 0 and len(facts) == 0
                else logger.info
            )
            stats_level(
                "memory_extraction_filter_stats",
                extra={
                    "fact.total_candidates": total_candidates,
                    "fact.accepted_count": len(facts),
                    "fact.dropped_invalid": drop_counts.get("invalid", 0),
                    "fact.dropped_empty_content": drop_counts.get("empty_content", 0),
                    "fact.dropped_low_confidence": drop_counts.get("low_confidence", 0),
                    "fact.dropped_secret": drop_counts.get("secret", 0),
                },
            )
            if not self._verification_enabled or not facts or len(messages) <= 1:
                return facts
            return await self._verify_facts_with_context(
                facts=facts,
                conversation_text=conversation_text,
            )

        except Exception as e:
            logger.warning("memory_extraction_failed", extra={"error.message": str(e)})
            return []

    async def _verify_facts_with_context(
        self,
        *,
        facts: list[ExtractedFact],
        conversation_text: str,
    ) -> list[ExtractedFact]:
        """Run second-pass LLM verification/rewrite against broader context."""
        fact_lines = []
        for idx, fact in enumerate(facts):
            fact_lines.append(
                json.dumps(
                    {
                        "index": idx,
                        "content": fact.content,
                        "subjects": fact.subjects,
                        "speaker": fact.speaker,
                    },
                    ensure_ascii=False,
                )
            )

        prompt = VERIFICATION_PROMPT.format(
            conversation=conversation_text,
            facts="\n".join(fact_lines),
        )

        try:
            response = await self._verification_llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._verification_model,
                max_tokens=512,
                temperature=0.0,
            )
            decisions = self._parse_verification_response(response.message.get_text())
            if not decisions:
                return facts

            # Keep only the latest valid decision per index and fail open
            # for any fact the model did not decide about.
            decisions_by_index: dict[int, dict[str, Any]] = {}
            for decision in decisions:
                idx = decision.get("index")
                verified = decision.get("verified")
                if verified is None:
                    # Backward-compatible parsing for older output shape.
                    verified = decision.get("grounded")
                if not isinstance(idx, int) or idx < 0 or idx >= len(facts):
                    continue
                if not isinstance(verified, bool):
                    continue
                decision["verified"] = verified
                decisions_by_index[idx] = decision

            # Fail open when verification output had no usable decisions.
            if not decisions_by_index:
                return facts

            verified_facts: list[ExtractedFact] = []
            dropped_count = 0
            rewritten_count = 0
            drop_reason_counts: Counter[str] = Counter()
            for idx, fact in enumerate(facts):
                decision = decisions_by_index.get(idx)
                if decision is None:
                    verified_facts.append(fact)
                    continue

                if decision["verified"] is not True:
                    dropped_count += 1
                    reason = decision.get("drop_reason")
                    if isinstance(reason, str) and reason in _VERIFICATION_DROP_REASONS:
                        drop_reason_counts[reason] += 1
                    else:
                        drop_reason_counts["unknown"] += 1
                    continue

                content = decision.get("content")
                if isinstance(content, str) and content.strip():
                    rewritten_content = content.strip()
                    if contains_secret(rewritten_content):
                        logger.debug(
                            "secret_filtered_from_verification",
                            extra={"content_preview": rewritten_content[:30]},
                        )
                        dropped_count += 1
                        drop_reason_counts["unsafe"] += 1
                        continue
                    if rewritten_content != fact.content:
                        rewritten_count += 1
                        fact.content = rewritten_content
                verified_facts.append(fact)

            logger.info(
                "memory_verification_stats",
                extra={
                    "fact.total_candidates": len(facts),
                    "fact.accepted_count": len(verified_facts),
                    "fact.rewritten_count": rewritten_count,
                    "fact.dropped_count": dropped_count,
                    "fact.decision_count": len(decisions_by_index),
                    "fact.dropped_unsupported": drop_reason_counts.get(
                        "unsupported", 0
                    ),
                    "fact.dropped_ambiguous": drop_reason_counts.get("ambiguous", 0),
                    "fact.dropped_meta_system": drop_reason_counts.get(
                        "meta_system", 0
                    ),
                    "fact.dropped_stale_status": drop_reason_counts.get(
                        "stale_status", 0
                    ),
                    "fact.dropped_low_utility": drop_reason_counts.get(
                        "low_utility", 0
                    ),
                    "fact.dropped_unsafe": drop_reason_counts.get("unsafe", 0),
                    "fact.dropped_unknown_reason": drop_reason_counts.get("unknown", 0),
                },
            )
            return verified_facts
        except Exception:
            logger.warning("memory_verification_failed", exc_info=True)
            return facts

    def _parse_verification_response(self, response_text: str) -> list[dict[str, Any]]:
        """Parse the LLM verification response."""
        text = response_text.strip()
        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug(
                "Failed to parse verification response as JSON: %s", text[:200]
            )
            return []

        if not isinstance(data, list):
            return []

        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _build_owner_section(owner_names: list[str] | None) -> str:
        if not owner_names:
            return ""
        names_str = ", ".join(f'"{n}"' for n in owner_names)
        return f"""
## The user (owner)
The following names refer to the user themselves: {names_str}
Facts about these names are facts about the user - use subjects: [] (empty array).
Do NOT put the user's own name in subjects - subjects is for OTHER people in their life.
"""

    @staticmethod
    def _build_existing_memories_section(
        existing_memories: list[str] | None,
    ) -> str:
        if not existing_memories:
            return ""
        memory_list = "\n".join(f"- {m}" for m in existing_memories[:20])
        return f"""## Already in memory (don't re-extract):
{memory_list}
"""

    @staticmethod
    def _build_datetime_section(current_datetime: datetime | None) -> str:
        if not current_datetime:
            return ""
        formatted_dt = current_datetime.strftime("%B %d, %Y at %H:%M")
        weekday = current_datetime.strftime("%A")
        return f"""
## Current date and time
The current datetime is: {weekday}, {formatted_dt}

CRITICAL: Convert ALL relative time references to absolute dates in extracted facts:
- "this weekend" \u2192 "the weekend of [actual date]"
- "next Tuesday" \u2192 "[actual weekday, Month Day, Year]"
- "tomorrow" \u2192 "[actual Month Day, Year]"
- "last week" \u2192 "the week of [actual date range]"
- "in 2 days" \u2192 "[actual Month Day, Year]"

This ensures memories remain meaningful when recalled later.
"""

    def _format_conversation(
        self,
        messages: list[Message],
        speaker_info: SpeakerInfo | None = None,
    ) -> str:
        """Format messages into an XML-tagged transcript for the extraction prompt."""
        lines = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue

            text = msg.get_text()
            if not text.strip():
                continue

            if len(text) > 2000:
                text = text[:2000] + "..."

            if msg.role == Role.USER:
                if speaker_info:
                    label = speaker_info.format_label()
                    lines.append(f"<user>\n{label}: {text}\n</user>")
                else:
                    lines.append(f"<user>\n{text}\n</user>")
            else:
                lines.append(f"<assistant>\n{text}\n</assistant>")

        return "\n\n".join(lines)

    def _parse_extraction_response(
        self,
        response_text: str,
        *,
        drop_counts: Counter[str] | None = None,
    ) -> list[ExtractedFact]:
        """Parse the LLM's JSON response into ExtractedFact objects."""
        text = response_text.strip()
        counters = drop_counts or Counter()

        # Strip markdown code fences if present
        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

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
                counters["invalid"] += 1
                continue
            fact = self._parse_fact_item(item, drop_counts=counters)
            if fact is not None:
                facts.append(fact)
        return facts

    def _parse_fact_item(
        self,
        item: dict[str, Any],
        *,
        drop_counts: Counter[str] | None = None,
    ) -> ExtractedFact | None:
        """Parse a single fact dict from the LLM response. Returns None if invalid."""
        counters = drop_counts or Counter()
        content = item.get("content", "").strip()
        if not content:
            counters["empty_content"] += 1
            return None

        confidence = float(item.get("confidence", 0.0))
        if confidence < self._confidence_threshold:
            counters["low_confidence"] += 1
            return None

        if contains_secret(content):
            counters["secret"] += 1
            logger.debug(
                "secret_filtered_from_extraction",
                extra={"content_preview": content[:30]},
            )
            return None

        subjects = item.get("subjects", [])
        if not isinstance(subjects, list):
            subjects = []
        subjects = [str(s) for s in subjects if s]

        try:
            memory_type = MemoryType(item.get("type", "knowledge"))
        except ValueError:
            memory_type = MemoryType.KNOWLEDGE

        speaker = item.get("speaker")
        if speaker:
            speaker = str(speaker).strip().lstrip("@") or None

        sensitivity = None
        sensitivity_str = item.get("sensitivity")
        if sensitivity_str:
            try:
                sensitivity = Sensitivity(sensitivity_str)
            except ValueError:
                pass

        portable = item.get("portable", True)
        if not isinstance(portable, bool):
            portable = True

        aliases_raw = item.get("aliases", {})
        aliases: dict[str, list[str]] = {}
        if isinstance(aliases_raw, dict):
            for name, alias_list in aliases_raw.items():
                if isinstance(name, str) and isinstance(alias_list, list):
                    cleaned = [
                        str(a).strip() for a in alias_list if a and str(a).strip()
                    ]
                    if cleaned:
                        aliases[name.strip()] = cleaned

        return ExtractedFact(
            content=content,
            subjects=subjects,
            shared=bool(item.get("shared", False)),
            confidence=confidence,
            memory_type=memory_type,
            speaker=speaker,
            sensitivity=sensitivity,
            portable=portable,
            aliases=aliases,
        )
