"""Extraction pipeline for processing facts from conversations.

Provides a clear contract for the extraction workflow:
1. Extract facts from conversation using LLM
2. Resolve person references to person IDs
3. Return facts with resolved person_ids ready for storage

This formalizes the extraction logic previously embedded in the Agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ash.core.filters import OwnerMatchers
from ash.store.types import ExtractedFact, MemoryType

if TYPE_CHECKING:
    from ash.llm.types import Message
    from ash.memory.extractor import MemoryExtractor
    from ash.store.store import Store

logger = logging.getLogger(__name__)


@dataclass
class ResolvedFact:
    """A fact with person references resolved to IDs.

    This is the output of the extraction pipeline, ready for storage.
    """

    # Original fact content
    content: str
    memory_type: MemoryType
    confidence: float

    # Resolved person IDs (empty for self-facts)
    subject_person_ids: list[str] = field(default_factory=list)

    # Attribution
    speaker_username: str | None = None
    speaker_display_name: str | None = None
    speaker_person_id: str | None = None

    # Privacy and scope
    shared: bool = False
    sensitivity: str | None = None
    portable: bool = True

    # Tracking
    newly_created_person_ids: list[str] = field(default_factory=list)
    observed_at: datetime | None = None


@dataclass
class ExtractionResult:
    """Result of the extraction pipeline.

    Contains resolved facts and any side effects (newly created people, etc).
    """

    facts: list[ResolvedFact]
    newly_created_person_ids: list[str] = field(default_factory=list)


class ExtractionPipeline:
    """Pipeline for extracting and resolving facts from conversations.

    Coordinates between the MemoryExtractor (LLM-based extraction) and
    the Store (person resolution). Returns facts with person_ids already
    resolved, ready for direct storage.

    Usage:
        pipeline = ExtractionPipeline(extractor, store)
        result = await pipeline.extract_and_resolve(
            messages=messages,
            user_id="user-123",
            speaker_username="alice",
        )
        for fact in result.facts:
            await store.add_memory(
                content=fact.content,
                subject_person_ids=fact.subject_person_ids,
                ...
            )
    """

    def __init__(
        self,
        extractor: MemoryExtractor,
        store: Store,
        confidence_threshold: float = 0.7,
    ):
        """Initialize extraction pipeline.

        Args:
            extractor: LLM-based fact extractor.
            store: Store for person resolution.
            confidence_threshold: Minimum confidence to include a fact.
        """
        self._extractor = extractor
        self._store = store
        self._confidence_threshold = confidence_threshold

    async def extract_and_resolve(
        self,
        messages: list[Message],
        user_id: str,
        speaker_username: str | None = None,
        speaker_display_name: str | None = None,
        chat_id: str | None = None,
        existing_memories: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract facts from messages and resolve person references.

        Args:
            messages: Conversation messages to analyze.
            user_id: Owner user ID for scoping.
            speaker_username: Username of the speaker for attribution.
            speaker_display_name: Display name of the speaker.
            chat_id: Optional chat ID for group scoping.
            existing_memories: Already-stored facts to avoid duplicates.

        Returns:
            ExtractionResult with resolved facts and metadata.
        """
        from ash.core.filters import build_owner_matchers
        from ash.memory.extractor import SpeakerInfo

        # Build owner names for filtering self-references
        owner_names: list[str] = []
        if speaker_username:
            owner_names.append(speaker_username)
        if speaker_display_name and speaker_display_name not in owner_names:
            owner_names.append(speaker_display_name)

        # Build speaker info for extraction
        speaker_info = SpeakerInfo(
            user_id=user_id,
            username=speaker_username,
            display_name=speaker_display_name,
        )

        # Extract raw facts
        raw_facts = await self._extractor.extract_from_conversation(
            messages=messages,
            existing_memories=existing_memories,
            owner_names=owner_names if owner_names else None,
            speaker_info=speaker_info,
            current_datetime=datetime.now(UTC),
        )

        logger.info(
            "Extracted %d facts from conversation (speaker=%s)",
            len(raw_facts),
            speaker_username,
        )

        # Resolve speaker's person ID for linking self-facts
        speaker_person_id: str | None = None
        if speaker_username:
            try:
                pids = await self._store.find_person_ids_for_username(speaker_username)
                if pids:
                    speaker_person_id = next(iter(pids))
            except Exception:
                logger.debug("Failed to resolve speaker person ID", exc_info=True)

        # Build owner matchers for filtering
        owner_matchers = build_owner_matchers(owner_names)

        # Resolve each fact
        resolved_facts: list[ResolvedFact] = []
        all_newly_created: list[str] = []

        for fact in raw_facts:
            if fact.confidence < self._confidence_threshold:
                continue

            resolved = await self._resolve_fact(
                fact=fact,
                user_id=user_id,
                speaker_username=speaker_username,
                speaker_display_name=speaker_display_name,
                speaker_person_id=speaker_person_id,
                owner_matchers=owner_matchers,
            )

            if resolved:
                resolved_facts.append(resolved)
                all_newly_created.extend(resolved.newly_created_person_ids)

        return ExtractionResult(
            facts=resolved_facts,
            newly_created_person_ids=all_newly_created,
        )

    async def _resolve_fact(
        self,
        fact: ExtractedFact,
        user_id: str,
        speaker_username: str | None,
        speaker_display_name: str | None,
        speaker_person_id: str | None,
        owner_matchers: OwnerMatchers,
    ) -> ResolvedFact | None:
        """Resolve person references in a single fact.

        Args:
            fact: Extracted fact to resolve.
            user_id: Owner user ID.
            speaker_username: Username of the speaker.
            speaker_display_name: Display name of speaker.
            speaker_person_id: Pre-resolved speaker person ID.
            owner_matchers: Matchers for owner name detection.

        Returns:
            ResolvedFact with person IDs, or None if resolution fails.
        """
        from ash.core.filters import is_owner_name

        try:
            subject_person_ids: list[str] = []
            newly_created: list[str] = []

            # Resolve subject references
            if fact.subjects:
                for subject in fact.subjects:
                    # Skip owner references
                    if is_owner_name(subject, owner_matchers):
                        logger.debug("Skipping owner as subject: %s", subject)
                        continue

                    try:
                        result = await self._store.resolve_or_create_person(
                            created_by=user_id,
                            reference=subject,
                            content_hint=fact.content,
                            relationship_stated_by=speaker_username,
                        )
                        subject_person_ids.append(result.person_id)
                        if result.created:
                            newly_created.append(result.person_id)
                    except Exception:
                        logger.warning("Failed to resolve subject: %s", subject)

            # For RELATIONSHIP facts, attach the term to person records
            if fact.memory_type == MemoryType.RELATIONSHIP and subject_person_ids:
                rel_term = _extract_relationship_term(fact.content)
                if rel_term:
                    for pid in subject_person_ids:
                        try:
                            await self._store.add_relationship(
                                pid, rel_term, stated_by=speaker_username
                            )
                        except Exception:
                            logger.debug(
                                "Failed to add relationship %s to %s", rel_term, pid
                            )

            # Self-facts link to speaker's person record
            # (except RELATIONSHIP which links to the related person)
            final_subject_ids = subject_person_ids
            if (
                not subject_person_ids
                and speaker_person_id
                and fact.memory_type != MemoryType.RELATIONSHIP
            ):
                final_subject_ids = [speaker_person_id]

            # Validate speaker
            validated_speaker = _validate_speaker(fact.speaker)
            source_username = validated_speaker or speaker_username or user_id
            source_display_name = (
                speaker_display_name if source_username == speaker_username else None
            )

            return ResolvedFact(
                content=fact.content,
                memory_type=fact.memory_type,
                confidence=fact.confidence,
                subject_person_ids=final_subject_ids,
                speaker_username=source_username,
                speaker_display_name=source_display_name,
                speaker_person_id=speaker_person_id if not subject_person_ids else None,
                shared=fact.shared,
                sensitivity=fact.sensitivity.value if fact.sensitivity else None,
                portable=fact.portable,
                newly_created_person_ids=newly_created,
                observed_at=datetime.now(UTC),
            )

        except Exception:
            logger.debug(
                "Failed to resolve fact: %s",
                fact.content[:50],
                exc_info=True,
            )
            return None


def _extract_relationship_term(content: str) -> str | None:
    """Extract relationship term from content like 'Sarah is my wife'.

    Common patterns:
    - "X is my wife" -> "wife"
    - "my sister Sarah" -> "sister"
    - "X, my brother" -> "brother"
    """
    import re

    # Pattern: "X is my <relationship>"
    match = re.search(r"is (?:my|the) (\w+)", content, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Pattern: "my <relationship> X"
    match = re.search(r"(?:my|the) (\w+) \w+", content, re.IGNORECASE)
    if match and match.group(1).lower() not in ("favorite", "new", "old", "best"):
        return match.group(1).lower()

    return None


def _validate_speaker(speaker: str | None) -> str | None:
    """Validate and clean speaker attribution.

    Filters out invalid values that indicate assistant attribution.
    """
    if not speaker:
        return None

    speaker = speaker.strip().lower()

    # Invalid speaker values (assistant/system)
    invalid_speakers = {
        "assistant",
        "agent",
        "bot",
        "system",
        "ash",
        "ai",
        "claude",
        "gpt",
    }

    if speaker in invalid_speakers:
        return None

    return speaker
