"""Memory post-turn processing and background extraction orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ash.llm.types import Message as LLMMessage
from ash.llm.types import Role
from ash.memory.extractor import SpeakerInfo
from ash.memory.processing import (
    enrich_owner_names,
    ensure_self_person,
    process_extracted_facts,
)

if TYPE_CHECKING:
    from ash.core.session import SessionState
    from ash.memory.extractor import MemoryExtractor
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class MemoryPostprocessService:
    """Runs asynchronous memory extraction after a user turn."""

    def __init__(
        self,
        *,
        store: Store | None,
        extractor: MemoryExtractor | None,
        extraction_enabled: bool,
        min_message_length: int,
        debounce_seconds: int,
        context_messages: int,
        confidence_threshold: float,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._extraction_enabled = extraction_enabled
        self._min_message_length = min_message_length
        self._debounce_seconds = debounce_seconds
        self._context_messages = context_messages
        self._confidence_threshold = confidence_threshold
        self._last_extraction_time: float | None = None

    def maybe_schedule(
        self,
        *,
        user_message: str,
        session: SessionState,
        effective_user_id: str,
    ) -> None:
        if not self._should_extract(user_message):
            return

        task = asyncio.create_task(
            self._extract_background(
                session=session,
                user_id=effective_user_id,
                chat_id=session.chat_id,
            ),
            name="memory_extraction",
        )
        task.add_done_callback(self._handle_task_error)

    def _should_extract(self, user_message: str) -> bool:
        if not self._extraction_enabled:
            return False
        if not self._extractor or not self._store:
            return False
        if len(user_message) < self._min_message_length:
            return False
        if self._last_extraction_time is None:
            return True
        elapsed = time.time() - self._last_extraction_time
        return elapsed >= self._debounce_seconds

    @staticmethod
    def _handle_task_error(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (exc := task.exception()):
            logger.warning(
                "memory_extraction_task_failed", extra={"error.message": str(exc)}
            )

    async def _extract_background(
        self,
        *,
        session: SessionState,
        user_id: str,
        chat_id: str | None,
    ) -> None:
        if not self._extractor or not self._store:
            return

        try:
            self._last_extraction_time = time.time()

            existing_memories: list[str] = []
            try:
                recent = await self._store.list_memories(
                    owner_user_id=user_id,
                    chat_id=chat_id,
                    limit=20,
                )
                existing_memories = [m.content for m in recent]
            except Exception:
                logger.debug(
                    "Failed to get existing memories for extraction", exc_info=True
                )

            thread_messages: list[LLMMessage] = [
                msg
                for msg in session.messages
                if msg.role in (Role.USER, Role.ASSISTANT) and msg.get_text().strip()
            ]
            history_messages = self._load_recent_chat_history_messages(session)
            llm_messages = (history_messages + thread_messages)[
                -self._context_messages :
            ]
            if not llm_messages:
                return

            speaker_username = session.context.username
            speaker_display_name = session.context.display_name

            owner_names: list[str] = []
            if speaker_username:
                owner_names.append(speaker_username)
            if speaker_display_name and speaker_display_name not in owner_names:
                owner_names.append(speaker_display_name)
            speaker_info = SpeakerInfo(
                user_id=user_id,
                username=speaker_username,
                display_name=speaker_display_name,
            )

            speaker_person_id: str | None = None
            if speaker_username or speaker_display_name:
                effective_display = speaker_display_name or speaker_username
                assert effective_display is not None
                speaker_person_id = await self._ensure_self_person(
                    user_id=user_id,
                    username=speaker_username or "",
                    display_name=effective_display,
                )

            if speaker_person_id and self._store:
                await enrich_owner_names(
                    self._store,
                    owner_names,
                    speaker_person_id,
                )

            facts = await self._extractor.extract_from_conversation(
                messages=llm_messages,
                existing_memories=existing_memories,
                owner_names=owner_names if owner_names else None,
                speaker_info=speaker_info,
                current_datetime=datetime.now(UTC),
            )

            logger.info(
                "facts_extracted",
                extra={
                    "count": len(facts),
                    "fact.speaker": speaker_info.username if speaker_info else None,
                },
            )
            for fact in facts:
                logger.info(
                    "fact_extracted",
                    extra={
                        "fact.content": fact.content[:80],
                        "fact.type": fact.memory_type.value,
                        "fact.confidence": fact.confidence,
                        "fact.subjects": fact.subjects,
                        "fact.speaker": fact.speaker,
                    },
                )

            graph_chat_id: str | None = None
            if session.provider and session.chat_id and self._store:
                chat_entry = self._store.graph.find_chat_by_provider(
                    session.provider, session.chat_id
                )
                if chat_entry:
                    graph_chat_id = chat_entry.id
                logger.debug(
                    "memory_extraction_chat_lookup: provider=%s chat_id=%s graph_chat_id=%s chat_count=%d",
                    session.provider,
                    session.chat_id,
                    graph_chat_id,
                    len(self._store.graph.chats),
                )

            await process_extracted_facts(
                facts=facts,
                store=self._store,
                user_id=user_id,
                chat_id=chat_id,
                speaker_username=speaker_username,
                speaker_display_name=speaker_display_name,
                speaker_person_id=speaker_person_id,
                owner_names=owner_names,
                source="background_extraction",
                confidence_threshold=self._confidence_threshold,
                graph_chat_id=graph_chat_id,
                chat_type=session.context.chat_type,
            )
        except Exception:
            logger.warning("Background memory extraction failed", exc_info=True)

    def _load_recent_chat_history_messages(
        self,
        session: SessionState,
    ) -> list[LLMMessage]:
        """Load recent chat history as extraction context."""
        if not session.provider or not session.chat_id:
            return []

        from ash.chats.history import read_recent_chat_history

        entries = read_recent_chat_history(
            session.provider,
            session.chat_id,
            limit=self._context_messages * 2,
        )
        messages: list[LLMMessage] = []
        for entry in entries:
            content = entry.content.strip()
            if not content:
                continue
            if entry.role == "user":
                prefix = ""
                if entry.username:
                    display = f" ({entry.display_name})" if entry.display_name else ""
                    prefix = f"@{entry.username}{display}: "
                elif entry.display_name:
                    prefix = f"{entry.display_name}: "
                messages.append(
                    LLMMessage(role=Role.USER, content=f"{prefix}{content}")
                )
            else:
                messages.append(LLMMessage(role=Role.ASSISTANT, content=content))
        return messages

    async def _ensure_self_person(
        self,
        *,
        user_id: str,
        username: str,
        display_name: str,
    ) -> str | None:
        if not self._store:
            return None
        return await ensure_self_person(
            self._store,
            user_id,
            username,
            display_name,
        )

    async def ensure_self_person(
        self,
        *,
        user_id: str,
        username: str,
        display_name: str,
    ) -> str | None:
        """Public wrapper used by tests."""
        return await self._ensure_self_person(
            user_id=user_id,
            username=username,
            display_name=display_name,
        )
