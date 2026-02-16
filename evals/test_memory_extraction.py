"""Memory extraction evaluation tests.

Tests that explicit "remember" requests store facts with proper classification
(type, subjects, person linking) via the extraction pipeline.

Run with: uv run pytest evals/test_memory_extraction.py -v -s -m eval
Requires ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables.
"""

import asyncio
import logging
from pathlib import Path

import pytest

from ash.core.agent import AgentComponents
from ash.core.session import SessionState
from ash.llm.base import LLMProvider
from ash.store.types import MemoryType
from evals.judge import LLMJudge
from evals.runner import get_case_by_id, load_eval_suite
from evals.types import EvalConfig

logger = logging.getLogger(__name__)

CASES_DIR = Path(__file__).parent / "cases"
EXTRACTION_CASES = CASES_DIR / "memory_extraction.yaml"


def _make_session(
    session_id: str,
    user_id: str = "eval-user",
    username: str = "david",
    display_name: str = "David Cramer",
    chat_id: str = "eval-chat-extract",
) -> SessionState:
    """Create a SessionState for extraction eval."""
    session = SessionState(
        session_id=session_id,
        provider="telegram",
        chat_id=chat_id,
        user_id=user_id,
    )
    session.context.username = username
    session.context.display_name = display_name
    session.context.chat_type = "private"
    return session


async def _drain_extraction_tasks() -> None:
    """Wait for all background memory extraction tasks to complete."""
    await asyncio.sleep(0)
    for task in asyncio.all_tasks():
        if task.get_name() == "memory_extraction":
            await task


async def _dump_state(agent: AgentComponents) -> None:
    """Log extracted memories and people for eval debugging."""
    if agent.memory_manager:
        store = agent.memory_manager
        memories = await store.get_all_memories()
        logger.info("=== Extracted memories (%d) ===", len(memories))
        for m in memories:
            from ash.graph.edges import get_subject_person_ids

            subjects = get_subject_person_ids(store.graph, m.id)
            logger.info(
                "  [%s] %s (type=%s, owner=%s, subjects=%s, source=%s)",
                m.id[:8],
                m.content[:80],
                m.memory_type.value,
                m.owner_user_id,
                subjects,
                m.source_username,
            )

    if agent.person_manager:
        people = await agent.person_manager.list_people()
        logger.info("=== People records (%d) ===", len(people))
        for p in people:
            alias_strs = [a.value if hasattr(a, "value") else str(a) for a in p.aliases]
            rel_strs = [f"{r.relationship}(by={r.stated_by})" for r in p.relationships]
            logger.info(
                "  [%s] %s (aliases=%s, relationships=%s)",
                p.id[:8],
                p.name,
                alias_strs,
                rel_strs,
            )


@pytest.mark.eval
class TestMemoryExtractionEvals:
    """Memory extraction evaluation tests."""

    @pytest.mark.asyncio
    async def test_extraction_relationship_stored(
        self,
        eval_memory_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """'Remember Sarah is my sister' stores a relationship memory with person link."""
        agent = eval_memory_agent.agent

        session = _make_session("eval-extract-relationship")
        response = await agent.process_message(
            "Remember that Sarah is my sister",
            session=session,
            user_id="eval-user",
        )
        await _drain_extraction_tasks()
        await _dump_state(eval_memory_agent)

        # Structural checks: verify memory was stored with correct type
        store = eval_memory_agent.memory_manager
        assert store is not None

        memories = await store.get_all_memories()
        sarah_memories = [
            m
            for m in memories
            if "sarah" in m.content.lower() and "sister" in m.content.lower()
        ]
        assert len(sarah_memories) >= 1, (
            f"Expected at least 1 memory about Sarah being sister, got {len(sarah_memories)}. "
            f"All memories: {[m.content for m in memories]}"
        )

        # Check type classification
        sarah_mem = sarah_memories[0]
        assert sarah_mem.memory_type == MemoryType.RELATIONSHIP, (
            f"Expected relationship type, got {sarah_mem.memory_type.value}"
        )

        # Check person record exists
        people = await store.list_people()
        sarah_people = [p for p in people if "sarah" in p.name.lower()]
        assert len(sarah_people) >= 1, (
            f"Expected a person record for Sarah, got {[p.name for p in people]}"
        )

        # Judge the response quality
        suite = load_eval_suite(EXTRACTION_CASES)
        case = get_case_by_id(suite, "extraction_relationship_stored")
        judge = LLMJudge(judge_llm, EvalConfig())
        result = await judge.evaluate(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
        )

        logger.info("Response: %s", response.text)
        logger.info("Judge: passed=%s, score=%s", result.passed, result.score)
        logger.info("Reasoning: %s", result.reasoning)

        assert result.passed, (
            f"extraction_relationship_stored failed (score={result.score}): {result.reasoning}"
        )

    @pytest.mark.asyncio
    async def test_extraction_preference_stored(
        self,
        eval_memory_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """'Remember I prefer dark mode' stores a preference memory."""
        agent = eval_memory_agent.agent

        session = _make_session("eval-extract-preference")
        response = await agent.process_message(
            "Remember that I prefer dark mode in all apps",
            session=session,
            user_id="eval-user",
        )
        await _drain_extraction_tasks()
        await _dump_state(eval_memory_agent)

        # Structural checks
        store = eval_memory_agent.memory_manager
        assert store is not None

        memories = await store.get_all_memories()
        dark_mode_memories = [m for m in memories if "dark mode" in m.content.lower()]
        assert len(dark_mode_memories) >= 1, (
            f"Expected at least 1 memory about dark mode, got {len(dark_mode_memories)}. "
            f"All memories: {[m.content for m in memories]}"
        )

        pref_mem = dark_mode_memories[0]
        assert pref_mem.memory_type == MemoryType.PREFERENCE, (
            f"Expected preference type, got {pref_mem.memory_type.value}"
        )

        # Judge
        suite = load_eval_suite(EXTRACTION_CASES)
        case = get_case_by_id(suite, "extraction_preference_stored")
        judge = LLMJudge(judge_llm, EvalConfig())
        result = await judge.evaluate(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
        )

        logger.info("Response: %s", response.text)
        logger.info("Judge: passed=%s, score=%s", result.passed, result.score)

        assert result.passed, (
            f"extraction_preference_stored failed (score={result.score}): {result.reasoning}"
        )

    @pytest.mark.asyncio
    async def test_extraction_recall_after_store(
        self,
        eval_memory_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """A fact stored via 'remember' is recallable in a new session."""
        agent = eval_memory_agent.agent

        # Session 1: Store the fact
        session1 = _make_session("eval-extract-store", chat_id="eval-chat-store")
        await agent.process_message(
            "Remember that Sarah is my sister",
            session=session1,
            user_id="eval-user",
        )
        await _drain_extraction_tasks()
        await _dump_state(eval_memory_agent)

        # Session 2: Recall in a fresh session
        session2 = _make_session("eval-extract-recall", chat_id="eval-chat-recall")
        suite = load_eval_suite(EXTRACTION_CASES)
        case = get_case_by_id(suite, "extraction_recall_after_store")

        response = await agent.process_message(
            case.prompt,
            session=session2,
            user_id="eval-user",
        )

        judge = LLMJudge(judge_llm, EvalConfig())
        result = await judge.evaluate(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
        )

        logger.info("Response: %s", response.text)
        logger.info("Judge: passed=%s, score=%s", result.passed, result.score)
        logger.info("Reasoning: %s", result.reasoning)

        assert result.passed, (
            f"extraction_recall_after_store failed (score={result.score}): {result.reasoning}"
        )
