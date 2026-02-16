"""Identity resolution evaluation tests.

Simulates a Telegram group chat where multiple users mention people,
then verifies the agent resolves identities and recalls cross-user
facts in a separate private session.

Run with: uv run pytest evals/test_identity.py -v -s -m eval
Requires ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables.
"""

import asyncio
import logging
from pathlib import Path

import pytest

from ash.core.agent import AgentComponents
from ash.core.session import SessionState
from ash.llm.base import LLMProvider
from evals.judge import LLMJudge
from evals.runner import get_case_by_id, load_eval_suite
from evals.types import EvalConfig

logger = logging.getLogger(__name__)

CASES_DIR = Path(__file__).parent / "cases"
IDENTITY_CASES = CASES_DIR / "identity.yaml"


def _make_group_session(
    session_id: str,
    user_id: str,
    username: str,
    display_name: str,
) -> SessionState:
    """Create a SessionState emulating a Telegram group chat message."""
    session = SessionState(
        session_id=session_id,
        provider="telegram",
        chat_id="eval-group-123",
        user_id=user_id,
    )
    session.context.username = username
    session.context.display_name = display_name
    session.context.chat_type = "supergroup"
    session.context.chat_title = "Eval Test Group"
    return session


def _make_private_session(
    session_id: str,
    user_id: str,
    username: str,
    display_name: str,
) -> SessionState:
    """Create a SessionState emulating a Telegram private chat."""
    session = SessionState(
        session_id=session_id,
        provider="telegram",
        chat_id=f"eval-private-{user_id}",
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


async def _dump_state(agent: "AgentComponents") -> None:
    """Log extracted memories and people for eval debugging."""
    if agent.memory_manager:
        store = agent.memory_manager
        memories = await store.get_all_memories()
        logger.info("=== Extracted memories (%d) ===", len(memories))
        for m in memories:
            logger.info(
                "  [%s] %s (type=%s, owner=%s, subjects=%s, source=%s)",
                m.id[:8],
                m.content[:80],
                m.memory_type.value,
                m.owner_user_id,
                m.subject_person_ids,
                m.source_username,
            )

    if agent.person_manager:
        people = await agent.person_manager.list_people()
        logger.info("=== People records (%d) ===", len(people))
        for p in people:
            alias_strs = [a.value if hasattr(a, "value") else str(a) for a in p.aliases]
            rel_strs = [
                f"{r.term}(by={r.stated_by})" if hasattr(r, "term") else str(r)
                for r in p.relationships
            ]
            logger.info(
                "  [%s] %s (aliases=%s, relationships=%s)",
                p.id[:8],
                p.name,
                alias_strs,
                rel_strs,
            )


async def _seed_group_chat(agent: "AgentComponents") -> None:
    """Seed a group chat with David mentioning his wife and Sukhpreet speaking."""
    a = agent.agent

    # David mentions his wife Sukhpreet
    session_david = _make_group_session(
        session_id="eval-group-david",
        user_id="user-david",
        username="dcramer",
        display_name="David Cramer",
    )
    await a.process_message(
        "My wife Sukhpreet loves hiking, we went on a great trail last weekend.",
        session=session_david,
        user_id="user-david",
    )
    await _drain_extraction_tasks()

    # Sukhpreet (@sksembhi) speaks for herself
    session_sukhpreet = _make_group_session(
        session_id="eval-group-sukhpreet",
        user_id="user-sukhpreet",
        username="sksembhi",
        display_name="Sukhpreet Sembhi",
    )
    await a.process_message(
        "I'm training for a marathon next month, so excited!",
        session=session_sukhpreet,
        user_id="user-sukhpreet",
    )
    await _drain_extraction_tasks()

    # Dump state for debugging
    await _dump_state(agent)


async def _judge_case(
    agent: "AgentComponents",
    judge_llm: LLMProvider,
    case_id: str,
    session: SessionState,
    user_id: str,
) -> None:
    """Run a case prompt, judge it, and assert it passes."""
    suite = load_eval_suite(IDENTITY_CASES)
    case = get_case_by_id(suite, case_id)

    response = await agent.agent.process_message(
        case.prompt,
        session=session,
        user_id=user_id,
    )

    judge = LLMJudge(judge_llm, EvalConfig())
    result = await judge.evaluate(
        case=case,
        response_text=response.text,
        tool_calls=response.tool_calls,
    )

    logger.info("[%s] Response: %s", case_id, response.text)
    logger.info("[%s] Judge: passed=%s, score=%s", case_id, result.passed, result.score)
    logger.info("[%s] Reasoning: %s", case_id, result.reasoning)
    if result.criteria_scores:
        logger.info("[%s] Criteria: %s", case_id, result.criteria_scores)

    assert result.passed, f"{case_id} failed (score={result.score}): {result.reasoning}"


@pytest.mark.eval
class TestIdentityEvals:
    """Identity resolution evaluation tests."""

    @pytest.mark.asyncio
    async def test_identity_group_chat_recall(
        self,
        eval_memory_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Multi-user group chat: David's wife Sukhpreet recalled across users."""
        await _seed_group_chat(eval_memory_agent)

        session = _make_private_session(
            session_id="eval-private-recall",
            user_id="user-david",
            username="dcramer",
            display_name="David Cramer",
        )
        await _judge_case(
            eval_memory_agent,
            judge_llm,
            "identity_group_chat_recall",
            session,
            "user-david",
        )

    @pytest.mark.asyncio
    async def test_identity_relationship_term_recall(
        self,
        eval_memory_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """'Tell me about my wife' should resolve to Sukhpreet via relationship."""
        await _seed_group_chat(eval_memory_agent)

        session = _make_private_session(
            session_id="eval-private-wife",
            user_id="user-david",
            username="dcramer",
            display_name="David Cramer",
        )
        await _judge_case(
            eval_memory_agent,
            judge_llm,
            "identity_relationship_term_recall",
            session,
            "user-david",
        )

    @pytest.mark.asyncio
    async def test_identity_the_prefix_resolution(
        self,
        eval_memory_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """'The boss' should resolve via prefix stripping to the boss relationship."""
        agent = eval_memory_agent.agent

        # Seed: David says "my boss Marcus is intense about deadlines"
        session_david = _make_group_session(
            session_id="eval-group-boss",
            user_id="user-david",
            username="dcramer",
            display_name="David Cramer",
        )
        await agent.process_message(
            "My boss Marcus is intense about deadlines, we had a long sprint review today.",
            session=session_david,
            user_id="user-david",
        )
        await _drain_extraction_tasks()
        await _dump_state(eval_memory_agent)

        session = _make_private_session(
            session_id="eval-private-boss",
            user_id="user-david",
            username="dcramer",
            display_name="David Cramer",
        )
        await _judge_case(
            eval_memory_agent,
            judge_llm,
            "identity_the_prefix_resolution",
            session,
            "user-david",
        )

    @pytest.mark.asyncio
    async def test_identity_self_not_leaked(
        self,
        eval_memory_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Agent should not describe the user as a third-party known person."""
        await _seed_group_chat(eval_memory_agent)

        session = _make_private_session(
            session_id="eval-private-self",
            user_id="user-david",
            username="dcramer",
            display_name="David Cramer",
        )
        await _judge_case(
            eval_memory_agent,
            judge_llm,
            "identity_self_not_leaked",
            session,
            "user-david",
        )
