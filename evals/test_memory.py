"""Memory recall evaluation tests.

Tests end-to-end memory flow: facts stated in one session are recalled
in a separate session via semantic search.

Run with: uv run pytest evals/test_memory.py -v -s -m eval
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
from evals.runner import load_eval_suite
from evals.types import EvalConfig

logger = logging.getLogger(__name__)

CASES_DIR = Path(__file__).parent / "cases"
MEMORY_CASES = CASES_DIR / "memory.yaml"


@pytest.mark.eval
class TestMemoryEvals:
    """Memory recall evaluation tests."""

    @pytest.mark.asyncio
    async def test_memory_recall(
        self,
        eval_memory_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Facts stated in session 1 are recalled in session 2."""
        agent = eval_memory_agent.agent

        # --- Session 1: State personal facts ---
        session1 = SessionState(
            session_id="eval-memory-ingestion",
            provider="eval",
            chat_id="eval-chat-1",
            user_id="eval-user",
        )
        await agent.process_message(
            "My favorite programming language is Rust and I live in Portland, Oregon.",
            session=session1,
            user_id="eval-user",
        )

        # Wait for background memory extraction to complete
        await asyncio.sleep(0)
        for task in asyncio.all_tasks():
            if task.get_name() == "memory_extraction":
                await task

        # Dump extracted state for debugging
        if eval_memory_agent.memory_manager:
            store = eval_memory_agent.memory_manager
            memories = await store.get_all_memories()
            logger.info("=== Extracted memories (%d) ===", len(memories))
            for m in memories:
                logger.info(
                    "  [%s] %s (type=%s, owner=%s)",
                    m.id[:8],
                    m.content[:80],
                    m.memory_type.value,
                    m.owner_user_id,
                )

        # --- Session 2: Recall in a fresh session ---
        session2 = SessionState(
            session_id="eval-memory-recall",
            provider="eval",
            chat_id="eval-chat-2",
            user_id="eval-user",
        )

        suite = load_eval_suite(MEMORY_CASES)
        recall_case = suite.cases[0]

        response = await agent.process_message(
            recall_case.prompt,
            session=session2,
            user_id="eval-user",
        )

        # Judge the recall
        judge = LLMJudge(judge_llm, EvalConfig())
        result = await judge.evaluate(
            case=recall_case,
            response_text=response.text,
            tool_calls=response.tool_calls,
        )

        logger.info("Response: %s", response.text)
        logger.info("Judge: passed=%s, score=%s", result.passed, result.score)
        logger.info("Reasoning: %s", result.reasoning)
        if result.criteria_scores:
            logger.info("Criteria: %s", result.criteria_scores)

        assert result.passed, (
            f"Memory recall failed (score={result.score}): {result.reasoning}"
        )
