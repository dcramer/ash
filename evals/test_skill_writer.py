"""Skill writer behavior evaluation tests.

End-to-end test that the skill-writer agent can research, plan, and
implement a skill with correct phase compliance and artifact quality.

Run with: uv run pytest evals/test_skill_writer.py -v -s
Requires ANTHROPIC_API_KEY environment variable.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from ash.agents.base import AgentContext
from ash.agents.builtin.skill_writer import SkillWriterAgent
from ash.agents.executor import AgentExecutor
from ash.config import AshConfig
from ash.core.agent import AgentComponents
from ash.llm.base import LLMProvider
from evals.judge import LLMJudge
from evals.runner import (
    extract_phase_tool_calls,
    run_agent_to_completion,
)
from evals.types import EvalCase

CASES_DIR = Path(__file__).parent / "cases"


@pytest.fixture
async def skill_writer_executor(
    eval_config: AshConfig,
    eval_agent: AgentComponents,
) -> AsyncGenerator[tuple[SkillWriterAgent, AgentExecutor], None]:
    """Create skill-writer agent with executor for multi-turn testing."""
    agent = SkillWriterAgent()
    executor = AgentExecutor(
        llm_provider=eval_agent.llm,
        tool_executor=eval_agent.tool_executor,
        config=eval_config,
    )
    yield agent, executor


@pytest.mark.eval
class TestSkillWriterE2E:
    """End-to-end test with phase compliance and artifact judging."""

    @pytest.mark.asyncio
    async def test_url_fetcher_skill(
        self,
        skill_writer_executor: tuple[SkillWriterAgent, AgentExecutor],
        judge_llm: LLMProvider,
        eval_workspace_path: Path,
    ) -> None:
        """Test creating a URL fetcher skill - verifiable Python-based skill.

        This is a specific task with verifiable output:
        - Input: "Create a skill that fetches a URL and returns the page title"
        - Expected: Python script using httpx/requests, extracts <title> tag

        Tests both phase compliance AND artifact quality.
        """
        agent, executor = skill_writer_executor

        context = AgentContext(
            session_id="test-url-fetcher",
            chat_id="test-chat",
            user_id="test-user",
        )

        print("\n=== Creating URL fetcher skill ===")
        final_result, all_results = await run_agent_to_completion(
            executor,
            agent,
            "Create a skill that fetches a URL and returns the page title",
            context,
        )

        print(f"Completed in {len(all_results)} phases")

        # Check phase compliance (research/plan should be read-only)
        phase_tools = extract_phase_tool_calls(all_results)
        for i, phase_name in enumerate(["research", "plan"]):
            if i < len(phase_tools):
                tools = {tc["name"] for tc in phase_tools[i]}
                assert "bash" not in tools, f"{phase_name} phase used bash"
                assert "write_file" not in tools, f"{phase_name} phase wrote files"
                print(f"{phase_name} phase tools: {tools}")

        # Verify skill was created
        skills_dir = eval_workspace_path / "skills"
        skill_files = list(skills_dir.rglob("SKILL.md")) if skills_dir.exists() else []
        assert skill_files, "No skill was created"

        skill_dir = skill_files[0].parent

        # Gather artifacts
        artifact_contents = []
        for f in skill_dir.rglob("*"):
            if f.is_file() and f.suffix in (".md", ".py", ".sh"):
                try:
                    artifact_contents.append(f"=== {f.name} ===\n{f.read_text()}")
                except UnicodeDecodeError:
                    continue

        artifact_text = "\n\n".join(artifact_contents)
        print(f"Created skill in: {skill_dir.name}")
        print(f"Artifact:\n{artifact_text[:1500]}...")

        # Judge the artifact
        print("\n=== LLM Judge evaluation ===")
        judge_case = EvalCase(
            id="url_fetcher_artifact",
            description="Evaluate URL fetcher skill artifact",
            prompt="Create a skill that fetches a URL and returns the page title",
            expected_behavior="""
                Given "https://example.com", the skill should return "Example Domain".

                The artifact must:
                1. Have SKILL.md with frontmatter (description)
                2. Have a Python script with PEP 723 deps (httpx or requests)
                3. Script should fetch URL and extract <title> tag
                4. Instructions should have a placeholder for user input
                   (e.g., <user_message>, <url>, or similar)
            """,
            criteria=[
                "Has Python script with HTTP library (httpx/requests/urllib)",
                "Script extracts title from HTML (<title> tag)",
                "Uses PEP 723 inline dependencies if external libs needed",
                "Instructions have placeholder for user input in bash command",
            ],
        )

        judge = LLMJudge(judge_llm)
        judge_result = await judge.evaluate(
            case=judge_case,
            response_text=artifact_text,
            tool_calls=[],
        )

        print(f"Judge passed: {judge_result.passed}")
        print(f"Judge score: {judge_result.score}")
        print(f"Judge reasoning: {judge_result.reasoning}")
        if judge_result.criteria_scores:
            print(f"Criteria: {judge_result.criteria_scores}")

        if judge_result.judge_error:
            pytest.skip(f"Judge error: {judge_result.reasoning}")

        assert judge_result.passed, (
            f"URL fetcher skill failed: {judge_result.reasoning}"
        )
