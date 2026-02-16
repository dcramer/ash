"""Skill writer behavior evaluation tests.

End-to-end test that the skill-writer bundled skill can create
a skill with correct SKILL.md format and artifact quality.

Run with: uv run pytest evals/test_skill_writer.py -v -s -m eval
Requires ANTHROPIC_API_KEY environment variable.
"""

from pathlib import Path

import pytest

from ash.agents.base import AgentContext
from ash.agents.executor import AgentExecutor
from ash.config import AshConfig
from ash.core.agent import AgentComponents
from ash.llm.base import LLMProvider
from ash.skills import SkillRegistry
from ash.tools.builtin.skills import SkillAgent
from evals.judge import LLMJudge
from evals.runner import run_agent_to_completion
from evals.types import EvalCase


@pytest.fixture
def skill_writer_agent(eval_agent: AgentComponents) -> SkillAgent:
    """Get the skill-writer bundled skill as a SkillAgent."""
    registry: SkillRegistry = eval_agent.skill_registry
    skill = registry.get("skill-writer")
    return SkillAgent(skill)


@pytest.mark.eval
class TestSkillWriterE2E:
    """End-to-end test with artifact quality judging."""

    @pytest.mark.asyncio
    async def test_url_fetcher_skill(
        self,
        skill_writer_agent: SkillAgent,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
        eval_workspace_path: Path,
        eval_config: AshConfig,
    ) -> None:
        """Test creating a URL fetcher skill - verifiable Python-based skill."""
        executor = AgentExecutor(
            llm_provider=eval_agent.llm,
            tool_executor=eval_agent.tool_executor,
            config=eval_config,
        )

        context = AgentContext(
            session_id="test-url-fetcher",
            chat_id="test-chat",
            user_id="test-user",
        )

        print("\n=== Creating URL fetcher skill ===")
        final_result, all_results = await run_agent_to_completion(
            executor,
            skill_writer_agent,
            "Create a skill that fetches a URL and returns the page title",
            context,
        )

        print(f"Completed in {len(all_results)} phases")

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
                1. Have SKILL.md with frontmatter (description, authors, rationale)
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
