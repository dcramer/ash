"""Skill writer behavior evaluation tests.

These tests evaluate the skill-writer agent's behavior across phases:
- Research phase: Must be read-only (no bash, no file writes)
- Plan phase: Must be read-only (no bash, no file writes)
- Implementation phase: Can write files and execute code

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
from evals.runner import (
    check_phase_constraints,
    extract_phase_tool_calls,
    get_case_by_id,
    load_eval_suite,
    run_agent_to_completion,
)

CASES_DIR = Path(__file__).parent / "cases"
SKILL_WRITER_CASES = CASES_DIR / "skill_writer.yaml"


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
class TestSkillWriterPhaseCompliance:
    """Phase compliance tests - research/plan must be read-only."""

    @pytest.mark.asyncio
    async def test_research_phase_read_only(
        self,
        skill_writer_executor: tuple[SkillWriterAgent, AgentExecutor],
        eval_workspace_path: Path,
    ) -> None:
        """Research phase should not use bash or write files."""
        agent, executor = skill_writer_executor
        suite = load_eval_suite(SKILL_WRITER_CASES)
        case = get_case_by_id(suite, "phases_are_read_only")

        context = AgentContext(
            session_id="test-research",
            chat_id="test-chat",
            user_id="test-user",
        )

        # Run until first checkpoint (end of research phase)
        result = await executor.execute(agent, case.prompt, context)

        # Extract tool calls from the checkpoint session
        if result.checkpoint:
            from evals.runner import extract_tool_calls_from_session

            tool_calls = extract_tool_calls_from_session(result.checkpoint.session_json)
            tool_names = {tc["name"] for tc in tool_calls}
        else:
            tool_names = set()

        # Should have used web_search, web_fetch, or delegated to research agent
        research_tools = {"web_search", "web_fetch", "use_agent"}
        assert tool_names & research_tools, (
            f"Research phase should use web_search, web_fetch, or use_agent. "
            f"Tools used: {tool_names}"
        )

        # Should NOT have used bash or write_file
        assert "bash" not in tool_names, "Research phase used bash"
        assert "write_file" not in tool_names, "Research phase wrote files"

        # Should have checkpointed
        assert result.checkpoint is not None, "Research phase should checkpoint"

        print(f"\nResearch phase tools: {tool_names}")
        print(f"Checkpoint prompt: {result.checkpoint.prompt[:200]}...")

    @pytest.mark.asyncio
    async def test_plan_phase_read_only(
        self,
        skill_writer_executor: tuple[SkillWriterAgent, AgentExecutor],
        eval_workspace_path: Path,
    ) -> None:
        """Plan phase should not use bash or write files."""
        agent, executor = skill_writer_executor
        suite = load_eval_suite(SKILL_WRITER_CASES)
        case = get_case_by_id(suite, "phases_are_read_only")

        context = AgentContext(
            session_id="test-plan",
            chat_id="test-chat",
            user_id="test-user",
        )

        # Run to completion (auto-approve checkpoints)
        final_result, all_results = await run_agent_to_completion(
            executor, agent, case.prompt, context
        )

        # Extract tools per phase
        phase_tools = extract_phase_tool_calls(all_results)

        # Check phase constraints
        violations = check_phase_constraints(case, phase_tools)
        if violations:
            for phase_idx, msg in violations:
                print(f"Phase {phase_idx} violation: {msg}")

        # Phase 0 = research, Phase 1 = plan
        if len(phase_tools) >= 2:
            plan_tools = phase_tools[1]
            plan_tool_names = {tc["name"] for tc in plan_tools}

            assert "bash" not in plan_tool_names, "Plan phase used bash"
            assert "write_file" not in plan_tool_names, "Plan phase wrote files"

            print(f"\nPlan phase tools: {plan_tool_names}")

        assert not violations, f"Phase constraint violations: {violations}"


@pytest.mark.eval
class TestSkillWriterQuality:
    """Skill quality tests - created skills should produce correct output."""

    @pytest.mark.asyncio
    async def test_word_counter_skill(
        self,
        skill_writer_executor: tuple[SkillWriterAgent, AgentExecutor],
        judge_llm: LLMProvider,
        eval_workspace_path: Path,
    ) -> None:
        """Test creating a word-counter skill and judging the artifact.

        This is a specific, verifiable task:
        - Input: "Create a skill that counts words in text"
        - Expected output: A skill that, when given text, returns a word count

        The judge evaluates whether the created artifact would correctly
        count words when invoked.
        """
        from evals.judge import LLMJudge
        from evals.types import EvalCase

        agent, executor = skill_writer_executor

        context = AgentContext(
            session_id="test-word-counter",
            chat_id="test-chat",
            user_id="test-user",
        )

        print("\n=== Creating word-counter skill ===")
        final_result, all_results = await run_agent_to_completion(
            executor,
            agent,
            "Create a skill that counts words in text",
            context,
        )

        print(f"Completed in {len(all_results)} phases")

        # Verify skill was created
        skills_dir = eval_workspace_path / "skills"
        skill_files = list(skills_dir.rglob("SKILL.md")) if skills_dir.exists() else []
        assert skill_files, "No skill was created"

        skill_path = skill_files[0]
        skill_dir = skill_path.parent

        # Gather all skill artifacts
        artifact_contents = []
        for f in skill_dir.rglob("*"):
            if f.is_file() and f.suffix in (".md", ".py", ".sh"):
                try:
                    artifact_contents.append(f"=== {f.name} ===\n{f.read_text()}")
                except UnicodeDecodeError:
                    continue

        artifact_text = "\n\n".join(artifact_contents)
        print(f"Created skill in: {skill_dir.name}")
        print(f"Artifact:\n{artifact_text[:1000]}...")

        # Judge the artifact
        print("\n=== LLM Judge evaluation ===")
        judge_case = EvalCase(
            id="word_counter_artifact",
            description="Evaluate word-counter skill artifact",
            prompt="Create a skill that counts words in text",
            expected_behavior="""
                Given input "hello world foo bar", the skill should return 4.
                Given input "The quick brown fox", the skill should return 4.

                The artifact must:
                1. Have SKILL.md with frontmatter containing description
                2. Have imperative instructions (not passive docs)
                3. Use <user_message> placeholder for the input text
                4. If Python: script should split text and count words
            """,
            criteria=[
                "Would return correct word count for 'hello world foo bar' (4 words)",
                "SKILL.md has valid frontmatter with description",
                "Instructions reference user input via placeholder",
                "Logic correctly counts words (splits on whitespace)",
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
            f"Word-counter skill failed: {judge_result.reasoning}"
        )


@pytest.mark.eval
class TestSkillWriterE2E:
    """End-to-end tests with phase compliance and artifact judging."""

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
        from evals.judge import LLMJudge
        from evals.types import EvalCase

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
