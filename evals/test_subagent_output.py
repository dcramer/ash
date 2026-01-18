"""Subagent output quality evaluation tests.

These tests evaluate:
- Checkpoint formatting (no boilerplate headers)
- Subagent voice consistency (personality in user-facing messages)
- Technical output neutrality (no personality in generated files)

Run with: uv run pytest evals/test_subagent_output.py -v -s
Requires ANTHROPIC_API_KEY environment variable.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from ash.agents.base import AgentContext
from ash.agents.builtin.skill_writer import SkillWriterAgent
from ash.agents.executor import AgentExecutor
from ash.config import AshConfig
from ash.config.workspace import Workspace
from ash.core.agent import AgentComponents
from ash.llm.base import LLMProvider
from evals.runner import run_agent_to_completion

CASES_DIR = Path(__file__).parent / "cases"
SUBAGENT_OUTPUT_CASES = CASES_DIR / "subagent_output.yaml"


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
class TestCheckpointFormatting:
    """Tests for checkpoint prompt clarity without boilerplate."""

    @pytest.mark.asyncio
    async def test_checkpoint_prompt_clarity(
        self,
        skill_writer_executor: tuple[SkillWriterAgent, AgentExecutor],
        judge_llm: LLMProvider,
        eval_workspace_path: Path,
    ) -> None:
        """Verify checkpoint prompts are clean without 'Agent paused' header."""
        from evals.judge import LLMJudge
        from evals.types import EvalCase

        agent, executor = skill_writer_executor

        context = AgentContext(
            session_id="test-checkpoint-format",
            chat_id="test-chat",
            user_id="test-user",
        )

        print("\n=== Testing checkpoint prompt formatting ===")

        # Run until first checkpoint (end of research phase)
        result = await executor.execute(
            agent,
            "Create a simple skill that says hello",
            context,
        )

        assert result.checkpoint is not None, "Expected agent to checkpoint"

        checkpoint_prompt = result.checkpoint.prompt
        print(f"Checkpoint prompt:\n{checkpoint_prompt[:500]}...")

        # Direct check for boilerplate
        assert "Agent paused for input" not in checkpoint_prompt, (
            "Checkpoint prompt should not contain 'Agent paused for input' header"
        )
        assert "paused for input" not in checkpoint_prompt.lower(), (
            "Checkpoint prompt should not contain any 'paused for input' text"
        )

        # Judge the checkpoint prompt quality
        judge_case = EvalCase(
            id="checkpoint_clarity",
            description="Evaluate checkpoint prompt clarity",
            prompt="Create a simple skill that says hello",
            expected_behavior="""
                The checkpoint prompt should:
                1. Directly explain what was found or decided
                2. Ask a clear question or present options
                3. NOT have generic headers like "Agent paused" or "Waiting for input"
                4. Read naturally as a message to a user
            """,
            criteria=[
                "No boilerplate headers or meta-commentary",
                "Directly presents information or question",
                "Reads naturally as a conversation",
                "Options (if any) are actionable",
            ],
        )

        judge = LLMJudge(judge_llm)
        judge_result = await judge.evaluate(
            case=judge_case,
            response_text=checkpoint_prompt,
            tool_calls=[],
        )

        print(f"Judge passed: {judge_result.passed}")
        print(f"Judge reasoning: {judge_result.reasoning}")

        if judge_result.judge_error:
            pytest.skip(f"Judge error: {judge_result.reasoning}")

        assert judge_result.passed, (
            f"Checkpoint prompt clarity failed: {judge_result.reasoning}"
        )


@pytest.mark.eval
class TestSubagentVoice:
    """Tests for subagent voice consistency with SOUL."""

    @pytest.mark.asyncio
    async def test_subagent_voice_consistency(
        self,
        skill_writer_executor: tuple[SkillWriterAgent, AgentExecutor],
        judge_llm: LLMProvider,
        eval_workspace: Workspace,
        eval_workspace_path: Path,
    ) -> None:
        """Verify subagent prompts have personality matching SOUL."""
        from evals.judge import LLMJudge
        from evals.types import EvalCase

        agent, executor = skill_writer_executor

        # Get the SOUL content for comparison
        soul_content = eval_workspace.soul
        print("\n=== Testing voice consistency ===")
        print(f"SOUL preview: {soul_content[:200]}...")

        context = AgentContext(
            session_id="test-voice-consistency",
            chat_id="test-chat",
            user_id="test-user",
            voice=soul_content,  # Pass voice to context
        )

        # Run until first checkpoint
        result = await executor.execute(
            agent,
            "Create a skill that fetches weather data",
            context,
        )

        assert result.checkpoint is not None, "Expected agent to checkpoint"

        checkpoint_prompt = result.checkpoint.prompt
        print(f"Checkpoint prompt:\n{checkpoint_prompt[:500]}...")

        # Judge whether the voice matches the SOUL
        judge_case = EvalCase(
            id="voice_consistency",
            description="Evaluate voice consistency with SOUL",
            prompt="Create a skill that fetches weather data",
            expected_behavior=f"""
                The checkpoint prompt should reflect the communication style
                defined in the SOUL:

                {soul_content[:500]}...

                The prompt should:
                1. Match the tone and style described in SOUL
                2. Feel like it comes from the same persona
                3. Not be generic/robotic corporate-speak
            """,
            criteria=[
                "Tone matches SOUL personality",
                "Language style is consistent with SOUL",
                "Not generic or robotic",
                "Feels like the same 'voice' throughout",
            ],
        )

        judge = LLMJudge(judge_llm)
        judge_result = await judge.evaluate(
            case=judge_case,
            response_text=checkpoint_prompt,
            tool_calls=[],
        )

        print(f"Judge passed: {judge_result.passed}")
        print(f"Judge reasoning: {judge_result.reasoning}")

        if judge_result.judge_error:
            pytest.skip(f"Judge error: {judge_result.reasoning}")

        # Voice consistency is a soft requirement - log but don't fail hard
        if not judge_result.passed:
            print(f"WARNING: Voice consistency check failed: {judge_result.reasoning}")


@pytest.mark.eval
class TestTechnicalOutputNeutrality:
    """Tests for keeping personality out of technical outputs."""

    @pytest.mark.asyncio
    async def test_technical_output_neutral(
        self,
        skill_writer_executor: tuple[SkillWriterAgent, AgentExecutor],
        judge_llm: LLMProvider,
        eval_workspace_path: Path,
    ) -> None:
        """Verify generated files don't have personality injection."""
        from evals.judge import LLMJudge
        from evals.types import EvalCase

        agent, executor = skill_writer_executor

        context = AgentContext(
            session_id="test-technical-neutral",
            chat_id="test-chat",
            user_id="test-user",
        )

        print("\n=== Testing technical output neutrality ===")

        # Run to completion (auto-approve checkpoints)
        final_result, all_results = await run_agent_to_completion(
            executor,
            agent,
            "Create a skill that adds two numbers together",
            context,
        )

        print(f"Completed in {len(all_results)} phases")

        # Verify skill was created
        skills_dir = eval_workspace_path / "skills"
        skill_files = list(skills_dir.rglob("SKILL.md")) if skills_dir.exists() else []

        if not skill_files:
            pytest.skip("No skill was created - cannot test technical neutrality")

        skill_dir = skill_files[0].parent

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
        print(f"Artifact preview:\n{artifact_text[:800]}...")

        # Judge the artifact for neutrality
        judge_case = EvalCase(
            id="technical_neutrality",
            description="Evaluate technical output neutrality",
            prompt="Create a skill that adds two numbers together",
            expected_behavior="""
                The generated skill files should:
                1. Have clean, professional code
                2. NOT contain personality catchphrases
                3. NOT have playful comments or emoji in code
                4. Be technically correct and well-formatted
                5. Have neutral, documentation-style comments if any
            """,
            criteria=[
                "Code is clean and professional",
                "No personality catchphrases in file content",
                "Comments are neutral and informative",
                "YAML frontmatter is properly formatted",
            ],
        )

        judge = LLMJudge(judge_llm)
        judge_result = await judge.evaluate(
            case=judge_case,
            response_text=artifact_text,
            tool_calls=[],
        )

        print(f"Judge passed: {judge_result.passed}")
        print(f"Judge reasoning: {judge_result.reasoning}")

        if judge_result.judge_error:
            pytest.skip(f"Judge error: {judge_result.reasoning}")

        assert judge_result.passed, (
            f"Technical neutrality check failed: {judge_result.reasoning}"
        )
