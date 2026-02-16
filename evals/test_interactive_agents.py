"""Interactive subagent stack evaluation tests.

Tests the end-to-end flow: main agent delegates to a skill via use_skill,
ChildActivated raises, execute_turn drives the child, child calls complete,
and the result cascades back to the parent.

Run with: uv run pytest evals/test_interactive_agents.py -v -s -m eval
Requires ANTHROPIC_API_KEY environment variable.
"""

import pytest

from ash.agents.types import (
    AgentStack,
    ChildActivated,
    TurnAction,
)
from ash.core.agent import AgentComponents
from ash.core.session import SessionState
from ash.skills.types import SkillDefinition


@pytest.fixture
def eval_agent_with_test_skill(
    eval_agent: AgentComponents,
) -> AgentComponents:
    """Register a simple test skill on the eval agent's skill registry."""
    skill = SkillDefinition(
        name="quick-calc",
        description="Performs simple calculations and returns results",
        instructions=(
            "You are a calculator assistant.\n"
            "The user will ask you a math question.\n"
            "Compute the answer and call the `complete` tool with the result.\n"
            "Do NOT use any other tools. Just compute the answer mentally and complete.\n"
        ),
        allowed_tools=["complete", "send_message"],
        max_iterations=3,
    )
    eval_agent.skill_registry.register(skill)
    return eval_agent


@pytest.mark.eval
class TestInteractiveSubagentStack:
    """End-to-end test of the interactive subagent stack."""

    @pytest.mark.asyncio
    async def test_skill_delegation_and_completion(
        self,
        eval_agent_with_test_skill: AgentComponents,
    ) -> None:
        """Main agent delegates to a skill, child completes, result cascades back."""
        components = eval_agent_with_test_skill
        agent = components.agent
        executor = components.agent_executor
        assert executor is not None, "AgentExecutor must be configured"

        session = SessionState(
            session_id="eval-interactive",
            provider="eval",
            chat_id="eval-chat",
            user_id="eval-user",
        )

        # Phase 1: Send a message that should trigger skill delegation.
        # The main agent should recognise quick-calc and call use_skill.
        prompt = (
            "Use the quick-calc skill to compute: what is 17 * 23? "
            "Report the result back to me."
        )

        with pytest.raises(ChildActivated) as exc_info:
            await agent.process_message(
                user_message=prompt,
                session=session,
                user_id="eval-user",
            )

        ca = exc_info.value
        assert ca.main_frame is not None, "ChildActivated must carry main_frame"
        assert ca.child_frame is not None, "ChildActivated must carry child_frame"
        assert ca.child_frame.agent_type == "skill"
        assert "quick-calc" in ca.child_frame.agent_name

        print(f"\nChild activated: {ca.child_frame.agent_name}")
        print(f"Main frame agent: {ca.main_frame.agent_name}")

        # Phase 2: Run the orchestration loop (simulating provider behaviour).
        stack = AgentStack()
        stack.push(ca.main_frame)
        stack.push(ca.child_frame)

        entry_user_message = None
        entry_tool_result = None
        final_text = None
        max_loop = 20  # safety limit

        for _ in range(max_loop):
            top = stack.top
            assert top is not None

            result = await executor.execute_turn(
                top,
                user_message=entry_user_message,
                tool_result=entry_tool_result,
            )
            entry_user_message = None
            entry_tool_result = None

            print(
                f"  [{top.agent_name}] action={result.action.name} text={result.text[:120] if result.text else ''!r}"
            )

            if result.action == TurnAction.SEND_TEXT:
                # If top is main agent, it produced a final response
                if top.agent_type == "main":
                    stack.pop()
                    final_text = result.text
                    break
                # Subagent wants user interaction — not expected for quick-calc
                # but handle gracefully
                final_text = result.text
                break

            elif result.action == TurnAction.COMPLETE:
                completed = stack.pop()
                print(f"  Completed: {completed.agent_name} → {result.text!r}")
                if stack.is_empty:
                    final_text = result.text
                    break
                # Inject result into parent's pending tool_use
                assert completed.parent_tool_use_id is not None
                entry_tool_result = (
                    completed.parent_tool_use_id,
                    result.text,
                    False,
                )

            elif result.action == TurnAction.CHILD_ACTIVATED:
                assert result.child_frame is not None
                stack.push(result.child_frame)

            elif result.action == TurnAction.MAX_ITERATIONS:
                failed = stack.pop()
                if stack.is_empty:
                    pytest.fail(f"Agent {failed.agent_name} hit max iterations")
                entry_tool_result = (
                    failed.parent_tool_use_id,
                    "Agent reached maximum iterations.",
                    True,
                )

            elif result.action == TurnAction.ERROR:
                pytest.fail(f"Agent error: {result.text}")

        assert stack.is_empty, f"Stack should be empty, has {stack.depth} frames"
        assert final_text is not None, "Should have produced a final response"

        print(f"\nFinal response:\n{final_text}")

        # The response should contain the answer (17 * 23 = 391)
        assert "391" in final_text, (
            f"Expected '391' in final response but got: {final_text}"
        )
