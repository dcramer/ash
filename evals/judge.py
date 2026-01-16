"""LLM-as-judge implementation for evaluating agent responses."""

import json
import logging
from typing import Any

from ash.llm.base import LLMProvider
from ash.llm.types import Message, Role
from evals.types import EvalCase, JudgeResult

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing whether an AI assistant's response meets the expected behavior.

You will be given:
1. The user's prompt
2. The expected behavior description
3. Specific criteria to evaluate (if any)
4. The assistant's response text
5. The tools the assistant called (if any)

Your task is to evaluate whether the response satisfies the expected behavior and criteria.

Respond with a JSON object containing:
- "passed": boolean - true if the response adequately meets expectations
- "score": number between 0.0 and 1.0 - overall quality score
- "reasoning": string - brief explanation of your judgment
- "criteria_scores": object - scores (0.0-1.0) for each specific criterion if provided

Be fair but thorough. A response passes if it demonstrates the expected behavior, even if not perfect.
Focus on whether the assistant understood the request and took appropriate action.

IMPORTANT: Respond ONLY with valid JSON, no other text."""


async def judge_response(
    llm: LLMProvider,
    case: EvalCase,
    response_text: str,
    tool_calls: list[dict[str, Any]],
    *,
    model: str = "claude-sonnet-4-5",
) -> JudgeResult:
    """Judge an agent's response against the expected behavior.

    Args:
        llm: LLM provider to use for judging.
        case: The evaluation case being judged.
        response_text: The text response from the agent.
        tool_calls: List of tool calls made by the agent.
        model: Model to use for judging (default: Haiku for cost efficiency).

    Returns:
        JudgeResult with pass/fail status, score, and reasoning.
    """
    # Format tool calls for the judge
    tools_summary = ""
    if tool_calls:
        tools_list = []
        for tc in tool_calls:
            tool_desc = f"- {tc['name']}"
            if tc.get("input"):
                tool_desc += f": {json.dumps(tc['input'], default=str)[:200]}"
            if tc.get("is_error"):
                tool_desc += " [ERROR]"
            tools_list.append(tool_desc)
        tools_summary = "\n".join(tools_list)
    else:
        tools_summary = "(no tools called)"

    # Format criteria
    criteria_text = ""
    if case.criteria:
        criteria_text = "\n".join(f"- {c}" for c in case.criteria)
    else:
        criteria_text = "(no specific criteria)"

    # Format expected tools
    expected_tools_text = ""
    if case.expected_tools:
        expected_tools_text = (
            f"\n\nExpected tools to be called: {', '.join(case.expected_tools)}"
        )

    prompt = f"""## User Prompt
{case.prompt}

## Expected Behavior
{case.expected_behavior}

## Specific Criteria
{criteria_text}
{expected_tools_text}

## Assistant's Response
{response_text or "(no text response)"}

## Tools Called
{tools_summary}

Evaluate whether this response meets the expected behavior. Respond with JSON only."""

    try:
        response = await llm.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            model=model,
            system=JUDGE_SYSTEM_PROMPT,
            temperature=0,  # Deterministic judging
            max_tokens=1024,
        )

        result_text = response.message.get_text().strip()

        # Try to extract JSON from the response
        # Sometimes the model might wrap it in markdown code blocks
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            # Remove first and last lines (```json and ```)
            result_text = "\n".join(lines[1:-1])

        result_data = json.loads(result_text)

        return JudgeResult(
            passed=result_data.get("passed", False),
            score=float(result_data.get("score", 0.0)),
            reasoning=result_data.get("reasoning", "No reasoning provided"),
            criteria_scores=result_data.get("criteria_scores", {}),
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse judge response as JSON: {e}")
        return JudgeResult(
            passed=False,
            score=0.0,
            reasoning=f"Judge response parsing failed: {e}",
            criteria_scores={},
        )
    except Exception as e:
        logger.error(f"Judge evaluation failed: {e}")
        return JudgeResult(
            passed=False,
            score=0.0,
            reasoning=f"Judge evaluation error: {e}",
            criteria_scores={},
        )
