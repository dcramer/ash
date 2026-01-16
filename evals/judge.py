"""LLM-as-judge implementation for evaluating agent responses."""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from ash.llm.base import LLMProvider
from ash.llm.types import Message, Role
from evals.types import EvalCase, EvalConfig, JudgeResult

logger = logging.getLogger(__name__)

# Improved judge prompt with concrete rubric
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing whether an AI assistant's response meets expected behavior.

## Input Format
You will receive:
1. User's original prompt
2. Expected behavior description
3. Specific evaluation criteria (if any)
4. Expected tools to be called (if any)
5. Assistant's response text
6. Tools the assistant actually called

## Evaluation Rubric

### Pass/Fail Determination
A response PASSES if ALL of the following are true:
- The assistant understood the user's intent correctly
- The response directly addresses the request
- Any required tools were called (if expected_tools specified)
- No critical errors occurred during tool execution
- The response does not contain fabricated information

A response FAILS if ANY of the following are true:
- The assistant misunderstood the request
- Required tools were not called when expected
- Tool calls resulted in errors that weren't handled
- The response is evasive or doesn't address the request
- The assistant hallucinated information

### Scoring Guidelines (0.0-1.0)
- 1.0: Perfect response, all criteria fully met
- 0.8-0.9: Good response, minor issues or room for improvement
- 0.6-0.7: Acceptable response, some criteria partially met
- 0.4-0.5: Marginal response, significant gaps
- 0.2-0.3: Poor response, major issues
- 0.0-0.1: Failed response, doesn't meet requirements

### Criteria Evaluation
For each specific criterion, score independently:
- 1.0: Criterion fully satisfied
- 0.5: Criterion partially satisfied
- 0.0: Criterion not satisfied

## Response Format
Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):
{
  "passed": boolean,
  "score": number (0.0-1.0),
  "reasoning": "Brief explanation of judgment with specific evidence",
  "criteria_scores": {"criterion_name": score, ...}
}"""


class Judge(ABC):
    """Abstract base class for judges."""

    @abstractmethod
    async def evaluate(
        self,
        case: EvalCase,
        response_text: str,
        tool_calls: list[dict[str, Any]],
    ) -> JudgeResult:
        """Evaluate an agent's response.

        Args:
            case: The evaluation case.
            response_text: The text response from the agent.
            tool_calls: List of tool calls made by the agent.

        Returns:
            JudgeResult with pass/fail status, score, and reasoning.
        """
        ...


class LLMJudge(Judge):
    """LLM-based judge implementation."""

    def __init__(
        self,
        llm: LLMProvider,
        config: EvalConfig | None = None,
    ):
        """Initialize the LLM judge.

        Args:
            llm: LLM provider to use for judging.
            config: Eval configuration (uses defaults if not provided).
        """
        self.llm = llm
        self.config = config or EvalConfig()

    async def evaluate(
        self,
        case: EvalCase,
        response_text: str,
        tool_calls: list[dict[str, Any]],
    ) -> JudgeResult:
        """Evaluate with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.config.retry_attempts):
            try:
                return await self._evaluate_once(case, response_text, tool_calls)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    f"Judge JSON parse failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e}"
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Judge evaluation failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e}"
                )

            # Exponential backoff between retries
            if attempt < self.config.retry_attempts - 1:
                delay = self.config.retry_base_delay * (2**attempt)
                await asyncio.sleep(delay)

        # All retries exhausted - return error result
        error_type = (
            "parse_error"
            if isinstance(last_error, json.JSONDecodeError)
            else "api_error"
        )
        return JudgeResult(
            passed=False,
            score=0.0,
            reasoning=f"Judge failed after {self.config.retry_attempts} attempts: {last_error}",
            criteria_scores={},
            judge_error=True,
            error_type=error_type,
        )

    async def _evaluate_once(
        self,
        case: EvalCase,
        response_text: str,
        tool_calls: list[dict[str, Any]],
    ) -> JudgeResult:
        """Single evaluation attempt."""
        prompt = self._build_prompt(case, response_text, tool_calls)

        response = await self.llm.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            model=self.config.judge_model,
            system=JUDGE_SYSTEM_PROMPT,
            temperature=self.config.judge_temperature,
            max_tokens=self.config.judge_max_tokens,
        )

        result_text = response.message.get_text().strip()
        result_data = self._parse_json_response(result_text)

        return JudgeResult(
            passed=result_data.get("passed", False),
            score=float(result_data.get("score", 0.0)),
            reasoning=result_data.get("reasoning", "No reasoning provided"),
            criteria_scores=result_data.get("criteria_scores", {}),
        )

    def _build_prompt(
        self,
        case: EvalCase,
        response_text: str,
        tool_calls: list[dict[str, Any]],
    ) -> str:
        """Build the evaluation prompt."""
        # Format tool calls
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

        return f"""## User Prompt
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

Evaluate whether this response meets the expected behavior."""

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM response with robust handling.

        Handles various formats:
        - Plain JSON
        - JSON wrapped in ```json ... ```
        - JSON wrapped in ``` ... ```
        - JSON with leading/trailing text
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        # Pattern handles ```json, ```JSON, or just ```
        code_block_pattern = r"```(?:json|JSON)?\s*\n?(.*?)\n?```"
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in the text
        # Look for content between first { and last }
        brace_pattern = r"\{.*\}"
        match = re.search(brace_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # All parsing attempts failed
        raise json.JSONDecodeError(
            "Could not extract valid JSON from response", text, 0
        )


# Backwards-compatible function interface
async def judge_response(
    llm: LLMProvider,
    case: EvalCase,
    response_text: str,
    tool_calls: list[dict[str, Any]],
    *,
    model: str = "claude-sonnet-4-5",
    config: EvalConfig | None = None,
) -> JudgeResult:
    """Judge an agent's response against the expected behavior.

    This is a convenience wrapper around LLMJudge for backwards compatibility.

    Args:
        llm: LLM provider to use for judging.
        case: The evaluation case being judged.
        response_text: The text response from the agent.
        tool_calls: List of tool calls made by the agent.
        model: Model to use for judging (overrides config if provided).
        config: Eval configuration.

    Returns:
        JudgeResult with pass/fail status, score, and reasoning.
    """
    if config is None:
        config = EvalConfig(judge_model=model)
    elif model != "claude-sonnet-4-5":  # Only override if explicitly set
        config = EvalConfig(
            judge_model=model,
            judge_temperature=config.judge_temperature,
            judge_max_tokens=config.judge_max_tokens,
            retry_attempts=config.retry_attempts,
            retry_base_delay=config.retry_base_delay,
        )

    judge = LLMJudge(llm, config)
    return await judge.evaluate(case, response_text, tool_calls)
