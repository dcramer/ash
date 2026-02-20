"""Eval execution utilities."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ash.core.agent import Agent
from ash.core.session import SessionState
from ash.llm.base import LLMProvider
from evals.judge import LLMJudge, check_forbidden_tools
from evals.types import (
    Assertions,
    EvalCase,
    EvalConfig,
    EvalSuite,
    JudgeResult,
    SessionConfig,
    SetupStep,
    SuiteDefaults,
)

if TYPE_CHECKING:
    from ash.agents.base import Agent as SubAgent
    from ash.agents.base import AgentContext, AgentResult
    from ash.agents.executor import AgentExecutor
    from ash.core.types import AgentComponents

logger = logging.getLogger(__name__)


def load_eval_suite(path: Path) -> EvalSuite:
    """Load an eval suite from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed EvalSuite.
    """
    with path.open() as f:
        data = yaml.safe_load(f)

    return EvalSuite.model_validate(data)


def discover_eval_suites(cases_dir: Path | None = None) -> list[Path]:
    """Auto-discover eval suite YAML files.

    Args:
        cases_dir: Directory to search (defaults to evals/cases).

    Returns:
        List of paths to YAML suite files.
    """
    if cases_dir is None:
        cases_dir = Path(__file__).parent / "cases"

    if not cases_dir.exists():
        logger.warning(f"Cases directory not found: {cases_dir}")
        return []

    suites = list(cases_dir.glob("*.yaml")) + list(cases_dir.glob("*.yml"))
    return sorted(suites)


def get_case_by_id(suite: EvalSuite, case_id: str) -> EvalCase:
    """Get a specific case from a suite by ID.

    Args:
        suite: The eval suite.
        case_id: ID of the case to find.

    Returns:
        The matching EvalCase.

    Raises:
        ValueError: If no case with the given ID exists.
    """
    for case in suite.cases:
        if case.id == case_id:
            return case
    raise ValueError(f"Case '{case_id}' not found in suite '{suite.name}'")


@dataclass
class EvalResult:
    """Result of running a single eval case."""

    case: EvalCase
    response_text: str
    tool_calls: list[dict[str, Any]]
    judge_result: JudgeResult
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Whether the eval passed (excludes judge errors)."""
        return self.judge_result.passed and self.error is None

    @property
    def score(self) -> float:
        """Score from the judge."""
        return self.judge_result.score if self.error is None else 0.0

    @property
    def is_judge_error(self) -> bool:
        """Whether this result is due to a judge error, not an actual failure."""
        return self.judge_result.judge_error


@dataclass
class EvalReport:
    """Report from running an eval suite."""

    suite_name: str
    config: EvalConfig = field(default_factory=EvalConfig)
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of cases."""
        return len(self.results)

    @property
    def valid_results(self) -> list[EvalResult]:
        """Results excluding judge errors (for accurate metrics)."""
        return [r for r in self.results if not r.is_judge_error]

    @property
    def passed(self) -> int:
        """Number of passed cases."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        """Number of failed cases (excluding judge errors)."""
        return sum(1 for r in self.valid_results if not r.passed)

    @property
    def judge_errors(self) -> int:
        """Number of cases that failed due to judge errors."""
        return len(self.results) - len(self.valid_results)

    @property
    def accuracy(self) -> float:
        """Accuracy as a fraction (0.0 to 1.0).

        Excludes judge errors from the calculation to give true accuracy.
        """
        valid = self.valid_results
        if not valid:
            return 0.0
        return sum(1 for r in valid if r.passed) / len(valid)

    @property
    def average_score(self) -> float:
        """Average score across all cases (excluding judge errors)."""
        valid = self.valid_results
        if not valid:
            return 0.0
        return sum(r.score for r in valid) / len(valid)

    def failed_cases(self) -> list[EvalResult]:
        """Get list of failed cases (excluding judge errors)."""
        return [r for r in self.valid_results if not r.passed]

    def judge_error_cases(self) -> list[EvalResult]:
        """Get list of cases that failed due to judge errors."""
        return [r for r in self.results if r.is_judge_error]


async def run_eval_case(
    agent: Agent,
    case: EvalCase,
    judge_llm: LLMProvider,
    *,
    session: SessionState | None = None,
    config: EvalConfig | None = None,
    judge_model: str | None = None,
) -> EvalResult:
    """Run a single eval case and judge the result.

    Args:
        agent: The agent to test.
        case: The eval case to run.
        judge_llm: LLM provider to use for judging.
        session: Optional session state (creates fresh one if not provided).
        config: Eval configuration (uses defaults if not provided).
        judge_model: Override judge model (deprecated, use config instead).

    Returns:
        EvalResult with the response and judgment.
    """
    from dataclasses import replace

    if config is None:
        config = EvalConfig()
    if judge_model is not None:
        config = replace(config, judge_model=judge_model)

    # Create fresh session if not provided
    if session is None:
        session = SessionState(
            session_id=f"eval-{case.id}",
            provider="eval",
            chat_id="eval-chat",
            user_id="eval-user",
        )

    try:
        # Run the agent
        response = await agent.send_message(
            user_message=case.prompt,
            session=session,
            user_id="eval-user",
        )

        # Pre-judge: Check forbidden tools deterministically
        forbidden_result = check_forbidden_tools(case, response.tool_calls)
        if forbidden_result:
            return EvalResult(
                case=case,
                response_text=response.text,
                tool_calls=response.tool_calls,
                judge_result=forbidden_result,
            )

        # Judge the response with LLM
        judge = LLMJudge(judge_llm, config)
        judge_result = await judge.evaluate(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
        )

        return EvalResult(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
            judge_result=judge_result,
        )

    except Exception as e:
        logger.error(f"Eval case {case.id} failed with error: {e}")
        return EvalResult(
            case=case,
            response_text="",
            tool_calls=[],
            judge_result=JudgeResult(
                passed=False,
                score=0.0,
                reasoning=f"Execution error: {e}",
                criteria_scores={},
            ),
            error=str(e),
        )


async def run_eval_suite(
    agent: Agent,
    suite: EvalSuite,
    judge_llm: LLMProvider,
    *,
    config: EvalConfig | None = None,
    judge_model: str | None = None,
) -> EvalReport:
    """Run all cases in an eval suite.

    Args:
        agent: The agent to test.
        suite: The eval suite to run.
        judge_llm: LLM provider to use for judging.
        config: Eval configuration.
        judge_model: Override judge model (deprecated, use config instead).

    Returns:
        EvalReport with all results.
    """
    from dataclasses import replace

    if config is None:
        config = EvalConfig()
    if judge_model is not None:
        config = replace(config, judge_model=judge_model)

    report = EvalReport(suite_name=suite.name, config=config)

    for case in suite.cases:
        logger.info(f"Running eval case: {case.id}")
        result = await run_eval_case(
            agent=agent,
            case=case,
            judge_llm=judge_llm,
            config=config,
        )
        report.results.append(result)

        if result.passed:
            status = "PASSED"
        elif result.is_judge_error:
            status = "JUDGE_ERROR"
        else:
            status = "FAILED"
        logger.info(f"Case {case.id}: {status} (score: {result.score:.2f})")

    # Log summary
    if report.judge_errors > 0:
        logger.warning(
            f"Suite '{suite.name}': {report.judge_errors} cases had judge errors"
        )

    return report


# --- v2.0 YAML-driven orchestration ---


def build_session_state(
    config: SessionConfig,
    defaults: SuiteDefaults,
    *,
    case_id: str = "eval",
) -> SessionState:
    """Build a SessionState by merging case config with suite defaults.

    Args:
        config: Per-case session configuration.
        defaults: Suite-level defaults.
        case_id: Case ID for generating fallback session_id.

    Returns:
        Configured SessionState.
    """
    default_session = defaults.session

    import uuid

    session_id = f"eval-{case_id}-{uuid.uuid4().hex[:8]}"
    provider = config.provider or default_session.provider or "eval"
    chat_id = config.chat_id or default_session.chat_id or "eval-chat"
    user_id = config.user_id or default_session.user_id or "eval-user"

    session = SessionState(
        session_id=session_id,
        provider=provider,
        chat_id=chat_id,
        user_id=user_id,
    )

    # Apply context fields
    username = config.username or default_session.username
    display_name = config.display_name or default_session.display_name
    chat_type = config.chat_type or default_session.chat_type
    chat_title = config.chat_title or default_session.chat_title

    if username:
        session.context.username = username
    if display_name:
        session.context.display_name = display_name
    if chat_type:
        session.context.chat_type = chat_type
    if chat_title:
        session.context.chat_title = chat_title

    return session


def _build_setup_session(
    step: SetupStep,
    defaults: SuiteDefaults,
) -> SessionState:
    """Build a SessionState from a setup step, merging with defaults."""
    config = SessionConfig(
        provider=step.provider,
        chat_id=step.chat_id,
        user_id=step.user_id,
        username=step.username,
        display_name=step.display_name,
        chat_type=step.chat_type,
        chat_title=step.chat_title,
    )
    return build_session_state(config, defaults, case_id="setup")


async def drain_extraction_tasks() -> None:
    """Wait for all background memory extraction tasks to complete."""
    await asyncio.sleep(0)
    for task in asyncio.all_tasks():
        if task.get_name() == "memory_extraction":
            await task


async def dump_state(components: "AgentComponents") -> None:
    """Log extracted memories and people for eval debugging."""
    if components.memory_manager:
        store = components.memory_manager
        memories = await store.get_all_memories()
        logger.info("=== Extracted memories (%d) ===", len(memories))
        for m in memories:
            from ash.graph.edges import get_learned_in_chat, get_subject_person_ids

            subjects = get_subject_person_ids(store.graph, m.id)
            learned_in = get_learned_in_chat(store.graph, m.id)
            learned_chat = store.graph.chats.get(learned_in) if learned_in else None
            learned_info = (
                f"learned_in={learned_chat.chat_type}({learned_chat.provider_id})"
                if learned_chat
                else "learned_in=None"
            )
            logger.info(
                "  [%s] %s (type=%s, owner=%s, subjects=%s, source=%s, %s)",
                m.id[:8],
                m.content[:80],
                m.memory_type.value,
                m.owner_user_id,
                subjects,
                m.source_username,
                learned_info,
            )

        # Log chat entries
        logger.info("=== Chat entries (%d) ===", len(store.graph.chats))
        for cid, chat in store.graph.chats.items():
            logger.info(
                "  [%s] provider=%s, provider_id=%s, chat_type=%s",
                cid[:8],
                chat.provider,
                chat.provider_id,
                chat.chat_type,
            )

    if components.person_manager:
        people = await components.person_manager.list_people()
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


async def run_setup_steps(
    components: "AgentComponents",
    steps: list[SetupStep],
    defaults: SuiteDefaults,
) -> None:
    """Execute setup steps: send seeding messages and optionally drain extraction.

    Args:
        components: Agent components to use.
        steps: Setup steps to execute.
        defaults: Suite defaults for session config.
    """
    agent = components.agent
    for step in steps:
        session = _build_setup_session(step, defaults)
        user_id = step.user_id or defaults.session.user_id or "eval-user"

        # Ensure ChatEntry graph node exists so LEARNED_IN edges can be created
        # during memory extraction (mirrors what the Telegram session handler does)
        if components.memory_manager and session.chat_id and session.provider:
            try:
                await components.memory_manager.ensure_chat(
                    provider=session.provider,
                    provider_id=session.chat_id,
                    chat_type=session.context.chat_type,
                    title=session.context.chat_title,
                )
            except Exception:
                logger.debug("eval_chat_upsert_failed", exc_info=True)

        for message in step.messages:
            await agent.send_message(
                user_message=message,
                session=session,
                user_id=user_id,
                agent_executor=components.agent_executor,
            )
        if step.drain_extraction:
            await drain_extraction_tasks()

    await dump_state(components)


async def check_structural_assertions(
    components: "AgentComponents",
    assertions: Assertions,
) -> list[str]:
    """Check structural assertions against the memory/people store.

    Args:
        components: Agent components with memory/person managers.
        assertions: Structural assertions to check.

    Returns:
        List of failure messages (empty if all passed).
    """
    failures: list[str] = []

    if assertions.memories and components.memory_manager:
        store = components.memory_manager
        memories = await store.get_all_memories()

        for ma in assertions.memories:
            # Find memories matching content_contains
            matching = memories
            for keyword in ma.content_contains:
                matching = [m for m in matching if keyword.lower() in m.content.lower()]

            if not matching:
                failures.append(
                    f"No memory found containing {ma.content_contains}. "
                    f"All memories: {[m.content for m in memories]}"
                )
                continue

            if ma.memory_type:
                typed = [m for m in matching if m.memory_type.value == ma.memory_type]
                if not typed:
                    failures.append(
                        f"Memory matching {ma.content_contains} found but type "
                        f"is {matching[0].memory_type.value}, expected {ma.memory_type}"
                    )

    if assertions.people:
        if not components.memory_manager:
            failures.append("No memory manager available for people assertions")
        else:
            people = await components.memory_manager.list_people()

            for pa in assertions.people:
                matching = [
                    p for p in people if pa.name_contains.lower() in p.name.lower()
                ]
                if not matching:
                    failures.append(
                        f"No person record found containing '{pa.name_contains}'. "
                        f"People: {[p.name for p in people]}"
                    )

    return failures


async def run_yaml_eval_case(
    components: "AgentComponents",
    suite: EvalSuite,
    case: EvalCase,
    judge_llm: LLMProvider,
    *,
    config: EvalConfig | None = None,
) -> list[EvalResult]:
    """Run a single YAML-defined eval case through the full orchestration.

    Handles setup, session building, single/multi-turn execution,
    extraction draining, structural assertions, and judging.

    Args:
        components: Agent components (from fixture).
        suite: The eval suite containing defaults and setup.
        case: The eval case to run.
        judge_llm: LLM provider for judging.
        config: Eval configuration.

    Returns:
        List of EvalResult (one per turn).
    """
    if config is None:
        config = EvalConfig()

    defaults = suite.defaults
    agent = components.agent

    # 1. Run setup steps
    if case.setup is not None:
        # Per-case setup replaces suite setup
        await run_setup_steps(components, case.setup, defaults)
    elif suite.setup and not case.skip_suite_setup:
        await run_setup_steps(components, suite.setup, defaults)

    # 2. Build eval session from merged config
    case_session_config = case.session or SessionConfig()
    session = build_session_state(case_session_config, defaults, case_id=case.id)

    # Ensure ChatEntry exists for the test case chat so RPC handlers can
    # resolve chat_type (needed for DM-source privacy filtering)
    if components.memory_manager and session.chat_id and session.provider:
        try:
            await components.memory_manager.ensure_chat(
                provider=session.provider,
                provider_id=session.chat_id,
                chat_type=session.context.chat_type,
                title=session.context.chat_title,
            )
        except Exception:
            logger.debug("eval_case_chat_upsert_failed", exc_info=True)

    # 3. Determine user_id for message sending
    user_id = case_session_config.user_id or defaults.session.user_id or "eval-user"

    # 4. Determine if we should drain extraction
    should_drain = defaults.drain_extraction

    results: list[EvalResult] = []

    if case.turns:
        # Multi-turn case
        for turn in case.turns:
            turn_case = EvalCase(
                id=f"{case.id}_turn{len(results) + 1}",
                description=case.description,
                prompt=turn.prompt,
                expected_behavior=turn.expected_behavior,
                criteria=turn.criteria,
            )

            result = await _execute_and_judge(
                agent,
                turn_case,
                session,
                user_id,
                judge_llm,
                config,
                agent_executor=components.agent_executor,
            )
            results.append(result)

        if should_drain:
            await drain_extraction_tasks()
    else:
        # Single-turn case
        result = await _execute_and_judge(
            agent,
            case,
            session,
            user_id,
            judge_llm,
            config,
            agent_executor=components.agent_executor,
        )
        results.append(result)

        if should_drain:
            await drain_extraction_tasks()

    # 5. Structural assertions
    if case.assertions:
        assertion_failures = await check_structural_assertions(
            components, case.assertions
        )
        if assertion_failures:
            # Create a failing result for structural assertion failures
            results.append(
                EvalResult(
                    case=EvalCase(
                        id=f"{case.id}_assertions",
                        description="Structural assertions",
                        prompt=case.prompt,
                    ),
                    response_text="",
                    tool_calls=[],
                    judge_result=JudgeResult(
                        passed=False,
                        score=0.0,
                        reasoning="Structural assertion failures: "
                        + "; ".join(assertion_failures),
                        criteria_scores={},
                    ),
                    error="; ".join(assertion_failures),
                )
            )

    return results


async def _execute_and_judge(
    agent: Agent,
    case: EvalCase,
    session: SessionState,
    user_id: str,
    judge_llm: LLMProvider,
    config: EvalConfig,
    *,
    agent_executor: "AgentExecutor | None" = None,
) -> EvalResult:
    """Execute a single prompt and judge the response."""
    try:
        response = await agent.send_message(
            user_message=case.prompt,
            session=session,
            user_id=user_id,
            agent_executor=agent_executor,
        )

        # Pre-judge: forbidden tools check
        forbidden_result = check_forbidden_tools(case, response.tool_calls)
        if forbidden_result:
            return EvalResult(
                case=case,
                response_text=response.text,
                tool_calls=response.tool_calls,
                judge_result=forbidden_result,
            )

        # LLM judge
        judge = LLMJudge(judge_llm, config)
        judge_result = await judge.evaluate(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
        )

        logger.info("[%s] Response: %s", case.id, response.text)
        logger.info(
            "[%s] Judge: passed=%s, score=%s",
            case.id,
            judge_result.passed,
            judge_result.score,
        )
        logger.info("[%s] Reasoning: %s", case.id, judge_result.reasoning)
        if judge_result.criteria_scores:
            logger.info("[%s] Criteria: %s", case.id, judge_result.criteria_scores)

        return EvalResult(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
            judge_result=judge_result,
        )

    except Exception as e:
        logger.error(f"Eval case {case.id} failed with error: {e}")
        return EvalResult(
            case=case,
            response_text="",
            tool_calls=[],
            judge_result=JudgeResult(
                passed=False,
                score=0.0,
                reasoning=f"Execution error: {e}",
                criteria_scores={},
            ),
            error=str(e),
        )


# Multi-turn evaluation support for agents with checkpoints


@dataclass
class MultiTurnEvalResult:
    """Result from running a multi-turn eval (agent with checkpoints)."""

    case: EvalCase
    final_result: "AgentResult"
    phase_results: list["AgentResult"]  # Results per phase (between checkpoints)
    phase_tool_calls: list[list[dict[str, Any]]]  # Tool calls per phase
    total_iterations: int

    @property
    def all_tool_calls(self) -> list[dict[str, Any]]:
        """Flatten all tool calls across phases."""
        return [tc for phase in self.phase_tool_calls for tc in phase]


async def run_agent_to_completion(
    executor: "AgentExecutor",
    agent: "SubAgent",
    input_message: str,
    context: "AgentContext",
    *,
    max_checkpoints: int = 10,
    auto_approve: str = "Proceed",
) -> tuple["AgentResult", list["AgentResult"]]:
    """Run an agent through all checkpoints to completion.

    Automatically approves each checkpoint with the specified response,
    allowing multi-phase agents (like skill-writer) to run to completion.

    Args:
        executor: The agent executor.
        agent: The agent to run.
        input_message: Initial user message/task.
        context: Execution context.
        max_checkpoints: Maximum checkpoints before giving up.
        auto_approve: Response to send at each checkpoint.

    Returns:
        Tuple of (final_result, all_intermediate_results_including_final)
    """
    results: list[AgentResult] = []
    result = await executor.execute(agent, input_message, context)
    results.append(result)

    checkpoint_count = 0
    while result.checkpoint and checkpoint_count < max_checkpoints:
        result = await executor.execute(
            agent,
            input_message,
            context,
            resume_from=result.checkpoint,
            user_response=auto_approve,
        )
        results.append(result)
        checkpoint_count += 1

    return result, results


def extract_tool_calls_from_session(session_json: str) -> list[dict[str, Any]]:
    """Extract tool calls from a serialized session.

    Parses the session JSON and extracts all tool_use blocks from
    assistant messages.

    Args:
        session_json: JSON serialized SessionState.

    Returns:
        List of tool call dicts with 'name' and 'input' keys.
    """
    import json

    data = json.loads(session_json)
    tool_calls: list[dict[str, Any]] = []

    for message in data.get("messages", []):
        if message.get("role") != "assistant":
            continue

        content = message.get("content", [])
        if isinstance(content, str):
            continue

        for block in content:
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                        "id": block.get("id", ""),
                    }
                )

    return tool_calls


def extract_phase_tool_calls(
    results: list["AgentResult"],
) -> list[list[dict[str, Any]]]:
    """Extract tool calls for each phase from multi-turn results.

    Each phase ends with an interrupt checkpoint. Returns list of tool call
    lists, one per phase.

    Args:
        results: List of AgentResults from run_agent_to_completion.

    Returns:
        List of tool call lists, one per phase.
    """
    phases: list[list[dict[str, Any]]] = []
    prev_tool_count = 0

    for result in results:
        # Get tool calls from checkpoint session if available
        if result.checkpoint:
            all_calls = extract_tool_calls_from_session(result.checkpoint.session_json)
            # Extract only the new calls since last phase
            phase_calls = all_calls[prev_tool_count:]
            phases.append(phase_calls)
            prev_tool_count = len(all_calls)
        elif not result.is_error:
            # Final result (no checkpoint) - no session to extract from
            # The final phase's tool calls are harder to get without checkpoint
            phases.append([])

    return phases


def check_phase_constraints(
    case: EvalCase,
    phase_tool_calls: list[list[dict[str, Any]]],
) -> list[tuple[int, str]]:
    """Check if phase constraints are violated.

    Args:
        case: The eval case with optional phase_constraints.
        phase_tool_calls: Tool calls per phase from extract_phase_tool_calls.

    Returns:
        List of (phase_index, violation_message) tuples.
    """
    violations: list[tuple[int, str]] = []
    if not case.phase_constraints:
        return violations

    phase_names = list(case.phase_constraints.keys())
    for i, phase_name in enumerate(phase_names):
        if i >= len(phase_tool_calls):
            break

        constraint = case.phase_constraints[phase_name]
        tools = phase_tool_calls[i]
        used = {tc["name"] for tc in tools}

        # Check forbidden tools
        forbidden_used = used & set(constraint.forbidden_tools)
        if forbidden_used:
            violations.append(
                (i, f"Phase '{phase_name}' used forbidden tools: {forbidden_used}")
            )

        # Check expected tools
        missing = set(constraint.expected_tools) - used
        if missing:
            violations.append(
                (i, f"Phase '{phase_name}' missing expected tools: {missing}")
            )

    return violations
