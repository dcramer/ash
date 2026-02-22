"""Pydantic models for eval case structure."""

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field


class PhaseConstraint(BaseModel):
    """Constraints for a specific phase in multi-turn evals."""

    expected_tools: list[str] = Field(
        default_factory=list,
        description="Tools that SHOULD be called in this phase",
    )
    forbidden_tools: list[str] = Field(
        default_factory=list,
        description="Tools that MUST NOT be called in this phase",
    )
    must_checkpoint: bool = Field(
        default=False,
        description="Whether this phase must end with an interrupt/checkpoint",
    )


@dataclass
class EvalConfig:
    """Configuration for eval runs.

    Centralizes all configurable values to avoid hardcoding throughout the system.
    """

    # Judge configuration
    judge_model: str = "gpt-5.2"
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024

    # Retry configuration
    retry_attempts: int = 3
    retry_base_delay: float = 1.0  # seconds, with exponential backoff

    # Accuracy thresholds (can be overridden per-suite)
    accuracy_threshold: float = 0.80

    # Case discovery
    cases_dir: Path = field(default_factory=lambda: Path("evals/cases"))
    auto_discover_cases: bool = True

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.cases_dir, str):
            self.cases_dir = Path(self.cases_dir)


# --- v2.0 YAML schema models ---


class SessionConfig(BaseModel):
    """Session configuration for eval cases.

    session_id is always auto-generated — never specify it manually.
    """

    provider: str | None = None
    chat_id: str | None = None
    user_id: str | None = None
    username: str | None = None
    display_name: str | None = None
    chat_type: str | None = None
    chat_title: str | None = None


class SetupStep(BaseModel):
    """A setup step that seeds messages into an agent before the eval prompt.

    session_id is always auto-generated — never specify it manually.
    """

    provider: str | None = None
    chat_id: str | None = None
    user_id: str | None = None
    username: str | None = None
    display_name: str | None = None
    chat_type: str | None = None
    chat_title: str | None = None
    messages: list[str] = Field(default_factory=list)
    drain_extraction: bool = False


class EvalTurn(BaseModel):
    """A single turn in a multi-turn eval case."""

    prompt: str
    expected_behavior: str = ""
    criteria: list[str] = Field(default_factory=list)


class MemoryAssertion(BaseModel):
    """Structural assertion about stored memories."""

    content_contains: list[str] = Field(default_factory=list)
    memory_type: str | None = None


class PersonAssertion(BaseModel):
    """Structural assertion about stored person records."""

    name_contains: str


class Assertions(BaseModel):
    """Structural assertions to check after eval execution."""

    memories: list[MemoryAssertion] = Field(default_factory=list)
    people: list[PersonAssertion] = Field(default_factory=list)


class SuiteDefaults(BaseModel):
    """Default configuration for all cases in a suite."""

    agent: str = "default"
    drain_extraction: bool = False
    session: SessionConfig = Field(default_factory=SessionConfig)


class EvalCase(BaseModel):
    """A single evaluation case."""

    id: str = Field(description="Unique identifier for the case")
    description: str = Field(
        default="", description="Human-readable description of the test"
    )
    prompt: str = Field(default="", description="User message to send to the agent")
    expected_behavior: str = Field(
        default="",
        description="Description of what the agent should do",
    )
    criteria: list[str] = Field(
        default_factory=list,
        description="Specific criteria the judge should evaluate",
    )
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Tools that should be called (if any)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering/grouping cases",
    )
    forbidden_tools: list[str] = Field(
        default_factory=list,
        description="Tools that MUST NOT be called (auto-fail if used)",
    )
    disallowed_tool_result_substrings: list[str] = Field(
        default_factory=list,
        description="Tool result substrings that MUST NOT appear (auto-fail if seen)",
    )
    phase_constraints: dict[str, PhaseConstraint] | None = Field(
        default=None,
        description="Per-phase tool constraints for multi-turn evals",
    )
    input_data: dict[str, str] | None = Field(
        default=None,
        description="Optional input data to pass to the agent context",
    )

    # v2.0 fields
    session: SessionConfig | None = Field(
        default=None,
        description="Per-case session config (merges with suite defaults)",
    )
    setup: list[SetupStep] | None = Field(
        default=None,
        description="Per-case setup steps (replaces suite setup when present)",
    )
    skip_suite_setup: bool = Field(
        default=False,
        description="Skip suite-level setup for this case",
    )
    turns: list[EvalTurn] | None = Field(
        default=None,
        description="Multi-turn conversation (mutually exclusive with prompt)",
    )
    assertions: Assertions | None = Field(
        default=None,
        description="Structural assertions to check after execution",
    )
    agent: str | None = Field(
        default=None,
        description="Per-case agent override (default | memory)",
    )


class EvalSuite(BaseModel):
    """A suite of evaluation cases."""

    schema_version: str = Field(
        default="1.0",
        description="Schema version for forward compatibility",
    )
    name: str = Field(description="Name of the eval suite")
    description: str = Field(
        default="", description="Description of what this suite tests"
    )
    accuracy_threshold: float | None = Field(
        default=None,
        description="Suite-specific accuracy threshold (overrides default)",
    )
    cases: list[EvalCase] = Field(
        default_factory=list, description="List of eval cases"
    )

    # v2.0 fields
    defaults: SuiteDefaults = Field(
        default_factory=SuiteDefaults,
        description="Default configuration for all cases",
    )
    setup: list[SetupStep] | None = Field(
        default=None,
        description="Suite-level setup steps (run before each case)",
    )


class JudgeResult(BaseModel):
    """Result from the LLM judge."""

    passed: bool = Field(description="Whether the response passed the evaluation")
    score: float = Field(ge=0.0, le=1.0, description="Overall score from 0.0 to 1.0")
    reasoning: str = Field(description="Explanation of the judgment")
    criteria_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion scores (0.0 to 1.0)",
    )
    judge_error: bool = Field(
        default=False,
        description="True if the result is due to a judge error, not an actual evaluation failure",
    )
    error_type: str | None = Field(
        default=None,
        description="Type of error if judge_error is True (e.g., 'parse_error', 'api_error', 'timeout')",
    )
