"""Pydantic models for eval case structure."""

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field


@dataclass
class EvalConfig:
    """Configuration for eval runs.

    Centralizes all configurable values to avoid hardcoding throughout the system.
    """

    # Judge configuration
    judge_model: str = "claude-sonnet-4-5"
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


class EvalCase(BaseModel):
    """A single evaluation case."""

    id: str = Field(description="Unique identifier for the case")
    description: str = Field(description="Human-readable description of the test")
    prompt: str = Field(description="User message to send to the agent")
    expected_behavior: str = Field(
        description="Description of what the agent should do"
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
