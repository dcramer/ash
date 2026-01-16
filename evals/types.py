"""Pydantic models for eval case structure."""

from pydantic import BaseModel, Field


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

    name: str = Field(description="Name of the eval suite")
    description: str = Field(
        default="", description="Description of what this suite tests"
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
