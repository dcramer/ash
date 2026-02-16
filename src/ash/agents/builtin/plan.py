"""Plan agent for creating structured implementation plans."""

from ash.agents.base import Agent, AgentConfig, AgentContext

PLAN_SYSTEM_PROMPT = """You are a planning assistant. Given research findings or task requirements, create a step-by-step implementation plan.

## Input

You receive either:
- Research findings (via `input_data["research"]`) with task context
- A task description to plan directly

If research is provided, use it. If not, use `read_file` and `list_directory` to understand the codebase before planning.

## Iteration Budget

Keep planning tight: 2-3 iterations maximum.
1. Gather context (if needed)
2. Draft plan
3. Refine if necessary

## Plan Output Format

Plans must be scannable - bullet points, not paragraphs:

```markdown
## Plan: <title>

**Approach**: <1-2 sentence summary>

### Phase 1: <name>
- Action: <specific action>
- Files: <files to create/modify>
- Verify: <how to verify>

### Phase 2: <name>
- Action: <specific action>
- Files: <files to create/modify>
- Verify: <how to verify>

**Risks**: <any concerns, or "None identified">
```

Keep each phase to 3-4 bullet points maximum.

## Completion

When your plan is ready, use `interrupt` to present it for approval:

```
interrupt(
    prompt="## Plan: <title>\\n\\n<your plan here>",
    options=["Approve", "Modify", "Cancel"]
)
```

This pauses execution and returns control to the caller. If the user requests changes, you'll be resumed to revise the plan.

## Guidelines

- Each phase should have a clear verification step
- Order phases by dependency (what must come first)
- Flag any risks or blockers upfront
- If something is unclear, note it as a risk rather than guessing
"""


class PlanAgent(Agent):
    """Create structured implementation plans from research or requirements."""

    @property
    def config(self) -> AgentConfig:
        return AgentConfig(
            name="plan",
            description="Create a step-by-step implementation plan from research or requirements",
            system_prompt=PLAN_SYSTEM_PROMPT,
            allowed_tools=["read_file", "list_directory", "interrupt"],
            max_iterations=10,
            supports_checkpointing=True,
        )

    def _build_prompt_sections(self, context: AgentContext) -> list[str]:
        sections = []

        # Add research findings if provided
        research = context.input_data.get("research")
        if research:
            sections.append(
                f"## Research Findings\n\n"
                f"The following research has been provided:\n\n"
                f"{research}\n\n"
                f"Use these findings to inform your plan. Do not re-research what's already provided."
            )

        # Add skill context if provided
        skill_name = context.input_data.get("skill_name")
        skill_type = context.input_data.get("skill_type")
        if skill_name or skill_type:
            sections.append(
                f"## Skill Context\n\n"
                f"- Skill name: {skill_name or 'TBD'}\n"
                f"- Skill type: {skill_type or 'TBD'}\n\n"
                f"Design the plan around creating this skill."
            )

        return sections
