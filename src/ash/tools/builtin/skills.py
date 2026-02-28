"""Skill invocation tool."""

import logging
from typing import TYPE_CHECKING, Any

from ash.agents.base import Agent
from ash.agents.types import AgentConfig, AgentContext, ChildActivated, StackFrame
from ash.core.types import SkillInstructionAugmenter
from ash.skills.types import SkillDefinition
from ash.tools.base import Tool, ToolContext, ToolResult, format_subagent_result

if TYPE_CHECKING:
    from ash.agents import AgentExecutor
    from ash.config import AshConfig
    from ash.skills import SkillRegistry

logger = logging.getLogger(__name__)

# Built-in skills that are handled specially (not loaded from SKILL.md files)
BUILTIN_SKILLS: dict[str, str] = {}

# Wrapper guidance prepended to all skill system prompts
SKILL_AGENT_WRAPPER = """You are a skill executor. Your job is to run the skill instructions below and report results.

## How to Execute

1. Follow the instructions in the skill definition
2. Run any commands or tools specified
3. Report what happened - include actual output

## Handling Errors

When a command fails or returns an error:
- Report the error message to the user
- STOP by default - do not attempt to fix, debug, or work around the problem unless the user explicitly asks you to do so
- The user will decide whether to invoke the skill-writer to fix the skill

**NEVER do any of the following when something fails:**
- Read the script source to understand why it failed
- Copy or modify script files
- Use sed, awk, or other tools to edit files
- Write inline scripts to diagnose the issue
- Try alternative approaches not in the instructions

If the skill is broken, say so and stop unless the user explicitly asks for repair/debugging.

## Output

When your task is finished, call `complete` with the final output:
- `complete({"result": "<final output>"})`

This is required so control returns to the parent agent.

- Include actual command output, not just summaries
- If something failed, include the error message
- Be concise - the user wants results, not a narrative

For long-running tasks, use `send_message` for progress updates only (e.g., "Processing file 3 of 10...").
Never use `send_message` for the final result - the final result must go through `complete`.

---

"""


def format_skill_result(content: str, skill_name: str) -> str:
    """Format skill result with structured tags for LLM clarity."""

    return format_subagent_result(content, "skill", skill_name)


class SkillAgent(Agent):
    """Ephemeral agent wrapper for a skill definition.

    Converts a SkillDefinition into an Agent so it can be executed
    via AgentExecutor with the standard agent loop.
    """

    def __init__(
        self,
        skill: SkillDefinition,
        model_override: str | None = None,
        instruction_augmenter: "SkillInstructionAugmenter | None" = None,
        sandbox_skill_dir: str | None = None,
    ) -> None:
        """Initialize skill agent.

        Args:
            skill: Skill definition to wrap.
            model_override: Optional model alias to override skill's default.
            instruction_augmenter: Optional callback returning extra instruction lines.
            sandbox_skill_dir: Sandbox container path to this skill's directory.
        """
        self._skill = skill
        self._model_override = model_override
        self._instruction_augmenter = instruction_augmenter
        self._sandbox_skill_dir = sandbox_skill_dir

    @property
    def config(self) -> AgentConfig:
        """Return agent configuration derived from skill."""
        return AgentConfig(
            name=f"skill:{self._skill.name}",
            description=self._skill.description,
            system_prompt=self._skill.instructions,
            allowed_tools=self._skill.allowed_tools,
            max_iterations=self._skill.max_iterations,
            model=self._model_override or self._skill.model,
            is_skill_agent=True,
        )

    def build_system_prompt(self, context: AgentContext) -> str:
        """Build system prompt with wrapper guidance and optional context injection.

        Args:
            context: Execution context with optional user-provided context.

        Returns:
            System prompt string with wrapper + skill instructions + context.
        """
        # Start with wrapper guidance
        prompt = SKILL_AGENT_WRAPPER

        # Add provenance if available
        if self._skill.authors or self._skill.rationale:
            prompt += f"## Skill: {self._skill.name}\n"
            if self._skill.authors:
                prompt += f"**Authors:** {', '.join(self._skill.authors)}\n"
            if self._skill.rationale:
                prompt += f"**Rationale:** {self._skill.rationale}\n"
            prompt += "\n"

        # Add skill instructions
        prompt += self._skill.instructions

        # Tell the skill agent where its co-located files live in the sandbox
        if self._sandbox_skill_dir:
            prompt += "\n\n## Skill Directory\n\n"
            prompt += f"Your skill files are at `{self._sandbox_skill_dir}/`. "
            prompt += (
                "Relative paths in your instructions resolve against this directory."
            )

        # Inject integration-provided additional context
        if self._instruction_augmenter:
            extra_lines = self._instruction_augmenter(self._skill.name)
            if extra_lines:
                prompt += "\n\n## Additional Context\n\n"
                prompt += "\n".join(extra_lines)

        # Inject user-provided context if available
        user_context = context.input_data.get("context", "")
        if user_context:
            prompt += f"\n\n## Context\n\n{user_context}"

        # Add shared environment context (sandbox, runtime, tool guidance)
        if context.shared_prompt:
            prompt += f"\n\n{context.shared_prompt}"

        # Add voice guidance for user-facing messages
        if context.voice:
            prompt += f"""

## Communication Style (for user-facing messages only)

{context.voice}

IMPORTANT: Apply this style ONLY to interrupt() prompts that users will see.
Do NOT apply it to tool outputs, file content, or technical results."""

        return prompt


class UseSkillTool(Tool):
    """Invoke a skill with isolated execution.

    Skills run as subagents with their own LLM loops, tool restrictions,
    and scoped environments (API keys injected from config).
    """

    def __init__(
        self,
        registry: "SkillRegistry",
        executor: "AgentExecutor",
        config: "AshConfig",
        voice: str | None = None,
        subagent_context: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            registry: Skill registry to look up skills.
            executor: Agent executor to run skill agents.
            config: Application configuration for skill settings.
            voice: Optional communication style for user-facing skill messages.
            subagent_context: Shared prompt context (sandbox, runtime, tool guidance) for subagents.
        """
        self._registry = registry
        self._executor = executor
        self._config = config
        self._voice = voice
        self._subagent_context = subagent_context
        self._capability_manager: Any | None = None
        self._skill_instruction_augmenter: SkillInstructionAugmenter | None = None

    def set_shared_prompt(self, prompt: str | None) -> None:
        """Update shared prompt context used for skill execution."""
        self._subagent_context = prompt

    def set_capability_manager(self, manager: Any | None) -> None:
        """Attach host capability manager for skill capability preflight checks."""
        self._capability_manager = manager

    def set_skill_instruction_augmenter(
        self, augmenter: SkillInstructionAugmenter | None
    ) -> None:
        """Attach integration skill instruction augmenter for skill execution."""
        self._skill_instruction_augmenter = augmenter

    @property
    def name(self) -> str:
        return "use_skill"

    @property
    def description(self) -> str:
        # Combine registry skills with built-in skills
        skill_names = [s.name for s in self._registry.list_available()]
        skill_names.extend(BUILTIN_SKILLS.keys())
        if not skill_names:
            return "Invoke a skill (none available)"
        skill_list = ", ".join(sorted(set(skill_names)))
        return f"Invoke a skill with isolated execution. Available: {skill_list}"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "Name of the skill to invoke",
                },
                "message": {
                    "type": "string",
                    "description": "Task/message for the skill to work on",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context to help the skill understand the task",
                },
            },
            "required": ["skill", "message"],
        }

    def _build_skill_environment(
        self,
        skill: SkillDefinition,
        skill_config: Any,
        *,
        base_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build environment dict for skill execution."""
        env: dict[str, str] = dict(base_env or {})
        if not skill_config:
            return env
        config_env = skill_config.get_env_vars()
        for var_name in skill.env:
            if var_name in config_env:
                env[var_name] = config_env[var_name]
            else:
                logger.warning(
                    "skill_env_var_missing",
                    extra={
                        "skill.name": skill.name,
                        "env.var": var_name,
                    },
                )
        return env

    def _resolve_allowed_chat_ids(self, skill_config: Any) -> set[str]:
        """Resolve effective allowed chat IDs (per-skill override -> defaults)."""
        allowlist_value: Any = None

        if skill_config and getattr(skill_config, "allow_chat_ids", None) is not None:
            allowlist_value = skill_config.allow_chat_ids
        else:
            defaults = getattr(self._config, "skill_defaults", None)
            if defaults is not None:
                allowlist_value = getattr(defaults, "allow_chat_ids", None)

        if allowlist_value is None:
            raw_allowlist: list[str] = []
        elif isinstance(allowlist_value, str):
            raw_allowlist = [allowlist_value]
        elif isinstance(allowlist_value, (list, tuple, set)):
            raw_allowlist = [str(item) for item in allowlist_value]
        else:
            raw_allowlist = []

        return {
            str(chat_id).strip() for chat_id in raw_allowlist if str(chat_id).strip()
        }

    def _validate_skill_access(
        self,
        skill: SkillDefinition,
        skill_config: Any,
        context: ToolContext | None,
    ) -> str | None:
        """Return an access-denied error string when skill invocation is blocked."""
        # Architecture/spec reference: specs/skills.md
        chat_id = (context.chat_id if context else None) or None
        chat_type_raw = context.metadata.get("chat_type") if context else None
        chat_type = str(chat_type_raw).strip().lower() if chat_type_raw else None

        allowed_chat_types = [t.strip().lower() for t in skill.allowed_chat_types if t]
        if skill.sensitive and not allowed_chat_types:
            allowed_chat_types = ["private"]

        if allowed_chat_types:
            if not chat_type:
                return (
                    f"Skill '{skill.name}' requires chat context and is only available in: "
                    f"{', '.join(sorted(set(allowed_chat_types)))}"
                )
            if chat_type not in allowed_chat_types:
                return (
                    f"Skill '{skill.name}' is only available in: "
                    f"{', '.join(sorted(set(allowed_chat_types)))}"
                )

        allowed_chat_ids = self._resolve_allowed_chat_ids(skill_config)
        if allowed_chat_ids:
            normalized_chat_id = str(chat_id).strip() if chat_id else ""
            if not normalized_chat_id or normalized_chat_id not in allowed_chat_ids:
                return (
                    f"Skill '{skill.name}' is not enabled for this chat "
                    "(configure [skills.defaults].allow_chat_ids or "
                    f"[skills.{skill.name}].allow_chat_ids)."
                )

        return None

    async def _validate_required_capabilities(
        self,
        skill: SkillDefinition,
        context: ToolContext | None,
    ) -> str | None:
        """Return an error if skill-declared capabilities are unavailable."""
        required = sorted({item.strip() for item in skill.capabilities if item.strip()})
        if not required:
            return None

        if context is None or not context.user_id:
            return (
                f"Skill '{skill.name}' requires verified user context for "
                "capability access."
            )

        manager = self._capability_manager
        if manager is None:
            return (
                f"Skill '{skill.name}' requires capabilities but capability manager "
                "is not available."
            )

        chat_type_raw = context.metadata.get("chat_type")
        chat_type = str(chat_type_raw).strip() if chat_type_raw else None
        try:
            visible = await manager.list_capabilities(
                user_id=context.user_id,
                chat_type=chat_type,
                include_unavailable=False,
            )
        except Exception as e:
            code = getattr(e, "code", "capability_backend_unavailable")
            return f"Capability preflight failed ({code}): {e}"

        visible_ids = {str(item.get("id")) for item in visible if item.get("id")}
        missing = [
            capability_id
            for capability_id in required
            if capability_id not in visible_ids
        ]
        if not missing:
            return None

        return (
            f"Skill '{skill.name}' requires unavailable capabilities in this context: "
            f"{', '.join(missing)}"
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        skill_name = input_data.get("skill")
        message = input_data.get("message")
        user_context = input_data.get("context", "")

        if not skill_name:
            return ToolResult.error("Missing required field: skill")

        if not message:
            return ToolResult.error("Missing required field: message")

        if not self._registry.has(skill_name):
            self._registry.reload_all(self._config.workspace)
            if not self._registry.has(skill_name):
                # Include built-in skills in available list
                available = set(self._registry.list_names())
                available.update(BUILTIN_SKILLS.keys())
                return ToolResult.error(
                    f"Skill '{skill_name}' not found. Available: {', '.join(sorted(available))}"
                )

        skill = self._registry.get(skill_name)
        skill_config = self._config.skills.get(skill_name)

        if skill_config and not skill_config.enabled:
            return ToolResult.error(f"Skill '{skill_name}' is disabled in config")

        access_error = self._validate_skill_access(skill, skill_config, context)
        if access_error:
            return ToolResult.error(access_error)

        capability_error = await self._validate_required_capabilities(skill, context)
        if capability_error:
            return ToolResult.error(capability_error)

        if skill.env:
            config_env = skill_config.get_env_vars() if skill_config else {}
            missing = [var for var in skill.env if var not in config_env]
            if missing:
                return ToolResult.error(
                    f"Skill '{skill_name}' requires configuration.\n\n"
                    f"Add to ~/.ash/config.toml:\n\n"
                    f"[skills.{skill_name}]\n"
                    + "\n".join(f'{var} = "your-value-here"' for var in missing)
                )

        inherited_env = dict(context.env) if context else {}
        env = self._build_skill_environment(
            skill,
            skill_config,
            base_env=inherited_env,
        )
        model_override = skill_config.model if skill_config else None

        # Compute sandbox container path for this skill's directory
        from ash.skills.types import compute_sandbox_skill_dir

        sb_dir = compute_sandbox_skill_dir(skill, self._config.sandbox.mount_prefix)

        agent = SkillAgent(
            skill,
            model_override=model_override,
            instruction_augmenter=self._skill_instruction_augmenter,
            sandbox_skill_dir=sb_dir,
        )

        if context:
            agent_context = AgentContext.from_tool_context(
                context,
                input_data={"context": user_context},
                voice=self._voice,
                shared_prompt=self._subagent_context,
            )
        else:
            agent_context = AgentContext(
                input_data={"context": user_context},
                voice=self._voice,
                shared_prompt=self._subagent_context,
            )

        # Get session info from context for subagent logging
        session_manager, tool_use_id = (
            context.get_session_info() if context else (None, None)
        )

        # Build child frame and raise ChildActivated for interactive stack handling.
        # The orchestrator will run all turns (including the first).
        agent_config = agent.config
        overrides = self._config.agents.get(agent_config.name)
        model_alias = (overrides.model if overrides else None) or agent_config.model
        resolved_model: str | None = None
        if model_alias:
            try:
                resolved_model = self._config.get_model(model_alias).model
            except Exception:
                logger.warning(
                    "model_resolution_failed",
                    extra={
                        "model.alias": model_alias,
                        "skill.name": skill_name,
                    },
                )

        logger.info(
            "skill_invoked",
            extra={
                "skill": skill_name,
                "model": resolved_model or model_alias or "default",
                "message_len": len(message),
                "message_preview": message[:200],
            },
        )

        # Start agent session for logging
        agent_session_id: str | None = None
        if session_manager and tool_use_id:
            agent_session_id = await session_manager.start_agent_session(
                parent_tool_use_id=tool_use_id,
                agent_type="skill",
                agent_name=agent_config.name,
            )

        # Build child session with initial message
        from ash.core.session import SessionState
        from ash.sessions.types import generate_id

        child_session = SessionState(
            session_id=f"agent-{agent_config.name}-{agent_context.session_id or 'unknown'}",
            provider=agent_context.provider or "",
            chat_id=agent_context.chat_id or "",
            user_id=agent_context.user_id or "",
        )
        child_session.add_user_message(message)

        system_prompt = agent.build_system_prompt(agent_context)

        child_frame = StackFrame(
            frame_id=generate_id(),
            agent_name=agent_config.name,
            agent_type="skill",
            session=child_session,
            system_prompt=system_prompt,
            context=agent_context,
            model=resolved_model,
            environment=env,
            max_iterations=agent_config.max_iterations,
            effective_tools=agent_config.get_effective_tools(),
            is_skill_agent=True,
            voice=self._voice,
            parent_tool_use_id=tool_use_id,
            agent_session_id=agent_session_id,
        )

        raise ChildActivated(child_frame)
