"""System prompt builder with full context."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.config import AshConfig, Workspace
    from ash.memory.manager import RetrievedContext
    from ash.skills import SkillRegistry
    from ash.tools import ToolRegistry


@dataclass
class RuntimeInfo:
    """Runtime information for system prompt."""

    os: str | None = None
    arch: str | None = None
    python: str | None = None
    model: str | None = None
    provider: str | None = None
    timezone: str | None = None
    time: str | None = None

    @classmethod
    def from_environment(
        cls,
        model: str | None = None,
        provider: str | None = None,
        timezone: str | None = None,
    ) -> RuntimeInfo:
        """Create RuntimeInfo from current environment.

        Args:
            model: Current model name.
            provider: Current provider name.
            timezone: User's timezone.

        Returns:
            RuntimeInfo with environment details.
        """
        return cls(
            os=platform.system(),
            arch=platform.machine(),
            python=platform.python_version(),
            model=model,
            provider=provider,
            timezone=timezone,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


@dataclass
class PromptContext:
    """Context for building system prompts."""

    runtime: RuntimeInfo | None = None
    memory: RetrievedContext | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)


class SystemPromptBuilder:
    """Build system prompts with full context.

    Constructs system prompts with:
    - Base identity (SOUL.md)
    - Available tools with descriptions
    - Available skills with descriptions
    - Model aliases
    - Workspace info
    - Sandbox configuration
    - Runtime info (OS, model, time, etc.)
    - Memory context (knowledge, past conversations)
    """

    def __init__(
        self,
        workspace: Workspace,
        tool_registry: ToolRegistry,
        skill_registry: SkillRegistry,
        config: AshConfig,
    ):
        """Initialize prompt builder.

        Args:
            workspace: Loaded workspace with personality.
            tool_registry: Registry of available tools.
            skill_registry: Registry of available skills.
            config: Application configuration.
        """
        self._workspace = workspace
        self._tools = tool_registry
        self._skills = skill_registry
        self._config = config

    def build(self, context: PromptContext | None = None) -> str:
        """Build complete system prompt.

        Args:
            context: Optional context with runtime info and memory.

        Returns:
            Complete system prompt string.
        """
        context = context or PromptContext()
        parts: list[str] = []

        # 1. Base identity (SOUL.md)
        if self._workspace.soul:
            parts.append(self._workspace.soul)

        # 2. Tools section
        tools_section = self._build_tools_section()
        if tools_section:
            parts.append(f"\n\n{tools_section}")

        # 3. Skills section
        skills_section = self._build_skills_section()
        if skills_section:
            parts.append(f"\n\n{skills_section}")

        # 4. Model aliases
        aliases_section = self._build_model_aliases_section()
        if aliases_section:
            parts.append(f"\n\n{aliases_section}")

        # 5. Workspace info
        workspace_section = self._build_workspace_section()
        if workspace_section:
            parts.append(f"\n\n{workspace_section}")

        # 6. Sandbox info
        sandbox_section = self._build_sandbox_section()
        if sandbox_section:
            parts.append(f"\n\n{sandbox_section}")

        # 7. Runtime info
        if context.runtime:
            runtime_section = self._build_runtime_section(context.runtime)
            if runtime_section:
                parts.append(f"\n\n{runtime_section}")

        # 8. Memory context
        if context.memory:
            memory_section = self._build_memory_section(context.memory)
            if memory_section:
                parts.append(f"\n\n{memory_section}")

        return "".join(parts)

    def _build_tools_section(self) -> str:
        """Build tools documentation section.

        Returns:
            Tools section string or empty if no tools.
        """
        tool_defs = self._tools.get_definitions()
        if not tool_defs:
            return ""

        lines = [
            "## Available Tools",
            "",
            "The following tools are available for use:",
            "",
        ]

        for tool_def in tool_defs:
            name = tool_def["name"]
            desc = tool_def["description"]
            # Truncate long descriptions for prompt efficiency
            if len(desc) > 150:
                desc = desc[:147] + "..."
            lines.append(f"- **{name}**: {desc}")

        return "\n".join(lines)

    def _build_skills_section(self) -> str:
        """Build skills listing section.

        Returns:
            Skills section string.
        """
        lines = [
            "## Skills",
            "",
            "Skills are reusable behaviors that combine instructions with tools. "
            "Invoke them with `use_skill`. To create new skills, use the `manage-skill` skill.",
            "",
        ]

        # List existing skills if any
        available_skills = list(self._skills)
        if available_skills:
            lines.append("### Available Skills")
            lines.append("")
            for skill in available_skills:
                lines.append(f"- **{skill.name}**: {skill.description}")
        else:
            lines.append("*No skills available.*")

        return "\n".join(lines)

    def _build_model_aliases_section(self) -> str:
        """Build model aliases section.

        Returns:
            Model aliases section or empty if only default model.
        """
        aliases = self._config.list_models()
        if len(aliases) <= 1:
            return ""

        lines = [
            "## Model Aliases",
            "",
            "Available model configurations:",
            "",
        ]

        for alias in aliases:
            model = self._config.get_model(alias)
            lines.append(f"- `{alias}`: {model.provider}/{model.model}")

        return "\n".join(lines)

    def _build_workspace_section(self) -> str:
        """Build workspace info section.

        Returns:
            Workspace section string.
        """
        lines = [
            "## Workspace",
            "",
            f"Working directory: {self._config.workspace}",
        ]
        return "\n".join(lines)

    def _build_sandbox_section(self) -> str:
        """Build sandbox configuration section.

        Returns:
            Sandbox section string.
        """
        sandbox = self._config.sandbox

        lines = [
            "## Sandbox",
            "",
            "Commands execute in a Docker sandbox with security restrictions.",
            f"- Workspace access: {sandbox.workspace_access}",
            f"- Memory limit: {sandbox.memory_limit}",
            f"- Timeout: {sandbox.timeout}s",
        ]

        if sandbox.network_mode == "none":
            lines.append("- Network: isolated (no external access)")
        else:
            lines.append("- Network: bridge (has external access)")

        return "\n".join(lines)

    def _build_runtime_section(self, runtime: RuntimeInfo) -> str:
        """Build runtime information section.

        Args:
            runtime: Runtime information.

        Returns:
            Runtime section string.
        """
        info_parts = []
        if runtime.os:
            arch_suffix = f" ({runtime.arch})" if runtime.arch else ""
            info_parts.append(f"os={runtime.os}{arch_suffix}")
        if runtime.python:
            info_parts.append(f"python={runtime.python}")
        if runtime.model:
            info_parts.append(f"model={runtime.model}")
        if runtime.provider:
            info_parts.append(f"provider={runtime.provider}")

        lines = ["## Runtime", ""]

        if info_parts:
            lines.append(f"Runtime: {' | '.join(info_parts)}")

        if runtime.timezone or runtime.time:
            tz = runtime.timezone or "system"
            time = runtime.time or "unknown"
            lines.append(f"Timezone: {tz}, Current time: {time}")

        return "\n".join(lines)

    def _build_memory_section(self, memory: RetrievedContext) -> str:
        """Build memory context section.

        Args:
            memory: Retrieved memory context.

        Returns:
            Memory section string or empty if no context.
        """
        context_items: list[str] = []
        for item in memory.knowledge:
            context_items.append(f"- [Knowledge] {item.content}")
        for item in memory.messages:
            context_items.append(f"- [Past conversation] {item.content}")

        if context_items:
            header = (
                "## Relevant Context from Memory\n\n"
                "The following information has been automatically retrieved. "
                "Use it directly - no need to call the recall tool.\n\n"
            )
            return header + "\n".join(context_items)

        return ""
