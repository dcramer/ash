"""Integration runtime and hooks for harness composition.

Spec contract: specs/subsystems.md (Integration Hooks), specs/integrations.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from ash.core.types import (
    IncomingMessagePreprocessor,
    PromptContextAugmenter,
    SandboxEnvAugmenter,
    SkillInstructionAugmenter,
)

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.core.prompt import PromptContext
    from ash.core.session import SessionState
    from ash.core.types import AgentComponents, MessagePostprocessHook
    from ash.providers.base import IncomingMessage
    from ash.rpc.server import RPCServer

T = TypeVar("T")
IntegrationMode = Literal["serve", "chat", "eval"]
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IntegrationContext:
    """Shared runtime context passed to integration contributors."""

    config: AshConfig
    components: AgentComponents
    mode: IntegrationMode
    sessions_path: Path | None = None
    # Runtime-owned env projected into sandbox tool execution hooks.
    # Spec contract: specs/subsystems.md (Integration Hooks), specs/rpc.md.
    sandbox_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IntegrationHealthSnapshot:
    """Operational health summary for integration runtime."""

    configured_count: int
    active_count: int
    failed_setup: tuple[str, ...]
    hook_failures: dict[str, int]

    @property
    def is_degraded(self) -> bool:
        return bool(self.failed_setup or self.hook_failures)


class IntegrationContributor:
    """Base class for integration contributors."""

    name = "integration"
    priority = 1000

    async def setup(self, context: IntegrationContext) -> None:
        """Initialize contributor state."""
        return None

    async def on_startup(self, context: IntegrationContext) -> None:
        """Run startup lifecycle hook."""
        return None

    async def on_shutdown(self, context: IntegrationContext) -> None:
        """Run shutdown lifecycle hook."""
        return None

    def register_rpc_methods(
        self,
        server: RPCServer,
        context: IntegrationContext,
    ) -> None:
        """Register RPC methods."""
        return None

    def augment_prompt_context(
        self,
        prompt_context: PromptContext,
        session: SessionState,
        context: IntegrationContext,
    ) -> PromptContext:
        """Augment structured prompt context."""
        return prompt_context

    def augment_sandbox_env(
        self,
        env: dict[str, str],
        session: SessionState,
        effective_user_id: str,
        context: IntegrationContext,
    ) -> dict[str, str]:
        """Augment sandbox env for tool execution."""
        return env

    async def preprocess_incoming_message(
        self,
        message: IncomingMessage,
        context: IntegrationContext,
    ) -> IncomingMessage:
        """Preprocess provider incoming messages before session processing."""
        return message

    def augment_skill_instructions(
        self,
        skill_name: str,
        context: IntegrationContext,
    ) -> list[str]:
        """Return additional instruction lines to append when a skill is invoked."""
        return []

    async def on_message_postprocess(
        self,
        user_message: str,
        session: SessionState,
        effective_user_id: str,
        context: IntegrationContext,
    ) -> None:
        """Run post-turn work after a user message is processed."""
        return None


class IntegrationRuntime:
    """Deterministic integration pipeline runtime."""

    def __init__(self, contributors: list[IntegrationContributor] | None = None):
        self._contributors = tuple(
            sorted(contributors or [], key=lambda c: (c.priority, c.name))
        )
        self._active_contributors = self._contributors
        self._failed_setup: dict[str, int] = {}
        self._hook_failure_counts: dict[str, int] = {}

    @property
    def contributors(self) -> tuple[IntegrationContributor, ...]:
        return self._contributors

    @property
    def active_contributors(self) -> tuple[IntegrationContributor, ...]:
        return self._active_contributors

    def _log_hook_failure(
        self,
        *,
        hook_name: str,
        contributor: IntegrationContributor,
    ) -> None:
        key = f"{contributor.name}.{hook_name}"
        self._hook_failure_counts[key] = self._hook_failure_counts.get(key, 0) + 1
        logger.warning(
            "integration_hook_failed",
            extra={
                "integration.name": contributor.name,
                "integration.priority": contributor.priority,
                "integration.hook": hook_name,
            },
            exc_info=True,
        )

    async def setup(self, context: IntegrationContext) -> None:
        # Spec contract: specs/subsystems.md (Integration Hooks)
        # Keep integration failures isolated so one contributor doesn't break all.
        active: list[IntegrationContributor] = []
        self._failed_setup.clear()
        for contributor in self._contributors:
            try:
                await contributor.setup(context)
            except Exception:
                self._failed_setup[contributor.name] = (
                    self._failed_setup.get(contributor.name, 0) + 1
                )
                self._log_hook_failure(hook_name="setup", contributor=contributor)
                continue
            active.append(contributor)
        self._active_contributors = tuple(active)

    def health_snapshot(self) -> IntegrationHealthSnapshot:
        return IntegrationHealthSnapshot(
            configured_count=len(self._contributors),
            active_count=len(self._active_contributors),
            failed_setup=tuple(sorted(self._failed_setup)),
            hook_failures=dict(sorted(self._hook_failure_counts.items())),
        )

    async def on_startup(self, context: IntegrationContext) -> None:
        for contributor in self._active_contributors:
            try:
                await contributor.on_startup(context)
            except Exception:
                self._log_hook_failure(hook_name="on_startup", contributor=contributor)

    async def on_shutdown(self, context: IntegrationContext) -> None:
        for contributor in reversed(self._active_contributors):
            try:
                await contributor.on_shutdown(context)
            except Exception:
                self._log_hook_failure(hook_name="on_shutdown", contributor=contributor)

    def register_rpc_methods(
        self, server: RPCServer, context: IntegrationContext
    ) -> None:
        for contributor in self._active_contributors:
            try:
                contributor.register_rpc_methods(server, context)
            except Exception:
                self._log_hook_failure(
                    hook_name="register_rpc_methods",
                    contributor=contributor,
                )

    def _build_hooks(self, factory: Callable[[IntegrationContributor], T]) -> list[T]:
        return [factory(contributor) for contributor in self._active_contributors]

    def prompt_context_augmenters(
        self, context: IntegrationContext
    ) -> list[PromptContextAugmenter]:
        def _factory(contributor: IntegrationContributor) -> PromptContextAugmenter:
            def _hook(
                prompt_context: PromptContext,
                session: SessionState,
            ) -> PromptContext:
                try:
                    return contributor.augment_prompt_context(
                        prompt_context, session, context
                    )
                except Exception:
                    self._log_hook_failure(
                        hook_name="augment_prompt_context",
                        contributor=contributor,
                    )
                    return prompt_context

            return _hook

        return self._build_hooks(_factory)

    def skill_instruction_augmenter(
        self, context: IntegrationContext
    ) -> SkillInstructionAugmenter:
        """Return a closure that collects skill instruction augmentations from contributors."""

        def _augmenter(skill_name: str) -> list[str]:
            lines: list[str] = []
            for contributor in self._active_contributors:
                try:
                    lines.extend(
                        contributor.augment_skill_instructions(skill_name, context)
                    )
                except Exception:
                    self._log_hook_failure(
                        hook_name="augment_skill_instructions",
                        contributor=contributor,
                    )
            return lines

        return _augmenter

    def sandbox_env_augmenters(
        self, context: IntegrationContext
    ) -> list[SandboxEnvAugmenter]:
        def _runtime_env_hook(
            env: dict[str, str],
            _session: SessionState,
            _effective_user_id: str,
        ) -> dict[str, str]:
            if not context.sandbox_env:
                return env
            merged = dict(env)
            merged.update(context.sandbox_env)
            return merged

        def _factory(contributor: IntegrationContributor) -> SandboxEnvAugmenter:
            def _hook(
                env: dict[str, str],
                session: SessionState,
                effective_user_id: str,
            ) -> dict[str, str]:
                try:
                    return contributor.augment_sandbox_env(
                        env, session, effective_user_id, context
                    )
                except Exception:
                    self._log_hook_failure(
                        hook_name="augment_sandbox_env",
                        contributor=contributor,
                    )
                    return env

            return _hook

        hooks = [_runtime_env_hook]
        hooks.extend(self._build_hooks(_factory))
        return hooks

    def message_postprocess_hooks(
        self, context: IntegrationContext
    ) -> list[MessagePostprocessHook]:
        def _factory(contributor: IntegrationContributor) -> MessagePostprocessHook:
            async def _hook(
                user_message: str,
                session: SessionState,
                effective_user_id: str,
            ) -> None:
                try:
                    await contributor.on_message_postprocess(
                        user_message,
                        session,
                        effective_user_id,
                        context,
                    )
                except Exception:
                    self._log_hook_failure(
                        hook_name="on_message_postprocess",
                        contributor=contributor,
                    )

            return _hook

        return self._build_hooks(_factory)

    def incoming_message_preprocessors(
        self, context: IntegrationContext
    ) -> list[IncomingMessagePreprocessor]:
        def _factory(
            contributor: IntegrationContributor,
        ) -> IncomingMessagePreprocessor:
            async def _hook(message: IncomingMessage) -> IncomingMessage:
                try:
                    return await contributor.preprocess_incoming_message(
                        message, context
                    )
                except Exception:
                    self._log_hook_failure(
                        hook_name="preprocess_incoming_message",
                        contributor=contributor,
                    )
                    return message

            return _hook

        return self._build_hooks(_factory)
