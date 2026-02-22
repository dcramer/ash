"""Integration runtime and hooks for harness composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ash.core.types import PromptContextAugmenter, SandboxEnvAugmenter

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.core.prompt import PromptContext
    from ash.core.session import SessionState
    from ash.core.types import AgentComponents, MessagePostprocessHook
    from ash.rpc.server import RPCServer


@dataclass(slots=True)
class IntegrationContext:
    """Shared runtime context passed to integration contributors."""

    config: AshConfig
    components: AgentComponents
    mode: Literal["serve", "chat", "eval"]
    sessions_path: Path | None = None


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

    @property
    def contributors(self) -> tuple[IntegrationContributor, ...]:
        return self._contributors

    async def setup(self, context: IntegrationContext) -> None:
        for contributor in self._contributors:
            await contributor.setup(context)

    async def on_startup(self, context: IntegrationContext) -> None:
        for contributor in self._contributors:
            await contributor.on_startup(context)

    async def on_shutdown(self, context: IntegrationContext) -> None:
        for contributor in reversed(self._contributors):
            await contributor.on_shutdown(context)

    def register_rpc_methods(
        self, server: RPCServer, context: IntegrationContext
    ) -> None:
        for contributor in self._contributors:
            contributor.register_rpc_methods(server, context)

    def prompt_context_augmenters(
        self, context: IntegrationContext
    ) -> list[PromptContextAugmenter]:
        hooks: list[PromptContextAugmenter] = []
        for contributor in self._contributors:

            def _hook(
                prompt_context: PromptContext,
                session: SessionState,
                *,
                _contributor: IntegrationContributor = contributor,
            ) -> PromptContext:
                return _contributor.augment_prompt_context(
                    prompt_context, session, context
                )

            hooks.append(_hook)
        return hooks

    def sandbox_env_augmenters(
        self, context: IntegrationContext
    ) -> list[SandboxEnvAugmenter]:
        hooks: list[SandboxEnvAugmenter] = []
        for contributor in self._contributors:

            def _hook(
                env: dict[str, str],
                session: SessionState,
                effective_user_id: str,
                *,
                _contributor: IntegrationContributor = contributor,
            ) -> dict[str, str]:
                return _contributor.augment_sandbox_env(
                    env, session, effective_user_id, context
                )

            hooks.append(_hook)
        return hooks

    def message_postprocess_hooks(
        self, context: IntegrationContext
    ) -> list[MessagePostprocessHook]:
        hooks: list[MessagePostprocessHook] = []
        for contributor in self._contributors:

            async def _hook(
                user_message: str,
                session: SessionState,
                effective_user_id: str,
                *,
                _contributor: IntegrationContributor = contributor,
            ) -> None:
                await _contributor.on_message_postprocess(
                    user_message,
                    session,
                    effective_user_id,
                    context,
                )

            hooks.append(_hook)
        return hooks
