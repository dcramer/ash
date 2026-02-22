"""Integration runtime and hooks for harness composition.

Spec contract: specs/subsystems.md (Integration Hooks), specs/integrations.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from ash.core.types import (
    IncomingMessagePreprocessor,
    PromptContextAugmenter,
    SandboxEnvAugmenter,
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
        for contributor in self._contributors:
            try:
                await contributor.setup(context)
            except Exception:
                self._log_hook_failure(hook_name="setup", contributor=contributor)
                continue
            active.append(contributor)
        self._active_contributors = tuple(active)

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

    def sandbox_env_augmenters(
        self, context: IntegrationContext
    ) -> list[SandboxEnvAugmenter]:
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

        return self._build_hooks(_factory)

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
