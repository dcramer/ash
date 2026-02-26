"""Capabilities integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks), specs/capabilities.md.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ash.core.prompt import PromptContext
from ash.core.prompt_keys import TOOL_ROUTING_RULES_KEY
from ash.core.session import SessionState
from ash.integrations.runtime import IntegrationContext, IntegrationContributor

if TYPE_CHECKING:
    from ash.capabilities import CapabilityProvider
    from ash.config import AshConfig, CapabilityProviderConfig

logger = logging.getLogger(__name__)


class CapabilitiesIntegration(IntegrationContributor):
    """Owns capability manager setup and RPC surface registration."""

    name = "capabilities"
    priority = 255

    async def setup(self, context: IntegrationContext) -> None:
        from ash.capabilities import create_capability_manager

        components = context.components
        manager = getattr(components, "capability_manager", None)
        providers = list(getattr(components, "capability_providers", None) or [])
        providers.extend(_build_configured_capability_providers(context.config))
        if manager is None:
            manager = await create_capability_manager()
            components.capability_manager = manager

        for provider in providers:
            try:
                await manager.register_provider(provider)
            except Exception as e:
                logger.warning(
                    "capability_provider_register_failed",
                    extra={
                        "provider.namespace": getattr(provider, "namespace", "unknown"),
                        "error.message": str(e),
                    },
                )

        tool_registry = getattr(components, "tool_registry", None)
        if tool_registry is not None and hasattr(tool_registry, "has"):
            try:
                if tool_registry.has("use_skill") and hasattr(tool_registry, "get"):
                    skill_tool = tool_registry.get("use_skill")
                    setter = getattr(skill_tool, "set_capability_manager", None)
                    if callable(setter):
                        setter(manager)
            except Exception:
                logger.warning("capability_skill_wiring_failed", exc_info=True)

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.capability import register_capability_methods

        manager = getattr(context.components, "capability_manager", None)
        if manager is None:
            return
        register_capability_methods(server, manager)

    def augment_prompt_context(
        self,
        prompt_context: PromptContext,
        session: SessionState,
        context: IntegrationContext,
    ) -> PromptContext:
        _ = session
        _ = context
        lines = prompt_context.extra_context.setdefault(TOOL_ROUTING_RULES_KEY, [])
        if isinstance(lines, list):
            line = (
                "- For sensitive external integrations (email/calendar), use "
                "`ash-sb capability` so identity scope is enforced by "
                "`ASH_CONTEXT_TOKEN`; do not request raw credential env vars."
            )
            if line not in lines:
                lines.append(line)
        return prompt_context


def _build_configured_capability_providers(
    config: AshConfig,
) -> list[CapabilityProvider]:
    providers: list[CapabilityProvider] = []
    for provider_name, provider_config in sorted(config.capabilities.providers.items()):
        if not provider_config.enabled:
            continue
        try:
            provider = _create_capability_provider(
                provider_name=provider_name,
                config=provider_config,
            )
        except Exception as e:
            logger.warning(
                "capability_provider_load_failed",
                extra={
                    "provider_name": provider_name,
                    "namespace": provider_config.namespace or provider_name,
                    "command": provider_config.command,
                    "error.message": str(e),
                },
            )
            continue
        providers.append(provider)
    return providers


def _create_capability_provider(
    *,
    provider_name: str,
    config: CapabilityProviderConfig,
) -> CapabilityProvider:
    from ash.capabilities.providers import SubprocessCapabilityProvider

    return SubprocessCapabilityProvider(
        namespace=config.namespace or provider_name,
        command=config.command,
        timeout_seconds=config.timeout_seconds,
    )
