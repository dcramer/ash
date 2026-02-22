"""Default integration contributor composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ash.integrations.memory import MemoryIntegration
from ash.integrations.runtime_rpc import RuntimeRPCIntegration
from ash.integrations.scheduling import SchedulingIntegration

if TYPE_CHECKING:
    from ash.agents import AgentExecutor
    from ash.integrations.runtime import IntegrationContributor
    from ash.scheduling.handler import MessageRegistrar, MessageSender


@dataclass(slots=True)
class DefaultIntegrations:
    """Concrete integration set for a runtime mode."""

    contributors: list[IntegrationContributor]
    scheduling: SchedulingIntegration | None = None


def create_default_integrations(
    *,
    mode: Literal["serve", "chat", "eval"],
    include_memory: bool = True,
    schedule_file: Path | None = None,
    logs_path: Path | None = None,
    timezone: str = "UTC",
    senders: dict[str, MessageSender] | None = None,
    registrars: dict[str, MessageRegistrar] | None = None,
    agent_executor: AgentExecutor | None = None,
) -> DefaultIntegrations:
    """Build the default integration contributors for a runtime mode."""
    contributors: list[IntegrationContributor] = []
    schedule_integration: SchedulingIntegration | None = None

    if mode == "serve":
        if logs_path is None or schedule_file is None:
            raise ValueError("serve integrations require logs_path and schedule_file")
        contributors.append(RuntimeRPCIntegration(logs_path))
        if include_memory:
            contributors.append(MemoryIntegration())
        schedule_integration = SchedulingIntegration(
            schedule_file,
            timezone=timezone,
            senders=senders,
            registrars=registrars,
            agent_executor=agent_executor,
        )
        contributors.append(schedule_integration)
    elif mode == "chat":
        if include_memory:
            contributors.append(MemoryIntegration())
    elif mode == "eval":
        if schedule_file is None:
            raise ValueError("eval integrations require schedule_file")
        schedule_integration = SchedulingIntegration(schedule_file)
        contributors.append(schedule_integration)
        if include_memory:
            contributors.append(MemoryIntegration())
    else:
        raise ValueError(f"unsupported integration mode: {mode}")

    return DefaultIntegrations(
        contributors=contributors,
        scheduling=schedule_integration,
    )
