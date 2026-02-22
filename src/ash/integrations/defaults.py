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


def _create_chat_integrations(*, include_memory: bool) -> DefaultIntegrations:
    contributors: list[IntegrationContributor] = []
    if include_memory:
        contributors.append(MemoryIntegration())
    return DefaultIntegrations(contributors=contributors)


def _create_eval_integrations(
    *,
    include_memory: bool,
    schedule_file: Path | None,
) -> DefaultIntegrations:
    if schedule_file is None:
        raise ValueError("eval integrations require schedule_file")

    scheduling = SchedulingIntegration(schedule_file)
    contributors: list[IntegrationContributor] = [scheduling]
    if include_memory:
        contributors.append(MemoryIntegration())
    return DefaultIntegrations(contributors=contributors, scheduling=scheduling)


def _create_serve_integrations(
    *,
    include_memory: bool,
    schedule_file: Path | None,
    logs_path: Path | None,
    timezone: str,
    senders: dict[str, MessageSender] | None,
    registrars: dict[str, MessageRegistrar] | None,
    agent_executor: AgentExecutor | None,
) -> DefaultIntegrations:
    if logs_path is None or schedule_file is None:
        raise ValueError("serve integrations require logs_path and schedule_file")

    scheduling = SchedulingIntegration(
        schedule_file,
        timezone=timezone,
        senders=senders,
        registrars=registrars,
        agent_executor=agent_executor,
    )

    contributors: list[IntegrationContributor] = [RuntimeRPCIntegration(logs_path)]
    if include_memory:
        contributors.append(MemoryIntegration())
    contributors.append(scheduling)
    return DefaultIntegrations(contributors=contributors, scheduling=scheduling)


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
    if mode == "serve":
        return _create_serve_integrations(
            include_memory=include_memory,
            schedule_file=schedule_file,
            logs_path=logs_path,
            timezone=timezone,
            senders=senders,
            registrars=registrars,
            agent_executor=agent_executor,
        )
    if mode == "chat":
        return _create_chat_integrations(include_memory=include_memory)
    if mode == "eval":
        return _create_eval_integrations(
            include_memory=include_memory,
            schedule_file=schedule_file,
        )
    raise ValueError(f"unsupported integration mode: {mode}")
