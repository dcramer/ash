"""Default integration contributor composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ash.integrations.browser import BrowserIntegration
from ash.integrations.image import ImageIntegration
from ash.integrations.memory import MemoryIntegration
from ash.integrations.runtime import IntegrationMode
from ash.integrations.runtime_rpc import RuntimeRPCIntegration
from ash.integrations.scheduling import SchedulingIntegration
from ash.integrations.todo import TodoIntegration

if TYPE_CHECKING:
    from ash.agents import AgentExecutor
    from ash.integrations.runtime import IntegrationContributor
    from ash.scheduling.handler import MessagePersister, MessageRegistrar, MessageSender


@dataclass(slots=True)
class DefaultIntegrations:
    """Concrete integration set for a runtime mode."""

    contributors: list[IntegrationContributor]
    scheduling: SchedulingIntegration | None = None


def _create_chat_integrations(
    *,
    include_memory: bool,
    include_image: bool,
    include_browser: bool,
    include_todo: bool,
    graph_dir: Path,
) -> DefaultIntegrations:
    contributors: list[IntegrationContributor] = []
    if include_image:
        contributors.append(ImageIntegration())
    if include_browser:
        contributors.append(BrowserIntegration())
    if include_todo:
        contributors.append(TodoIntegration(graph_dir=graph_dir))
    if include_memory:
        contributors.append(MemoryIntegration())
    return DefaultIntegrations(contributors=contributors)


def _create_eval_integrations(
    *,
    include_memory: bool,
    include_image: bool,
    include_browser: bool,
    include_todo: bool,
    graph_dir: Path,
) -> DefaultIntegrations:
    scheduling = SchedulingIntegration(graph_dir)
    contributors: list[IntegrationContributor] = [scheduling]
    if include_image:
        contributors.append(ImageIntegration())
    if include_browser:
        contributors.append(BrowserIntegration())
    if include_todo:
        contributors.append(TodoIntegration(graph_dir=graph_dir, schedule_enabled=True))
    if include_memory:
        contributors.append(MemoryIntegration())
    return DefaultIntegrations(contributors=contributors, scheduling=scheduling)


def _create_serve_integrations(
    *,
    include_memory: bool,
    include_image: bool,
    include_browser: bool,
    include_todo: bool,
    graph_dir: Path,
    logs_path: Path | None,
    timezone: str,
    senders: dict[str, MessageSender] | None,
    registrars: dict[str, MessageRegistrar] | None,
    persisters: dict[str, MessagePersister] | None,
    agent_executor: AgentExecutor | None,
) -> DefaultIntegrations:
    if logs_path is None:
        raise ValueError("serve integrations require logs_path")

    scheduling = SchedulingIntegration(
        graph_dir,
        timezone=timezone,
        senders=senders,
        registrars=registrars,
        persisters=persisters,
        agent_executor=agent_executor,
    )

    contributors: list[IntegrationContributor] = [RuntimeRPCIntegration(logs_path)]
    if include_image:
        contributors.append(ImageIntegration())
    if include_browser:
        contributors.append(BrowserIntegration())
    if include_todo:
        contributors.append(TodoIntegration(graph_dir=graph_dir, schedule_enabled=True))
    if include_memory:
        contributors.append(MemoryIntegration())
    contributors.append(scheduling)
    return DefaultIntegrations(contributors=contributors, scheduling=scheduling)


def create_default_integrations(
    *,
    mode: IntegrationMode,
    include_memory: bool = True,
    include_image: bool = True,
    include_browser: bool = True,
    include_todo: bool = True,
    graph_dir: Path | None = None,
    logs_path: Path | None = None,
    timezone: str = "UTC",
    senders: dict[str, MessageSender] | None = None,
    registrars: dict[str, MessageRegistrar] | None = None,
    persisters: dict[str, MessagePersister] | None = None,
    agent_executor: AgentExecutor | None = None,
) -> DefaultIntegrations:
    """Build the default integration contributors for a runtime mode."""
    if graph_dir is None:
        from ash.config.paths import get_graph_dir

        graph_dir = get_graph_dir()
    if mode == "serve":
        return _create_serve_integrations(
            include_memory=include_memory,
            include_image=include_image,
            include_browser=include_browser,
            include_todo=include_todo,
            graph_dir=graph_dir,
            logs_path=logs_path,
            timezone=timezone,
            senders=senders,
            registrars=registrars,
            persisters=persisters,
            agent_executor=agent_executor,
        )
    if mode == "chat":
        return _create_chat_integrations(
            include_memory=include_memory,
            include_image=include_image,
            include_browser=include_browser,
            include_todo=include_todo,
            graph_dir=graph_dir,
        )
    if mode == "eval":
        return _create_eval_integrations(
            include_memory=include_memory,
            include_image=include_image,
            include_browser=include_browser,
            include_todo=include_todo,
            graph_dir=graph_dir,
        )
    raise ValueError(f"unsupported integration mode: {mode}")
