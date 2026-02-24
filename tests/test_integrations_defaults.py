from __future__ import annotations

from pathlib import Path

import pytest

from ash.integrations import (
    BrowserIntegration,
    CapabilitiesIntegration,
    ImageIntegration,
    MemoryIntegration,
    RuntimeRPCIntegration,
    SchedulingIntegration,
    TodoIntegration,
    create_default_integrations,
)


def test_create_default_integrations_chat_includes_memory() -> None:
    result = create_default_integrations(mode="chat")

    assert len(result.contributors) == 5
    assert isinstance(result.contributors[0], ImageIntegration)
    assert isinstance(result.contributors[1], BrowserIntegration)
    assert isinstance(result.contributors[2], CapabilitiesIntegration)
    assert isinstance(result.contributors[3], TodoIntegration)
    assert isinstance(result.contributors[4], MemoryIntegration)
    assert result.scheduling is None


def test_create_default_integrations_chat_can_disable_memory() -> None:
    result = create_default_integrations(mode="chat", include_memory=False)

    assert len(result.contributors) == 4
    assert isinstance(result.contributors[0], ImageIntegration)
    assert isinstance(result.contributors[1], BrowserIntegration)
    assert isinstance(result.contributors[2], CapabilitiesIntegration)
    assert isinstance(result.contributors[3], TodoIntegration)
    assert result.scheduling is None


def test_create_default_integrations_chat_can_disable_todo() -> None:
    result = create_default_integrations(mode="chat", include_todo=False)

    assert len(result.contributors) == 4
    assert isinstance(result.contributors[0], ImageIntegration)
    assert isinstance(result.contributors[1], BrowserIntegration)
    assert isinstance(result.contributors[2], CapabilitiesIntegration)
    assert isinstance(result.contributors[3], MemoryIntegration)
    assert result.scheduling is None


def test_create_default_integrations_eval_uses_graph_dir_by_default() -> None:
    result = create_default_integrations(mode="eval")
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_eval_order() -> None:
    result = create_default_integrations(
        mode="eval",
        include_memory=True,
    )

    assert len(result.contributors) == 6
    assert isinstance(result.contributors[0], SchedulingIntegration)
    assert isinstance(result.contributors[1], ImageIntegration)
    assert isinstance(result.contributors[2], BrowserIntegration)
    assert isinstance(result.contributors[3], CapabilitiesIntegration)
    assert isinstance(result.contributors[4], TodoIntegration)
    assert isinstance(result.contributors[5], MemoryIntegration)
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_eval_can_disable_memory() -> None:
    result = create_default_integrations(
        mode="eval",
        include_memory=False,
    )

    assert len(result.contributors) == 5
    assert isinstance(result.contributors[0], SchedulingIntegration)
    assert isinstance(result.contributors[1], ImageIntegration)
    assert isinstance(result.contributors[2], BrowserIntegration)
    assert isinstance(result.contributors[3], CapabilitiesIntegration)
    assert isinstance(result.contributors[4], TodoIntegration)
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_eval_can_disable_todo() -> None:
    result = create_default_integrations(
        mode="eval",
        include_todo=False,
    )

    assert len(result.contributors) == 5
    assert isinstance(result.contributors[0], SchedulingIntegration)
    assert isinstance(result.contributors[1], ImageIntegration)
    assert isinstance(result.contributors[2], BrowserIntegration)
    assert isinstance(result.contributors[3], CapabilitiesIntegration)
    assert isinstance(result.contributors[4], MemoryIntegration)
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_serve_requires_paths() -> None:
    with pytest.raises(ValueError, match="logs_path"):
        create_default_integrations(mode="serve")


def test_create_default_integrations_serve_order() -> None:
    result = create_default_integrations(
        mode="serve",
        include_memory=True,
        logs_path=Path("logs"),
    )

    assert len(result.contributors) == 7
    assert isinstance(result.contributors[0], RuntimeRPCIntegration)
    assert isinstance(result.contributors[1], ImageIntegration)
    assert isinstance(result.contributors[2], BrowserIntegration)
    assert isinstance(result.contributors[3], CapabilitiesIntegration)
    assert isinstance(result.contributors[4], TodoIntegration)
    assert isinstance(result.contributors[5], MemoryIntegration)
    assert isinstance(result.contributors[6], SchedulingIntegration)
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_serve_can_disable_memory() -> None:
    result = create_default_integrations(
        mode="serve",
        include_memory=False,
        logs_path=Path("logs"),
    )

    assert len(result.contributors) == 6
    assert isinstance(result.contributors[0], RuntimeRPCIntegration)
    assert isinstance(result.contributors[1], ImageIntegration)
    assert isinstance(result.contributors[2], BrowserIntegration)
    assert isinstance(result.contributors[3], CapabilitiesIntegration)
    assert isinstance(result.contributors[4], TodoIntegration)
    assert isinstance(result.contributors[5], SchedulingIntegration)
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_serve_can_disable_todo() -> None:
    result = create_default_integrations(
        mode="serve",
        include_todo=False,
        logs_path=Path("logs"),
    )

    assert len(result.contributors) == 6
    assert isinstance(result.contributors[0], RuntimeRPCIntegration)
    assert isinstance(result.contributors[1], ImageIntegration)
    assert isinstance(result.contributors[2], BrowserIntegration)
    assert isinstance(result.contributors[3], CapabilitiesIntegration)
    assert isinstance(result.contributors[4], MemoryIntegration)
    assert isinstance(result.contributors[5], SchedulingIntegration)
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="unsupported integration mode"):
        create_default_integrations(
            mode="bad-mode",  # type: ignore[arg-type]
            include_memory=True,
        )
