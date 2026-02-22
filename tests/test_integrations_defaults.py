from __future__ import annotations

from pathlib import Path

import pytest

from ash.integrations import (
    MemoryIntegration,
    RuntimeRPCIntegration,
    SchedulingIntegration,
    create_default_integrations,
)


def test_create_default_integrations_chat_includes_memory() -> None:
    result = create_default_integrations(mode="chat")

    assert len(result.contributors) == 1
    assert isinstance(result.contributors[0], MemoryIntegration)
    assert result.scheduling is None


def test_create_default_integrations_chat_can_disable_memory() -> None:
    result = create_default_integrations(mode="chat", include_memory=False)

    assert result.contributors == []
    assert result.scheduling is None


def test_create_default_integrations_eval_requires_schedule_file() -> None:
    with pytest.raises(ValueError, match="schedule_file"):
        create_default_integrations(mode="eval")


def test_create_default_integrations_eval_order() -> None:
    result = create_default_integrations(
        mode="eval",
        include_memory=True,
        schedule_file=Path("schedule.jsonl"),
    )

    assert len(result.contributors) == 2
    assert isinstance(result.contributors[0], SchedulingIntegration)
    assert isinstance(result.contributors[1], MemoryIntegration)
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_eval_can_disable_memory() -> None:
    result = create_default_integrations(
        mode="eval",
        include_memory=False,
        schedule_file=Path("schedule.jsonl"),
    )

    assert len(result.contributors) == 1
    assert isinstance(result.contributors[0], SchedulingIntegration)
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_serve_requires_paths() -> None:
    with pytest.raises(ValueError, match="logs_path"):
        create_default_integrations(mode="serve", schedule_file=Path("schedule.jsonl"))


def test_create_default_integrations_serve_order() -> None:
    result = create_default_integrations(
        mode="serve",
        include_memory=True,
        schedule_file=Path("schedule.jsonl"),
        logs_path=Path("logs"),
    )

    assert len(result.contributors) == 3
    assert isinstance(result.contributors[0], RuntimeRPCIntegration)
    assert isinstance(result.contributors[1], MemoryIntegration)
    assert isinstance(result.contributors[2], SchedulingIntegration)
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_serve_can_disable_memory() -> None:
    result = create_default_integrations(
        mode="serve",
        include_memory=False,
        schedule_file=Path("schedule.jsonl"),
        logs_path=Path("logs"),
    )

    assert len(result.contributors) == 2
    assert isinstance(result.contributors[0], RuntimeRPCIntegration)
    assert isinstance(result.contributors[1], SchedulingIntegration)
    assert isinstance(result.scheduling, SchedulingIntegration)


def test_create_default_integrations_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="unsupported integration mode"):
        create_default_integrations(
            mode="bad-mode",  # type: ignore[arg-type]
            include_memory=True,
        )
