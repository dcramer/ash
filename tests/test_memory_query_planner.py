from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock

import pytest

from ash.config import AshConfig
from ash.config.models import ConfigError, ModelConfig
from ash.llm import LLMProvider
from ash.llm.types import CompletionResponse, Message, Role
from ash.memory.query_planner import LLMQueryPlanner, resolve_query_planner_runtime


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(role=Role.ASSISTANT, content=text),
            )
        )


@pytest.mark.asyncio
async def test_llm_query_planner_rewrites_single_query() -> None:
    llm = _FakeLLM('{"query":"location home city weather context"}')
    planner = LLMQueryPlanner(
        llm=cast(LLMProvider, llm),
        model="gpt-5-mini",
        retrieval_limit=25,
    )

    query = await planner.plan(
        user_message="check weather",
        chat_type="private",
        sender_username="dcramer",
    )

    assert query.query == "location home city weather context"
    assert query.max_results == 25
    assert query.supplemental_queries == ()


@pytest.mark.asyncio
async def test_llm_query_planner_includes_lookup_queries() -> None:
    llm = _FakeLLM(
        '{"query":"weather now","lookup_queries":["where user lives","user city","where user lives"]}'
    )
    planner = LLMQueryPlanner(
        llm=cast(LLMProvider, llm),
        model="gpt-5-mini",
        retrieval_limit=25,
        max_lookup_queries=2,
    )

    query = await planner.plan(
        user_message="what's the weather like",
        chat_type="group",
        sender_username="dcramer",
    )

    assert query.query == "weather now"
    assert query.supplemental_queries == ("where user lives", "user city")


@pytest.mark.asyncio
async def test_llm_query_planner_falls_back_to_user_query_on_invalid_json() -> None:
    llm = _FakeLLM("not-json")
    planner = LLMQueryPlanner(
        llm=cast(LLMProvider, llm),
        model="gpt-5-mini",
        retrieval_limit=25,
    )

    query = await planner.plan(
        user_message="check weather",
        chat_type="private",
        sender_username="dcramer",
    )

    assert query.query == "check weather"
    assert query.max_results == 25


def test_resolve_query_planner_runtime_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AshConfig(
        models={
            "default": ModelConfig(provider="openai", model="gpt-5.2"),
            "fast": ModelConfig(provider="openai", model="gpt-5-mini"),
        }
    )
    marker = object()
    monkeypatch.setattr(
        AshConfig,
        "create_llm_provider_for_model",
        lambda _self, _alias: marker,
    )

    llm, model = resolve_query_planner_runtime(
        config=config,
        requested_alias="fast",
        default_alias="default",
    )

    assert llm is marker
    assert model == "gpt-5-mini"


def test_resolve_query_planner_runtime_uses_default_alias_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AshConfig(
        models={
            "default": ModelConfig(provider="openai", model="gpt-5.2"),
            "fast": ModelConfig(provider="openai", model="gpt-5-mini"),
        }
    )
    marker = object()
    calls: list[str] = []
    monkeypatch.setattr(
        AshConfig,
        "create_llm_provider_for_model",
        lambda _self, alias: (calls.append(alias), marker)[1],
    )

    llm, model = resolve_query_planner_runtime(
        config=config,
        requested_alias=None,
        default_alias="default",
    )

    assert llm is marker
    assert model == "gpt-5.2"
    assert calls == ["default"]


def test_resolve_query_planner_runtime_unknown_alias_raises() -> None:
    config = AshConfig(
        models={"default": ModelConfig(provider="openai", model="gpt-5.2")}
    )
    with pytest.raises(ConfigError):
        resolve_query_planner_runtime(
            config=config,
            requested_alias="openai:gpt-5-mini",
            default_alias="default",
        )
