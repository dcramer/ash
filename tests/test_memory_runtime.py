from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr

from ash.config import AshConfig
from ash.config.models import EmbeddingsConfig, ModelConfig
from ash.memory.runtime import initialize_memory_runtime


def _config(tmp_path: Path) -> AshConfig:
    return AshConfig(
        workspace=tmp_path / "workspace",
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
        embeddings=EmbeddingsConfig(provider="openai", model="text-embedding-3-small"),
    )


@pytest.mark.asyncio
async def test_initialize_memory_runtime_disables_without_graph_dir(
    tmp_path: Path,
) -> None:
    runtime = await initialize_memory_runtime(
        config=_config(tmp_path),
        graph_dir=None,
        model_alias="default",
    )

    assert runtime.store is None
    assert runtime.extractor is None


@pytest.mark.asyncio
async def test_initialize_memory_runtime_builds_store_and_extractor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    fake_store = cast(Any, SimpleNamespace())
    set_llm_calls: list[tuple[object, str]] = []
    fake_store.set_llm = lambda llm, model: set_llm_calls.append((llm, model))

    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        AshConfig,
        "resolve_embeddings_api_key",
        lambda _self: SecretStr("embed-key"),
    )
    monkeypatch.setattr(
        AshConfig,
        "_resolve_provider_api_key",
        lambda _self, _provider: None,
    )

    def _fake_create_registry(**kwargs: Any) -> object:
        calls["registry_kwargs"] = kwargs
        return object()

    monkeypatch.setattr("ash.memory.runtime.create_registry", _fake_create_registry)

    async def _fake_create_store(**kwargs: Any) -> object:
        calls["store_kwargs"] = kwargs
        return fake_store

    monkeypatch.setattr("ash.memory.runtime.create_store", _fake_create_store)

    fake_extraction_llm = object()
    monkeypatch.setattr(
        AshConfig,
        "create_llm_provider_for_model",
        lambda _self, _alias: fake_extraction_llm,
    )

    class _FakeExtractor:
        def __init__(
            self,
            *,
            llm: object,
            model: str,
            confidence_threshold: float,
        ) -> None:
            calls["extractor_args"] = {
                "llm": llm,
                "model": model,
                "confidence_threshold": confidence_threshold,
            }

    monkeypatch.setattr("ash.memory.runtime.MemoryExtractor", _FakeExtractor)

    runtime = await initialize_memory_runtime(
        config=config,
        graph_dir=tmp_path / "graph",
        model_alias="default",
    )

    assert runtime.store is fake_store
    assert runtime.extractor is not None
    assert calls["registry_kwargs"]["openai_api_key"] == "embed-key"
    assert calls["store_kwargs"]["graph_dir"] == tmp_path / "graph"
    assert calls["extractor_args"]["llm"] is fake_extraction_llm
    assert set_llm_calls == [(fake_extraction_llm, "gpt-5-mini")]


@pytest.mark.asyncio
async def test_initialize_memory_runtime_tolerates_extractor_init_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    fake_store = cast(Any, SimpleNamespace(set_llm=lambda *_args: None))

    monkeypatch.setattr(
        AshConfig,
        "resolve_embeddings_api_key",
        lambda _self: SecretStr("embed-key"),
    )
    monkeypatch.setattr(
        AshConfig,
        "_resolve_provider_api_key",
        lambda _self, _provider: None,
    )
    monkeypatch.setattr(
        "ash.memory.runtime.create_registry", lambda **_kwargs: object()
    )

    async def _fake_create_store(**_kwargs: Any) -> object:
        return fake_store

    monkeypatch.setattr("ash.memory.runtime.create_store", _fake_create_store)
    monkeypatch.setattr(
        AshConfig,
        "create_llm_provider_for_model",
        lambda _self, _alias: object(),
    )
    monkeypatch.setattr("ash.memory.runtime.MemoryExtractor", lambda **_kwargs: 1 / 0)

    runtime = await initialize_memory_runtime(
        config=config,
        graph_dir=tmp_path / "graph",
        model_alias="default",
    )

    assert runtime.store is fake_store
    assert runtime.extractor is None


@pytest.mark.asyncio
async def test_initialize_memory_runtime_can_skip_extractor_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    fake_store = cast(Any, SimpleNamespace())

    monkeypatch.setattr(
        AshConfig,
        "resolve_embeddings_api_key",
        lambda _self: SecretStr("embed-key"),
    )
    monkeypatch.setattr(
        AshConfig,
        "_resolve_provider_api_key",
        lambda _self, _provider: None,
    )
    monkeypatch.setattr(
        "ash.memory.runtime.create_registry", lambda **_kwargs: object()
    )

    async def _fake_create_store(**_kwargs: Any) -> object:
        return fake_store

    monkeypatch.setattr("ash.memory.runtime.create_store", _fake_create_store)
    monkeypatch.setattr(
        AshConfig,
        "create_llm_provider_for_model",
        lambda _self, _alias: 1 / 0,
    )

    runtime = await initialize_memory_runtime(
        config=config,
        graph_dir=tmp_path / "graph",
        model_alias="default",
        initialize_extractor=False,
    )

    assert runtime.store is fake_store
    assert runtime.extractor is None
