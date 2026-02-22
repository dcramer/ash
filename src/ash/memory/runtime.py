"""Memory runtime bootstrap for store and extractor wiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from ash.llm import create_registry
from ash.memory.extractor import MemoryExtractor
from ash.store import create_store

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.llm import LLMProvider
    from ash.store.store import Store


@dataclass(slots=True)
class MemoryRuntime:
    """Initialized memory runtime state."""

    store: Store | None = None
    extractor: MemoryExtractor | None = None


async def initialize_memory_runtime(
    *,
    config: AshConfig,
    graph_dir: Path | None,
    model_alias: str,
    initialize_extractor: bool = True,
    logger: logging.Logger | None = None,
) -> MemoryRuntime:
    """Initialize graph store and optional extraction runtime for memory subsystem."""
    log = logger or logging.getLogger(__name__)

    if not graph_dir:
        log.info("memory_tools_disabled", extra={"config.reason": "no_graph_directory"})
        return MemoryRuntime()

    if not config.embeddings:
        log.info(
            "memory_tools_disabled",
            extra={"config.reason": "embeddings_not_configured"},
        )
        return MemoryRuntime()

    store: Store | None = None
    extractor: MemoryExtractor | None = None

    try:
        embeddings_key = config.resolve_embeddings_api_key()
        if not embeddings_key:
            log.info(
                "memory_tools_disabled",
                extra={
                    "config.reason": "no_api_key",
                    "embeddings.provider": config.embeddings.provider,
                },
            )
            return MemoryRuntime()

        config.get_model("default")
        openai_key = config._resolve_provider_api_key("openai")
        anthropic_key = config._resolve_provider_api_key("anthropic")
        if config.embeddings.provider == "openai" and not openai_key:
            openai_key = embeddings_key

        llm_registry = create_registry(
            anthropic_api_key=anthropic_key.get_secret_value()
            if anthropic_key
            else None,
            openai_api_key=openai_key.get_secret_value() if openai_key else None,
        )
        store = await create_store(
            graph_dir=graph_dir,
            llm_registry=llm_registry,
            embedding_model=config.embeddings.model,
            embedding_provider=config.embeddings.provider,
            max_entries=config.memory.max_entries,
        )
        log.debug("Store initialized")
    except ValueError as error:
        log.debug("Memory disabled: %s", error)
        return MemoryRuntime()
    except Exception:
        log.warning("Failed to initialize graph store", exc_info=True)
        return MemoryRuntime()

    if not initialize_extractor or not config.memory.extraction_enabled:
        return MemoryRuntime(store=store)

    extraction_model_alias = config.memory.extraction_model or model_alias
    try:
        extraction_model_config = config.get_model(extraction_model_alias)
        extraction_llm = config.create_llm_provider_for_model(extraction_model_alias)
        grounding_llm, grounding_model_name = _resolve_grounding_model_runtime(
            config=config
        )
        extractor = MemoryExtractor(
            llm=extraction_llm,
            model=extraction_model_config.model,
            confidence_threshold=config.memory.extraction_confidence_threshold,
            grounding_enabled=config.memory.extraction_grounding_enabled,
            grounding_llm=grounding_llm,
            grounding_model=grounding_model_name,
        )
        log.debug(
            "Memory extractor initialized (model=%s grounding_model=%s)",
            extraction_model_config.model,
            grounding_model_name,
        )
        store.set_llm(extraction_llm, extraction_model_config.model)
    except Exception:
        log.warning("Failed to initialize memory extractor", exc_info=True)

    return MemoryRuntime(store=store, extractor=extractor)


def _resolve_grounding_model_runtime(
    *,
    config: AshConfig,
) -> tuple[LLMProvider, str]:
    """Resolve provider + concrete model name for grounding.

    Supports:
    - None: use configured `default` model alias.
    - Alias: use a configured `[models.<alias>]`.
    - Provider-qualified: `openai:gpt-5.2`, `anthropic:claude-sonnet-4-5`.
    - Plain model name: use default model's provider with provided model name.
    """
    requested = config.memory.extraction_grounding_model
    if requested is None:
        alias = "default"
        model_config = config.get_model(alias)
        return config.create_llm_provider_for_model(alias), model_config.model

    requested = requested.strip()
    if not requested:
        alias = "default"
        model_config = config.get_model(alias)
        return config.create_llm_provider_for_model(alias), model_config.model

    if requested in config.models:
        model_config = config.get_model(requested)
        return config.create_llm_provider_for_model(requested), model_config.model

    if ":" in requested:
        provider, model_name = requested.split(":", 1)
        provider = provider.strip()
        model_name = model_name.strip()
        if provider not in {"anthropic", "openai", "openai-oauth"}:
            raise ValueError(
                "Invalid extraction_grounding_model provider. "
                "Expected anthropic/openai/openai-oauth."
            )
        if not model_name:
            raise ValueError("Invalid extraction_grounding_model: missing model name.")
        typed_provider = cast(Literal["anthropic", "openai", "openai-oauth"], provider)
        return config.create_llm_provider_for_provider(typed_provider), model_name

    default_provider = config.default_model.provider
    return config.create_llm_provider_for_provider(default_provider), requested
