"""LLM-based planner for memory retrieval query rewriting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ash.llm.types import Message, Role

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.llm import LLMProvider

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\\s*\\n(.*?)```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class PlannedMemoryQuery:
    """Single planned retrieval query and fetch budget."""

    query: str
    max_results: int
    supplemental_queries: tuple[str, ...] = ()


class MemoryQueryPlanner:
    """Interface for planning memory retrieval."""

    async def plan(
        self,
        *,
        user_message: str,
        chat_type: str | None,
        sender_username: str | None,
        recent_messages: tuple[str, ...] = (),
    ) -> PlannedMemoryQuery:
        raise NotImplementedError


class LLMQueryPlanner(MemoryQueryPlanner):
    """Use a fast LLM call to rewrite one retrieval query per turn."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        model: str,
        retrieval_limit: int,
        max_lookup_queries: int = 2,
    ) -> None:
        self._llm = llm
        self._model = model
        self._retrieval_limit = max(1, retrieval_limit)
        self._max_lookup_queries = max(0, max_lookup_queries)

    async def plan(
        self,
        *,
        user_message: str,
        chat_type: str | None,
        sender_username: str | None,
        recent_messages: tuple[str, ...] = (),
    ) -> PlannedMemoryQuery:
        planner_input = {
            "message": user_message,
            "chat_type": chat_type or "unknown",
            "sender_username": sender_username,
            "recent_messages": list(recent_messages),
        }

        response = await self._llm.complete(
            messages=[
                Message(role=Role.USER, content=json.dumps(planner_input)),
            ],
            model=self._model,
            system=(
                "You are a memory retrieval planner. "
                "Given the latest user message and recent chat context, decide what memories "
                "should be looked up to answer the user's actual inquiry well. "
                "Return ONLY JSON with this shape: "
                '{"query": string, "lookup_queries": string[]}. '
                "Rules: "
                "1) query should capture the user's direct request. "
                "2) lookup_queries should be additional memory lookups for missing background "
                "facts needed to answer well. "
                "3) Use recent_messages to resolve pronouns/references before choosing lookups. "
                "4) When the request is underspecified, include lookup_queries that recover "
                "relevant user context (for example location, timezone, preferences, ongoing plans, "
                "or people context). "
                "5) Keep lookup_queries concise and non-duplicative. "
                "6) Do not include explanations or markdown."
            ),
            max_tokens=200,
            temperature=0,
        )

        text = response.message.get_text().strip()
        payload = _parse_json_object(text)
        raw_query = payload.get("query")
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        if not query:
            query = user_message
        supplemental_queries = _parse_lookup_queries(
            payload.get("lookup_queries"),
            primary_query=query,
            limit=self._max_lookup_queries,
        )

        return PlannedMemoryQuery(
            query=query,
            max_results=self._retrieval_limit,
            supplemental_queries=supplemental_queries,
        )


def _parse_json_object(text: str) -> dict[str, object]:
    """Parse JSON object from plain text or fenced code output."""
    if not text:
        return {}

    candidates = [text]
    if match := _JSON_BLOCK_RE.search(text):
        candidates.insert(0, match.group(1).strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return cast(dict[str, object], data)

    return {}


def _parse_lookup_queries(
    value: object,
    *,
    primary_query: str,
    limit: int,
) -> tuple[str, ...]:
    if limit <= 0:
        return ()
    if not isinstance(value, list):
        return ()

    deduped: list[str] = []
    seen = {primary_query.casefold()}
    for item in value:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if not candidate:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= limit:
            break
    return tuple(deduped)


def resolve_query_planner_runtime(
    *,
    config: AshConfig,
    requested_alias: str | None,
    default_alias: str,
) -> tuple[LLMProvider, str]:
    """Resolve planner provider+model from a configured model alias."""
    alias = (
        requested_alias.strip()
        if requested_alias and requested_alias.strip()
        else default_alias
    )
    model_config = config.get_model(alias)
    return config.create_llm_provider_for_model(alias), model_config.model
