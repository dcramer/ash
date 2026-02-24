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


class MemoryQueryPlanner:
    """Interface for planning memory retrieval."""

    async def plan(
        self,
        *,
        user_message: str,
        chat_type: str | None,
        sender_username: str | None,
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
    ) -> None:
        self._llm = llm
        self._model = model
        self._retrieval_limit = max(1, retrieval_limit)

    async def plan(
        self,
        *,
        user_message: str,
        chat_type: str | None,
        sender_username: str | None,
    ) -> PlannedMemoryQuery:
        planner_input = {
            "message": user_message,
            "chat_type": chat_type or "unknown",
            "sender_username": sender_username,
        }

        response = await self._llm.complete(
            messages=[
                Message(role=Role.USER, content=json.dumps(planner_input)),
            ],
            model=self._model,
            system=(
                "Rewrite the user message into ONE memory retrieval query that maximizes recall "
                "for relevant user context (location/preferences/people/tasks/recent context) "
                'without changing intent. Return ONLY JSON: {"query": string}.'
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

        return PlannedMemoryQuery(
            query=query,
            max_results=self._retrieval_limit,
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
