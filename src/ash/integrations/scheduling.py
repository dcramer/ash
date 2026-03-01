"""Scheduling integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from ash.integrations.runtime import IntegrationContext, IntegrationContributor
from ash.llm.types import Message, Role

if TYPE_CHECKING:
    from ash.agents import AgentExecutor
    from ash.scheduling.handler import (
        MessagePersister,
        MessageRegistrar,
        MessageSender,
    )


class SchedulingIntegration(IntegrationContributor):
    """Owns scheduling storage, RPC wiring, and optional dispatch lifecycle."""

    name = "scheduling"
    priority = 300

    def __init__(
        self,
        graph_dir: Path,
        *,
        timezone: str = "UTC",
        senders: dict[str, MessageSender] | None = None,
        registrars: dict[str, MessageRegistrar] | None = None,
        persisters: dict[str, MessagePersister] | None = None,
        agent_executor: AgentExecutor | None = None,
    ) -> None:
        self._graph_dir = graph_dir
        self._timezone = timezone
        self._senders = senders or {}
        self._registrars = registrars or {}
        self._persisters = persisters or {}
        self._agent_executor = agent_executor
        self._store = None
        self._watcher = None

    @property
    def store(self):
        return self._store

    async def setup(self, context: IntegrationContext) -> None:
        from ash.scheduling import ScheduledTaskHandler, ScheduleStore, ScheduleWatcher

        self._store = ScheduleStore(self._graph_dir)
        if self._senders:
            self._watcher = ScheduleWatcher(self._store, timezone=self._timezone)
            schedule_handler = ScheduledTaskHandler(
                context.components.agent,
                self._senders,
                self._registrars,
                self._persisters,
                timezone=self._timezone,
                agent_executor=self._agent_executor,
            )
            self._watcher.add_handler(schedule_handler.handle)

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.schedule import register_schedule_methods

        if self._store is None:
            return

        async def parse_time_with_llm(time_text: str, timezone: str):
            return await self._parse_time_with_llm(
                context=context,
                time_text=time_text,
                timezone=timezone,
            )

        register_schedule_methods(
            server,
            self._store,
            parse_time_with_llm=parse_time_with_llm,
        )

    def augment_skill_instructions(
        self,
        skill_name: str,
        context: IntegrationContext,
    ) -> list[str]:
        if skill_name != "schedule":
            return []
        return []

    async def on_startup(self, context: IntegrationContext) -> None:
        if self._watcher is not None:
            await self._watcher.start()

    async def on_shutdown(self, context: IntegrationContext) -> None:
        if self._watcher is not None:
            await self._watcher.stop()

    async def _parse_time_with_llm(
        self,
        *,
        context: IntegrationContext,
        time_text: str,
        timezone: str,
    ) -> datetime | None:
        """Resolve an absolute UTC time from free-form text via LLM."""
        try:
            tz = ZoneInfo(timezone)
        except Exception:
            tz = ZoneInfo("UTC")

        model_alias = "fast" if "fast" in context.config.models else "default"
        model_config = context.config.get_model(model_alias)
        llm = context.config.create_llm_provider_for_model(model_alias)

        now_local = datetime.now(UTC).astimezone(tz)
        system_prompt = (
            "Convert user time text to one absolute datetime. "
            'Return JSON only: {"trigger_at": "<ISO8601 with offset or Z>"} '
            'or {"trigger_at": null} when ambiguous/invalid.'
        )
        user_prompt = (
            f"timezone={tz.key}\n"
            f"now_local={now_local.isoformat()}\n"
            f"time_text={time_text}\n"
            "Prefer future times."
        )

        try:
            response = await llm.complete(
                messages=[Message(role=Role.USER, content=user_prompt)],
                system=system_prompt,
                model=model_config.model,
                max_tokens=120,
            )
        except Exception:
            return None

        payload = _extract_first_json_object(response.message.get_text())
        if payload is None:
            return None

        trigger_at = payload.get("trigger_at")
        if trigger_at is None:
            return None
        if not isinstance(trigger_at, str) or not trigger_at.strip():
            return None

        try:
            parsed = datetime.fromisoformat(trigger_at.strip().replace("Z", "+00:00"))
        except ValueError:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        return parsed.astimezone(UTC)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned)
        except Exception:
            return None
    if isinstance(value, dict):
        return value
    return None
