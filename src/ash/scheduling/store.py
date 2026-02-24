"""Schedule store backed by ash.graph schedule nodes."""

from __future__ import annotations

import fcntl
import logging
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, TypeVar, cast

from ash.graph.edges import (
    SCHEDULE_FOR_CHAT,
    SCHEDULE_FOR_USER,
    create_schedule_for_chat_edge,
    create_schedule_for_user_edge,
)
from ash.graph.persistence import GraphPersistence, hydrate_graph
from ash.scheduling.types import ScheduleEntry, register_schedule_graph_schema

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class ScheduleStore:
    """Graph-backed storage for schedule entries."""

    def __init__(self, graph_dir: Path) -> None:
        register_schedule_graph_schema()
        # Backward-compatible path normalization for callers/tests that still
        # pass a legacy schedule file path (e.g. ".../schedule.jsonl").
        if graph_dir.suffix == ".jsonl":
            graph_dir = graph_dir.parent / "graph"
        self._graph_dir = graph_dir
        self._persistence = GraphPersistence(self._graph_dir)
        self._lock_file = self._graph_dir / ".schedule.lock"

    @property
    def graph_dir(self) -> Path:
        return self._graph_dir

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_entries(self) -> list[ScheduleEntry]:
        graph = self._load_graph()
        raw = cast(dict[str, Any], graph.get_node_collection("schedule_entry"))
        entries = [entry for entry in raw.values() if isinstance(entry, ScheduleEntry)]
        for i, entry in enumerate(entries):
            entry.line_number = i
        return entries

    def get_entry(self, entry_id: str) -> ScheduleEntry | None:
        graph = self._load_graph()
        raw = graph.get_node_collection("schedule_entry").get(entry_id)
        return raw if isinstance(raw, ScheduleEntry) else None

    def get_stats(self, timezone: str = "UTC") -> dict[str, Any]:
        entries = self.get_entries()
        periodic_count = sum(1 for e in entries if e.is_periodic)
        due_count = sum(1 for e in entries if e.is_due(timezone))
        return {
            "graph_dir": str(self._graph_dir),
            "total": len(entries),
            "one_shot": len(entries) - periodic_count,
            "periodic": periodic_count,
            "due": due_count,
        }

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_entry(self, entry: ScheduleEntry) -> None:
        if not entry.id:
            entry.id = uuid.uuid4().hex[:8]
        entry_id = entry.id

        def mutate(entries: dict[str, ScheduleEntry]) -> None:
            entries[entry_id] = entry

        self._mutate_graph(mutate, dirty_edges=True)

    def remove_entry(self, entry_id: str) -> bool:
        removed = False

        def mutate(entries: dict[str, ScheduleEntry]) -> None:
            nonlocal removed
            removed = entry_id in entries
            if removed:
                entries.pop(entry_id, None)

        self._mutate_graph(mutate, dirty_edges=removed)
        return removed

    def update_entry(
        self,
        entry_id: str,
        message: str | None = None,
        trigger_at: datetime | None = None,
        cron: str | None = None,
        timezone: str | None = None,
    ) -> ScheduleEntry | None:
        if message is None and trigger_at is None and cron is None and timezone is None:
            raise ValueError("At least one updatable field must be provided")

        updated: ScheduleEntry | None = None

        def mutate(entries: dict[str, ScheduleEntry]) -> None:
            nonlocal updated
            entry = entries.get(entry_id)
            if not entry:
                return
            updated = _apply_updates(
                entry,
                message=message,
                trigger_at=trigger_at,
                cron=cron,
                timezone=timezone,
            )
            entries[entry_id] = updated

        self._mutate_graph(mutate)
        return updated

    def clear_all(self) -> int:
        removed = 0

        def mutate(entries: dict[str, ScheduleEntry]) -> None:
            nonlocal removed
            removed = len(entries)
            entries.clear()

        self._mutate_graph(mutate, dirty_edges=removed > 0)
        return removed

    def remove_and_update(
        self,
        remove_ids: set[str],
        updates: dict[str, ScheduleEntry],
    ) -> None:
        def mutate(entries: dict[str, ScheduleEntry]) -> None:
            for entry_id in remove_ids:
                entries.pop(entry_id, None)
            for entry_id, entry in updates.items():
                entries[entry_id] = entry

        self._mutate_graph(mutate, dirty_edges=bool(remove_ids))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _file_lock(self, file: IO) -> Iterator[None]:
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)

    def _load_graph(self):
        self._graph_dir.mkdir(parents=True, exist_ok=True)
        raw_data = self._persistence.load_raw_sync()
        return hydrate_graph(raw_data)

    def _mutate_graph(
        self,
        mutate: Callable[[dict[str, ScheduleEntry]], _T | None],
        *,
        dirty_edges: bool = False,
    ) -> _T | None:
        self._graph_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)

        with self._lock_file.open("a+") as lockf:
            with self._file_lock(lockf):
                graph = self._load_graph()
                entries = cast(
                    dict[str, ScheduleEntry],
                    graph.get_node_collection("schedule_entry"),
                )
                previous_ids = set(entries)

                result = mutate(entries)

                # Rebuild schedule nodes and schedule edges from canonical entries.
                current_ids = set(entries)
                removed_ids = previous_ids - current_ids
                for entry_id in removed_ids:
                    graph.remove_node("schedule_entry", entry_id)
                for entry_id, entry in entries.items():
                    if graph.get_node(entry_id) is None:
                        graph.add_node("schedule_entry", entry)
                    self._sync_entry_edges(graph, entry)

                dirty = {"schedules"}
                if dirty_edges or removed_ids or entries:
                    dirty.add("edges")
                self._persist_graph(graph, dirty=dirty)
                return result

    def _sync_entry_edges(self, graph, entry: ScheduleEntry) -> None:
        if not entry.id:
            return
        for edge in list(graph.get_outgoing(entry.id, edge_type=SCHEDULE_FOR_CHAT)):
            graph.remove_edge(edge.id)
        for edge in list(graph.get_outgoing(entry.id, edge_type=SCHEDULE_FOR_USER)):
            graph.remove_edge(edge.id)
        if entry.chat_id:
            graph.add_edge(create_schedule_for_chat_edge(entry.id, entry.chat_id))
        if entry.user_id:
            graph.add_edge(create_schedule_for_user_edge(entry.id, entry.user_id))

    def _persist_graph(self, graph, *, dirty: set[str]) -> None:
        self._persistence.mark_dirty(*dirty)
        self._persistence.flush_sync(graph)


def _apply_updates(
    entry: ScheduleEntry,
    message: str | None = None,
    trigger_at: datetime | None = None,
    cron: str | None = None,
    timezone: str | None = None,
) -> ScheduleEntry:
    """Apply updates to an entry with validation.

    Raises:
        ValueError: If the update is invalid.
    """
    if trigger_at is not None and entry.cron is not None:
        raise ValueError("Cannot change periodic entry to one-shot")
    if cron is not None and entry.trigger_at is not None:
        raise ValueError("Cannot change one-shot entry to periodic")

    if trigger_at is not None and trigger_at <= datetime.now(UTC):
        raise ValueError("trigger_at must be in the future")

    if cron is not None:
        try:
            from croniter import croniter

            croniter(cron)
        except Exception as e:
            raise ValueError(f"Invalid cron expression: {e}") from e

    if message is not None:
        entry.message = message
    if trigger_at is not None:
        entry.trigger_at = trigger_at
    if cron is not None:
        entry.cron = cron
    if timezone is not None:
        entry.timezone = timezone

    return entry
