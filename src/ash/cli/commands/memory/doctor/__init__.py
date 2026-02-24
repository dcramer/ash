"""Doctor command for memory diagnostics and repair."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ash.cli.commands.memory.doctor.attribution import memory_doctor_attribution
from ash.cli.commands.memory.doctor.backfill_subjects import (
    memory_doctor_backfill_subjects,
)
from ash.cli.commands.memory.doctor.contradictions import memory_doctor_contradictions
from ash.cli.commands.memory.doctor.dedup import memory_doctor_dedup
from ash.cli.commands.memory.doctor.embed_missing import memory_doctor_embed_missing
from ash.cli.commands.memory.doctor.fix_names import memory_doctor_fix_names
from ash.cli.commands.memory.doctor.normalize_semantics import (
    memory_doctor_normalize_semantics,
)
from ash.cli.commands.memory.doctor.prune_missing_provenance import (
    memory_doctor_provenance_audit,
    memory_doctor_prune_missing_provenance,
)
from ash.cli.commands.memory.doctor.quality import memory_doctor_quality
from ash.cli.commands.memory.doctor.reclassify import memory_doctor_reclassify
from ash.cli.commands.memory.doctor.self_facts import memory_doctor_self_facts

if TYPE_CHECKING:
    from ash.config.models import AshConfig
    from ash.store.store import Store


@dataclass(frozen=True)
class DoctorPipelineStage:
    """A single stage in the doctor-all repair pipeline."""

    name: str
    run: Callable[[Store, AshConfig, bool], Awaitable[None]]


def _doctor_repair_pipeline() -> list[DoctorPipelineStage]:
    """Shared memory-doctor repair stages for the `all` command."""

    async def _self_facts(store: Store, _config: AshConfig, force: bool) -> None:
        await memory_doctor_self_facts(store, force=force)

    async def _attribution(store: Store, _config: AshConfig, force: bool) -> None:
        await memory_doctor_attribution(store, force=force)

    async def _fix_names(store: Store, _config: AshConfig, force: bool) -> None:
        await memory_doctor_fix_names(store, force=force)

    async def _quality(store: Store, config: AshConfig, force: bool) -> None:
        await memory_doctor_quality(store, config, force=force)

    async def _backfill_subjects(store: Store, config: AshConfig, force: bool) -> None:
        await memory_doctor_backfill_subjects(store, force=force, config=config)

    async def _normalize_semantics(
        store: Store, _config: AshConfig, force: bool
    ) -> None:
        await memory_doctor_normalize_semantics(store, force=force)

    async def _reclassify(store: Store, config: AshConfig, force: bool) -> None:
        await memory_doctor_reclassify(store, config, force=force)

    async def _dedup(store: Store, config: AshConfig, force: bool) -> None:
        await memory_doctor_dedup(store, config, force=force)

    async def _contradictions(store: Store, config: AshConfig, force: bool) -> None:
        await memory_doctor_contradictions(store, config, force=force)

    return [
        DoctorPipelineStage(name="self-facts", run=_self_facts),
        DoctorPipelineStage(name="attribution", run=_attribution),
        DoctorPipelineStage(name="fix-names", run=_fix_names),
        DoctorPipelineStage(name="quality", run=_quality),
        DoctorPipelineStage(name="backfill-subjects", run=_backfill_subjects),
        DoctorPipelineStage(name="normalize-semantics", run=_normalize_semantics),
        DoctorPipelineStage(name="reclassify", run=_reclassify),
        DoctorPipelineStage(name="dedup", run=_dedup),
        DoctorPipelineStage(name="contradictions", run=_contradictions),
    ]


async def memory_doctor_all(store: Store, config: AshConfig, force: bool) -> None:
    """Run the full repair pipeline with a shared stage ordering."""
    for stage in _doctor_repair_pipeline():
        await stage.run(store, config, force)


__all__ = [
    "DoctorPipelineStage",
    "memory_doctor_all",
    "memory_doctor_attribution",
    "memory_doctor_backfill_subjects",
    "memory_doctor_contradictions",
    "memory_doctor_dedup",
    "memory_doctor_embed_missing",
    "memory_doctor_fix_names",
    "memory_doctor_normalize_semantics",
    "memory_doctor_provenance_audit",
    "memory_doctor_prune_missing_provenance",
    "memory_doctor_quality",
    "memory_doctor_reclassify",
    "memory_doctor_self_facts",
]
