"""Doctor command for memory diagnostics and repair."""

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

__all__ = [
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
