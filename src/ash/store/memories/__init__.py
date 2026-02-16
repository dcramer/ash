"""Memory operations split into focused modules.

Re-exports MemoryOpsMixin for backward compatibility with store.py imports.
"""

from ash.store.memories.crud import MemoryCrudMixin
from ash.store.memories.eviction import MemoryEvictionMixin
from ash.store.memories.lifecycle import MemoryLifecycleMixin


class MemoryOpsMixin(
    MemoryCrudMixin,
    MemoryLifecycleMixin,
    MemoryEvictionMixin,
):
    """Combined mixin for all memory operations.

    Split implementation:
    - crud.py: add_memory, get_memory, list_memories, delete_memory, batch_update
    - lifecycle.py: gc, forget_person, archive_memories, get_supersession_chain
    - eviction.py: enforce_max_entries, compact, clear, rebuild_index, remap
    """


__all__ = [
    "MemoryOpsMixin",
]
