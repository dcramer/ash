# Subsystems

> Modular components with clear boundaries, defined interfaces, and isolated responsibilities

## Intent

Subsystems solve the problem of complexity growth. As the codebase expands, tight coupling makes changes risky and testing difficult. Subsystems provide:

1. **Clear boundaries** - Each subsystem owns a specific domain with defined responsibilities
2. **Stable interfaces** - Consumers depend on public API, not implementation details
3. **Isolated testing** - Subsystems can be tested independently with mocked dependencies
4. **Maintainability** - Changes inside a subsystem don't ripple across the codebase

A subsystem is NOT:
- A microservice (subsystems live in the same process)
- A plugin (subsystems are core functionality, not optional extensions)
- Just a directory (subsystems have explicit contracts)

## Outcomes

### Each subsystem has a single responsibility

| Subsystem | Responsibility | NOT responsible for |
|-----------|----------------|---------------------|
| memory | Long-term fact storage and retrieval | Conversation history, session state |
| sessions | Conversation persistence and context | Fact extraction, semantic search |
| scheduling | Deferred task execution | Task content, routing |

### Consumers use public API only

Imports should come from the subsystem root, not internal modules:

```python
# Good - public API
from ash.memory import MemoryManager, create_memory_manager

# Avoid - internal implementation
from ash.memory.store import MemoryStore
```

Internal components may be exposed for advanced composition but are not part of the stable contract.

### Subsystems are independently testable

Each subsystem can be tested with:
- Mocked dependencies (database, LLM, other subsystems)
- Real dependencies in integration tests
- No reliance on other subsystems' internals

### Dependencies flow one direction

```
core/agent.py  →  subsystem/  →  db/models.py
                              →  llm/
tools/         →  subsystem/
cli/           →  subsystem/
```

Subsystems:
- MAY depend on foundational layers (db, llm, config)
- MUST NOT depend on other subsystems directly
- MUST NOT depend on core/agent (that's the orchestrator)

When subsystems need to interact, the agent orchestrates or events are used.

## Structure

Each subsystem follows a consistent layout:

```
src/ash/{subsystem}/
    __init__.py        # Public API with docstring
    types.py           # Public types (dataclasses, enums)
    manager.py         # Primary facade + factory function
    {internal}.py      # Implementation modules
```

### `__init__.py` - Public contract

Documents what consumers can depend on:

```python
"""One-line description.

Public API:
- PrimaryManager: Main entry point
- create_primary_manager: Factory function

Types:
- PublicType1, PublicType2

Internal (for composition):
- InternalComponent1, InternalComponent2
"""

from ash.{subsystem}.manager import PrimaryManager, create_primary_manager
from ash.{subsystem}.types import PublicType1, PublicType2

__all__ = [...]
```

### Factory function

Encapsulates internal wiring so consumers don't need to know about components:

```python
async def create_memory_manager(
    db_session: AsyncSession,
    llm_registry: LLMRegistry,
    ...
) -> MemoryManager:
    """Create fully-wired manager."""
    # Internal wiring hidden from consumers
```

### Types in `types.py`

Public types live in one place, not scattered across implementation files:
- Dataclasses for results and context
- Enums for status/state
- TypedDicts for complex parameters

## Current Subsystems

| Subsystem | Status | Spec |
|-----------|--------|------|
| memory | Complete | [specs/memory.md](memory.md) |
| people | Complete | [specs/people.md](people.md) |
| sessions | Needs refactor | - |
| scheduling | Complete | [specs/schedule.md](schedule.md) |

## Verification

For each subsystem:

- [ ] Has outcome-focused spec in `specs/`
- [ ] Public API documented in `__init__.py`
- [ ] Types centralized in `types.py`
- [ ] Factory function for wiring
- [ ] Tests pass with mocked dependencies
- [ ] No imports from other subsystems
- [ ] Consumers import from root, not internal modules
