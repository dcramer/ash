# Graph Store

> In-memory knowledge graph backed by JSONL files and numpy vectors, with explicit typed edges and multi-hop retrieval.

## Intent

The graph store is the unified storage and retrieval layer for Ash. All memories, people, users, and chats are nodes in an in-memory `KnowledgeGraph`. All relationships (memory-about-person, user-is-person, supersession chains, merge chains) are explicit typed `Edge` objects with their own adjacency indexes.

**Design principles:**
- **JSONL source of truth** — Human-readable, atomic writes via tempfile + `os.replace()`
- **In-memory graph** — All queries run against dicts and adjacency lists, no SQL
- **Numpy vectors** — Brute-force cosine similarity, no C extension dependencies
- **Explicit edges** — All relationships are first-class Edge objects with temporal metadata
- **Multi-hop retrieval** — BFS traversal discovers related memories across graph hops

**Not:**
- A graph database (no query language, no transactions)
- An RDF/property graph system (simple typed edges, not triples)

## On-Disk Layout

```
~/.ash/graph/
├── memories.jsonl           # One MemoryEntry per line
├── people.jsonl             # One PersonEntry per line
├── users.jsonl              # One UserEntry per line
├── chats.jsonl              # One ChatEntry per line
├── edges.jsonl              # All typed edges
└── embeddings/
    ├── memories.npy         # float32 array, shape (N, 1536)
    └── memories.ids.json    # row index → memory_id mapping
```

**Persistence strategy:**
- Startup: Load all JSONL → build in-memory KnowledgeGraph + adjacency indexes
- Mutation: Update in-memory → rewrite affected JSONL file atomically
- Vectors: Numpy `.npy` files + JSON ID mapping, saved after embedding changes
- Scale: ~1MB per 2000 entries, <10ms to rewrite a file

## Graph Model

### Nodes

| Type | File | Dataclass | Purpose |
|------|------|-----------|---------|
| **Memory** | `memories.jsonl` | `MemoryEntry` | Facts, preferences, observations |
| **Person** | `people.jsonl` | `PersonEntry` | Identity entities (people in user's life) |
| **User** | `users.jsonl` | `UserEntry` | Provider identity (Telegram user, etc.) |
| **Chat** | `chats.jsonl` | `ChatEntry` | Chat/channel from a provider |

### Edge Types

All edges are stored in `edges.jsonl` and indexed in in-memory adjacency lists.

| Edge Type | Direction | Purpose |
|-----------|-----------|---------|
| `ABOUT` | Memory → Person | Memory is about this person |
| `STATED_BY` | Memory → Person | Person stated this fact |
| `SUPERSEDES` | Memory → Memory | New memory replaces old |
| `IS_PERSON` | User → Person | User identity maps to person |
| `MERGED_INTO` | Person → Person | Duplicate person merged into primary |
| `HAS_RELATIONSHIP` | Person → Person | Relationship link between people |

**Edge dataclass** (`ash.graph.graph.Edge`):
```python
@dataclass
class Edge:
    id: str
    edge_type: str          # ABOUT, SUPERSEDES, etc.
    source_type: str        # "memory", "person", "user"
    source_id: str
    target_type: str
    target_id: str
    weight: float = 1.0
    properties: dict | None = None
    created_at: datetime | None = None
    valid_from: datetime | None = None
    invalid_at: datetime | None = None  # Soft-delete timestamp
    created_by: str | None = None
```

### Dual-Write Pattern

During the FK → edge migration, both FK fields and edges are maintained:
- **Writes** create/update both the FK field (e.g., `memory.subject_person_ids`) and the corresponding edge (e.g., `ABOUT`)
- **Reads** use edge queries via adjacency lists for key paths (retrieval, lifecycle, supersession)
- **Backfill** auto-populates edges from FK fields on first load when `edges.jsonl` is empty

### Key Traversals

| Query | Method | Via |
|-------|--------|-----|
| Memories about person X | `get_memories_about_person(graph, pid)` | Incoming ABOUT edges |
| Person for user | `get_person_for_user(graph, uid)` | Outgoing IS_PERSON edge |
| Users for person | `get_users_for_person(graph, pid)` | Incoming IS_PERSON edges |
| Supersession chain | `get_supersession_targets(graph, mid)` | Outgoing SUPERSEDES edges |
| Merge chain | `follow_merge_chain(graph, pid)` | Outgoing MERGED_INTO edges |
| Multi-hop discovery | `bfs_traverse(graph, seeds, max_hops=2)` | All edges (excl. SUPERSEDES) |

## Architecture

### KnowledgeGraph (`graph/graph.py`)

In-memory data structure holding all nodes and edges with adjacency indexes.

```
KnowledgeGraph
├── memories: dict[str, MemoryEntry]
├── people: dict[str, PersonEntry]
├── users: dict[str, UserEntry]
├── chats: dict[str, ChatEntry]
├── edges: dict[str, Edge]
├── _outgoing: defaultdict[str, list[str]]   # node_id → [edge_ids]
├── _incoming: defaultdict[str, list[str]]   # node_id → [edge_ids]
├── _edges_by_type: defaultdict[str, list[str]]
└── _node_type: dict[str, str]               # node_id → "memory"|"person"|...
```

Operations: `add_edge()`, `remove_edge()`, `invalidate_edge()`, `get_outgoing()`, `get_incoming()`.

### GraphPersistence (`graph/persistence.py`)

JSONL load/save with atomic writes. Each node type has its own `.jsonl` file.

- `load()` — Reads all JSONL files, builds KnowledgeGraph, auto-backfills edges
- `save_memories()`, `save_people()`, `save_users()`, `save_chats()`, `save_edges()` — Atomic rewrite

### NumpyVectorIndex (`graph/vectors.py`)

Brute-force cosine similarity using numpy matrix multiplication.

- `search(query_embedding, limit)` → `list[tuple[str, float]]`
- `add(node_id, embedding)`, `remove(node_id)`, `has(node_id)`
- `save(path)` / `load(path)` — Persists as `.npy` + `.ids.json`

At Ash's scale (thousands of memories, 1536-dim), search takes ~1-3ms.

### Edge Helpers (`graph/edges.py`)

Factory functions and query helpers for typed edges:
- Factories: `create_about_edge()`, `create_supersedes_edge()`, `create_is_person_edge()`, `create_merged_into_edge()`
- Queries: `get_subject_person_ids()`, `get_memories_about_person()`, `get_person_for_user()`, `get_merged_into()`, `follow_merge_chain()`

### BFS Traversal (`graph/traversal.py`)

Multi-hop graph traversal for discovery:

```python
def bfs_traverse(
    graph: KnowledgeGraph,
    seed_ids: set[str],
    max_hops: int = 2,
    exclude_edge_types: set[str] | None = None,  # Default: {SUPERSEDES}
    filter_fn: Callable[[str, Edge], bool] | None = None,
) -> list[TraversalResult]:
```

Follows both outgoing and incoming edges. Returns `TraversalResult` with `node_id`, `node_type`, `hops`, and `path` (edge IDs).

### Store (`store/store.py`)

Unified facade composing all mixins. Constructor:

```python
Store(
    graph: KnowledgeGraph,
    persistence: GraphPersistence,
    vector_index: NumpyVectorIndex,
    embedding_generator: EmbeddingGenerator,
    llm: LLMProvider | None = None,
    max_entries: int | None = None,
)
```

Factory: `create_store(graph_dir, llm_registry, ...)` handles loading, wiring, and migration.

## Retrieval Pipeline

4-stage pipeline in `store/retrieval.py`:

```
Stage 1: Vector search         → Scoped to user/chat, returns ranked memories
Stage 2: Cross-context          → ABOUT edges for participants, privacy-filtered
Stage 3: Multi-hop BFS          → 2-hop traversal from seed persons, hop-scored
Stage 4: RRF fusion             → Reciprocal rank fusion across stages
```

**Stage 3 detail:**
- Seeds = person IDs from stage 1/2 results + participants
- BFS 2 hops via adjacency lists, skips SUPERSEDES edges
- Memory nodes: apply privacy/scope filter, score by hop distance (0.5 for 1-hop, 0.3 for 2-hop)

**Stage 4 (RRF) detail:**
- Each stage produces a ranked list
- RRF score = Σ 1/(k + rank) where k=60
- Results appearing in multiple stages get boosted
- Falls back to similarity sort when only one stage has results

## Privacy Model

Cross-context memory retrieval respects sensitivity levels:

| Sensitivity | Private chat | Group chat |
|-------------|-------------|------------|
| PUBLIC | Visible | Visible |
| PERSONAL | Visible if querier is subject | Visible if querier is subject |
| SENSITIVE | Visible if querier is subject | **Hidden** |

The `portable` flag controls whether a memory can appear outside its original context. Non-portable memories are excluded from cross-context retrieval.

## Module Structure

```
src/ash/graph/
├── __init__.py          # Exports
├── graph.py             # KnowledgeGraph, Edge dataclass
├── edges.py             # Edge type constants, factories, query helpers
├── persistence.py       # JSONL load/save, edge backfill
├── vectors.py           # NumpyVectorIndex
└── traversal.py         # BFS multi-hop traversal

src/ash/store/
├── __init__.py          # Exports: create_store
├── store.py             # Store class (mixin composition)
├── types.py             # MemoryEntry, PersonEntry, UserEntry, ChatEntry, etc.
├── search.py            # SearchMixin (vector search + context retrieval)
├── retrieval.py         # RetrievalPipeline (4-stage with BFS + RRF)
├── supersession.py      # SupersessionMixin (conflict detection, hearsay)
├── users.py             # UserChatOpsMixin
├── hearsay.py           # HearsayMixin
├── trust.py             # TrustMixin
├── memories/
│   ├── crud.py          # MemoryCrudMixin
│   ├── helpers.py       # Shared memory helpers
│   ├── lifecycle.py     # GC, expiration, archival, forget_person
│   └── eviction.py      # Max entries, compaction, remap
└── people/
    ├── crud.py          # PeopleCrudMixin
    ├── helpers.py       # Normalization, sorting
    ├── resolution.py    # PeopleResolutionMixin (find, fuzzy match)
    ├── relationships.py # RelationshipsMixin
    └── dedup.py         # PeopleDedupMixin (merge, follow chain)
```
