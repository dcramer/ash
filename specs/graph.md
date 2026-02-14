# Graph Store

> Unified graph architecture that combines memory and people into a single store with O(1) traversals

## Intent

The graph store unifies memory and people management into a single facade (`GraphStore`) backed by an in-memory graph index (`GraphIndex`). It solves three problems:

1. **Unified API** - One store handles memory CRUD, person resolution, user/chat management, and cross-context retrieval. Eliminates the circular dependency between MemoryManager and PersonManager.
2. **O(1) traversals** - Graph index provides constant-time lookups for "memories about person X", "user for this provider_id", "people known by user" instead of O(n) linear scans.
3. **First-class identity nodes** - Users and Chats are proper graph nodes (not just string IDs scattered across memory fields), enabling identity resolution and cross-provider linking.

The graph store is NOT:
- A graph database (JSONL files are the source of truth, graph index is derived)
- A replacement for the vector index (semantic search still uses sqlite-vec)
- An RDF/property graph system (edges are fields on nodes, not separate entities)

## Filesystem Layout

```
~/.ash/
├── config.toml
├── graph/                    # Source of truth — node JSONL files only
│   ├── memories.jsonl        #   Memory nodes
│   ├── people.jsonl          #   Person nodes
│   ├── users.jsonl           #   User identity nodes
│   └── chats.jsonl           #   Chat/channel nodes
├── index/                    # Derived data (rebuildable from graph/)
│   ├── embeddings.jsonl      #   Vector embeddings (expensive to recompute)
│   └── vectors.db            #   sqlite-vec index (rebuildable from embeddings)
├── sessions/                 # Conversation transcripts
├── chats/                    # Per-chat operational state
│   └── {provider}/{chat_id}/
│       └── state.json        # ChatState (references graph node IDs)
├── skills/
│   └── state/                # Per-skill operational state
├── skills.installed/
├── run/
├── logs/
├── cache/uv/
└── schedule.jsonl
```

**Key principle**: `graph/` contains source-of-truth JSONL files. `index/` contains derived data that can be rebuilt. Everything else is operational state.

## Graph Model

### Nodes

| Type | File | Schema | Purpose |
|------|------|--------|---------|
| **Memory** | `memories.jsonl` | `MemoryEntry` | Facts, preferences, observations |
| **Person** | `people.jsonl` | `PersonEntry` | Identity entities (people in user's life) |
| **User** | `users.jsonl` | `UserEntry` | Provider identity (Telegram user, etc.) |
| **Chat** | `chats.jsonl` | `ChatEntry` | Chat/channel from a provider |

**UserEntry** bridges provider identity to Person:
- `provider_id` is the stable anchor (e.g., Telegram numeric ID)
- `username` is mostly-stable (can change)
- `display_name` is unstable (can change anytime)
- `person_id` links to the Person record (IS_PERSON edge)

**ChatEntry** represents a chat/channel:
- `provider_id` is the provider's chat ID
- `title`, `chat_type` are mutable metadata

### Edges

Edges are stored as fields on nodes. The GraphIndex extracts them into adjacency lists at build time.

| Edge | Direction | Stored As | Forward Query | Reverse Query |
|------|-----------|-----------|---------------|---------------|
| **ABOUT** | Memory → Person | `memory.subject_person_ids` | "who is this about?" | "memories about X?" |
| **OWNED_BY** | Memory → User | `memory.owner_user_id` (provider_id) | "who owns this?" | "X's memories?" |
| **IN_CHAT** | Memory → Chat | `memory.chat_id` (provider_id) | "which chat?" | "memories in chat X?" |
| **STATED_BY** | Memory → User | `memory.source_username` (username) | "who said this?" | "what did X say?" |
| **SUPERSEDES** | Memory → Memory | `memory.superseded_by_id` | "what replaced this?" | "what did X replace?" |
| **KNOWS** | User → Person | `person.relationships[].stated_by` | "who does user know?" | "who knows X?" |
| **IS_PERSON** | User → Person | `user.person_id` | "user's person record?" | "whose self-record?" |
| **MERGED_INTO** | Person → Person | `person.merged_into` | "merged into who?" | "who merged into X?" |

**Note**: Existing fields hold provider IDs/usernames, not node UUIDs. GraphIndex resolves these via lookup tables at build time.

### Key Traversals

| Query | Traversal | Complexity |
|-------|-----------|------------|
| Memories about person X | `graph.memories_about(person_id)` → reverse ABOUT | O(1) |
| Resolve username → person | `graph.resolve_user_by_username()` → IS_PERSON | O(1) |
| Cross-context retrieval | person_ids → `memories_about()` → filter privacy/portable | O(k) per person |
| People known by user | `graph.people_known_by_user(user_id)` → KNOWS | O(1) |
| Hearsay candidates | person_ids → `memories_about()` → filter source_username | O(k) per person |

## Architecture

### GraphIndex (`graph/index.py`)

In-memory adjacency lists built from node lists. Derived data — can be rebuilt from JSONL files in milliseconds for hundreds of nodes.

```
GraphIndex
├── _outgoing: dict[EdgeType, dict[str, set[str]]]  # source → targets
├── _incoming: dict[EdgeType, dict[str, set[str]]]   # target → sources
├── _provider_to_user: dict[str, str]                 # provider_id → user.id
├── _username_to_user: dict[str, str]                 # username → user.id
└── _chat_provider_to_id: dict[str, str]              # chat provider_id → chat.id
```

**Rebuild trigger**: Lazy invalidation via JSONL mtime checks. Any mutation sets `_graph_built = False`, and the next access triggers a full rebuild from `get_all_memories()` + cached people/users/chats.

### GraphStore (`graph/store.py`)

Unified facade composing existing storage layers:

```
GraphStore
├── _store: FileMemoryStore        # Memory CRUD (JSONL)
├── _index: VectorIndex            # Semantic search (sqlite-vec)
├── _embeddings: EmbeddingGenerator # Embedding generation
├── _people_jsonl: TypedJSONL[PersonEntry]
├── _user_jsonl: TypedJSONL[UserEntry]
├── _chat_jsonl: TypedJSONL[ChatEntry]
├── _graph: GraphIndex             # In-memory graph
├── _memory_by_id: dict[str, MemoryEntry]  # ID → memory cache
└── _llm: LLMProvider | None       # For fuzzy matching, supersession
```

**Public API surface**:
- Memory: `add_memory()`, `search()`, `get_context_for_message()`, `delete_memory()`, `gc()`, `forget_person()`
- Person: `create_person()`, `find_person()`, `resolve_or_create_person()`, `merge_people()`, `find_person_ids_for_username()`
- User/Chat: `ensure_user()`, `ensure_chat()` (upserts)
- Graph: `get_graph()` (async, ensures built)

### Factory

`create_graph_store()` handles all wiring:
1. Filesystem migration (old layout → new layout)
2. Data migration (extract User/Chat nodes from existing memories)
3. SQLite → JSONL migration (legacy)
4. Component creation (FileMemoryStore, VectorIndex, EmbeddingGenerator)
5. Index rebuild if needed

## Module Structure

```
src/ash/graph/
├── __init__.py      # Exports: GraphStore, create_graph_store, types
├── types.py         # EdgeType, UserEntry, ChatEntry
├── index.py         # GraphIndex (in-memory adjacency lists)
├── store.py         # GraphStore (unified facade)
└── migration.py     # Auto-migration (filesystem + data)

src/ash/memory/      # Storage internals (used by GraphStore)
├── file_store.py    # FileMemoryStore (JSONL CRUD)
├── index.py         # VectorIndex (sqlite-vec)
├── embeddings.py    # EmbeddingGenerator
├── jsonl.py         # TypedJSONL[T] generic store
├── types.py         # MemoryEntry, MemoryType, etc.
└── extractor.py     # MemoryExtractor (LLM-based)

src/ash/people/
└── types.py         # PersonEntry, AliasEntry, RelationshipClaim
```

## Privacy Model

Cross-context memory retrieval respects sensitivity levels:

| Sensitivity | Private chat | Group chat |
|-------------|-------------|------------|
| PUBLIC | Visible | Visible |
| PERSONAL | Visible if querier is subject | Visible if querier is subject |
| SENSITIVE | Visible if querier is subject | **Hidden** |

The `portable` flag controls whether a memory can appear outside its original context. Non-portable memories (e.g., "Bob is presenting next") are excluded from cross-context retrieval.
