# Memory Data Model

## Graph Model

The memory subsystem is best understood as a graph. Memory entries are nodes; attributions, subjects, and lifecycle links are edges.

### Nodes

| Node Type | Backed By | Key Fields |
|-----------|-----------|------------|
| **Memory** | `memories` table (SQLite) | id, content, memory_type |

### Edges (owned by memory subsystem)

| Edge | From → To | Stored As | Metadata |
|------|-----------|-----------|----------|
| **ABOUT** | Memory → Person | `subject_person_ids` list | — |
| **STATED_BY** | Memory → username | `source_username` field | source_display_name |
| **OWNED_BY** | Memory → user_id | `owner_user_id` field | — |
| **IN_CHAT** | Memory → chat_id | `chat_id` field | — |
| **SUPERSEDES** | Memory → Memory | `superseded_by_id` field | superseded_at |
| **LEARNED_IN** | Memory → Chat | Graph edge | Where a memory was first learned |
| **PARTICIPATES_IN** | Person → Chat | Graph edge | Chat membership (best-effort) |

### Key Traversals

| Query | Traversal |
|-------|-----------|
| "what about Bob?" | Bob(person) ←ABOUT← memories →filter(privacy, portable)→ **results** |
| "search Alice" | "Alice" →resolve→ Person →ABOUT← memories + Person →HAS_RELATIONSHIP→ related →ABOUT← memories → **results** |
| hearsay check | memory →STATED_BY→ user; memory →ABOUT→ persons; overlap? → fact, else hearsay |
| cross-context | person ←ABOUT← memories →filter(portable=true, privacy)→ **results** |
| contextual disclosure | memory →LEARNED_IN→ chat ←PARTICIPATES_IN← partner? → include/exclude |

See [specs/people.md](../people.md) for Person-owned edges (SELF, KNOWS, ALIAS, MERGED_INTO).

## Storage Architecture

Memory uses a **SQLite-primary** architecture. A single SQLite database is the source of truth for all data:

```
~/.ash/data/
└── ash.db                    # All data: memories, people, users, embeddings
```

### Tables

| Table | Purpose |
|-------|---------|
| `memories` | All memory entries (active + archived, distinguished by `archived_at`) |
| `memory_subjects` | ABOUT edges: memory → person_id |
| `memory_embeddings` | sqlite-vec virtual table for vector search |
| `people` | Person entities |
| `person_aliases` | Alias entries for people |
| `person_relationships` | Relationship claims for people |
| `users` | User records with username → person_id links |

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| SQLite is source of truth | Single file, ACID transactions, no sync issues |
| sqlite-vec for vectors | Efficient similarity search via virtual table |
| Archive in-place | Active + archived in same table; `archived_at` field distinguishes them |
| Export for portability | `ash db export` produces JSONL for backup/migration |


## Schema

### Memory Entry

The `memories` table stores all memory entries. Subject links are in the `memory_subjects` join table.

```json
{
  "id": "abc-123",
  "version": 1,
  "content": "User prefers dark mode",
  "memory_type": "preference",
  "created_at": "2026-02-09T10:00:00+00:00",
  "observed_at": null,
  "owner_user_id": "user-1",
  "chat_id": null,
  "subject_person_ids": [],
  "source": "user",
  "source_username": "david",
  "source_display_name": "David Cramer",
  "source_session_id": null,
  "source_message_id": null,
  "extraction_confidence": null,
  "sensitivity": null,
  "expires_at": null,
  "superseded_at": null,
  "superseded_by_id": null,
  "archived_at": null,
  "archive_reason": null,
  "metadata": null
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | UUID primary key |
| `version` | Yes | Schema version (currently 1) |
| `content` | Yes | The memory text |
| `memory_type` | Yes | Type classification |
| `created_at` | Yes | When written to storage |
| `observed_at` | No | When fact was observed (for delayed extraction) |
| `owner_user_id` | One of these | Personal scope |
| `chat_id` | required | Group scope |
| `subject_person_ids` | Yes | Who this is about (empty list if nobody) |
| `source` | Yes | "user" / "extraction" / "background_extraction" / "cli" / "rpc" |
| `source_username` | No | Username/handle of who stated this fact |
| `source_display_name` | No | Display name of source user |
| `source_session_id` | No | Session ID for extraction tracing |
| `source_message_id` | No | Message UUID for extraction tracing |
| `extraction_confidence` | No | 0.0-1.0 confidence score |
| `sensitivity` | No | "public" / "personal" / "sensitive" (null = public) |
| `expires_at` | No | Explicit TTL |
| `superseded_at` | No | When marked superseded |
| `superseded_by_id` | No | What memory replaced this one |
| `archived_at` | No | When archived (null = active) |
| `portable` | No | Whether cross-context traversal is allowed (default true) |
| `archive_reason` | No | "superseded" / "expired" / "ephemeral_decay" / "forgotten" / "user_deleted" |

### Embedding Record

Embeddings are stored in the `memory_embeddings` sqlite-vec virtual table. Each row maps a `memory_id` to a float32 vector blob used for similarity search.

### Person Entry

See [specs/people.md](../people.md) for the canonical person schema. Summary:

```json
{
  "id": "person-456",
  "version": 1,
  "created_by": "123456789",
  "name": "Sarah",
  "relationships": [
    {"relationship": "wife", "stated_by": "123456789", "created_at": "2026-01-15T10:00:00+00:00"}
  ],
  "aliases": [
    {"value": "my wife", "added_by": "123456789", "created_at": "2026-01-15T10:00:00+00:00"}
  ],
  "merged_into": null,
  "created_at": "2026-01-15T10:00:00+00:00",
  "updated_at": "2026-01-15T10:00:00+00:00",
  "metadata": null
}
```

### Extracted Fact (Internal)

Used during background extraction, before facts are converted to MemoryEntry:

```json
{
  "content": "User prefers dark mode",
  "subjects": [],
  "shared": false,
  "confidence": 0.85,
  "memory_type": "preference",
  "speaker": "@david (David Cramer)",
  "sensitivity": "public"
}
```

| Field | Description |
|-------|-------------|
| `content` | The fact content |
| `subjects` | Person references mentioned (e.g., "Sarah", "my wife") |
| `shared` | Whether this is a group fact |
| `confidence` | Extraction confidence (0.0-1.0) |
| `memory_type` | Assigned memory type |
| `speaker` | Who stated this fact (format: `@username (Display Name)`) |
| `sensitivity` | Privacy classification ("public" / "personal" / "sensitive") |
| `portable` | Whether this fact should be retrievable cross-context (default true) |
