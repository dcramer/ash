# Memory

> Long-term knowledge persistence so the agent remembers facts across conversations

## Intent

The memory subsystem solves a fundamental problem: conversations are ephemeral, but users expect the agent to remember important facts about them. Memory provides:

1. **Automatic context** - Relevant facts surface before each response without user intervention
2. **Explicit storage** - Users can tell the agent to remember specific things
3. **Person awareness** - Facts can be attributed to people in the user's life
4. **Knowledge evolution** - New information supersedes outdated facts
5. **Smart decay** - Ephemeral facts expire naturally based on their type

**Scale expectations:** Tens of users in a chat, each with thousands of memories. Operations like doctor, search, and GC must handle this without degrading.

Memory is NOT for:
- Conversation history (that's sessions)
- Temporary task state (that's session context)
- Credentials or secrets (security risk)

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

See [specs/people.md](people.md) for Person-owned edges (SELF, KNOWS, ALIAS, MERGED_INTO).

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

## Memory Types

Memories are classified into types that determine their lifecycle:

### Long-Lived Types (no automatic expiration)

| Type | Description | Examples |
|------|-------------|----------|
| `preference` | User likes, dislikes, preferences | "prefers dark mode", "favorite color is blue" |
| `identity` | Facts about the user themselves | "works as a software engineer", "lives in SF" |
| `relationship` | People in user's life | "Sarah is my wife", "has a dog named Max" |
| `knowledge` | Factual information | "project X uses Python", "company uses Slack" |

### Ephemeral Types (decay over time)

| Type | Default TTL | Description | Examples |
|------|-------------|-------------|----------|
| `context` | 7 days | Current situation/state | "working on project X" |
| `event` | 30 days | Past occurrences | "had dinner with Sarah Tuesday" |
| `task` | 14 days | Things to do/remember | "needs to call dentist" |
| `observation` | 3 days | Fleeting observations | "mentioned being tired" |

### Type Assignment

Types are assigned during:
1. **Explicit remember**: User says "remember X" → LLM infers type from content
2. **Background extraction**: LLM extracts and classifies during conversation
3. **CLI/RPC add**: Optional `type` parameter, defaults to `knowledge`

## Outcomes

### The agent recalls relevant context

When a user asks about something they've told the agent before, the agent should know it without being reminded.

| User says | Agent should |
|-----------|--------------|
| "Remember I'm allergic to peanuts" | Store this fact |
| (later) "What should I avoid eating?" | Mention peanut allergy |
| "My wife's name is Sarah" | Store fact, create Person entity |
| (later) "What's my wife's name?" | Answer "Sarah" |

### Users can explicitly store facts

When users say "remember that..." / "don't forget..." / "save this...", the agent stores facts by calling `ash-sb memory add` in the sandbox. This routes through `MemoryManager`, which generates embeddings, indexes the memory, and detects supersession automatically. Features:
- Supports batch storage (agent makes multiple `ash-sb memory add` calls)
- Supports expiration ("remember for 2 weeks")
- Supports person attribution ("remember Sarah likes...")

### The agent can search memory

Memory search happens in two ways:
1. **Automatic context injection** - Before each response, the retrieval pipeline gathers relevant memories and includes them in the system prompt
2. **Explicit search** - `ash memory search <query>` and `memory.search` RPC perform hybrid search

Search combines two retrieval strategies:
- **Vector search** - Semantic similarity via embeddings (OpenAI text-embedding-3-small)
- **Person-graph search** - If the query resolves to a person (by name, username, or alias), retrieves memories linked via ABOUT edges, plus memories about related people (1-hop via HAS_RELATIONSHIP)

Results from both strategies are merged (deduplicated, highest score kept) and ranked by similarity.

### Facts about people are tracked

When facts relate to people in the user's life:
- Person entities are created/resolved automatically
- References like "my wife", "Sarah", "my boss" resolve to the same person
- System prompt includes up to 50 most recently active people for context
- Memory results show who facts are about

### Old information gets superseded

When new information conflicts with old:
- "Favorite color is red" then "favorite color is blue" → only blue is retrieved
- Superseded facts archived in-place (`archived_at` set) but excluded from active search
- Subject-specific facts don't conflict with general facts

**Supersession process:**
1. Vector search finds candidates with similarity ≥ 0.75
2. LLM verifies that memories actually conflict (reduces false positives)
3. Old memory marked as superseded with reference to replacement
4. GC later archives superseded memory in-place (`archived_at` + `archive_reason` set)

### Memory doesn't grow unbounded

- Optional `max_entries` cap with smart eviction
- Expired memories cleaned up automatically
- Garbage collection removes superseded/expired entries
- Ephemeral types decay based on their TTL

**GC algorithm:**
1. Scan active memories (where `archived_at` is null):
   - Explicit expiration (`expires_at` passed)
   - Superseded memories (`superseded_at` set)
   - Ephemeral types past their default TTL (measured from `observed_at`, falling back to `created_at`)
2. Set `archived_at` + `archive_reason` on each (archive-in-place)
3. Remove embeddings from vector index

**Compaction:** Over time, archived entries accumulate. `ash memory compact --force` permanently removes archived entries older than 90 days (configurable via `--older-than`).

## Scoping

Memories have visibility scope controlled by two fields:

| Scope | `owner_user_id` | `chat_id` | Visible to |
|-------|-----------------|-----------|------------|
| Personal | Set | NULL | Only the user who created it |
| Group | NULL | Set | Everyone in the chat |

**Scoping rules:**
- Personal memories: `owner_user_id` is set, `chat_id` is NULL
- Group memories: `owner_user_id` is NULL, `chat_id` is set
- When retrieving, a user sees their personal memories + any group memories for the current chat

| Scope | Use case |
|-------|----------|
| Personal | Default - "I like coffee" |
| Group | Team facts - "Our standup is at 9am" |

## Authorization

Best-effort authorization prevents accidental cross-user data access. When context is provided:

| Operation | Rule |
|-----------|------|
| Delete memory | Must be owner (personal) or member of chat (group) |
| Update person | Must be the person's creator (`created_by`) or the person themselves |
| Add person alias | Must be the person's creator (`created_by`) or the person themselves |
| Reference person in memory | Any user (people are globally visible) |
| Get memories about person | Filtered to accessible memories (privacy rules apply) |

**Design notes:**
- Authorization is best-effort, not security boundary (sandbox can't be fully trusted)
- Operations without context (e.g., internal GC) skip checks
- Goal: prevent accidental mistakes, not malicious actors

## Privacy & Sensitivity

Memories can be classified by sensitivity level to control sharing:

### Sensitivity Levels

| Level | Description | Sharing Rules |
|-------|-------------|---------------|
| `public` | Can be shared anywhere (default) | No restrictions |
| `personal` | Personal details not for group disclosure | Only with subject or memory owner |
| `sensitive` | High privacy (medical, financial, mental health) | Only in private chat with subject |

### Examples

| Content | Sensitivity | Why |
|---------|-------------|-----|
| "Prefers dark mode" | public | General preference |
| "Looking for a new job" | personal | Sensitive career info |
| "Has anxiety" | sensitive | Mental health |
| "Salary is $150k" | sensitive | Financial |
| "Is pregnant, due August" | sensitive | Medical/health |
| "Has a dog named Max" | public | General fact |

### Privacy Filtering

Privacy rules depend on the chat context:

**Group chats:**

| Sensitivity | Subject is participant | Subject is NOT participant |
|-------------|----------------------|--------------------------|
| PUBLIC | Shown | Shown |
| PERSONAL | Shown | Excluded |
| SENSITIVE | Shown | Excluded |

**Private chats (DMs) — contextual disclosure:**

A memory is disclosable if any of:
- Memory is ABOUT the DM partner
- Memory was STATED_BY the DM partner
- DM partner has a PARTICIPATES_IN edge to the memory's LEARNED_IN chat (they were present when it was said)
- Memory is a self-memory (no subjects)
- Memory has no LEARNED_IN edge (legacy data — fail-open)

Otherwise, memories about third parties are excluded.

**Cross-context retrieval:**

1. **PUBLIC**: Always shown
2. **PERSONAL**: Only shown to the subject person or memory owner
3. **SENSITIVE**: Only shown in private chat with the subject

### Backward Compatibility

- `sensitivity` field defaults to `null`, treated as `public`
- Existing memories work unchanged

## Cross-Context Retrieval

Facts learned about a person in one context (e.g., group chat) can be recalled in another context (e.g., private chat with that person), subject to privacy rules.

### How It Works

1. When Alice mentions "@bob loves pizza" in a group chat:
   - Memory stored with `subject_person_ids=[bob_person_id]`
   - Person record links username "bob" to the person entity

2. When Bob starts a private chat:
   - System finds memories where Bob is a subject (across all owners)
   - Privacy filter applies: sensitive facts only shown to Bob himself
   - Facts from other users become available context

### Cross-Context Query

`find_memories_by_subject(person_ids)` searches across ALL owners:
- Caller resolves person IDs first via `PersonManager`
- Returns memories where that person is a subject
- Only includes **portable** memories (enduring facts about a person, not chat-operational)
- Optional `exclude_owner_user_id` to avoid double-counting

### Privacy in Groups

When someone asks about a person in a group chat:
- **PUBLIC** facts: shown
- **PERSONAL** facts: shown only if the subject is a chat participant
- **SENSITIVE** facts: shown only if the subject is a chat participant

### Self-Query Example

When Bob asks "what do you know about me?" in different contexts:

| Context | PUBLIC | PERSONAL | SENSITIVE |
|---------|--------|----------|-----------|
| Private chat with Bob | ✓ | ✓ | ✓ |
| Group chat (Bob asking) | ✓ | ✓ | ✗ |
| Group chat (others asking) | ✓ | ✗ | ✗ |

## Retrieval Pipeline

Automatic context injection uses a multi-stage retrieval pipeline:

| Stage | Input | Method | Score |
|-------|-------|--------|-------|
| 1. Primary search | User message | Hybrid vector + person-graph search | Vector similarity (0-1) or 0.75/0.55 for graph |
| 2. Cross-context | Participant person IDs | ABOUT edges across all owners | 0.7 fixed |
| 3. Multi-hop BFS | Seed persons from stages 1-2 | 2-hop graph traversal | 0.5 (1-hop) / 0.3 (2-hop) |
| 4. RRF fusion | All stage results | Reciprocal Rank Fusion (K=60) | Combined RRF score |

Stage 2 excludes the querying user's own memories (covered by stage 1). Stage 3 skips SUPERSEDES edges and filters by portable/privacy. RRF fusion boosts memories appearing in multiple stages.

## Background Extraction

Optionally, facts are extracted automatically from conversations:
- Runs after each exchange (debounced)
- Extracts preferences, facts about people, important dates
- Skips assistant actions, temporary context, credentials
- Confidence threshold filters low-quality extractions

## Secrets Filtering

Memory will NEVER store credentials or secrets. Three-layer defense:

1. **Extraction prompt** - LLM instructed to reject secrets during extraction
2. **Post-extraction filter** - Regex patterns catch secrets before storage
3. **Centralized filter in MemoryManager** - Final check catches all entry points (CLI, RPC, direct)

### Automatically Rejected

| Type | Examples |
|------|----------|
| Passwords | "my password is X", "passwd: hunter2" |
| API Keys | sk-..., ghp_..., gho_..., AKIA... |
| SSN | 123-45-6789 |
| Credit Cards | 16-digit numbers with optional separators |
| Private Keys | -----BEGIN PRIVATE KEY----- |
| Slack Tokens | xoxb-..., xoxp-..., xoxs-... |

Even if the user explicitly asks to remember these, they will be rejected with an error.

## Temporal Context

Facts with relative time references are automatically converted to absolute dates during extraction.

### How It Works

1. Current datetime is passed to the extraction prompt
2. LLM instructed to rewrite relative references → absolute dates
3. `observed_at` field records when the fact was stated

### Examples

| User says | Stored as |
|-----------|-----------|
| "this weekend" | "the weekend of Feb 15-16, 2026" |
| "next Tuesday" | "Tuesday, Feb 18, 2026" |
| "tomorrow" | "Feb 12, 2026" |
| "in 2 days" | "Feb 14, 2026" |

This ensures memories remain meaningful when recalled weeks or months later.

## Multi-User Attribution

Memory supports tracking WHO provided each fact, enabling trust-based reasoning:

### Source Attribution Fields

| Field | Description |
|-------|-------------|
| `source_username` | Username/handle of who stated this fact |
| `source_display_name` | Display name of the source user |
| `subject_person_ids` | Who the memory is ABOUT (third parties) |

### Trust Model

| Source == Subject? | Type | Trustworthiness |
|-------------------|------|-----------------|
| Yes (speaking about self) | **FACT** | High - first-person claim |
| No (speaking about others) | **HEARSAY** | Lower - second-hand claim |

### Examples

| Who said it | Content | Source User | Subjects | Trust |
|-------------|---------|-------------|----------|-------|
| David | "I like pizza" | david | [] | FACT |
| David | "Bob likes pasta" | david | [bob] | HEARSAY |
| Bob | "I like pasta" | bob | [] | FACT |

### CLI Display

The `ash memory list` command shows:
- **About**: Subject person(s), or source user if subjects is empty (speaking about self)
- **Source**: Who provided the information (`@username`)
- **Trust**: "fact" or "hearsay"

The `ash memory show` command displays full attribution details.

### Extraction with Speaker Identity

During background extraction, messages are formatted with speaker identity:
```
@david (David Cramer): I like pizza
@bob: Bob prefers pasta
Assistant: Great choices!
```

The LLM then attributes each extracted fact to the appropriate speaker.

## Multi-Subject Facts

Facts can be about multiple people simultaneously. The `subject_person_ids` list supports multiple entries, creating ABOUT edges to each person.

### Joint Facts

When a fact inherently involves multiple people as participants (not just one person reporting about another), all participants should be subjects:

| Statement | Subjects | Why |
|-----------|----------|-----|
| "Alice and Bob are starting a company" | [Alice, Bob] | Joint venture, both are participants |
| "The team is relocating to Austin" | [Alice, Bob, Carol] | Affects all team members |
| "My coworker and I are working on project X" | [coworker] + speaker | Joint effort |

### Speaker as Subject

When the speaker is one of multiple participants in a joint fact, they should be included as a subject. The processing pipeline only treats the speaker as a self-fact (subjects=[]) when they are the SOLE subject — not when they are one of several.

| Statement | subjects (extraction) | subject_person_ids (after processing) |
|-----------|----------------------|--------------------------------------|
| "I like pizza" | [] | [speaker_pid] (self-fact injection) |
| "Alice and I started a company" | ["Alice", speaker_name] | [alice_pid, speaker_pid] |
| "Alice started a company" | ["Alice"] | [alice_pid] |

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

See [specs/people.md](people.md) for the canonical person schema. Summary:

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

## Forget Person

When a user requests to be forgotten, or a person record needs to be purged:

1. Find all active memories with ABOUT edges to this person (`subject_person_ids` contains person_id)
2. Archive in-place: set `archived_at` + `archive_reason="forgotten"`
3. Remove from vector index
4. If `delete_person_record=True`, delete the person node

Available via:
- `memory.forget_person` RPC method
- `ash memory forget --person <id>` CLI command

## Portable Memories

Not all group-scoped memories about a person should cross contexts. The `portable` field controls whether a memory's ABOUT edge is traversable from outside the originating chat.

| Content | Portable | Why |
|---------|----------|-----|
| "Bob loves pizza" | true | Enduring trait about the person |
| "Bob is presenting next" | false | Chat-operational, only relevant here |
| "Sarah's birthday is March 15" | true | Enduring fact |
| "Sarah will send the report by EOD" | false | Ephemeral task context |

**Default:** `true` for backward compatibility. Extraction prompt guides the LLM to set `portable=false` for chat-operational facts.

## Known Limitations

- **Contradictory group facts**: no authority model — last-write-wins. Two users stating conflicting facts about the same subject both persist.
- **Hearsay directionality**: Self-facts have the speaker's person_id injected into `subject_person_ids` for graph traversal (so they're discoverable when asking "what do you know about Bob?"). Hearsay supersession uses the original extraction state (no subjects = self-fact) to trigger cross-scope lookup, not the post-injection `subject_person_ids`.

## Configuration

```toml
[memory]
auto_gc = true              # Clean up on startup
max_entries = 1000          # Cap on active memories (optional)
extraction_enabled = true   # Background extraction
extraction_confidence_threshold = 0.7
```

## RPC Interface (Sandbox Access)

Tools running in the sandbox can access memory via RPC:

| Method | Purpose | Parameters |
|--------|---------|------------|
| `memory.search` | Semantic search | `query` (required), `limit`, `user_id`, `chat_id` |
| `memory.add` | Add a memory | `content` (required), `source`, `expires_days`, `user_id`, `chat_id`, `subjects` |
| `memory.list` | List recent memories | `limit`, `include_expired`, `user_id`, `chat_id` |
| `memory.delete` | Delete a memory | `memory_id` (required), `user_id`, `chat_id` |
| `memory.forget_person` | Archive all memories about a person | `person_id` (required), `delete_person_record` (default false) |

The sandbox CLI (`ash-sb memory`) wraps these RPC calls with environment-provided `ASH_USER_ID` and `ASH_CHAT_ID`. Authorization checks use these values to verify ownership before mutations.

## Rebuild & Repair

If the vector index is missing embeddings (e.g., after a crash or manual DB edit), it can be rebuilt:

```bash
uv run ash memory rebuild-index
```

**How rebuild works:**
1. Query active memories from the `memories` table
2. Check which memory IDs already have embeddings in `memory_embeddings`
3. Generate embeddings via API only for missing entries
4. Insert new embeddings into the vector index

### Supersession History

View the chain of memories that led to a current memory:

```bash
# Show supersession history for a memory
uv run ash memory history <memory-id>
```

## Export, Import & Backup

The database can be exported, imported, and backed up using CLI commands:

### Export

Export all data (memories, people, users, chats) to a single JSONL file:

```bash
# Export to stdout
uv run ash db export > backup.jsonl

# Export to file
uv run ash db export --output backup.jsonl
```

Each line contains a record with a `_type` field indicating the entity type.

### Import

Import data from a previously exported JSONL file:

```bash
# Replace existing data
uv run ash db import backup.jsonl

# Merge with existing data
uv run ash db import backup.jsonl --merge
```

After import, run `ash memory rebuild-index` to rebuild the vector index.

### Backup

Create an atomic backup of the SQLite vector index using `VACUUM INTO`:

```bash
# Backup to timestamped file in ~/.ash/backups/
uv run ash db backup

# Backup to specific path
uv run ash db backup --output /tmp/ash-backup.db
```

The backup is a standalone database file that can be copied or restored directly.

## Verification

```bash
# Unit tests
uv run pytest tests/test_memory.py tests/test_memory_extractor.py tests/test_people.py -v

# Integration test
uv run ash chat "Remember my favorite color is blue"
uv run ash chat "What's my favorite color?"
# Should answer "blue"

uv run ash chat "Remember my wife Sarah likes Italian food"
uv run ash chat "What does my wife like?"
# Should mention Italian food and attribute to Sarah

# CLI inspection
uv run ash memory list
uv run ash memory search "favorite"
uv run ash memory gc

# Verify storage via CLI
uv run ash memory list
uv run ash db export | head -5

# Check supersession removes old memory
uv run ash chat "Remember my favorite color is red"
uv run ash memory search "favorite color"  # should show red
uv run ash chat "Remember my favorite color is blue"
uv run ash memory gc
uv run ash memory search "favorite color"  # should show only blue

# Rebuild index for missing embeddings
uv run ash memory rebuild-index

# Secrets filtering tests:
# 1. Via CLI - should reject
uv run ash memory add -q "my password is hunter2"
# Should error: "Memory content contains potential secrets"

# 2. Via CLI with API key - should reject
uv run ash memory add -q "API key: sk-abc123def456789"
# Should error: "Memory content contains potential secrets"

# 3. Normal content should work
uv run ash memory add -q "I prefer dark mode"
uv run ash memory search "dark mode"  # should find it

# Export/Import/Backup tests:
# 1. Export data
uv run ash db export --output /tmp/ash-export.jsonl
head -5 /tmp/ash-export.jsonl

# 2. Import data (with merge)
uv run ash db import /tmp/ash-export.jsonl --merge

# 3. Create backup
uv run ash db backup --output /tmp/ash-backup.db
ls -la /tmp/ash-backup.db
```

### Checklist

- [ ] Agent answers questions using previously stored facts
- [ ] `remember` tool stores facts when user explicitly requests
- [ ] Automatic context injection retrieves relevant memories before each response
- [ ] Person entities created for "my wife", "Sarah", etc.
- [ ] Known people appear in system prompt
- [ ] Memory results show subject attribution ("about Sarah")
- [ ] Conflicting facts supersede old versions (with LLM verification)
- [ ] Superseded facts archived but excluded from retrieval
- [ ] `max_entries` evicts oldest when exceeded
- [ ] `ash memory gc` removes expired/superseded/decayed entries
- [ ] Background extraction captures facts from conversations
- [ ] Extraction assigns memory types (preference, identity, etc.)
- [ ] Extraction skips low-confidence and duplicate facts
- [ ] Ephemeral memory types decay based on their TTL (from `observed_at`)
- [ ] SQLite database is the source of truth
- [ ] `ash memory rebuild-index` generates embeddings for missing entries
- [ ] `ash memory history <id>` shows supersession chain
- [ ] Extracted memories have `source_username` populated
- [ ] `ash memory list` shows About, Source, and Trust columns
- [ ] `ash memory show <id>` displays full attribution details
- [ ] Extraction classifies sensitivity (public/personal/sensitive)
- [ ] Sensitive memories not disclosed in group chats
- [ ] Personal memories only shown to subject or owner
- [ ] Cross-context retrieval finds facts from other owners
- [ ] Privacy filter applied to cross-context memories
- [ ] Secrets rejected via CLI: `ash memory add -q "password is hunter2"` → error
- [ ] Secrets rejected via RPC: agent storing API key → error
- [ ] Secrets filtered during extraction (LLM output post-filtered)
- [ ] Relative times converted to absolute dates during extraction
- [ ] `observed_at` field populated on extracted memories
- [ ] `ash memory search <query>` performs semantic search via MemoryManager
- [ ] `ash memory add` generates embeddings and indexes via MemoryManager
- [ ] `ash db export` exports all data to JSONL format
- [ ] `ash db import` imports data from exported JSONL
- [ ] `ash db backup` creates atomic SQLite backup with VACUUM INTO
