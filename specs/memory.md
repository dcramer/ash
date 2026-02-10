# Memory

> Long-term knowledge persistence so the agent remembers facts across conversations

## Intent

The memory subsystem solves a fundamental problem: conversations are ephemeral, but users expect the agent to remember important facts about them. Memory provides:

1. **Automatic context** - Relevant facts surface before each response without user intervention
2. **Explicit storage** - Users can tell the agent to remember specific things
3. **Person awareness** - Facts can be attributed to people in the user's life
4. **Knowledge evolution** - New information supersedes outdated facts
5. **Smart decay** - Ephemeral facts expire naturally based on their type

Memory is NOT for:
- Conversation history (that's sessions)
- Temporary task state (that's session context)
- Credentials or secrets (security risk)

## Storage Architecture

Memory uses a **filesystem-primary** architecture where JSONL files are the source of truth:

```
~/.ash/
├── people.jsonl                # Person entities (global)
├── memory/
│   ├── memories.jsonl          # Active memories (source of truth)
│   └── archive.jsonl           # Archived memories (append-only safety net)
└── data/
    └── memory.db               # Vector index only (rebuildable)
```

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| JSONL is source of truth | Human-readable, version-controllable, survives DB corruption |
| SQLite for vectors only | sqlite-vec provides fast similarity search, fully rebuildable |
| Embeddings in JSONL | Avoids API costs on rebuild; OpenAI embeddings aren't deterministic |
| Append-only archive | Never lose data; safety net for recovery |
| Atomic file writes | Write to temp file, then rename; prevents corruption on crash |

### Auto-Migration

On first run, if SQLite database exists but JSONL files don't, the system automatically migrates:
1. Exports all memories and people from SQLite to JSONL
2. Rebuilds the vector index
3. Original SQLite preserved as backup

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

The `remember` tool stores facts when users explicitly request it:
- "Remember that..." / "Don't forget..." / "Save this..."
- Supports batch storage (multiple facts at once)
- Supports expiration ("remember for 2 weeks")
- Supports person attribution ("remember Sarah likes...")

### The agent can search memory

Memory search happens in two ways:
1. **Automatic context injection** - Before each response, relevant memories are retrieved via semantic search and included in the system prompt
2. **Explicit search** - In the sandbox, `ash memory search <query>` searches stored facts via RPC

Search features:
- Semantic search (meaning, not just keywords)
- Filter by person ("what do I know about Sarah?")
- Returns relevant facts ranked by similarity

### Facts about people are tracked

When facts relate to people in the user's life:
- Person entities are created/resolved automatically
- References like "my wife", "Sarah", "my boss" resolve to the same person
- System prompt includes known people for context
- Memory results show who facts are about

### Old information gets superseded

When new information conflicts with old:
- "Favorite color is red" then "favorite color is blue" → only blue is retrieved
- Superseded facts preserved in archive but excluded from active search
- Subject-specific facts don't conflict with general facts

**Supersession process:**
1. Vector search finds candidates with similarity ≥ 0.75
2. LLM verifies that memories actually conflict (reduces false positives)
3. Old memory marked as superseded with reference to replacement
4. GC later archives superseded memory to `archive.jsonl`

### Memory doesn't grow unbounded

- Optional `max_entries` cap with smart eviction
- Expired memories cleaned up automatically
- Garbage collection removes superseded/expired entries
- Ephemeral types decay based on their TTL

**GC algorithm:**
1. Identify memories to archive:
   - Explicit expiration (`expires_at` passed)
   - Superseded memories (`superseded_at` set)
   - Ephemeral types past their default TTL
2. Append each to `archive.jsonl` with reason
3. Rewrite `memories.jsonl` without archived entries (atomic)
4. Remove embeddings from vector index

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
| Update person | Must be owner of the person entity |
| Add person alias | Must be owner of the person entity |
| Reference person in memory | Person must belong to memory owner |
| Get memories about person | Filtered to accessible memories |

**Design notes:**
- Authorization is best-effort, not security boundary (sandbox can't be fully trusted)
- Operations without context (e.g., internal GC) skip checks
- Goal: prevent accidental mistakes, not malicious actors

## Background Extraction

Optionally, facts are extracted automatically from conversations:
- Runs after each exchange (debounced)
- Extracts preferences, facts about people, important dates
- Skips assistant actions, temporary context, credentials
- Confidence threshold filters low-quality extractions

## JSONL Schema

### Memory Entry

```json
{
  "id": "abc-123",
  "version": 1,
  "content": "User prefers dark mode",
  "memory_type": "preference",
  "embedding": "BASE64_ENCODED_FLOAT32_ARRAY",
  "created_at": "2026-02-09T10:00:00+00:00",
  "observed_at": null,
  "owner_user_id": "user-1",
  "chat_id": null,
  "subject_person_ids": [],
  "source": "user",
  "source_session_id": null,
  "source_message_id": null,
  "extraction_confidence": null,
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
| `embedding` | Yes | Base64-encoded float32 array for vector search |
| `created_at` | Yes | When written to storage |
| `observed_at` | No | When fact was observed (for delayed extraction) |
| `owner_user_id` | One of these | Personal scope |
| `chat_id` | required | Group scope |
| `subject_person_ids` | Yes | Who this is about (empty list if nobody) |
| `source` | Yes | "user" / "extraction" / "cli" / "rpc" |
| `source_session_id` | No | Session ID for extraction tracing |
| `source_message_id` | No | Message UUID for extraction tracing |
| `extraction_confidence` | No | 0.0-1.0 confidence score |
| `expires_at` | No | Explicit TTL |
| `superseded_at` | No | When marked superseded |
| `superseded_by_id` | No | What memory replaced this one |
| `archived_at` | No | When moved to archive (archive only) |
| `archive_reason` | No | "superseded" / "expired" / "ephemeral_decay" |

### Person Entry

```json
{
  "id": "person-456",
  "version": 1,
  "owner_user_id": "user-1",
  "name": "Sarah",
  "relation": "wife",
  "aliases": ["my wife"],
  "created_at": "2026-01-15T10:00:00+00:00",
  "updated_at": null,
  "metadata": null
}
```

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

The sandbox CLI (`ash-sb memory`) wraps these RPC calls with environment-provided `ASH_USER_ID` and `ASH_CHAT_ID`. Authorization checks use these values to verify ownership before mutations.

## Rebuild & Repair

If the SQLite vector index is corrupted or missing, it can be rebuilt entirely from the JSONL source of truth:

```bash
# Rebuild vector index from JSONL
uv run ash memory rebuild-index
```

**How rebuild works:**
1. Delete existing SQLite database (if corrupted)
2. Create fresh database with sqlite-vec virtual table
3. Load all active memories from `memories.jsonl`
4. Insert embeddings (from stored base64, no API calls needed)

**Auto-recovery:** On startup, if SQLite is missing but JSONL exists, the index is automatically rebuilt.

### Supersession History

View the chain of memories that led to a current memory:

```bash
# Show supersession history for a memory
uv run ash memory history <memory-id>
```

## Verification

```bash
# Unit tests
uv run pytest tests/test_memory.py tests/test_memory_extractor.py tests/test_memory_file_store.py -v

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

# Verify JSONL storage
cat ~/.ash/memory/memories.jsonl | head -3

# Check supersession removes old memory
uv run ash chat "Remember my favorite color is red"
cat ~/.ash/memory/memories.jsonl | grep -c "favorite color"  # 1
uv run ash chat "Remember my favorite color is blue"
uv run ash memory gc
cat ~/.ash/memory/memories.jsonl | grep -c "favorite color"  # 1 (red archived)

# Verify extraction attribution
uv run ash chat "I really love hiking in the mountains"
cat ~/.ash/memory/memories.jsonl | grep source_session

# Rebuild index after corruption
rm ~/.ash/data/memory.db
uv run ash memory rebuild-index
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
- [ ] Ephemeral memory types decay based on their TTL
- [ ] `memories.jsonl` is the source of truth
- [ ] `ash memory rebuild-index` rebuilds SQLite from JSONL
- [ ] `ash memory history <id>` shows supersession chain
- [ ] Auto-migration from SQLite to JSONL on first run
