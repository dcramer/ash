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
| "Has a dog named Max" | public | General fact |

### Privacy Filtering

When retrieving memories:
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

`find_memories_about_user()` searches across ALL owners:
- Matches username to Person records via name/aliases
- Returns memories where that person is a subject
- Optional `exclude_owner_user_id` to avoid double-counting

### Privacy in Groups

When someone asks about a person in a group chat:
- **PUBLIC** facts: shown
- **PERSONAL** facts: shown only if the subject is asking
- **SENSITIVE** facts: NOT shown (group context)

### Self-Query Example

When Bob asks "what do you know about me?" in different contexts:

| Context | PUBLIC | PERSONAL | SENSITIVE |
|---------|--------|----------|-----------|
| Private chat with Bob | ✓ | ✓ | ✓ |
| Group chat (Bob asking) | ✓ | ✓ | ✗ |
| Group chat (others asking) | ✓ | ✗ | ✗ |

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
| `source_user_id` | Username/ID of who stated this fact |
| `source_user_name` | Display name of the source user |
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
  "source_user_id": "david",
  "source_user_name": "David Cramer",
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
| `embedding` | Yes | Base64-encoded float32 array for vector search |
| `created_at` | Yes | When written to storage |
| `observed_at` | No | When fact was observed (for delayed extraction) |
| `owner_user_id` | One of these | Personal scope |
| `chat_id` | required | Group scope |
| `subject_person_ids` | Yes | Who this is about (empty list if nobody) |
| `source` | Yes | "user" / "extraction" / "background_extraction" / "cli" / "rpc" |
| `source_user_id` | No | Who said/provided this fact (for multi-user attribution) |
| `source_user_name` | No | Display name of source user |
| `source_session_id` | No | Session ID for extraction tracing |
| `source_message_id` | No | Message UUID for extraction tracing |
| `extraction_confidence` | No | 0.0-1.0 confidence score |
| `sensitivity` | No | "public" / "personal" / "sensitive" (null = public) |
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
  "relationship": "wife",
  "aliases": ["my wife"],
  "created_at": "2026-01-15T10:00:00+00:00",
  "updated_at": null,
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

# Privacy tests:
# 1. Bob says "I have anxiety" in private
# 2. Verify sensitivity=sensitive in memories.jsonl
cat ~/.ash/memory/memories.jsonl | grep -A1 "anxiety" | grep sensitivity
# 3. In group chat, ask "tell me about Bob"
# 4. Verify sensitive fact is NOT disclosed
# 5. In private with Bob, ask same question
# 6. Verify sensitive fact IS disclosed

# Cross-context tests:
# 1. In group chat, Alice says "@bob loves pizza"
# 2. Verify memory stored with subject_person_ids containing bob
cat ~/.ash/memory/memories.jsonl | grep "loves pizza" | grep subject_person_ids
# 3. In private chat with Bob, ask "what do you know about me?"
# 4. Verify "loves pizza" fact is retrieved

# Secrets filtering tests:
# 1. Via CLI - should reject
uv run ash memory add -q "my password is hunter2"
# Should error: "Memory content contains potential secrets"

# 2. Via CLI with API key - should reject
uv run ash memory add -q "API key: sk-abc123def456789"
# Should error: "Memory content contains potential secrets"

# 3. Via chat - should NOT extract secrets
uv run ash chat "Remember my password is secret123"
cat ~/.ash/memory/memories.jsonl | grep -c "password"  # 0

# 4. Normal content should work
uv run ash memory add -q "I prefer dark mode"
cat ~/.ash/memory/memories.jsonl | grep "dark mode"  # found

# Temporal context tests:
# 1. Say something with relative time reference
uv run ash chat "I have a meeting this weekend"
# 2. Check extracted memory has absolute date
cat ~/.ash/memory/memories.jsonl | grep "meeting" | grep -E "Feb|weekend of"
# Should have converted "this weekend" to actual date

# 3. Verify observed_at is populated
cat ~/.ash/memory/memories.jsonl | grep "meeting" | grep "observed_at"
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
- [ ] Extracted memories have `source_user_id` populated
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
