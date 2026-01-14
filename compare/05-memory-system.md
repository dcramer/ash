# Memory System Comparison

This document compares memory systems across four agent codebases: **ash**, **archer**, **clawdbot**, and **pi-mono (mom)**.

## Overview

Memory systems in conversational agents serve several key responsibilities:

1. **Persistence** - Retaining information across sessions and restarts
2. **Retrieval** - Finding relevant information when needed
3. **Scoping** - Controlling who can see what information (personal vs shared)
4. **Conflict Resolution** - Handling contradictory or superseded information
5. **Extraction** - Automatically identifying memorable facts from conversations
6. **Relationship Tracking** - Managing information about people and entities

## Comparison Table

| Feature | ash | archer | clawdbot | pi-mono (mom) |
|---------|-----|--------|----------|---------------|
| **Storage Backend** | SQLite + sqlite-vec | MEMORY.md files | In-memory + research docs | MEMORY.md files |
| **Semantic Search** | Yes (embeddings) | No | No (proposed) | No |
| **Conflict Detection** | Yes (0.75 threshold) | Manual | No | Manual |
| **Person Tracking** | Yes (relationships, aliases) | No | No | No |
| **Memory Extraction** | Yes (background LLM) | Manual | No (proposed) | Manual |
| **Scoping** | Personal + Group | Global + Per-channel | Per-session | Global + Per-channel |
| **Expiration** | Yes (TTL support) | No | No | No |
| **Supersession** | Yes (automatic) | No | No | No |

## Detailed Analysis

### ash (Python)

**Architecture**: ash implements a sophisticated, structured memory system with multiple layers.

**Core Components**:

- `MemoryManager` - Orchestrates retrieval and persistence
- `MemoryStore` - SQLite-based data access layer
- `SemanticRetriever` - Vector search using sqlite-vec
- `MemoryExtractor` - Background LLM for fact extraction
- `EmbeddingGenerator` - OpenAI embeddings (1536 dimensions)

**Storage Model**:

```python
# Memory table with scoping and lifecycle
Memory:
    id: str
    content: str
    source: str
    expires_at: datetime | None
    superseded_at: datetime | None
    superseded_by_id: str | None
    owner_user_id: str | None  # Personal scope
    chat_id: str | None        # Group scope
    subject_person_ids: list[str] | None  # Entity links
```

**Semantic Search Implementation**:

```python
# From retrieval.py - Vector search with sqlite-vec
async def search_memories(
    self,
    query: str,
    limit: int = 10,
    owner_user_id: str | None = None,
    chat_id: str | None = None,
) -> list[SearchResult]:
    query_embedding = await self._embeddings.embed(query)

    sql = text("""
        SELECT me.memory_id, m.content,
               vec_distance_cosine(me.embedding, :query_embedding) as distance
        FROM memory_embeddings me
        JOIN memories m ON me.memory_id = m.id
        WHERE m.superseded_at IS NULL
        ORDER BY distance ASC
        LIMIT :limit
    """)
```

**Conflict Detection**:

```python
# From manager.py - Automatic supersession at 0.75 similarity
CONFLICT_SIMILARITY_THRESHOLD = 0.75

async def supersede_conflicting_memories(
    self,
    new_memory_id: str,
    new_content: str,
    owner_user_id: str | None = None,
) -> int:
    conflicts = await self.find_conflicting_memories(
        new_content=new_content,
        owner_user_id=owner_user_id,
    )
    for memory_id, similarity in conflicts:
        if similarity >= CONFLICT_SIMILARITY_THRESHOLD:
            await self._store.mark_memory_superseded(
                memory_id=memory_id,
                superseded_by_id=new_memory_id,
            )
```

**Person Tracking**:

```python
# From store.py - Relationship and alias management
Person:
    id: str
    owner_user_id: str
    name: str
    relation: str | None  # wife, boss, friend, etc.
    aliases: list[str]    # Alternative names

# Resolution handles "my wife", "Sarah", "@notzeeg"
async def find_person_by_reference(
    self,
    owner_user_id: str,
    reference: str,
) -> Person | None
```

**Background Memory Extraction**:

```python
# From extractor.py - LLM-powered fact extraction
EXTRACTION_PROMPT = """Analyze this conversation and identify facts:
- User preferences (likes, dislikes, habits)
- Facts about people in their life
- Important dates or events
- Explicit requests to remember something
- Corrections to previously known information

Return JSON array with:
- content: The fact (standalone, no unresolved pronouns)
- subjects: Names of people this is about
- shared: true if group knowledge
- confidence: 0.0-1.0
"""

async def extract_from_conversation(
    self,
    messages: list[Message],
    existing_memories: list[str] | None = None,
) -> list[ExtractedFact]
```

**Strengths**:
- Rich semantic search with vector embeddings
- Automatic conflict detection and supersession
- Person tracking with relationships and aliases
- Scoped memory (personal vs group)
- Background extraction with confidence scoring
- Memory expiration and garbage collection

**Weaknesses**:
- Requires OpenAI API for embeddings
- More complex infrastructure (SQLite + sqlite-vec)
- Higher operational overhead

---

### archer (TypeScript)

**Architecture**: archer uses a simple filesystem-based approach with Markdown files.

**Storage Model**:

```
./data/
  MEMORY.md                 # Global memory (shared across chats)
  <chat-id>/
      MEMORY.md             # Chat-specific memory
```

**Memory Loading**:

```typescript
// From agent.ts - Simple file-based memory
function getMemory(channelDir: string): string {
    const parts: string[] = [];

    // Read workspace-level memory (shared across all channels)
    const workspaceMemoryPath = join(channelDir, "..", "MEMORY.md");
    if (existsSync(workspaceMemoryPath)) {
        const content = readFileSync(workspaceMemoryPath, "utf-8").trim();
        if (content) {
            parts.push(`### Global Workspace Memory\n${content}`);
        }
    }

    // Read channel-specific memory
    const channelMemoryPath = join(channelDir, "MEMORY.md");
    if (existsSync(channelMemoryPath)) {
        const content = readFileSync(channelMemoryPath, "utf-8").trim();
        if (content) {
            parts.push(`### Channel-Specific Memory\n${content}`);
        }
    }

    return parts.join("\n\n");
}
```

**System Prompt Integration**:

```typescript
// From agent.ts - Memory in system prompt
return `You are archer, a Telegram bot personal assistant.

## Memory
Write to MEMORY.md files to persist context across conversations.
- Global (${workspacePath}/MEMORY.md): skills, preferences, project info
- Chat (${chatPath}/MEMORY.md): chat-specific decisions, ongoing work
Update when you learn something important.

### Current Memory
${memory}
`;
```

**Strengths**:
- Human-readable and editable
- Git-friendly (version control)
- No external dependencies
- Simple mental model
- Agent can self-manage memory via filesystem tools

**Weaknesses**:
- No semantic search (linear scan)
- No automatic conflict detection
- No person tracking
- Manual memory management by agent
- Limited scalability

---

### clawdbot (TypeScript)

**Architecture**: clawdbot currently has no dedicated memory system, relying on session persistence for continuity. Research documents propose a future Hindsight-inspired system.

**Current State**:

```typescript
// From conversation-store-memory.ts - Simple in-memory conversation store
export function createMSTeamsConversationStoreMemory(
  initial: MSTeamsConversationStoreEntry[] = [],
): MSTeamsConversationStore {
  const map = new Map<string, StoredConversationReference>();

  return {
    upsert: async (conversationId, reference) => {
      map.set(conversationId, reference);
    },
    get: async (conversationId) => {
      return map.get(conversationId) ?? null;
    },
    // ...
  };
}
```

**Proposed Architecture** (from research docs):

```
~/clawd/
  memory.md                    # Core facts + preferences
  memory/
    YYYY-MM-DD.md              # Daily logs (append-only)
  bank/                        # Curated memory pages
    world.md                   # Objective facts
    experience.md              # Agent's experiences
    opinions.md                # Subjective prefs + confidence
    entities/
      Peter.md
      The-Castle.md
```

**Proposed Retain/Recall/Reflect Loop**:

```markdown
## Retain (in daily logs)
- W @Peter: Currently in Marrakech (Nov 27-Dec 1, 2025)
- B @warelay: Fixed WS crash with try/catch wrapper
- O(c=0.95) @Peter: Prefers concise replies on WhatsApp

Type prefixes:
- W: World fact
- B: Biographical/experience
- O: Opinion with confidence
- S: Summary/observation
```

**Proposed Index**:

```
~/clawd/.memory/index.sqlite
- SQLite FTS5 for lexical search
- Optional embeddings for semantic search
- Rebuildable from Markdown source
```

**Strengths** (proposed):
- Human-readable Markdown source of truth
- Entity-centric organization
- Opinion confidence tracking
- Temporal queries support
- Offline-first design

**Weaknesses**:
- Currently unimplemented
- Complex reflection pipeline
- Manual curation required

---

### pi-mono (mom) (TypeScript)

**Architecture**: mom (from pi-mono) uses the same MEMORY.md pattern as archer, with workspace-level and channel-level files.

**Storage Model**:

```
/workspace/
  MEMORY.md                    # Global memory (all channels)
  <channel-id>/
      MEMORY.md                # Channel-specific memory
```

**Memory Loading** (identical to archer):

```typescript
// From agent.ts - Same pattern as archer
function getMemory(channelDir: string): string {
    const parts: string[] = [];

    const workspaceMemoryPath = join(channelDir, "..", "MEMORY.md");
    if (existsSync(workspaceMemoryPath)) {
        const content = readFileSync(workspaceMemoryPath, "utf-8").trim();
        if (content) {
            parts.push(`### Global Workspace Memory\n${content}`);
        }
    }

    const channelMemoryPath = join(channelDir, "MEMORY.md");
    if (existsSync(channelMemoryPath)) {
        const content = readFileSync(channelMemoryPath, "utf-8").trim();
        if (content) {
            parts.push(`### Channel-Specific Memory\n${content}`);
        }
    }

    return parts.join("\n\n");
}
```

**System Prompt Integration** (for Slack):

```typescript
// From agent.ts - Memory instructions
return `You are mom, a Slack bot assistant.

## Memory
Write to MEMORY.md files to persist context across conversations.
- Global (${workspacePath}/MEMORY.md): skills, preferences, project info
- Channel (${channelPath}/MEMORY.md): channel-specific decisions, ongoing work
Update when you learn something important.

### Current Memory
${memory}
`;
```

**Strengths**:
- Human-readable and editable
- Git-friendly
- Simple implementation
- Agent self-manages memory

**Weaknesses**:
- No semantic search
- No automatic conflict detection
- No person tracking
- Manual memory management

---

## Key Differences

### Storage Philosophy

| Codebase | Philosophy |
|----------|------------|
| **ash** | Structured database with embeddings - machine-optimized retrieval |
| **archer/mom** | Plain Markdown files - human-readable, agent-managed |
| **clawdbot** | Session-only (proposed: Markdown + derived index) |

### Retrieval Approach

| Codebase | Approach | Complexity |
|----------|----------|------------|
| **ash** | Vector similarity search | High |
| **archer/mom** | Full file injection into context | Low |
| **clawdbot** | None (proposed: FTS + embeddings) | Medium |

### Memory Lifecycle

| Codebase | Creation | Updates | Deletion |
|----------|----------|---------|----------|
| **ash** | Tool call or background extraction | Supersession | Expiration + GC |
| **archer/mom** | Agent writes to MEMORY.md | Agent edits file | Agent deletes |
| **clawdbot** | N/A | N/A | N/A |

### Scoping Model

| Codebase | Personal | Group | Global |
|----------|----------|-------|--------|
| **ash** | owner_user_id | chat_id | N/A |
| **archer** | N/A | Per-chat MEMORY.md | Workspace MEMORY.md |
| **mom** | N/A | Per-channel MEMORY.md | Workspace MEMORY.md |
| **clawdbot** | Per-session | N/A | N/A |

---

## Recommendations

### For Simple Use Cases

The **archer/mom** approach (MEMORY.md files) is recommended when:
- Memory volume is small (fits in context window)
- Human editability is important
- Simplicity is valued over features
- No external API dependencies desired

### For Advanced Use Cases

The **ash** approach (SQLite + embeddings) is recommended when:
- Large memory volumes expected
- Semantic search is important
- Automatic conflict resolution needed
- Person/entity tracking required
- Background extraction valuable

### Hybrid Approach

The **clawdbot** proposed architecture offers a middle ground:
- Markdown as source of truth (human-readable)
- SQLite index for efficient retrieval (machine-optimized)
- FTS5 for lexical search (no external APIs)
- Optional embeddings for semantic search

### Implementation Priorities

If building a new memory system:

1. **Start simple** - MEMORY.md files work for most cases
2. **Add FTS** - When linear search becomes slow
3. **Add embeddings** - When semantic relevance matters
4. **Add conflict detection** - When contradictory facts cause issues
5. **Add extraction** - When manual memory becomes burdensome

---

## Code References

| Codebase | Key Files |
|----------|-----------|
| **ash** | `src/ash/memory/manager.py`, `store.py`, `retrieval.py`, `extractor.py`, `embeddings.py`, `types.py` |
| **archer** | `src/agent.ts` (getMemory function) |
| **clawdbot** | `docs/experiments/research/memory.md`, `src/msteams/conversation-store-memory.ts` |
| **pi-mono** | `packages/mom/src/agent.ts` (getMemory function) |
