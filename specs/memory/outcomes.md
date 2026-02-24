# Memory Outcomes

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
