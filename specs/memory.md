# Memory

> Long-term knowledge persistence so the agent remembers facts across conversations

## Intent

The memory subsystem solves a fundamental problem: conversations are ephemeral, but users expect the agent to remember important facts about them. Memory provides:

1. **Automatic context** - Relevant facts surface before each response without user intervention
2. **Explicit storage** - Users can tell the agent to remember specific things
3. **Person awareness** - Facts can be attributed to people in the user's life
4. **Knowledge evolution** - New information supersedes outdated facts

Memory is NOT for:
- Conversation history (that's sessions)
- Temporary task state (that's session context)
- Credentials or secrets (security risk)

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
- Superseded facts preserved for audit but excluded from search
- Subject-specific facts don't conflict with general facts

### Memory doesn't grow unbounded

- Optional `max_entries` cap with smart eviction
- Expired memories cleaned up automatically
- Garbage collection removes superseded/expired entries

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

## Background Extraction

Optionally, facts are extracted automatically from conversations:
- Runs after each exchange (debounced)
- Extracts preferences, facts about people, important dates
- Skips assistant actions, temporary context, credentials
- Confidence threshold filters low-quality extractions

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
| `memory.delete` | Delete a memory | `memory_id` (required) |

The sandbox CLI (`ash memory`) wraps these RPC calls with environment-provided `user_id` and `chat_id`.

## Verification

```bash
# Unit tests
uv run pytest tests/test_memory.py tests/test_memory_extractor.py -v

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
```

### Checklist

- [ ] Agent answers questions using previously stored facts
- [ ] `remember` tool stores facts when user explicitly requests
- [ ] Automatic context injection retrieves relevant memories before each response
- [ ] Person entities created for "my wife", "Sarah", etc.
- [ ] Known people appear in system prompt
- [ ] Memory results show subject attribution ("about Sarah")
- [ ] Conflicting facts supersede old versions
- [ ] Superseded facts excluded from default retrieval
- [ ] `max_entries` evicts oldest when exceeded
- [ ] `ash memory gc` removes expired/superseded entries
- [ ] Background extraction captures facts from conversations
- [ ] Extraction skips low-confidence and duplicate facts
