# Memory

> Hybrid memory system with automatic context retrieval, explicit memory tools, and person-aware memories

Files: `src/ash/memory/manager.py`, `src/ash/memory/store.py`, `src/ash/memory/retrieval.py`, `src/ash/memory/embeddings.py`, `src/ash/tools/builtin/memory.py`, `src/ash/core/agent.py`, `src/ash/db/models.py`

## Requirements

### MUST

- Retrieve relevant context via semantic search before each LLM call
- Apply similarity threshold (default 0.3) to filter irrelevant messages
- Include top N memory entries regardless of similarity (personal assistant has small memory store)
- Include retrieved context (messages, memories) in system prompt
- Store conversation messages to database after each turn
- Index messages for semantic search via embeddings
- Link sessions to provider/chat_id/user_id
- Persist data across restarts
- Provide `remember` tool to store facts in memory
- Provide `recall` tool for explicit memory search
- Index memory entries for semantic search
- Support optional expiration on memory entries
- Track memory ownership (which user added it)
- Track memory subjects (which people the fact is about via JSON array)
- Support Person entities with name, relationship, and aliases
- Support batch storage of multiple facts in single remember tool call
- Include known people in system prompt for context
- Degrade gracefully if embedding service unavailable
- Mark conflicting memories as superseded when new memory is added (similarity >= 0.75)
- Filter out superseded memories from default retrieval
- Preserve superseded memories for history/audit

### SHOULD

- Limit retrieved context by token count
- Prioritize recent messages at equal relevance
- Include source attribution in retrieved context
- Include subject attribution (about X) in retrieved context
- Auto-extract person names from content when creating Person entities

### MAY

- Auto-extract facts from conversations to user profile
- Cache embeddings to avoid recomputation
- Add identity anchoring via external_id/external_provider on Person (for stable IDs across username changes)

## Data Models

### Person

```python
class Person(Base):
    id: str                    # UUID
    owner_user_id: str         # Which user owns this relationship
    name: str                  # "Sarah"
    relation: str | None   # "wife", "boss", "friend"
    aliases: list[str] | None  # ["my wife", "Sarah"]
    metadata_: dict | None
    created_at: datetime
    updated_at: datetime
```

### Memory

```python
class Memory(Base):
    id: str
    content: str
    source: str | None
    created_at: datetime
    expires_at: datetime | None
    metadata_: dict | None
    owner_user_id: str | None          # Who added this fact
    chat_id: str | None                # Which chat (NULL for personal)
    subject_person_ids: list[str] | None  # JSON array of Person IDs (who it's about)
    superseded_at: datetime | None     # When this memory was superseded
    superseded_by_id: str | None       # FK to Memory (newer version)
```

## Interface

### MemoryManager

```python
class MemoryManager:
    def __init__(self, store: MemoryStore, retriever: SemanticRetriever): ...

    async def get_context_for_message(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        max_messages: int = 5,
        max_memories: int = 10,
        min_message_similarity: float = 0.3,
    ) -> RetrievedContext: ...

    async def persist_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None: ...

    async def add_memory(
        self,
        content: str,
        source: str = "user",
        expires_at: datetime | None = None,
        expires_in_days: int | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
    ) -> Memory: ...

    async def search(
        self,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]: ...

    async def get_known_people(self, owner_user_id: str) -> list[Person]: ...

    async def find_person(self, owner_user_id: str, reference: str) -> Person | None: ...

    async def resolve_or_create_person(
        self,
        owner_user_id: str,
        reference: str,
        content_hint: str | None = None,
    ) -> PersonResolutionResult: ...
```

### RetrievedContext

```python
@dataclass
class RetrievedContext:
    messages: list[SearchResult]
    memories: list[SearchResult]  # Includes subject_name in metadata
```

### PersonResolutionResult

```python
@dataclass
class PersonResolutionResult:
    person_id: str
    created: bool
    person_name: str
```

### Tools

```python
# remember tool
{
    "name": "remember",
    "description": "Store a fact or preference in long-term memory",
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "A single fact to remember"},
            "facts": {
                "type": "array",
                "description": "Batch multiple facts in one call (preferred)",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "subjects": {"type": "array", "items": {"type": "string"}},
                        "expires_in_days": {"type": "integer"},
                        "shared": {"type": "boolean"}
                    },
                    "required": ["content"]
                }
            },
            "subjects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Who this fact is about (e.g., ['Sarah'], ['my wife', 'John'])"
            },
            "expires_in_days": {"type": "integer", "description": "Days until expiration"},
            "shared": {"type": "boolean", "description": "True for group/team facts"}
        }
    }
}

# recall tool
{
    "name": "recall",
    "description": "Search memory for relevant information",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"},
            "about": {"type": "string", "description": "Filter to person (e.g., 'my wife', 'Sarah')"}
        },
        "required": ["query"]
    }
}
```

## Behaviors

| Scenario | Behavior |
|----------|----------|
| Every message | Auto-retrieve relevant context (semantic search on user's message) |
| Auto-retrieval (messages) | Returns up to 5 messages above 0.3 similarity |
| Auto-retrieval (memories) | Returns up to 10 memory entries ranked by relevance with subject attribution |
| User says "remember my wife's name is Sarah" | Agent uses `remember` with subjects=["my wife"], creates Person entity |
| Subsequent "she likes Italian food" | Agent uses `remember` with subjects=["my wife"], links to existing Person |
| User says "remember Sarah and John are getting married" | Agent uses `remember` with subjects=["Sarah", "John"], links to both |
| Multiple facts at once | Agent uses `remember` with facts=[...] array for batch storage |
| User asks "what does my wife like?" | Agent may use `recall` with about="my wife" for targeted search |
| Low similarity messages | Filtered out (below 0.3 threshold) |
| Embedding service down | Log warning, continue without semantic search |
| No relevant context found | Proceed with empty context |

### Person Resolution

| Reference | Resolution |
|-----------|------------|
| "my wife" | Strip "my ", search by relationship="wife" |
| "Sarah" | Search by name |
| "my wife Sarah" | Extract name from content, create with name="Sarah", relationship="wife" |
| First mention | Create new Person entity |
| Subsequent mention | Find existing Person by name/relationship/alias |

### System Prompt Enhancement

When known people exist for a user, the system prompt includes:

```
## Known People

The user has told you about these people:

- **Sarah** (wife)
- **Michael** (boss)

Use these when interpreting references like 'my wife' or 'Sarah'.
```

Memory context includes subject attribution:

```
## Relevant Context from Memory

- [Memory (about Sarah)] Sarah likes Italian food
- [Memory] User prefers concise responses
```

### Memory Supersession

When a new memory conflicts with an existing memory (high semantic similarity in the same scope), the old memory is marked as superseded.

| Scenario | Behavior |
|----------|----------|
| New memory added | Check for conflicting memories (similarity >= 0.75) |
| Conflict detected | Mark old memory with `superseded_at` and `superseded_by_id` |
| Default retrieval | Exclude superseded memories |
| History access | Query superseded memories with `include_superseded=True` |
| Different subjects | Not considered conflicts (even if similar content) |

Example:
1. Store "User's favorite color is red"
2. Later store "User's favorite color is blue"
3. "...is red" gets `superseded_by_id` = "...is blue"
4. Only "...is blue" appears in retrieval

## Errors

| Condition | Response |
|-----------|----------|
| Embedding service unavailable | Log warning, skip retrieval, continue |
| Database unavailable | Fail request |
| No relevant context | Proceed with empty context |
| Remember tool fails | Return error to LLM |
| Person not found for filter | Return unfiltered results |

## Verification

```bash
uv run pytest tests/test_memory.py -v
uv run ash chat "Remember my wife's name is Sarah"
uv run ash chat "Remember she likes Italian food"
uv run ash chat "What does my wife like?"
```

- [ ] Person model exists in `src/ash/db/models.py`
- [ ] Memory model has owner_user_id, chat_id, subject_person_ids (JSON array), superseded_at, superseded_by_id
- [ ] Migration 002 adds chat_id, migration 003 adds supersession columns, migration 004 converts subject_person_id to subject_person_ids
- [ ] MemoryManager has person resolution methods
- [ ] MemoryManager.add_memory() checks for conflicts and supersedes old memories
- [ ] `remember` tool accepts subjects array and facts batch parameter
- [ ] `recall` tool accepts about filter
- [ ] Known people appear in system prompt
- [ ] Memories show subject attribution in context
- [ ] Agent calls `get_known_people()` before LLM call
- [ ] Superseded memories excluded from search_memories by default
- [ ] Superseded memories can be retrieved with include_superseded=True
