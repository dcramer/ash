# Memory

> Hybrid memory system with automatic context retrieval, explicit memory tools, and person-aware knowledge

Files: `src/ash/memory/manager.py`, `src/ash/memory/store.py`, `src/ash/memory/retrieval.py`, `src/ash/memory/embeddings.py`, `src/ash/tools/builtin/memory.py`, `src/ash/core/agent.py`, `src/ash/db/models.py`

## Requirements

### MUST

- Retrieve relevant context via semantic search before each LLM call
- Apply similarity threshold (default 0.3) to filter irrelevant messages
- Include top N knowledge entries regardless of similarity (personal assistant has small KB)
- Include retrieved context (messages, knowledge) in system prompt
- Store conversation messages to database after each turn
- Index messages for semantic search via embeddings
- Link sessions to provider/chat_id/user_id
- Persist data across restarts
- Provide `remember` tool to store facts in knowledge base
- Provide `recall` tool for explicit memory search
- Index knowledge entries for semantic search
- Support optional expiration on knowledge entries
- Track knowledge ownership (which user added it)
- Track knowledge subject (which person the fact is about)
- Support Person entities with name, relationship, and aliases
- Include known people in system prompt for context
- Degrade gracefully if embedding service unavailable

### SHOULD

- Limit retrieved context by token count
- Prioritize recent messages at equal relevance
- Include source attribution in retrieved context
- Include subject attribution (about X) in retrieved context
- Auto-extract person names from content when creating Person entities

### MAY

- Auto-extract facts from conversations to user profile
- Cache embeddings to avoid recomputation

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

### Knowledge (updated)

```python
class Knowledge(Base):
    id: str
    content: str
    source: str | None
    created_at: datetime
    expires_at: datetime | None
    metadata_: dict | None
    owner_user_id: str | None       # Who added this fact
    subject_person_id: str | None   # FK to Person (who it's about)
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
        max_knowledge: int = 10,
        min_message_similarity: float = 0.3,
    ) -> RetrievedContext: ...

    async def persist_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None: ...

    async def add_knowledge(
        self,
        content: str,
        source: str = "user",
        expires_at: datetime | None = None,
        owner_user_id: str | None = None,
        subject_person_id: str | None = None,
    ) -> Knowledge: ...

    async def search(
        self,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
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
    knowledge: list[SearchResult]  # Includes subject_name in metadata
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
            "content": {"type": "string", "description": "The fact to remember"},
            "subject": {"type": "string", "description": "Who this fact is about (e.g., 'my wife', 'boss')"},
            "expires_in_days": {"type": "integer", "description": "Days until expiration"}
        },
        "required": ["content"]
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
| Auto-retrieval (knowledge) | Returns up to 10 knowledge entries ranked by relevance with subject attribution |
| User says "remember my wife's name is Sarah" | Agent uses `remember` with subject="my wife", creates Person entity |
| Subsequent "she likes Italian food" | Agent uses `remember` with subject="my wife", links to existing Person |
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

Knowledge context includes subject attribution:

```
## Relevant Context from Memory

- [Knowledge (about Sarah)] Sarah likes Italian food
- [Knowledge] User prefers concise responses
```

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
- [ ] Knowledge model has owner_user_id and subject_person_id
- [ ] Migration 002 adds Person table and Knowledge columns
- [ ] MemoryManager has person resolution methods
- [ ] `remember` tool accepts subject parameter
- [ ] `recall` tool accepts about filter
- [ ] Known people appear in system prompt
- [ ] Knowledge shows subject attribution in context
- [ ] Agent calls `get_known_people()` before LLM call
