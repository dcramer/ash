# Memory

> Persistent memory with context retrieval for personalized conversations

Files: src/ash/memory/store.py, src/ash/memory/retrieval.py, src/ash/memory/embeddings.py, src/ash/core/agent.py

## Purpose

A personal assistant must remember past conversations, learn about the user, and retrieve relevant context to inform responses. Memory is not just storage - it's active retrieval integrated into the agent loop.

## Requirements

### MUST

**Persistence**
- Store all conversation messages to database after each turn
- Store sessions linked to provider/chat/user identifiers
- Persist across restarts

**Context Retrieval**
- Before each LLM call, retrieve relevant past context via semantic search
- Include retrieved context in the prompt (RAG pattern)
- Retrieve from both conversation history and knowledge base

**User Context**
- Track user profile with preferences and learned facts
- Include relevant user context in system prompt
- Update user understanding based on conversations

**Knowledge Base**
- Store knowledge entries with optional expiration
- Retrieve relevant knowledge based on query similarity
- Support manual knowledge insertion (via tool or API)

### SHOULD

- Limit context window by token count, not just message count
- Prioritize recent messages over old ones at equal relevance
- Chunk long documents for better retrieval
- Cache embeddings to avoid recomputation

### MAY

- Auto-extract facts about user from conversations
- Summarize old conversations to compress history
- Support multiple embedding providers
- Background indexing for large imports

## Integration

### Agent Loop with Memory

```
1. User sends message
2. Agent retrieves relevant context:
   - Semantic search over past messages
   - Semantic search over knowledge base
   - Load user profile
3. Agent builds prompt with retrieved context
4. LLM generates response (possibly with tools)
5. Agent persists:
   - User message + assistant response to database
   - Index new messages for future retrieval
6. Return response to user
```

### Context Injection

```python
# Before LLM call
relevant_messages = await retriever.search_messages(user_message, limit=5)
relevant_knowledge = await retriever.search_knowledge(user_message, limit=3)
user_profile = await store.get_user_profile(user_id)

# Build augmented system prompt
system = f"""
{base_system_prompt}

## About the user
{user_profile.notes}

## Relevant context
{format_retrieved_context(relevant_messages, relevant_knowledge)}
"""
```

## Interface

### MemoryManager (new - orchestrates retrieval and persistence)

```python
class MemoryManager:
    def __init__(
        self,
        store: MemoryStore,
        retriever: SemanticRetriever,
    ): ...

    async def get_context_for_message(
        self,
        session: Session,
        user_message: str,
        max_tokens: int = 2000,
    ) -> RetrievedContext: ...

    async def persist_turn(
        self,
        session: Session,
        user_message: str,
        assistant_response: str,
    ) -> None: ...

    async def get_user_context(self, user_id: str) -> str | None: ...
```

### MemoryStore (data access)

```python
class MemoryStore:
    # Sessions
    async def get_or_create_session(provider, chat_id, user_id) -> Session
    async def get_session(session_id) -> Session | None

    # Messages
    async def add_message(session_id, role, content, metadata) -> Message
    async def get_messages(session_id, limit, before) -> list[Message]

    # Knowledge
    async def add_knowledge(content, source, expires_at) -> Knowledge
    async def get_knowledge(limit, include_expired) -> list[Knowledge]

    # User Profiles
    async def get_or_create_user_profile(user_id, provider) -> UserProfile
    async def update_user_notes(user_id, notes) -> UserProfile | None
```

### SemanticRetriever (vector search)

```python
class SemanticRetriever:
    async def index_message(message_id, content) -> None
    async def index_knowledge(knowledge_id, content) -> None

    async def search_messages(query, session_id, limit) -> list[SearchResult]
    async def search_knowledge(query, limit) -> list[SearchResult]
    async def search_all(query, limit) -> list[SearchResult]
```

### Data Types

```python
@dataclass
class RetrievedContext:
    messages: list[SearchResult]
    knowledge: list[SearchResult]
    user_notes: str | None
    token_count: int

@dataclass
class SearchResult:
    id: str
    content: str
    similarity: float
    source_type: str  # "message" or "knowledge"
    metadata: dict | None
```

## Storage

### SQLite Tables

```sql
sessions (id, provider, chat_id, user_id, created_at, updated_at)
messages (id, session_id, role, content, created_at, token_count)
knowledge (id, content, source, created_at, expires_at)
user_profiles (user_id, provider, username, display_name, notes)
```

### Vector Tables (sqlite-vec)

```sql
message_embeddings (message_id, embedding FLOAT[1536])
knowledge_embeddings (knowledge_id, embedding FLOAT[1536])
```

## Behaviors

| Scenario | Behavior |
|----------|----------|
| First message in session | Create session, no past context retrieved |
| Subsequent messages | Retrieve relevant past messages from this + other sessions |
| User mentions preference | Should be extractable to user profile (MAY) |
| Knowledge query | Retrieve matching knowledge entries |
| Old expired knowledge | Excluded from retrieval by default |
| Context exceeds token limit | Truncate lowest-relevance items first |

## Errors

| Condition | Response |
|-----------|----------|
| Embedding service unavailable | Log warning, skip retrieval, continue without context |
| Database unavailable | Fail request (memory is required) |
| No relevant context found | Proceed with empty context (not an error) |

## Verification

```bash
uv run pytest tests/test_memory.py -v
uv run ash chat "Remember that I prefer concise responses"
uv run ash chat "What do you know about my preferences?"  # Should recall
```

- Conversation persists across CLI restarts
- Relevant past context appears in LLM prompts
- User profile notes are included in system prompt
- Knowledge retrieval returns semantically similar entries
