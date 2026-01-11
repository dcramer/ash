# Memory

> Hybrid memory system with automatic context retrieval and explicit memory tools

Files: `src/ash/memory/manager.py`, `src/ash/memory/store.py`, `src/ash/memory/retrieval.py`, `src/ash/memory/embeddings.py`, `src/ash/tools/builtin/memory.py`, `src/ash/core/agent.py`

## Requirements

### MUST

- Retrieve relevant context via semantic search before each LLM call
- Apply similarity threshold (default 0.3) to filter irrelevant messages
- Include top N knowledge entries regardless of similarity (personal assistant has small KB)
- Include retrieved context (messages, knowledge, user notes) in system prompt
- Store conversation messages to database after each turn
- Index messages for semantic search via embeddings
- Link sessions to provider/chat_id/user_id
- Persist data across restarts
- Provide `remember` tool to store facts in knowledge base
- Provide `recall` tool for explicit memory search
- Index knowledge entries for semantic search
- Support optional expiration on knowledge entries
- Degrade gracefully if embedding service unavailable

### SHOULD

- Limit retrieved context by token count
- Prioritize recent messages at equal relevance
- Include source attribution in retrieved context

### MAY

- Auto-extract facts from conversations to user profile
- Cache embeddings to avoid recomputation

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
    ) -> Knowledge: ...

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]: ...
```

### RetrievedContext

```python
@dataclass
class RetrievedContext:
    messages: list[SearchResult]
    knowledge: list[SearchResult]
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
            "expires_in_days": {"type": "integer", "description": "Days until expiration"}
        },
        "required": ["content"]
    }
}

# recall tool (MAY)
{
    "name": "recall",
    "description": "Search memory for relevant information",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"}
        },
        "required": ["query"]
    }
}
```

### Agent Integration

```python
class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        tool_executor: ToolExecutor,
        workspace: Workspace,
        memory_manager: MemoryManager | None = None,
        config: AgentConfig | None = None,
    ): ...
```

## Behaviors

| Scenario | Behavior |
|----------|----------|
| Every message | Auto-retrieve relevant context (semantic search on user's message) |
| Auto-retrieval (messages) | Returns up to 5 messages above 0.3 similarity |
| Auto-retrieval (knowledge) | Returns up to 10 knowledge entries ranked by relevance (no threshold) |
| User says "remember X" | Agent uses `remember` tool, stores to knowledge base |
| User asks about past topic | Context auto-retrieved if semantically similar to current message |
| User asks "what did we discuss about X" | Agent may use `recall` for targeted search |
| Low similarity messages | Filtered out (below 0.3 threshold) |
| Embedding service down | Log warning, continue without semantic search |
| No relevant context found | Proceed with empty context |

### When to Use `recall` vs Auto-Retrieval

| Situation | Approach |
|-----------|----------|
| Current message relates to past context | Auto-retrieval handles it |
| User explicitly asks to search memory | Use `recall` tool |
| Looking for something not related to current topic | Use `recall` tool |
| Want to search with custom query | Use `recall` tool |

## Errors

| Condition | Response |
|-----------|----------|
| Embedding service unavailable | Log warning, skip retrieval, continue |
| Database unavailable | Fail request |
| No relevant context | Proceed with empty context |
| Remember tool fails | Return error to LLM |

## Verification

```bash
uv run pytest tests/test_memory.py -v
uv run ash chat "Remember that I prefer concise responses"
uv run ash chat "What communication style do I prefer?"
```

- [ ] MemoryManager class exists in `src/ash/memory/manager.py`
- [ ] Agent accepts optional `memory_manager` parameter
- [ ] Agent calls `get_context_for_message()` before LLM call
- [ ] Agent calls `persist_turn()` after response
- [ ] Retrieved context appears in system prompt
- [ ] `remember` tool exists in `src/ash/tools/builtin/memory.py`
- [ ] `remember` tool stores and indexes knowledge
- [ ] Conversation persists across CLI restarts
- [ ] Semantic search returns relevant results
