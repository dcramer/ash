# Agent Core

> Orchestrates LLM interactions with agentic tool-use loop

Files: src/ash/core/agent.py, src/ash/core/session.py

## Requirements

### MUST

- Run agentic loop: LLM -> tools -> LLM until text response
- Limit tool iterations (default 10, configurable)
- Execute multiple tools per iteration if LLM requests them
- Pass tool results back to LLM for next iteration
- Track session state across conversation turns
- Support streaming responses with mid-stream tool execution
- Get system prompt from workspace configuration
- Return response with text, tool call history, and iteration count

### SHOULD

- Log tool executions for debugging
- Include tool execution status indicators in streaming output
- Handle empty LLM responses gracefully

### MAY

- Support parallel tool execution
- Add cost tracking for iterations
- Support tool execution timeout per-tool

## Interface

```python
@dataclass
class AgentConfig:
    model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    max_tool_iterations: int = 10

@dataclass
class AgentResponse:
    text: str
    tool_calls: list[dict[str, Any]]  # id, name, input, result, is_error
    iterations: int

class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        tool_executor: ToolExecutor,
        workspace: Workspace,
        config: AgentConfig | None = None,
    ): ...

    async def process_message(
        self,
        user_message: str,
        session: SessionState,
    ) -> AgentResponse: ...

    async def process_message_streaming(
        self,
        user_message: str,
        session: SessionState,
    ) -> AsyncIterator[str]: ...
```

```python
@dataclass
class SessionState:
    session_id: str
    provider: str
    chat_id: str
    user_id: str
    messages: list[Message]
    metadata: dict[str, Any]

    def add_user_message(content: str) -> Message
    def add_assistant_message(content: str | list[ContentBlock]) -> Message
    def add_tool_result(tool_use_id: str, content: str, is_error: bool = False) -> Message
    def get_messages_for_llm() -> list[Message]
    def get_pending_tool_uses() -> list[ToolUse]
    def to_json() / from_json() -> serialization
```

## Behaviors

| Scenario | Behavior |
|----------|----------|
| User message, no tools needed | Single LLM call, return text |
| User message, tools needed | LLM -> tool execution -> LLM -> text |
| Multiple tools requested | Execute all sequentially, combine results |
| Tool returns error | Pass error to LLM with is_error=True |
| Max iterations reached | Return message indicating limit reached |
| Streaming + tools | Yield text chunks, pause for tools, continue |
| Empty LLM response | Return early from streaming |

## Errors

| Condition | Response |
|-----------|----------|
| Max iterations exceeded | AgentResponse with limit message, iterations=max |
| Tool not found | Tool result with error, continue loop |
| Tool execution failure | Tool result with error, continue loop |
| LLM API error | Propagates to caller |

## Verification

```bash
uv run pytest tests/test_agent.py -v
uv run ash chat "What time is it?"  # No tools
uv run ash chat "Run: echo hello"   # Tool use
```

- Single-turn text response works
- Tool execution loop completes
- Streaming yields text chunks
- Tool indicators appear in streaming
- Max iteration limit enforced
