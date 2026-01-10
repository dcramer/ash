# LLM Providers

> Abstract interface for LLM completions, streaming, and embeddings

Files: src/ash/llm/base.py, src/ash/llm/types.py, src/ash/llm/anthropic.py, src/ash/llm/openai.py, src/ash/llm/registry.py

## Requirements

### MUST

- Define abstract provider interface (LLMProvider ABC)
- Support non-streaming completions with tools
- Support streaming completions with tools
- Support text embeddings generation
- Implement Anthropic Claude provider
- Implement OpenAI provider
- Registry for provider lookup by name
- Convert between internal types and provider-specific formats

### SHOULD

- Return token usage in completion response
- Include stop reason in response
- Stream tool use with start/delta/end events
- Support configurable model per request

### MAY

- Support additional providers (Ollama, etc.)
- Automatic retry on transient errors
- Token counting before API call

## Interface

```python
class LLMProvider(ABC):
    @property
    def name(self) -> str: ...
    @property
    def default_model(self) -> str: ...

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> CompletionResponse: ...

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]: ...

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]: ...
```

### Message Types

```python
class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class Message:
    role: Role
    content: str | list[ContentBlock]
    def get_text() -> str
    def get_tool_uses() -> list[ToolUse]

@dataclass
class TextContent:
    text: str

@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]

@dataclass
class ToolResult:
    tool_use_id: str
    content: str
    is_error: bool = False

@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
```

### Streaming Types

```python
class StreamEventType(Enum):
    TEXT_DELTA = "text_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_END = "tool_use_end"
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"
    ERROR = "error"

@dataclass
class StreamChunk:
    type: StreamEventType
    content: str | dict | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None

@dataclass
class CompletionResponse:
    message: Message
    usage: Usage | None = None
    stop_reason: str | None = None
    model: str | None = None

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
```

### Registry

```python
class LLMRegistry:
    def register(name: str, provider: LLMProvider) -> None
    def get(name: str) -> LLMProvider
    def has(name: str) -> bool
    def names() -> list[str]
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| Text message | Text response | Simple completion |
| Message + tools | Text or ToolUse blocks | May request tools |
| Stream request | StreamChunk iterator | Yields deltas |
| Stream + tools | Mixed text/tool chunks | Tool args in deltas |
| Embed texts | Float vectors | 1536 dims for OpenAI |

## Errors

| Condition | Response |
|-----------|----------|
| Invalid API key | AuthenticationError |
| Rate limit | RateLimitError (429) |
| Model not found | InvalidRequestError |
| Network failure | Propagates httpx error |
| Anthropic embed call | NotImplementedError (use OpenAI) |

## Verification

```bash
uv run pytest tests/test_llm_types.py -v
uv run ash chat "Hello"  # Uses configured provider
```

- Anthropic completions work
- OpenAI completions work
- Streaming yields chunks
- Tool use parsed correctly
- Embeddings generated (OpenAI)
