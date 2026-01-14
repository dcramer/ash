# Tool System Comparison

A comprehensive comparison of tool systems across four agent codebases: ash (Python), archer (TypeScript), clawdbot (TypeScript), and pi-mono (TypeScript).

## Overview

Tool systems are the execution layer that enables agents to interact with the external world. They bridge LLM outputs (tool calls) with actual system operations. Each codebase implements this differently based on their runtime environment, security requirements, and use cases.

| Aspect | ash | archer | clawdbot | pi-mono |
|--------|-----|--------|----------|---------|
| Language | Python | TypeScript | TypeScript | TypeScript |
| Schema System | Pydantic/JSON Schema | TypeBox | TypeBox | TypeBox |
| Core Tools | bash, read, write, web_search, web_fetch | bash, read, write, edit, attach | Inherits from pi-coding-agent + custom | bash, read, write, edit, grep, ls, find |
| Truncation Limits | 50KB / 4000 lines | 50KB / 2000 lines | 50KB / 2000 lines (via pi-coding-agent) | 50KB / 2000 lines |
| Sandbox Mode | Docker (mandatory) | None | Docker (optional) | None |
| Progress Updates | Via executor callback | Via onUpdate callback | Via onUpdate callback | Via onUpdate callback |
| Cancellation | No AbortSignal | AbortSignal | AbortSignal | AbortSignal |

## Detailed Analysis

### 1. ash (Python)

**Location:** `/home/dcramer/src/ash/src/ash/tools/`

ash uses an abstract class-based tool interface with mandatory Docker sandboxing. Tools are Python classes that inherit from a base `Tool` ABC.

#### Tool Interface

```python
# src/ash/tools/base.py
class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult: ...
```

The `ToolResult` dataclass provides success/error factory methods:

```python
@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, **metadata) -> "ToolResult": ...

    @classmethod
    def error(cls, message: str, **metadata) -> "ToolResult": ...
```

#### Executor with Timing/Logging

```python
# src/ash/tools/executor.py
class ToolExecutor:
    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        # Execute with timing
        start_time = time.monotonic()
        result = await tool.execute(input_data, context)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Single source of truth for tool logging
        logger.info(f"Tool: {tool_name} | {input_summary} | {duration_ms}ms")
```

#### Truncation

```python
# src/ash/tools/truncation.py
MAX_OUTPUT_BYTES = 50 * 1024  # 50KB
MAX_OUTPUT_LINES = 4000

def truncate_tail(output: str, ...) -> TruncationResult:
    """Keep last N lines/bytes - for bash output."""

def truncate_head(output: str, ...) -> TruncationResult:
    """Keep first N lines/bytes - for file reads."""
```

Truncated output is saved to temp files for later retrieval.

#### Docker Sandbox (Mandatory)

```python
# src/ash/tools/builtin/bash.py
class BashTool(Tool):
    """All commands run in an isolated container with:
    - Read-only root filesystem
    - All capabilities dropped
    - No privilege escalation
    - Process limits (fork bomb protection)
    - Memory limits
    - Non-root user execution
    - Optional gVisor runtime
    """
```

#### Tool Registry

```python
# src/ash/tools/registry.py
class ToolRegistry:
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool: ...
    def get_definitions(self) -> list[dict[str, Any]]: ...
```

---

### 2. archer (TypeScript)

**Location:** `/home/dcramer/src/archer/src/tools/`

archer uses factory functions that return `AgentTool` objects from pi-agent-core. Tools operate through a shared `Executor` interface for sandbox abstraction.

#### Tool Interface

```typescript
// Uses @mariozechner/pi-agent-core
interface AgentTool<TParameters extends TSchema> {
    name: string;
    label: string;
    description: string;
    parameters: TParameters;
    execute: (
        toolCallId: string,
        params: Static<TParameters>,
        signal?: AbortSignal,
    ) => Promise<{ content: ContentBlock[]; details?: any }>;
}
```

#### Schema Definition with TypeBox

```typescript
// src/tools/bash.ts
const bashSchema = Type.Object({
    label: Type.String({ description: "Brief description..." }),
    command: Type.String({ description: "Bash command to execute" }),
    timeout: Type.Optional(Type.Number({ description: "Timeout in seconds" })),
});
```

#### Truncation (2000 lines / 50KB)

```typescript
// src/tools/truncate.ts
export const DEFAULT_MAX_LINES = 2000;
export const DEFAULT_MAX_BYTES = 50 * 1024; // 50KB

export function truncateHead(content: string, options?: TruncationOptions): TruncationResult;
export function truncateTail(content: string, options?: TruncationOptions): TruncationResult;
```

#### AbortSignal Support

```typescript
// src/tools/bash.ts
execute: async (
    _toolCallId: string,
    { command, timeout },
    signal?: AbortSignal,  // Cancellation support
) => {
    const result = await executor.exec(command, { timeout, signal });
    // ...
}
```

#### Tool Factory Pattern

```typescript
// src/tools/index.ts
export function createMomTools(executor: Executor): AgentTool<any>[] {
    return [
        createReadTool(executor),
        createBashTool(executor),
        createEditTool(executor),
        createWriteTool(executor),
        attachTool,
    ];
}
```

---

### 3. clawdbot (TypeScript)

**Location:** `/home/dcramer/src/clawdbot/src/agents/`

clawdbot inherits from pi-coding-agent and extends with platform-specific tools. It adds sophisticated tool filtering, schema normalization for different providers, and optional Docker sandboxing.

#### Tool Composition

```typescript
// src/agents/pi-tools.ts
export function createClawdbotCodingTools(options?: {
    bash?: BashToolDefaults & ProcessToolDefaults;
    sandbox?: SandboxContext | null;
    config?: ClawdbotConfig;
    abortSignal?: AbortSignal;
    modelProvider?: string;  // Provider-specific quirks
}): AnyAgentTool[] {
    const base = codingTools.flatMap((tool) => {
        if (tool.name === readTool.name) {
            // Wrap with clawdbot-specific handling
            return [createClawdbotReadTool(freshReadTool)];
        }
        // ...
    });

    return [
        ...base,
        bashTool,
        processTool,
        createWhatsAppLoginTool(),  // Platform-specific
        ...createClawdbotTools({ ... }),
    ];
}
```

#### Tool Policy Filtering

```typescript
// Deny/allow lists for subagents
const DEFAULT_SUBAGENT_TOOL_DENY = [
    "sessions_list",
    "sessions_history",
    "sessions_send",
    "sessions_spawn",
];

function filterToolsByPolicy(
    tools: AnyAgentTool[],
    policy?: SandboxToolPolicy,
) { ... }
```

#### Schema Normalization for Multiple Providers

```typescript
// Handles Gemini/OpenAI schema differences
function normalizeToolParameters(tool: AnyAgentTool): AnyAgentTool {
    // Gemini rejects several JSON Schema keywords
    // OpenAI rejects schemas without top-level type: "object"
    return {
        ...tool,
        parameters: cleanSchemaForGemini(schema),
    };
}
```

#### Background Process Support

```typescript
// src/agents/bash-tools.ts
const bashSchema = Type.Object({
    command: Type.String(),
    yieldMs: Type.Optional(Type.Number({
        description: "Milliseconds to wait before backgrounding (default 10000)"
    })),
    background: Type.Optional(Type.Boolean({
        description: "Run in background immediately"
    })),
    timeout: Type.Optional(Type.Number()),
    elevated: Type.Optional(Type.Boolean({
        description: "Run on host with elevated permissions (if allowed)"
    })),
});
```

#### Process Management Tool

```typescript
// Companion tool for background bash sessions
export function createProcessTool(defaults?: ProcessToolDefaults): AgentTool<any> {
    // Actions: list, poll, log, write, kill, clear, remove
}
```

---

### 4. pi-mono (TypeScript)

**Location:** `/home/dcramer/src/pi-mono/packages/coding-agent/src/core/tools/`

pi-mono provides the foundational coding agent toolkit. It emphasizes pluggable operations for remote execution and comprehensive streaming support.

#### Tool Interface with Streaming

```typescript
// packages/agent/src/types.ts
export interface AgentTool<TParameters extends TSchema, TDetails = any> {
    label: string;
    execute: (
        toolCallId: string,
        params: Static<TParameters>,
        signal?: AbortSignal,
        onUpdate?: AgentToolUpdateCallback<TDetails>,  // Streaming updates
    ) => Promise<AgentToolResult<TDetails>>;
}

export type AgentToolUpdateCallback<T> = (partialResult: AgentToolResult<T>) => void;
```

#### Pluggable Operations for Remote Execution

```typescript
// packages/coding-agent/src/core/tools/bash.ts
export interface BashOperations {
    exec: (
        command: string,
        cwd: string,
        options: {
            onData: (data: Buffer) => void;  // Streaming output
            signal?: AbortSignal;
            timeout?: number;
        },
    ) => Promise<{ exitCode: number | null }>;
}

// Default: local shell
const defaultBashOperations: BashOperations = {
    exec: (command, cwd, { onData, signal, timeout }) => {
        const child = spawn(shell, [...args, command], { cwd, detached: true });
        child.stdout.on("data", onData);
        child.stderr.on("data", onData);
        // ...
    },
};
```

Same pattern for read operations:

```typescript
// packages/coding-agent/src/core/tools/read.ts
export interface ReadOperations {
    readFile: (absolutePath: string) => Promise<Buffer>;
    access: (absolutePath: string) => Promise<void>;
    detectImageMimeType?: (absolutePath: string) => Promise<string | null | undefined>;
}
```

#### Streaming Tool Output

```typescript
// Bash tool streams partial output during execution
execute: async (_toolCallId, { command, timeout }, signal, onUpdate) => {
    const handleData = (data: Buffer) => {
        // Stream partial output to callback
        if (onUpdate) {
            const truncation = truncateTail(fullText);
            onUpdate({
                content: [{ type: "text", text: truncation.content || "" }],
                details: { truncation, fullOutputPath: tempFilePath },
            });
        }
    };
    // ...
}
```

#### Tool Sets

```typescript
// packages/coding-agent/src/core/tools/index.ts
export const codingTools: Tool[] = [readTool, bashTool, editTool, writeTool];
export const readOnlyTools: Tool[] = [readTool, grepTool, findTool, lsTool];

export function createCodingTools(cwd: string, options?: ToolsOptions): Tool[];
export function createReadOnlyTools(cwd: string, options?: ToolsOptions): Tool[];
```

#### Image Handling in Read Tool

```typescript
// Auto-resize images to prevent token bloat
if (autoResizeImages) {
    const resized = await resizeImage({ type: "image", data: base64, mimeType });
    content = [
        { type: "text", text: `Read image file [${resized.mimeType}]\n${dimensionNote}` },
        { type: "image", data: resized.data, mimeType: resized.mimeType },
    ];
}
```

---

## Key Differences

### Schema Systems

| Codebase | Schema System | Schema Location |
|----------|---------------|-----------------|
| ash | Raw JSON Schema dicts | In `input_schema` property |
| archer | TypeBox | In `parameters` property |
| clawdbot | TypeBox (normalized) | Cleaned for Gemini/OpenAI compat |
| pi-mono | TypeBox | In `parameters` property |

### Sandbox Approaches

| Codebase | Sandbox | Configuration |
|----------|---------|---------------|
| ash | Docker (mandatory) | gVisor optional, network/memory limits |
| archer | None | Executor abstraction for potential sandboxing |
| clawdbot | Docker (optional) | Per-agent sandbox contexts |
| pi-mono | None | Operations abstraction for remote execution |

### Output Truncation Strategy

| Codebase | Max Lines | Max Bytes | Strategy |
|----------|-----------|-----------|----------|
| ash | 4000 | 50KB | Head for reads, tail for bash |
| archer | 2000 | 50KB | Head for reads, tail for bash |
| clawdbot | 2000 | 50KB | Inherits from pi-coding-agent |
| pi-mono | 2000 | 50KB | Head for reads, tail for bash |

### Cancellation Support

| Codebase | AbortSignal | Process Cleanup |
|----------|-------------|-----------------|
| ash | No | Via sandbox timeout |
| archer | Yes | killProcessTree on abort |
| clawdbot | Yes | Combined signals, process tree kill |
| pi-mono | Yes | killProcessTree on abort |

### Progress/Streaming Updates

| Codebase | Mechanism | Granularity |
|----------|-----------|-------------|
| ash | ExecutionCallback | Post-execution only |
| archer | onUpdate callback | Chunk-level streaming |
| clawdbot | onUpdate callback | Chunk-level streaming |
| pi-mono | onUpdate callback | Chunk-level streaming |

---

## Recommendations for ash

Based on this comparison, here are potential improvements ash could adopt:

### 1. Add AbortSignal Support

All TypeScript codebases support cancellation via AbortSignal. ash could add this to enable graceful command cancellation:

```python
class Tool(ABC):
    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
        abort_event: asyncio.Event | None = None,  # Add cancellation
    ) -> ToolResult: ...
```

### 2. Streaming Progress Updates

pi-mono's `onUpdate` callback enables real-time UI feedback during long operations. ash's `ExecutionCallback` only fires after completion. Consider adding streaming:

```python
class ToolExecutor:
    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
        on_progress: Callable[[str], None] | None = None,  # Streaming updates
    ) -> ToolResult: ...
```

### 3. Pluggable Operations Pattern

pi-mono's `BashOperations` and `ReadOperations` interfaces allow swapping implementations (local vs SSH vs container). ash could benefit from this for testing and remote execution:

```python
class BashOperations(Protocol):
    async def exec(
        self,
        command: str,
        cwd: str,
        timeout: int | None = None,
    ) -> tuple[str, int]: ...
```

### 4. Process Management Tool

clawdbot's `process` tool enables background task management (list, poll, kill). For long-running operations, this pattern is valuable. ash currently relies on shell job control within the sandbox.

### 5. Consider Adjusting Truncation Limits

ash uses 4000 lines while the others use 2000. The TypeScript codebases may have found 2000 to be a better balance for context window efficiency.

### 6. Tool Policy/Filtering

clawdbot's deny/allow list pattern for controlling which tools are available to subagents could be useful for ash's agent delegation features.

---

## Summary

Each codebase has evolved its tool system based on specific requirements:

- **ash**: Security-first with mandatory Docker sandboxing, class-based tools, centralized logging
- **archer**: Lightweight factory functions, shared executor abstraction, streaming support
- **clawdbot**: Multi-provider compatibility, sophisticated tool filtering, background process management
- **pi-mono**: Pluggable operations for remote execution, comprehensive streaming, foundation for others

The TypeScript codebases share significant code via pi-agent-core/pi-coding-agent, while ash stands alone with its Python implementation. The key architectural differences are:

1. **Sandboxing**: ash mandates Docker; others make it optional or delegate to executor abstractions
2. **Streaming**: TypeScript codebases stream output chunks; ash reports after completion
3. **Cancellation**: TypeScript uses AbortSignal; ash relies on timeouts
4. **Schema**: ash uses raw JSON Schema; TypeScript uses TypeBox for type safety
