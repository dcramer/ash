# Core Agent Loop Comparison

## Overview

The core agent loop is the central orchestrator in agentic systems - it manages the conversation flow between user prompts, LLM calls, and tool execution. Each iteration follows a pattern: receive input, call the LLM, check for tool calls, execute tools, and repeat until the LLM returns a final text response or an error/abort condition is met. The implementations compared here represent different approaches to handling this loop, from simple synchronous execution to sophisticated generator-based streaming with parallel tool execution.

## Comparison Table

| Feature | ash | archer | clawdbot | pi-mono (agent-loop) |
|---------|-----|--------|----------|---------------------|
| **Core File** | `src/ash/core/agent.py` | `src/agent.ts` | `src/agents/pi-embedded-runner.ts` | `packages/agent/src/agent-loop.ts` |
| **Language** | Python | TypeScript | TypeScript | TypeScript |
| **Iteration Limit** | 25 (configurable) | Inherited from pi-agent | No explicit limit | No explicit limit |
| **Tool Execution** | Sequential | Sequential (via pi-agent) | Sequential (via pi-agent) | Parallel with steering check |
| **Event System** | Callback-based (`on_tool_start`) | Event subscription | Event subscription + callbacks | Generator/EventStream (14+ types) |
| **Context Compaction** | LLM-generated summaries | Auto-compaction via pi-agent | Auto-compaction via pi-agent | Via `transformContext` hook |
| **Steering/Interruption** | None | Queue messages while running | Abort via `runAbortController` | `getSteeringMessages()` / `getFollowUpMessages()` |
| **Streaming** | Dual methods (sync/stream) | Inherits streaming | Partial reply callbacks | Full event streaming |
| **Model Failover** | None | None | Auth profile rotation | None (handled externally) |

## Detailed Analysis

### 1. ash (`src/ash/core/agent.py`)

**Architecture:** Python async class with a simple while-loop structure.

**Agent Loop (lines 592-688):**
```python
while iterations < self._config.max_tool_iterations:
    iterations += 1

    # Call LLM with pruned messages
    response = await self._llm.complete(
        messages=session.get_messages_for_llm(
            token_budget=setup.message_budget,
            recency_window=self._config.recency_window,
        ),
        model=self._config.model,
        tools=self._get_tool_definitions(),
        system=setup.system_prompt,
        max_tokens=self._config.max_tokens,
        temperature=self._config.temperature,
        thinking=self._config.thinking,
    )

    session.add_assistant_message(response.message.content)

    pending_tools = session.get_pending_tool_uses()
    if not pending_tools:
        final_text = response.message.get_text() or ""
        # ... return response
```

**Key Characteristics:**
- **MAX_TOOL_ITERATIONS = 25** (line 42): Hard limit prevents runaway loops
- **Sequential tool execution** (lines 643-669): Tools execute one at a time in order
- **Callback-based events**: Single `on_tool_start` callback before each tool
- **Compaction via LLM summarization** (lines 240-305): When context exceeds budget, older messages are summarized using an LLM call
- **Memory extraction**: Background task extracts facts from conversation (lines 419-520)
- **Two entry points**: `process_message()` for blocking, `process_message_streaming()` for streaming

**Tool Executor (`src/ash/tools/executor.py`):**
```python
async def execute(self, tool_name: str, input_data: dict, context: ToolContext):
    start_time = time.monotonic()
    try:
        result = await tool.execute(input_data, context)
    except Exception as e:
        result = ToolResult.error(f"Tool execution failed: {e}")
    duration_ms = int((time.monotonic() - start_time) * 1000)
    # ... logging and callbacks
```

---

### 2. archer (`src/agent.ts`)

**Architecture:** Thin wrapper around pi-coding-agent's `AgentSession`, with per-channel runner caching.

**Runner Creation (lines 408-447):**
```typescript
function createRunner(sandboxConfig: SandboxConfig, channelId: string, channelDir: string): AgentRunner {
    const agent = new Agent({
        initialState: {
            systemPrompt,
            model,
            thinkingLevel: "off",
            tools,
        },
        convertToLlm,
        getApiKey: async () => getAnthropicApiKey(authStorage),
    });

    const session = new AgentSession({
        agent,
        sessionManager: sessionManager as any,
        settingsManager: settingsManager as any,
        modelRegistry,
    });
```

**Event Subscription (lines 478-597):**
```typescript
session.subscribe(async (event) => {
    if (event.type === "tool_execution_start") {
        // Log and notify UI
    } else if (event.type === "tool_execution_end") {
        // Accumulate results
    } else if (event.type === "message_end") {
        // Handle assistant response
    } else if (event.type === "auto_compaction_start") {
        // Notify user
    }
});
```

**Key Characteristics:**
- **Per-channel runner caching** (lines 389-402): `getOrCreateRunner()` maintains one runner per Telegram channel
- **Delegates to pi-agent-core**: Does not implement its own loop
- **Event subscription model**: Subscribes once on creation, handles events during runs
- **System prompt rebuild**: Refreshes memory, skills, and context on each run (lines 634-645)
- **Session persistence**: Uses `MomSessionManager` for JSONL-based session storage

---

### 3. clawdbot (`src/agents/pi-embedded-runner.ts`)

**Architecture:** Production-grade wrapper around pi-agent with extensive error handling, model failover, and auth profile rotation.

**Main Run Function (lines 1026-1811):**
```typescript
export async function runEmbeddedPiAgent(params: {...}): Promise<EmbeddedPiRunResult> {
    while (true) {  // Retry loop for auth/model failover
        const thinkingLevel = mapThinkingLevel(thinkLevel);

        // ... setup session, tools, sandbox

        const { session } = await createAgentSession({
            model,
            thinkingLevel,
            systemPrompt,
            tools: builtInTools,
            customTools,
            sessionManager,
            // ...
        });

        await session.prompt(params.prompt, { images: params.images });

        // Handle failover conditions
        if (shouldRotate) {
            const rotated = await advanceAuthProfile();
            if (rotated) continue;
        }
    }
}
```

**Context Window Guard (lines 1115-1140):**
```typescript
const ctxGuard = evaluateContextWindowGuard({
    info: ctxInfo,
    warnBelowTokens: CONTEXT_WINDOW_WARN_BELOW_TOKENS,
    hardMinTokens: CONTEXT_WINDOW_HARD_MIN_TOKENS,
});
if (ctxGuard.shouldBlock) {
    throw new FailoverError(
        `Model context window too small (${ctxGuard.tokens} tokens).`,
        { reason: "unknown", provider, model: modelId },
    );
}
```

**Key Characteristics:**
- **Auth profile rotation** (lines 1148-1194): Cycles through multiple API keys on rate limits
- **Model failover** (lines 1589-1659): Falls back to configured alternatives on errors
- **Context window protection** (lines 1115-1140): Warns/blocks small context windows
- **Thinking level fallback** (lines 1555-1586): Retries with lower thinking levels on unsupported errors
- **Session pre-warming** (lines 445-461): Warms OS page cache for faster session loads
- **Lane-based queuing**: Uses `enqueueCommandInLane()` for serialized session access

---

### 4. pi-mono agent-loop (`packages/agent/src/agent-loop.ts`)

**Architecture:** Pure generator-based loop with fine-grained event streaming and parallel tool execution.

**Core Loop Structure (lines 104-198):**
```typescript
async function runLoop(
    currentContext: AgentContext,
    newMessages: AgentMessage[],
    config: AgentLoopConfig,
    signal: AbortSignal | undefined,
    stream: EventStream<AgentEvent, AgentMessage[]>,
): Promise<void> {
    let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

    while (true) {
        let hasMoreToolCalls = true;

        while (hasMoreToolCalls || pendingMessages.length > 0) {
            // Process pending steering messages
            if (pendingMessages.length > 0) {
                for (const message of pendingMessages) {
                    stream.push({ type: "message_start", message });
                    stream.push({ type: "message_end", message });
                    currentContext.messages.push(message);
                }
                pendingMessages = [];
            }

            // Stream assistant response
            const message = await streamAssistantResponse(...);

            if (message.stopReason === "error" || message.stopReason === "aborted") {
                stream.push({ type: "agent_end", messages: newMessages });
                return;
            }

            // Execute tools
            const toolCalls = message.content.filter((c) => c.type === "toolCall");
            hasMoreToolCalls = toolCalls.length > 0;

            if (hasMoreToolCalls) {
                const toolExecution = await executeToolCalls(/* ... */);
                // Check for steering messages mid-execution
            }

            pendingMessages = (await config.getSteeringMessages?.()) || [];
        }

        // Check for follow-up messages
        const followUpMessages = (await config.getFollowUpMessages?.()) || [];
        if (followUpMessages.length > 0) {
            pendingMessages = followUpMessages;
            continue;
        }
        break;
    }
}
```

**transformContext Hook (line 213):**
```typescript
let messages = context.messages;
if (config.transformContext) {
    messages = await config.transformContext(messages, signal);
}
const llmMessages = await config.convertToLlm(messages);
```

**Tool Execution with Steering (lines 294-378):**
```typescript
async function executeToolCalls(/* ... */): Promise<{
    toolResults: ToolResultMessage[];
    steeringMessages?: AgentMessage[];
}> {
    for (let index = 0; index < toolCalls.length; index++) {
        // ... execute tool

        // Check for steering messages - skip remaining tools if interrupted
        if (getSteeringMessages) {
            const steering = await getSteeringMessages();
            if (steering.length > 0) {
                steeringMessages = steering;
                const remainingCalls = toolCalls.slice(index + 1);
                for (const skipped of remainingCalls) {
                    results.push(skipToolCall(skipped, stream));
                }
                break;
            }
        }
    }
}
```

**Event Types (from `types.ts` lines 179-194):**
- `agent_start`, `agent_end`
- `turn_start`, `turn_end`
- `message_start`, `message_update`, `message_end`
- `tool_execution_start`, `tool_execution_update`, `tool_execution_end`

**Key Characteristics:**
- **Generator-based EventStream**: Returns iterable stream of events
- **transformContext hook**: Allows context manipulation before LLM calls (pruning, injection)
- **Steering messages**: User can interrupt tool execution mid-stream
- **Follow-up messages**: Queued messages processed after agent would stop
- **Parallel-ready architecture**: Tools execute sequentially but steering checks after each
- **No iteration limit**: Runs until complete or aborted
- **convertToLlm separation**: Clean abstraction between AgentMessage and LLM Message types

---

## Key Differences

### Iteration Limits
- **ash**: Explicit 25-iteration limit prevents runaway loops
- **Others**: Rely on timeout/abort mechanisms

### Tool Execution Model
- **ash**: Simple sequential execution with callback
- **pi-mono**: Sequential with mid-execution steering interrupts
- **archer/clawdbot**: Inherit from pi-agent

### Context Management
- **ash**: LLM-generated summaries replace old messages
- **pi-mono**: `transformContext` hook for flexible pruning/injection
- **clawdbot**: Context window guard blocks small models

### Steering/Interruption
- **ash**: No mid-run interruption capability
- **pi-mono**: Rich steering API (`getSteeringMessages`, `getFollowUpMessages`)
- **clawdbot**: Abort controller only

### Error Handling
- **ash**: Returns error in response, no retry
- **clawdbot**: Auth rotation, model failover, thinking level fallback
- **pi-mono**: Propagates errors via events

### Event Granularity
- **ash**: Single callback (`on_tool_start`)
- **pi-mono**: 14+ event types for full UI reactivity
- **archer/clawdbot**: Event subscription to pi-agent events

---

## Recommendations for ash

### 1. Add Steering/Interruption Support
The pi-mono `getSteeringMessages()` pattern allows users to interrupt long-running tool sequences. ash could add:
```python
async def process_message(
    self,
    ...,
    get_steering: Callable[[], Awaitable[list[str]]] | None = None,
):
    for tool_use in pending_tools:
        result = await self._tools.execute(...)

        # Check for steering after each tool
        if get_steering:
            steering = await get_steering()
            if steering:
                # Skip remaining tools, add steering message
                break
```

### 2. Richer Event System
Replace single `on_tool_start` callback with an event stream:
```python
@dataclass
class AgentEvent:
    type: Literal["turn_start", "turn_end", "tool_start", "tool_end", "message_start", "message_end"]
    data: dict[str, Any]

async def process_message_events(self, ...) -> AsyncIterator[AgentEvent]:
    yield AgentEvent(type="turn_start", data={})
    # ...
```

### 3. Context Transform Hook
Add a pre-LLM hook for flexible context manipulation:
```python
@dataclass
class AgentConfig:
    # ...
    transform_context: Callable[[list[Message]], Awaitable[list[Message]]] | None = None
```

This would allow:
- External RAG injection
- Custom pruning strategies
- Dynamic context augmentation

### 4. Context Window Guard
Adopt clawdbot's pattern to prevent OOM on small models:
```python
CONTEXT_WINDOW_HARD_MIN = 16_000
CONTEXT_WINDOW_WARN_BELOW = 32_000

def validate_context_window(model_context: int) -> None:
    if model_context < CONTEXT_WINDOW_HARD_MIN:
        raise ValueError(f"Model context too small: {model_context}")
```

### 5. Remove Iteration Limit (or Make Very High)
With proper abort/timeout mechanisms, the 25-iteration limit may be unnecessarily restrictive for complex multi-step tasks. Consider:
- Increasing to 100+
- Using timeout instead
- Making configurable per-request

### 6. Parallel Tool Execution
While ash's sequential execution is simpler, parallel execution could speed up independent tool calls:
```python
# Instead of:
for tool_use in pending_tools:
    result = await self._tools.execute(...)

# Consider:
results = await asyncio.gather(*[
    self._tools.execute(t.name, t.input, tool_context)
    for t in pending_tools
])
```
