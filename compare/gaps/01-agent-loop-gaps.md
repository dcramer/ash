# Agent Loop Gap Analysis

This document analyzes gaps between Ash's agent loop implementation and the reference implementations in pi-mono and Clawdbot.

**Files Analyzed:**
- Ash: `/home/dcramer/src/ash/src/ash/core/agent.py`
- Pi-mono: `/home/dcramer/src/pi-mono/packages/agent/src/agent-loop.ts`
- Clawdbot: `/home/dcramer/src/clawdbot/src/agents/pi-embedded-runner.ts`

---

## Gap 1: Parallel Tool Execution

### What Ash is Missing

Ash executes tools sequentially in a loop (lines 643-669 in `agent.py`):

```python
# agent.py lines 643-669
for tool_use in pending_tools:
    # Notify callback before execution
    if on_tool_start:
        await on_tool_start(tool_use.name, tool_use.input)

    result = await self._tools.execute(
        tool_use.name,
        tool_use.input,
        tool_context,
    )
    # ... add result to session
```

### Reference Implementation (Pi-mono)

Pi-mono executes tool calls with the ability to parallelize and interrupt mid-execution (lines 294-378 in `agent-loop.ts`):

```typescript
// agent-loop.ts lines 294-378
async function executeToolCalls(
    tools: AgentTool<any>[] | undefined,
    assistantMessage: AssistantMessage,
    signal: AbortSignal | undefined,
    stream: EventStream<AgentEvent, AgentMessage[]>,
    getSteeringMessages?: AgentLoopConfig["getSteeringMessages"],
): Promise<{ toolResults: ToolResultMessage[]; steeringMessages?: AgentMessage[] }> {
    const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
    const results: ToolResultMessage[] = [];

    for (let index = 0; index < toolCalls.length; index++) {
        const toolCall = toolCalls[index];
        // ... execute tool

        // Check for steering messages - skip remaining tools if user interrupted
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
    return { toolResults: results, steeringMessages };
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/core/agent.py`

### Concrete Python Code Changes

```python
# Add to imports
import asyncio
from typing import NamedTuple

# Add new type
class ToolExecutionResult(NamedTuple):
    tool_use: ToolUse
    result: Any
    error: bool


# Add new config option to AgentConfig (line 65)
@dataclass
class AgentConfig:
    # ... existing fields ...
    parallel_tool_execution: bool = True  # Execute multiple tools concurrently


# Replace the tool execution loop in process_message (lines 643-669)
async def _execute_tools_parallel(
    self,
    pending_tools: list[ToolUse],
    tool_context: ToolContext,
    on_tool_start: OnToolStartCallback | None = None,
) -> list[ToolExecutionResult]:
    """Execute tools in parallel."""

    async def execute_one(tool_use: ToolUse) -> ToolExecutionResult:
        if on_tool_start:
            await on_tool_start(tool_use.name, tool_use.input)
        result = await self._tools.execute(
            tool_use.name,
            tool_use.input,
            tool_context,
        )
        return ToolExecutionResult(
            tool_use=tool_use,
            result=result,
            error=result.is_error,
        )

    # Execute all tools concurrently
    results = await asyncio.gather(
        *[execute_one(tool) for tool in pending_tools],
        return_exceptions=True,
    )

    # Handle any exceptions
    execution_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            execution_results.append(ToolExecutionResult(
                tool_use=pending_tools[i],
                result=ToolResult(content=str(result), is_error=True),
                error=True,
            ))
        else:
            execution_results.append(result)

    return execution_results


# In process_message, replace sequential loop with:
if self._config.parallel_tool_execution and len(pending_tools) > 1:
    execution_results = await self._execute_tools_parallel(
        pending_tools, tool_context, on_tool_start
    )
    for exec_result in execution_results:
        tool_calls.append({
            "id": exec_result.tool_use.id,
            "name": exec_result.tool_use.name,
            "input": exec_result.tool_use.input,
            "result": exec_result.result.content,
            "is_error": exec_result.error,
        })
        session.add_tool_result(
            tool_use_id=exec_result.tool_use.id,
            content=exec_result.result.content,
            is_error=exec_result.error,
        )
else:
    # Keep sequential execution for single tools or when disabled
    for tool_use in pending_tools:
        # ... existing code ...
```

### Effort: Medium
### Priority: Medium

Parallel execution speeds up multi-tool turns significantly but requires careful handling of tool dependencies and shared state.

---

## Gap 2: Rich Event System

### What Ash is Missing

Ash has only a single callback type (line 40):

```python
# agent.py line 40
OnToolStartCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
```

No events for: agent_start, turn_start, message_start, message_update, message_end, turn_end, agent_end, tool_execution_update, tool_execution_end.

### Reference Implementation (Pi-mono)

Pi-mono has 14 event types (lines 179-194 in `types.ts`):

```typescript
// types.ts lines 179-194
export type AgentEvent =
    // Agent lifecycle
    | { type: "agent_start" }
    | { type: "agent_end"; messages: AgentMessage[] }
    // Turn lifecycle
    | { type: "turn_start" }
    | { type: "turn_end"; message: AgentMessage; toolResults: ToolResultMessage[] }
    // Message lifecycle
    | { type: "message_start"; message: AgentMessage }
    | { type: "message_update"; message: AgentMessage; assistantMessageEvent: AssistantMessageEvent }
    | { type: "message_end"; message: AgentMessage }
    // Tool execution lifecycle
    | { type: "tool_execution_start"; toolCallId: string; toolName: string; args: any }
    | { type: "tool_execution_update"; toolCallId: string; toolName: string; args: any; partialResult: any }
    | { type: "tool_execution_end"; toolCallId: string; toolName: string; result: any; isError: boolean };
```

Events are emitted throughout the loop:

```typescript
// agent-loop.ts lines 44-49
stream.push({ type: "agent_start" });
stream.push({ type: "turn_start" });
for (const prompt of prompts) {
    stream.push({ type: "message_start", message: prompt });
    stream.push({ type: "message_end", message: prompt });
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/core/agent.py`
- Create new file: `/home/dcramer/src/ash/src/ash/core/events.py`

### Concrete Python Code Changes

```python
# New file: /home/dcramer/src/ash/src/ash/core/events.py
"""Agent event types for rich lifecycle notifications."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, Union

if TYPE_CHECKING:
    from ash.llm.types import ContentBlock, Message


class AgentEventType(str, Enum):
    """All possible agent event types."""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    MESSAGE_START = "message_start"
    MESSAGE_UPDATE = "message_update"
    MESSAGE_END = "message_end"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_UPDATE = "tool_execution_update"
    TOOL_EXECUTION_END = "tool_execution_end"


@dataclass
class AgentStartEvent:
    type: str = "agent_start"


@dataclass
class AgentEndEvent:
    messages: list[Message]
    type: str = "agent_end"


@dataclass
class TurnStartEvent:
    iteration: int
    type: str = "turn_start"


@dataclass
class TurnEndEvent:
    message: Message
    tool_results: list[dict[str, Any]]
    type: str = "turn_end"


@dataclass
class MessageStartEvent:
    message: Message
    type: str = "message_start"


@dataclass
class MessageUpdateEvent:
    message: Message
    delta: str | None = None
    type: str = "message_update"


@dataclass
class MessageEndEvent:
    message: Message
    type: str = "message_end"


@dataclass
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    type: str = "tool_execution_start"


@dataclass
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    partial_result: Any
    type: str = "tool_execution_update"


@dataclass
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    duration_ms: int
    type: str = "tool_execution_end"


AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
]


class AgentEventHandler(Protocol):
    """Protocol for handling agent events."""

    async def __call__(self, event: AgentEvent) -> None:
        """Handle an agent event."""
        ...
```

```python
# In agent.py, add imports and modify Agent class:

from ash.core.events import (
    AgentEvent,
    AgentEventHandler,
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
)

# Replace OnToolStartCallback with more comprehensive handler
# Keep OnToolStartCallback for backward compatibility, but add:
OnAgentEventCallback = Callable[[AgentEvent], Awaitable[None]]


# Update process_message signature:
async def process_message(
    self,
    user_message: str,
    session: SessionState,
    user_id: str | None = None,
    on_tool_start: OnToolStartCallback | None = None,  # Deprecated, kept for compat
    on_event: OnAgentEventCallback | None = None,  # New rich event handler
    session_path: str | None = None,
) -> AgentResponse:

    # Helper to emit events
    async def emit(event: AgentEvent) -> None:
        if on_event:
            await on_event(event)

    # At start of processing:
    await emit(AgentStartEvent())

    # Before each LLM call:
    await emit(TurnStartEvent(iteration=iterations))

    # After getting response:
    await emit(MessageStartEvent(message=response.message))
    await emit(MessageEndEvent(message=response.message))

    # For tool execution:
    for tool_use in pending_tools:
        await emit(ToolExecutionStartEvent(
            tool_call_id=tool_use.id,
            tool_name=tool_use.name,
            args=tool_use.input,
        ))

        # Keep backward compat with on_tool_start
        if on_tool_start:
            await on_tool_start(tool_use.name, tool_use.input)

        start_time = time.monotonic()
        result = await self._tools.execute(...)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        await emit(ToolExecutionEndEvent(
            tool_call_id=tool_use.id,
            tool_name=tool_use.name,
            result=result.content,
            is_error=result.is_error,
            duration_ms=duration_ms,
        ))

    # After tool execution in turn:
    await emit(TurnEndEvent(message=response.message, tool_results=tool_calls))

    # At end:
    await emit(AgentEndEvent(messages=session.messages))
```

### Effort: Medium
### Priority: High

Rich events enable proper UI integration, logging, and debugging. Essential for building responsive interfaces.

---

## Gap 3: Steering Messages

### What Ash is Missing

Ash has no mechanism to inject messages mid-execution. The agent runs to completion without checking for user interruptions. There is no `getSteeringMessages` equivalent.

### Reference Implementation (Pi-mono)

Pi-mono checks for steering messages after each tool execution (lines 114, 162, 364-375 in `agent-loop.ts`):

```typescript
// types.ts lines 78-86
/**
 * Returns steering messages to inject into the conversation mid-run.
 *
 * Called after each tool execution to check for user interruptions.
 * If messages are returned, remaining tool calls are skipped and
 * these messages are added to the context before the next LLM call.
 */
getSteeringMessages?: () => Promise<AgentMessage[]>;

// agent-loop.ts lines 364-375
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
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/core/agent.py`

### Concrete Python Code Changes

```python
# Add to imports
from collections.abc import Callable

# Add type alias after OnToolStartCallback (line 40)
GetSteeringMessagesCallback = Callable[[], Awaitable[list[str]]]


# Add to AgentConfig
@dataclass
class AgentConfig:
    # ... existing fields ...
    check_steering_between_tools: bool = True  # Check for steering after each tool


# Update process_message signature
async def process_message(
    self,
    user_message: str,
    session: SessionState,
    user_id: str | None = None,
    on_tool_start: OnToolStartCallback | None = None,
    on_event: OnAgentEventCallback | None = None,
    get_steering_messages: GetSteeringMessagesCallback | None = None,
    session_path: str | None = None,
) -> AgentResponse:


# Replace tool execution loop (lines 643-669) with:
pending_steering: list[str] = []
skip_remaining_tools = False

for i, tool_use in enumerate(pending_tools):
    if skip_remaining_tools:
        # Mark remaining tools as skipped
        tool_calls.append({
            "id": tool_use.id,
            "name": tool_use.name,
            "input": tool_use.input,
            "result": "Skipped due to user message.",
            "is_error": True,
        })
        session.add_tool_result(
            tool_use_id=tool_use.id,
            content="Skipped due to user message.",
            is_error=True,
        )
        continue

    # Notify callback before execution
    if on_tool_start:
        await on_tool_start(tool_use.name, tool_use.input)

    result = await self._tools.execute(
        tool_use.name,
        tool_use.input,
        tool_context,
    )

    tool_calls.append({
        "id": tool_use.id,
        "name": tool_use.name,
        "input": tool_use.input,
        "result": result.content,
        "is_error": result.is_error,
    })

    session.add_tool_result(
        tool_use_id=tool_use.id,
        content=result.content,
        is_error=result.is_error,
    )

    # Check for steering messages after each tool (if enabled and callback provided)
    if (
        self._config.check_steering_between_tools
        and get_steering_messages
        and i < len(pending_tools) - 1  # Don't check after last tool
    ):
        steering = await get_steering_messages()
        if steering:
            pending_steering = steering
            skip_remaining_tools = True

# If we have pending steering messages, add them to context before next LLM call
if pending_steering:
    for msg in pending_steering:
        session.add_user_message(msg)
```

### Effort: Small
### Priority: High

Steering enables responsive agents that can be interrupted and redirected, critical for interactive use cases.

---

## Gap 4: Follow-up Messages

### What Ash is Missing

Ash has no mechanism to queue messages for processing after the current turn completes. When the agent finishes, it returns immediately.

### Reference Implementation (Pi-mono)

Pi-mono checks for follow-up messages when the agent would otherwise stop (lines 97, 185-190 in `agent-loop.ts` and `types.ts`):

```typescript
// types.ts lines 89-97
/**
 * Returns follow-up messages to process after the agent would otherwise stop.
 *
 * Called when the agent has no more tool calls and no steering messages.
 * If messages are returned, they're added to the context and the agent
 * continues with another turn.
 */
getFollowUpMessages?: () => Promise<AgentMessage[]>;

// agent-loop.ts lines 184-193
// Agent would stop here. Check for follow-up messages.
const followUpMessages = (await config.getFollowUpMessages?.()) || [];
if (followUpMessages.length > 0) {
    // Set as pending so inner loop processes them
    pendingMessages = followUpMessages;
    continue;
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/core/agent.py`

### Concrete Python Code Changes

```python
# Add type alias
GetFollowUpMessagesCallback = Callable[[], Awaitable[list[str]]]


# Update process_message signature
async def process_message(
    self,
    user_message: str,
    session: SessionState,
    user_id: str | None = None,
    on_tool_start: OnToolStartCallback | None = None,
    on_event: OnAgentEventCallback | None = None,
    get_steering_messages: GetSteeringMessagesCallback | None = None,
    get_follow_up_messages: GetFollowUpMessagesCallback | None = None,
    session_path: str | None = None,
) -> AgentResponse:


# Modify the main loop structure (starting around line 592)
# Change from:
#   while iterations < self._config.max_tool_iterations:
# To nested loops that check for follow-ups:

while True:  # Outer loop for follow-up messages
    while iterations < self._config.max_tool_iterations:
        iterations += 1

        # ... existing LLM call and tool execution ...

        # Check for tool uses
        pending_tools = session.get_pending_tool_uses()
        if not pending_tools:
            # No tool calls - check for follow-up messages before returning
            break  # Exit inner loop, will check follow-ups below

        # ... tool execution with steering checks ...

    # After inner loop exits (no more tools), check for follow-ups
    if get_follow_up_messages and iterations < self._config.max_tool_iterations:
        follow_ups = await get_follow_up_messages()
        if follow_ups:
            # Add follow-up messages and continue processing
            for msg in follow_ups:
                session.add_user_message(msg)
            continue  # Back to outer loop

    # No follow-ups, exit
    break

# Return final response
final_text = response.message.get_text() or ""
# ... rest of return logic ...
```

### Effort: Small
### Priority: Medium

Follow-up messages allow queuing additional user input while the agent is working, improving UX for rapid-fire requests.

---

## Gap 5: Context Transform Hook

### What Ash is Missing

Ash has no hook to transform messages before each LLM call. The messages go directly from the session to the LLM.

### Reference Implementation (Pi-mono)

Pi-mono applies an optional `transformContext` before converting messages for the LLM (lines 50-67 in `types.ts`, lines 211-215 in `agent-loop.ts`):

```typescript
// types.ts lines 50-67
/**
 * Optional transform applied to the context before `convertToLlm`.
 *
 * Use this for operations that work at the AgentMessage level:
 * - Context window management (pruning old messages)
 * - Injecting context from external sources
 */
transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;

// agent-loop.ts lines 211-215
// Apply context transform if configured (AgentMessage[] → AgentMessage[])
let messages = context.messages;
if (config.transformContext) {
    messages = await config.transformContext(messages, signal);
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/core/agent.py`

### Concrete Python Code Changes

```python
# Add import
from ash.llm.types import Message as LLMMessage

# Add type alias
TransformContextCallback = Callable[
    [list[LLMMessage]],
    Awaitable[list[LLMMessage]]
]


# Add to Agent.__init__ or as a parameter
class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        tool_executor: ToolExecutor,
        prompt_builder: SystemPromptBuilder,
        runtime: RuntimeInfo | None = None,
        memory_manager: MemoryManager | None = None,
        memory_extractor: MemoryExtractor | None = None,
        config: AgentConfig | None = None,
        transform_context: TransformContextCallback | None = None,  # NEW
    ):
        # ... existing init ...
        self._transform_context = transform_context


# In process_message, before the LLM call (around line 596):
# Get messages for LLM
messages = session.get_messages_for_llm(
    token_budget=setup.message_budget,
    recency_window=self._config.recency_window,
)

# Apply context transform if configured
if self._transform_context:
    messages = await self._transform_context(messages)

# Call LLM with transformed messages
response = await self._llm.complete(
    messages=messages,
    model=self._config.model,
    tools=self._get_tool_definitions(),
    system=setup.system_prompt,
    max_tokens=self._config.max_tokens,
    temperature=self._config.temperature,
    thinking=self._config.thinking,
)
```

### Alternative: Process-level Hook

For more flexibility, accept the hook in `process_message`:

```python
async def process_message(
    self,
    user_message: str,
    session: SessionState,
    user_id: str | None = None,
    on_tool_start: OnToolStartCallback | None = None,
    on_event: OnAgentEventCallback | None = None,
    get_steering_messages: GetSteeringMessagesCallback | None = None,
    get_follow_up_messages: GetFollowUpMessagesCallback | None = None,
    transform_context: TransformContextCallback | None = None,
    session_path: str | None = None,
) -> AgentResponse:
```

### Effort: Small
### Priority: Medium

Context transforms enable advanced use cases like dynamic context injection, custom pruning strategies, and RAG integration.

---

## Gap 6: Context Window Guard

### What Ash is Missing

Ash has no validation that the configured model has sufficient context window. It relies on the LLM provider to return an error.

### Reference Implementation (Clawdbot)

Clawdbot validates context window before starting the agent run (lines 1-84 in `context-window-guard.ts`, lines 1122-1140 in `pi-embedded-runner.ts`):

```typescript
// context-window-guard.ts
export const CONTEXT_WINDOW_HARD_MIN_TOKENS = 16_000;
export const CONTEXT_WINDOW_WARN_BELOW_TOKENS = 32_000;

export function evaluateContextWindowGuard(params: {
    info: ContextWindowInfo;
    warnBelowTokens?: number;
    hardMinTokens?: number;
}): ContextWindowGuardResult {
    // ...
    return {
        ...params.info,
        tokens,
        shouldWarn: tokens > 0 && tokens < warnBelow,
        shouldBlock: tokens > 0 && tokens < hardMin,
    };
}

// pi-embedded-runner.ts lines 1122-1140
const ctxGuard = evaluateContextWindowGuard({
    info: ctxInfo,
    warnBelowTokens: CONTEXT_WINDOW_WARN_BELOW_TOKENS,
    hardMinTokens: CONTEXT_WINDOW_HARD_MIN_TOKENS,
});
if (ctxGuard.shouldWarn) {
    log.warn(`low context window: ${provider}/${modelId} ctx=${ctxGuard.tokens}`);
}
if (ctxGuard.shouldBlock) {
    log.error(`blocked model (context window too small): ${provider}/${modelId}`);
    throw new FailoverError(
        `Model context window too small (${ctxGuard.tokens} tokens). Minimum is ${CONTEXT_WINDOW_HARD_MIN_TOKENS}.`,
        { reason: "unknown", provider, model: modelId },
    );
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/core/agent.py`
- Create new file: `/home/dcramer/src/ash/src/ash/core/context_guard.py`

### Concrete Python Code Changes

```python
# New file: /home/dcramer/src/ash/src/ash/core/context_guard.py
"""Context window validation for models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Constants
CONTEXT_WINDOW_HARD_MIN_TOKENS = 16_000
CONTEXT_WINDOW_WARN_BELOW_TOKENS = 32_000

ContextWindowSource = Literal["model", "config", "default"]


@dataclass
class ContextWindowInfo:
    """Information about a model's context window."""
    tokens: int
    source: ContextWindowSource


@dataclass
class ContextWindowGuardResult:
    """Result of context window validation."""
    tokens: int
    source: ContextWindowSource
    should_warn: bool
    should_block: bool


def resolve_context_window(
    model_context_window: int | None,
    config_context_budget: int | None,
    default_tokens: int = 100_000,
) -> ContextWindowInfo:
    """Resolve the effective context window from multiple sources."""
    if model_context_window and model_context_window > 0:
        return ContextWindowInfo(tokens=model_context_window, source="model")

    if config_context_budget and config_context_budget > 0:
        return ContextWindowInfo(tokens=config_context_budget, source="config")

    return ContextWindowInfo(tokens=default_tokens, source="default")


def evaluate_context_window_guard(
    info: ContextWindowInfo,
    warn_below_tokens: int = CONTEXT_WINDOW_WARN_BELOW_TOKENS,
    hard_min_tokens: int = CONTEXT_WINDOW_HARD_MIN_TOKENS,
) -> ContextWindowGuardResult:
    """Evaluate whether the context window is acceptable."""
    tokens = max(0, info.tokens)

    return ContextWindowGuardResult(
        tokens=tokens,
        source=info.source,
        should_warn=tokens > 0 and tokens < warn_below_tokens,
        should_block=tokens > 0 and tokens < hard_min_tokens,
    )


class ContextWindowTooSmallError(Exception):
    """Raised when model context window is below minimum."""

    def __init__(self, tokens: int, minimum: int, model: str | None = None):
        self.tokens = tokens
        self.minimum = minimum
        self.model = model
        msg = f"Model context window too small ({tokens} tokens). Minimum is {minimum}."
        if model:
            msg = f"{model}: {msg}"
        super().__init__(msg)
```

```python
# In agent.py, add to create_agent function (around line 908):

from ash.core.context_guard import (
    resolve_context_window,
    evaluate_context_window_guard,
    ContextWindowTooSmallError,
    CONTEXT_WINDOW_HARD_MIN_TOKENS,
    CONTEXT_WINDOW_WARN_BELOW_TOKENS,
)

async def create_agent(
    config: AshConfig,
    workspace: Workspace,
    db_session: AsyncSession | None = None,
    model_alias: str = "default",
) -> AgentComponents:
    # ... existing code to resolve model_config ...

    # Validate context window before proceeding
    ctx_info = resolve_context_window(
        model_context_window=getattr(model_config, 'context_window', None),
        config_context_budget=config.memory.context_token_budget,
    )

    ctx_guard = evaluate_context_window_guard(ctx_info)

    if ctx_guard.should_warn:
        logger.warning(
            f"Low context window for {model_config.model}: "
            f"{ctx_guard.tokens} tokens (warn threshold: {CONTEXT_WINDOW_WARN_BELOW_TOKENS})"
        )

    if ctx_guard.should_block:
        raise ContextWindowTooSmallError(
            tokens=ctx_guard.tokens,
            minimum=CONTEXT_WINDOW_HARD_MIN_TOKENS,
            model=model_config.model,
        )

    # ... rest of create_agent ...
```

### Effort: Small
### Priority: Low

Early validation prevents cryptic errors, but most LLM providers give clear error messages anyway.

---

## Gap 7: Configurable Iteration Limit

### What Ash is Missing

The iteration limit is defined as a module-level constant (line 42):

```python
# agent.py line 42
MAX_TOOL_ITERATIONS = 25
```

While `AgentConfig` does have `max_tool_iterations` (line 79), the module constant creates confusion and the default should be more clearly configurable.

### Reference Implementation

Both pi-mono and Clawdbot make iteration limits configurable through their configuration systems.

### Files to Modify

- `/home/dcramer/src/ash/src/ash/core/agent.py`
- `/home/dcramer/src/ash/src/ash/config.py` (if exists)

### Concrete Python Code Changes

```python
# In agent.py, remove or deprecate the module constant:

# OLD (line 42):
MAX_TOOL_ITERATIONS = 25

# NEW: Move default to AgentConfig and make it clear that's the source of truth

# Keep for backward compatibility but mark as deprecated:
MAX_TOOL_ITERATIONS = 25  # Deprecated: use AgentConfig.max_tool_iterations


# AgentConfig is already correct (line 79):
@dataclass
class AgentConfig:
    # ... other fields ...
    max_tool_iterations: int = 25  # This is already configurable!


# The issue is in create_agent - it doesn't expose max_tool_iterations in config
# Update create_agent to read from AshConfig (around line 1103):

# First, add to AshConfig (in config.py) if not present:
@dataclass
class AgentSettings:
    """Agent behavior settings."""
    max_tool_iterations: int = 25
    # ... other settings ...


# Then in create_agent, use it:
agent = Agent(
    llm=llm,
    tool_executor=tool_executor,
    prompt_builder=prompt_builder,
    runtime=runtime,
    memory_manager=memory_manager,
    memory_extractor=memory_extractor,
    config=AgentConfig(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        temperature=model_config.temperature,
        thinking=thinking_config,
        max_tool_iterations=config.agent.max_tool_iterations,  # From config
        context_token_budget=config.memory.context_token_budget,
        # ... rest ...
    ),
)
```

```toml
# Example config.toml addition:
[agent]
max_tool_iterations = 50  # Override default of 25
```

### Effort: Small
### Priority: Low

The functionality already exists in `AgentConfig`; this is just about making it more discoverable and configurable from the top-level config.

---

## Summary Table

| Gap | Description | Effort | Priority |
|-----|-------------|--------|----------|
| 1 | Parallel tool execution | Medium | Medium |
| 2 | Rich event system | Medium | High |
| 3 | Steering messages | Small | High |
| 4 | Follow-up messages | Small | Medium |
| 5 | Context transform hook | Small | Medium |
| 6 | Context window guard | Small | Low |
| 7 | Configurable iteration limit | Small | Low |

## Recommended Implementation Order

1. **Gap 3: Steering messages** - Small effort, high impact for interactive use
2. **Gap 2: Rich event system** - Foundation for UI and debugging
3. **Gap 4: Follow-up messages** - Complements steering for full message queue support
4. **Gap 5: Context transform hook** - Enables advanced customization
5. **Gap 1: Parallel tool execution** - Performance improvement
6. **Gap 6: Context window guard** - Nice-to-have validation
7. **Gap 7: Configurable iteration limit** - Minor cleanup
