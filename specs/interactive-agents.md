# Interactive Agents

> Stack-based model for interactive subagent communication

Files: src/ash/agents/types.py, src/ash/agents/executor.py, src/ash/tools/builtin/skills.py, src/ash/tools/builtin/complete.py, src/ash/providers/telegram/handlers/message_handler.py

## Overview

When the main agent invokes a skill or agent via `use_skill` / `use_agent`, the subagent **takes over the conversation thread**. The user talks directly to the subagent until it calls `complete(result)`, then control returns to the caller. Subagents can nest arbitrarily: main → skill-writer → research.

This replaces the batch execution model (where subagents ran to completion internally) with an interactive model managed by a provider-level orchestration loop.

## Requirements

### MUST

- Subagent text responses (no tool calls) are sent directly to the user
- User messages are routed to the top-of-stack subagent
- `complete(result)` tool pops the subagent and returns result to parent as `tool_result`
- Stack supports arbitrary nesting depth (main → skill → agent → ...)
- Parent resumes when child completes — may call LLM again, produce text, call more tools, or spawn another child
- Max iterations per frame are enforced; exceeding cascades error to parent
- `ChildActivated(BaseException)` propagates through `except Exception` handlers
- Main agent's paused state (session, iteration count) is preserved as a StackFrame
- Provider routes messages based on stack state: empty → main agent, non-empty → top of stack
- Built-in skills (claude-code) bypass the stack and use their own execution path

### SHOULD

- Log stack depth changes (push/pop) for debugging
- Provide clear error messages when max iterations are hit
- Clean up stack state when errors occur

### MAY

- Persist stack to state.json for process restart recovery (v2)
- Stream subagent responses (v2 — currently sends complete text)
- Support user cancellation ("cancel" / "stop") that unwinds the stack

## Data Model

### StackFrame

Each frame holds everything needed to pause and resume an agent's execution.

```python
@dataclass
class StackFrame:
    frame_id: str                     # Unique ID
    agent_name: str                   # e.g. "skill:research", "main"
    agent_type: str                   # "skill" | "agent" | "main"
    session: SessionState             # In-memory LLM conversation state
    system_prompt: str                # Cached for resumption
    context: AgentContext             # Routing context
    model: str | None = None          # Resolved model name
    environment: dict[str, str] | None = None  # Sandbox env vars
    iteration: int = 0
    max_iterations: int = 25
    effective_tools: list[str]        # Tool whitelist
    is_skill_agent: bool = False
    voice: str | None = None
    parent_tool_use_id: str | None = None  # Links to parent's pending tool_use
    agent_session_id: str | None = None    # For context.jsonl logging
```

`parent_tool_use_id` is the critical link: when this frame completes, its result becomes the `tool_result` for that ID in the parent's session.

### AgentStack / AgentStackManager

```python
@dataclass
class AgentStack:
    frames: list[StackFrame]
    # Properties: is_empty, depth, top
    # Methods: push(frame), pop() -> StackFrame

class AgentStackManager:
    """Manages stacks keyed by provider session_key."""
    # e.g. "telegram_123_456" → AgentStack
    def get_or_create(session_key: str) -> AgentStack
    def has_active(session_key: str) -> bool
    def clear(session_key: str) -> None
```

### TurnAction / TurnResult

```python
class TurnAction(Enum):
    SEND_TEXT           # Subagent produced text for user, waiting for reply
    COMPLETE            # Subagent called complete(), pop stack
    CHILD_ACTIVATED     # A child was pushed onto the stack
    INTERRUPT           # Plan agent interrupt (existing checkpoint path)
    MAX_ITERATIONS      # Hit iteration limit
    ERROR               # Execution error

@dataclass
class TurnResult:
    action: TurnAction
    text: str = ""
    child_frame: StackFrame | None = None  # For CHILD_ACTIVATED
```

### ChildActivated

```python
class ChildActivated(BaseException):
    """Raised when a tool spawns an interactive child subagent.

    Uses BaseException (not Exception) so it propagates through
    ToolExecutor's generic except Exception handler.
    """
    child_frame: StackFrame
```

### complete tool

```python
class CompleteTool(Tool):
    name = "complete"
    description = "Signal that your task is complete and return a result."
    # input: {"result": "string"}
    # Intercepted by execute_turn — never actually runs execute()
```

Registered in the tool registry. Included for subagents, excluded from main agent.

## Execution Flow

### Phase 1: Main agent spawns a child

```
1. User sends message
2. Main agent calls process_message()
3. LLM returns tool_use for use_skill("research", "find APIs")
4. UseSkillTool.execute() builds a StackFrame and raises ChildActivated
5. ChildActivated propagates through ToolExecutor → Agent._execute_pending_tools()
6. Agent.process_message() catches ChildActivated, returns AgentResponse with:
   - child_activated=True
   - main_frame (main agent's paused SessionState as StackFrame)
   - child_frame (the new child's StackFrame)
7. Provider pushes main_frame then child_frame onto the stack
8. Provider enters orchestration loop, calls execute_turn(child_frame)
```

### Phase 2: User interacts with subagent

```
1. execute_turn calls LLM with child's session
2. LLM returns text (no tool calls) → TurnAction.SEND_TEXT
3. Provider sends text to user, returns (waits for next message)
4. Next user message arrives
5. Provider sees stack is non-empty → routes to _handle_stack_message
6. Orchestration loop calls execute_turn(top, user_message=msg.text)
7. LLM may call tools, produce text, or call complete
```

### Phase 3: Child completes, result cascades

```
1. Child's LLM calls complete(result="found 3 APIs")
2. execute_turn returns TurnResult(COMPLETE, text="found 3 APIs")
3. Orchestration loop pops child frame
4. Injects result as tool_result into parent: (parent_tool_use_id, "found 3 APIs", False)
5. Calls execute_turn(parent, tool_result=(...))
6. Parent's LLM receives the result, may produce text or call more tools
7. If parent produces text → SEND_TEXT, send to user
8. If parent is main agent and produces text → pop main frame, run memory extraction, clear stack
```

### Phase 4: Nested children

```
Stack: [main, skill-writer]
1. skill-writer calls use_agent("research", ...)
2. UseAgentTool raises ChildActivated
3. execute_turn returns TurnResult(CHILD_ACTIVATED, child_frame=research_frame)
4. Orchestration loop pushes research_frame → stack: [main, skill-writer, research]
5. Continues loop: execute_turn(research_frame)
6. research completes → pop → inject into skill-writer → resume
7. skill-writer may continue or complete → cascade to main
```

### Orchestration Loop (Provider)

The provider runs a while loop that processes TurnResults until it needs user input:

```python
while True:
    result = execute_turn(stack.top, user_message=..., tool_result=...)

    match result.action:
        case SEND_TEXT:
            send_to_user(result.text)
            if top.agent_type == "main":
                pop main, run memory extraction, clear stack
            return  # Wait for next user message

        case COMPLETE:
            completed = stack.pop()
            if stack.is_empty:
                send result.text, clear stack, return
            inject tool_result into parent
            continue  # Resume parent

        case CHILD_ACTIVATED:
            stack.push(result.child_frame)
            continue  # Run child's first turn

        case MAX_ITERATIONS:
            failed = stack.pop()
            if stack.is_empty:
                send error, return
            inject error tool_result into parent
            continue  # Cascade error

        case ERROR:
            # Same as MAX_ITERATIONS but with error text
```

Every iteration either `return`s (waiting for user input) or `continue`s (cascading). No recursion.

### execute_turn (AgentExecutor)

Runs one logical turn for a stack frame:

```python
async def execute_turn(frame, user_message=None, tool_result=None) -> TurnResult:
    session = frame.session

    if user_message: session.add_user_message(user_message)
    elif tool_result: session.add_tool_result(*tool_result)

    while frame.iteration < frame.max_iterations:
        frame.iteration += 1
        unresolved = _get_unresolved_tool_uses(session)

        if not unresolved:
            response = await llm.complete(session.messages, ...)
            session.add_assistant_message(response.content)
            if no tool_uses in response:
                return TurnResult(SEND_TEXT, text=response.text)
            unresolved = response.tool_uses

        for tool_use in unresolved:
            if tool_use.name == "complete":
                return TurnResult(COMPLETE, text=tool_use.input["result"])
            try:
                result = await tools.execute(tool_use.name, tool_use.input, ctx)
                session.add_tool_result(tool_use.id, result.content, result.is_error)
            except ChildActivated as ca:
                return TurnResult(CHILD_ACTIVATED, child_frame=ca.child_frame)

    return TurnResult(MAX_ITERATIONS)
```

### _get_unresolved_tool_uses

After a child completes and we inject a `tool_result`, the last message is a User(ToolResult) — not the assistant message with tool_uses. This helper walks backward to find the last assistant message and returns tool_uses that don't have matching results yet:

```python
def _get_unresolved_tool_uses(session) -> list[ToolUse]:
    # Walk backward to find last assistant message
    # Collect resolved tool_use_ids from subsequent messages
    # Return tool_uses without matching results
```

This handles the key case: parent had 3 tool_uses, #2 spawned a child. After child completes, #1 has a result, #2 just got one, #3 still needs execution.

## Child Spawning

### UseSkillTool

Does NOT run the first turn inline. Builds a `StackFrame` and raises `ChildActivated`. The orchestrator runs all turns including the first.

```python
async def execute(self, input_data, context):
    # Validate skill, check env vars, resolve model...
    child_session = SessionState(...)
    child_session.add_user_message(message)
    child_frame = StackFrame(
        agent_name=f"skill:{name}",
        agent_type="skill",
        session=child_session,
        parent_tool_use_id=context.tool_use_id,
        ...
    )
    raise ChildActivated(child_frame)
```

### Agent.process_message

Wraps `_execute_pending_tools()` to catch `ChildActivated`:

```python
try:
    tool_calls = await self._execute_pending_tools(...)
except ChildActivated as ca:
    main_frame = StackFrame(
        agent_name="main", agent_type="main",
        session=session, ...
    )
    return AgentResponse(
        child_activated=True,
        main_frame=main_frame,
        child_frame=ca.child_frame,
    )
```

## Stack Diagram

```
User message: "research Python async APIs"
─────────────────────────────────────────

Stack depth 0 (empty):
  → Main agent (Haiku) processes message
  → LLM calls use_skill("research", "Python async APIs")
  → UseSkillTool raises ChildActivated
  → AgentResponse(child_activated=True, main_frame, child_frame)

Stack depth 2: [main, skill:research]
  → execute_turn(skill:research) — first turn, session has initial message
  → LLM returns text: "I'll search for Python async APIs..."
  → SEND_TEXT → provider sends to user

User message: "focus on 3.13 specifically"
  → Provider sees stack active → routes to top
  → execute_turn(skill:research, user_message="focus on 3.13 specifically")
  → LLM calls web_search(...) → result → LLM continues
  → LLM calls complete("Found 3 APIs: ...")
  → COMPLETE

Stack depth 1: [main]
  → Inject "Found 3 APIs: ..." as tool_result for main's pending use_skill call
  → execute_turn(main, tool_result=(...))
  → Main agent's LLM sees the result, produces final text
  → SEND_TEXT → provider sends to user, pops main, clears stack

Stack depth 0 (empty):
  → Normal message routing resumes
```

## Session Logging

### context.jsonl

Subagent messages are tagged with `agent_session_id` linking them to their `AgentSessionEntry`. Uses existing `SessionManager.add_user_message()`, `add_assistant_message()`, `add_tool_use()`, `add_tool_result()` methods.

### AgentSessionCompleteEntry

Marks when a subagent finishes:

```python
@dataclass
class AgentSessionCompleteEntry:
    agent_session_id: str      # Links to AgentSessionEntry.id
    result: str                # Final result text
    is_error: bool = False
    created_at: datetime
    type: Literal["agent_session_complete"] = "agent_session_complete"
```

Written to context.jsonl when a frame calls `complete` or hits max iterations.

## Persistence (v2)

### StackFrameMeta (state.json)

```python
class StackFrameMeta(BaseModel):
    frame_id: str
    agent_session_id: str
    agent_name: str
    agent_type: str
    model: str | None = None
    iteration: int = 0
    max_iterations: int = 25
    parent_tool_use_id: str | None = None
    effective_tools: list[str] = []
    is_skill_agent: bool = False
    environment: dict[str, str] = {}
```

Added as `active_stack: list[StackFrameMeta] | None` on `SessionState` (Pydantic model in sessions/types.py).

### Reconstruction on restart (v2)

1. Load `state.json` → check `active_stack`
2. For each frame: filter `context.jsonl` by `agent_session_id`, rebuild session
3. Re-derive system prompt from agent/skill definition
4. Load into `AgentStackManager`

For v1: in-memory only. Active sessions are lost on process restart.

## What Stays Unchanged

- Plan agent's checkpoint/interrupt flow (separate code path, `supports_checkpointing=True`)
- `CheckpointState`, `_extract_checkpoint()`, checkpoint handler
- Existing `AgentExecutor.execute()` batch method (for non-interactive agents)
- Skill restriction: skills can't call `use_skill` (existing `is_skill_agent` check)
- Self-invocation prevention (existing check in executor)
- All session JSONL logging (subagent entries tagged with `agent_session_id`)
- Built-in skills (claude-code) use passthrough execution

## Behaviors

| Scenario | Behavior |
|----------|----------|
| Skill completes first turn | Stack push + immediate pop, result cascades to main |
| Skill needs interaction | Stack depth 2, user talks to skill |
| Nested: skill calls agent | Stack depth 3, user talks to inner agent |
| Inner agent completes | Cascades to skill, skill continues |
| Max iterations hit | Error cascades to parent via tool_result(is_error=True) |
| Main agent resumes | Makes more LLM calls, produces final text |
| Stack empty + message | Normal main agent flow |
| Stack active + message | Routes to top of stack |
| LLM error in turn | TurnAction.ERROR, cascades to parent |

## Errors

| Condition | Response |
|-----------|----------|
| Max iterations exceeded | Error tool_result cascades to parent |
| LLM API error | TurnAction.ERROR, cascades to parent |
| Tool execution error | Error tool_result added to session, LLM decides next step |
| Stack empty unexpectedly | Warning logged, return |
| Missing parent_tool_use_id on complete | Assertion error (programming bug) |

## Verification

```bash
uv run pytest tests/test_skill_execution.py -v
uv run pytest tests/test_message_handler.py -v
uv run pytest tests/test_agent.py -v

# Manual testing
# 1. Start server with a skill that needs interaction
# 2. Invoke skill via main agent
# 3. Verify user can talk to subagent
# 4. Verify complete() returns control to main agent
```

- ChildActivated propagates through except Exception handlers
- UseSkillTool raises ChildActivated with valid StackFrame
- execute_turn handles all TurnAction variants
- Orchestration loop cascades results correctly
- Main agent resumes after child completion
- Nested children cascade properly
