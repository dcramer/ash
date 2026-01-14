# Tool System Gap Analysis

This document analyzes 7 specific gaps between ash's tool system implementation and the reference implementations in pi-mono and archer.

## Files Analyzed

**Ash:**
- `/home/dcramer/src/ash/src/ash/tools/base.py` - Tool ABC, ToolContext, ToolResult
- `/home/dcramer/src/ash/src/ash/tools/executor.py` - ToolExecutor with logging
- `/home/dcramer/src/ash/src/ash/tools/truncation.py` - Output truncation utilities
- `/home/dcramer/src/ash/src/ash/tools/builtin/bash.py` - Sandboxed bash tool
- `/home/dcramer/src/ash/src/ash/tools/builtin/files.py` - read_file/write_file tools

**References:**
- `/home/dcramer/src/pi-mono/packages/agent/src/types.ts` - AgentTool interface with AbortSignal, onUpdate
- `/home/dcramer/src/pi-mono/packages/coding-agent/src/core/tools/bash.ts` - Bash with streaming, cancellation
- `/home/dcramer/src/pi-mono/packages/coding-agent/src/core/tools/edit.ts` - Edit tool with diff output
- `/home/dcramer/src/pi-mono/packages/coding-agent/src/core/tools/grep.ts` - Read-only grep tool
- `/home/dcramer/src/pi-mono/packages/coding-agent/src/core/tools/find.ts` - Read-only find tool
- `/home/dcramer/src/archer/src/tools/attach.ts` - File attachment for Telegram
- `/home/dcramer/src/archer/src/tools/edit.ts` - Edit tool with diff

---

## Gap 1: AbortSignal/Cancellation Support

### What Ash is Missing

Pi-mono tools accept an `AbortSignal` parameter for graceful cancellation. This allows:
- Interrupting long-running commands mid-execution
- Cancelling tool execution when a session ends
- User-initiated abort (Ctrl+C) that cleanly stops the current tool

Ash tools run to completion or timeout - there's no way to cancel mid-execution.

Current ash code (`base.py` lines 72-87):
```python
@abstractmethod
async def execute(
    self,
    input_data: dict[str, Any],
    context: ToolContext,
) -> ToolResult:
    """Execute the tool with the given input."""
    ...
```

Pi-mono's AgentTool interface (`types.ts` lines 157-166):
```typescript
export interface AgentTool<TParameters extends TSchema = TSchema, TDetails = any> extends Tool<TParameters> {
    label: string;
    execute: (
        toolCallId: string,
        params: Static<TParameters>,
        signal?: AbortSignal,  // <-- Cancellation support
        onUpdate?: AgentToolUpdateCallback<TDetails>,
    ) => Promise<AgentToolResult<TDetails>>;
}
```

### Reference

**Best implementation:** pi-mono bash tool (`bash.ts` lines 99-132)
```typescript
// Handle abort signal - kill entire process tree
const onAbort = () => {
    if (child.pid) {
        killProcessTree(child.pid);
    }
};

if (signal) {
    if (signal.aborted) {
        onAbort();
    } else {
        signal.addEventListener("abort", onAbort, { once: true });
    }
}

// Handle process exit
child.on("close", (code) => {
    if (timeoutHandle) clearTimeout(timeoutHandle);
    if (signal) signal.removeEventListener("abort", onAbort);

    if (signal?.aborted) {
        reject(new Error("aborted"));
        return;
    }
    // ... rest of handler
});
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/tools/base.py`
- `/home/dcramer/src/ash/src/ash/tools/executor.py`
- `/home/dcramer/src/ash/src/ash/tools/builtin/bash.py`
- `/home/dcramer/src/ash/src/ash/sandbox/executor.py`

### Proposed Changes

```python
# In base.py, add CancellationToken and modify ToolContext:

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CancellationToken:
    """Token for cooperative cancellation of tool execution.

    Similar to JavaScript's AbortSignal. Check `is_cancelled` periodically
    or register a callback with `on_cancel`.
    """

    _cancelled: bool = False
    _callbacks: list[Callable[[], None]] = field(default_factory=list)

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def cancel(self) -> None:
        """Request cancellation. Triggers all registered callbacks."""
        if self._cancelled:
            return
        self._cancelled = True
        for callback in self._callbacks:
            try:
                callback()
            except Exception:
                pass  # Don't let callback errors prevent other callbacks

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when cancelled.

        If already cancelled, callback is invoked immediately.
        """
        if self._cancelled:
            callback()
        else:
            self._callbacks.append(callback)

    def check(self) -> None:
        """Raise CancelledError if cancellation was requested.

        Call this periodically in long-running operations.
        """
        if self._cancelled:
            raise asyncio.CancelledError("Tool execution cancelled")


@dataclass
class ToolContext:
    """Context passed to tool execution."""

    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    # Cancellation support
    cancellation: CancellationToken | None = None
```

```python
# In base.py, update Tool ABC:

class Tool(ABC):
    """Abstract base class for tools."""

    @abstractmethod
    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool with the given input.

        Args:
            input_data: Tool input matching the input_schema.
            context: Execution context (includes cancellation token).

        Returns:
            Tool execution result.

        Note:
            Implementations should check context.cancellation.is_cancelled
            periodically for long-running operations and clean up resources
            when cancelled.
        """
        ...
```

```python
# In executor.py, add cancellation to execute():

class ToolExecutor:
    """Execute tools with logging, timing, error handling, and cancellation."""

    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of tool to execute.
            input_data: Tool input.
            context: Execution context.
            cancellation: Optional cancellation token.

        Returns:
            Tool result.
        """
        context = context or ToolContext()

        # Inject cancellation token into context
        if cancellation:
            context.cancellation = cancellation

            # Check if already cancelled before starting
            if cancellation.is_cancelled:
                return ToolResult.error("Execution cancelled before start")

        # Get tool
        try:
            tool = self._registry.get(tool_name)
        except KeyError:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult.error(f"Tool '{tool_name}' not found")

        # Execute with timing
        start_time = time.monotonic()
        try:
            result = await tool.execute(input_data, context)
        except asyncio.CancelledError:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.info(f"Tool: {tool_name} | cancelled after {duration_ms}ms")
            return ToolResult.error("Execution cancelled")
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            result = ToolResult.error(f"Tool execution failed: {e}")

        # ... rest of logging unchanged ...
        return result
```

```python
# In builtin/bash.py, add cancellation support:

import asyncio
from ash.tools.base import CancellationToken


class BashTool(Tool):
    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the bash command in the sandbox."""
        command = input_data.get("command")
        if not command:
            return ToolResult.error("Missing required parameter: command")

        timeout = input_data.get("timeout", 60)

        # Check cancellation before starting
        if context.cancellation and context.cancellation.is_cancelled:
            return ToolResult.error("Execution cancelled")

        try:
            return await self._execute_sandboxed(
                command,
                timeout,
                context.env,
                context.cancellation,
            )
        except asyncio.CancelledError:
            return ToolResult.error("Execution cancelled")
        except Exception as e:
            return ToolResult.error(f"Execution error: {e}")

    async def _execute_sandboxed(
        self,
        command: str,
        timeout: int,
        environment: dict[str, str] | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolResult:
        """Execute command in Docker sandbox with cancellation support."""

        # Create a task for the execution
        exec_task = asyncio.create_task(
            self._executor.execute(
                command,
                timeout=timeout,
                reuse_container=True,
                environment=environment,
            )
        )

        # If we have a cancellation token, set up the abort
        if cancellation:
            def on_cancel():
                if not exec_task.done():
                    exec_task.cancel()
            cancellation.on_cancel(on_cancel)

        try:
            result = await exec_task
        except asyncio.CancelledError:
            # Try to kill the container process
            await self._executor.kill_current()
            raise

        # ... rest of result handling unchanged ...
```

```python
# In sandbox/executor.py, add kill_current method:

class SandboxExecutor:
    """Execute commands in Docker sandbox."""

    async def kill_current(self) -> None:
        """Kill any currently running command in the sandbox.

        Used for cancellation - sends SIGKILL to the container's main process.
        """
        if not self._container:
            return

        try:
            # docker kill sends SIGKILL
            proc = await asyncio.create_subprocess_exec(
                "docker", "kill", self._container,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except Exception:
            pass  # Best effort - container may already be stopped
```

### Effort

**M** (1-2 days) - Requires threading cancellation through multiple layers and testing edge cases.

### Priority

**High** - Essential for responsive UX. Without cancellation, users must wait for timeout when they want to stop a command.

---

## Gap 2: Streaming Progress Updates

### What Ash is Missing

Pi-mono tools have an `onUpdate` callback for streaming partial results during execution. This enables:
- Live command output display as it runs
- Progress indicators for long operations
- Immediate feedback instead of waiting for completion

Ash tools only return the final result - no intermediate updates are possible.

Current ash ToolResult (`base.py` lines 28-44):
```python
@dataclass
class ToolResult:
    """Result from tool execution."""

    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
```

Pi-mono's streaming callback (`types.ts` lines 153-154):
```typescript
export type AgentToolUpdateCallback<T = any> = (partialResult: AgentToolResult<T>) => void;
```

### Reference

**Best implementation:** pi-mono bash tool (`bash.ts` lines 166-206)
```typescript
const handleData = (data: Buffer) => {
    totalBytes += data.length;

    // ... buffer management ...

    // Stream partial output to callback (truncated rolling buffer)
    if (onUpdate) {
        const fullBuffer = Buffer.concat(chunks);
        const fullText = fullBuffer.toString("utf-8");
        const truncation = truncateTail(fullText);
        onUpdate({
            content: [{ type: "text", text: truncation.content || "" }],
            details: {
                truncation: truncation.truncated ? truncation : undefined,
                fullOutputPath: tempFilePath,
            },
        });
    }
};
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/tools/base.py`
- `/home/dcramer/src/ash/src/ash/tools/executor.py`
- `/home/dcramer/src/ash/src/ash/tools/builtin/bash.py`

### Proposed Changes

```python
# In base.py, add progress callback type and modify context:

from typing import Callable, Protocol


class ProgressCallback(Protocol):
    """Callback for streaming tool execution progress.

    Called with partial results during long-running tool execution.
    """

    def __call__(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Report progress.

        Args:
            content: Current partial output (may be truncated).
            metadata: Optional progress metadata (bytes processed, etc).
        """
        ...


@dataclass
class ToolContext:
    """Context passed to tool execution."""

    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    cancellation: CancellationToken | None = None

    # Progress callback for streaming updates
    on_progress: ProgressCallback | None = None
```

```python
# In executor.py, add progress callback to execute():

# Type for progress callbacks
ProgressCallback = Callable[[str, dict[str, Any] | None], None]


class ToolExecutor:
    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
        cancellation: CancellationToken | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of tool to execute.
            input_data: Tool input.
            context: Execution context.
            cancellation: Optional cancellation token.
            on_progress: Optional callback for streaming progress updates.

        Returns:
            Tool result.
        """
        context = context or ToolContext()

        if cancellation:
            context.cancellation = cancellation
        if on_progress:
            context.on_progress = on_progress

        # ... rest unchanged ...
```

```python
# In builtin/bash.py, implement streaming:

import asyncio
from collections import deque


class BashTool(Tool):
    async def _execute_sandboxed(
        self,
        command: str,
        timeout: int,
        environment: dict[str, str] | None = None,
        cancellation: CancellationToken | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult:
        """Execute command in Docker sandbox with streaming output."""

        # Use a streaming executor that yields output chunks
        output_chunks: deque[str] = deque(maxlen=100)  # Rolling buffer
        total_bytes = 0

        async def stream_handler(chunk: str) -> None:
            """Handle output chunk from sandbox."""
            nonlocal total_bytes

            output_chunks.append(chunk)
            total_bytes += len(chunk.encode('utf-8'))

            # Stream to callback if provided
            if on_progress:
                # Combine recent chunks and truncate
                combined = "".join(output_chunks)
                truncation = truncate_tail(combined, save_full=False)
                on_progress(
                    truncation.content,
                    {
                        "bytes_so_far": total_bytes,
                        "truncated": truncation.truncated,
                    }
                )

        result = await self._executor.execute_streaming(
            command,
            timeout=timeout,
            reuse_container=True,
            environment=environment,
            on_output=stream_handler if on_progress else None,
        )

        # ... rest of result handling ...
```

```python
# In sandbox/executor.py, add streaming execute method:

class SandboxExecutor:
    async def execute_streaming(
        self,
        command: str,
        timeout: int = 60,
        reuse_container: bool = False,
        environment: dict[str, str] | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Execute command with streaming output.

        Args:
            command: Command to execute.
            timeout: Execution timeout in seconds.
            reuse_container: Whether to reuse existing container.
            environment: Additional environment variables.
            on_output: Callback for each output chunk.

        Returns:
            ExecutionResult after command completes.
        """
        # ... container setup unchanged ...

        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", container_id,
            "bash", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, **(environment or {})},
        )

        output_parts = []

        async def read_stream():
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                decoded = chunk.decode('utf-8', errors='replace')
                output_parts.append(decoded)
                if on_output:
                    on_output(decoded)

        try:
            await asyncio.wait_for(read_stream(), timeout=timeout)
            await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecutionResult(
                output="".join(output_parts),
                stderr="",
                exit_code=-1,
                timed_out=True,
            )

        return ExecutionResult(
            output="".join(output_parts),
            stderr="",
            exit_code=proc.returncode or 0,
            timed_out=False,
        )
```

### Effort

**M** (1-2 days) - Requires async streaming implementation and UI integration.

### Priority

**Medium** - Nice UX improvement but not blocking. Users can see final output after completion.

---

## Gap 3: Tool Result Details

### What Ash is Missing

Pi-mono separates tool results into:
- `content`: What goes to the LLM (text, images)
- `details`: What goes to UI/logging (diffs, truncation info, file paths)

This separation allows showing rich information in the UI without wasting LLM tokens.

Current ash ToolResult only has `content` for the LLM with `metadata` that's not surfaced to UI.

Pi-mono's AgentToolResult (`types.ts` lines 146-151):
```typescript
export interface AgentToolResult<T> {
    // Content blocks supporting text and images
    content: (TextContent | ImageContent)[];
    // Details to be displayed in a UI or logged
    details: T;
}
```

### Reference

**Best implementation:** pi-mono edit tool (`edit.ts` lines 186-194)
```typescript
resolve({
    content: [
        {
            type: "text",
            text: `Successfully replaced text in ${path}.`,
        },
    ],
    details: { diff: diffResult.diff, firstChangedLine: diffResult.firstChangedLine },
});
```

The UI can display the diff in a nice format while the LLM only sees the success message.

### Files to Modify

- `/home/dcramer/src/ash/src/ash/tools/base.py`
- All tool implementations that want to provide UI details

### Proposed Changes

```python
# In base.py, extend ToolResult:

from typing import TypeVar, Generic

T = TypeVar('T')


@dataclass
class ToolResult(Generic[T]):
    """Result from tool execution.

    Separates LLM-facing content from UI/logging details.
    """

    # Content for the LLM (string that goes into conversation)
    content: str

    # Whether execution resulted in an error
    is_error: bool = False

    # Metadata for internal tracking (exit codes, truncation info)
    metadata: dict[str, Any] = field(default_factory=dict)

    # UI/logging details (diffs, file previews, etc.)
    # Not sent to LLM - only for display purposes
    details: T | None = None

    @classmethod
    def success(
        cls,
        content: str,
        details: T | None = None,
        **metadata: Any,
    ) -> "ToolResult[T]":
        """Create a successful result."""
        return cls(content=content, is_error=False, metadata=metadata, details=details)

    @classmethod
    def error(cls, message: str, **metadata: Any) -> "ToolResult[None]":
        """Create an error result."""
        return cls(content=message, is_error=True, metadata=metadata, details=None)
```

```python
# Example usage in a hypothetical edit tool:

@dataclass
class EditDetails:
    """Details for edit tool results."""
    diff: str
    first_changed_line: int | None = None


class EditTool(Tool):
    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult[EditDetails]:
        """Execute text replacement."""
        # ... perform edit ...

        diff = generate_diff(old_content, new_content)

        return ToolResult.success(
            content=f"Successfully edited {file_path}",
            details=EditDetails(
                diff=diff,
                first_changed_line=first_line,
            ),
        )
```

```python
# In executor.py, pass details to callbacks:

ExecutionCallback = Callable[[str, dict[str, Any], ToolResult, int], None]

# The callback signature already receives full ToolResult,
# so UI can access result.details
```

### Effort

**S** (half day) - Simple dataclass change, backward compatible.

### Priority

**Medium** - Improves UI richness. Particularly valuable for edit tools where showing diffs is important.

---

## Gap 4: Edit Tool with Diff Output

### What Ash is Missing

Ash only has `read_file` and `write_file` tools. Pi-mono and archer have a dedicated `edit` tool that:
- Performs search-and-replace on specific text
- Returns a unified diff showing exactly what changed
- Validates the text is unique before replacing
- Handles line endings and BOM properly

This is safer than `write_file` because it shows what changed and prevents accidental overwrites.

Current ash approach requires reading the whole file, modifying, then writing - losing change visibility.

### Reference

**Best implementation:** pi-mono edit tool (`edit.ts`)
```typescript
export function createEditTool(cwd: string, options?: EditToolOptions): AgentTool<typeof editSchema> {
    return {
        name: "edit",
        label: "edit",
        description: "Edit a file by replacing exact text. The oldText must match exactly (including whitespace).",
        parameters: editSchema,
        execute: async (
            _toolCallId: string,
            { path, oldText, newText }: { path: string; oldText: string; newText: string },
            signal?: AbortSignal,
        ) => {
            // Read file, normalize line endings
            const { bom, text: content } = stripBom(rawContent);
            const normalizedContent = normalizeToLF(content);

            // Validate old text exists and is unique
            if (!normalizedContent.includes(normalizedOldText)) {
                throw new Error(`Could not find the exact text in ${path}.`);
            }

            const occurrences = normalizedContent.split(normalizedOldText).length - 1;
            if (occurrences > 1) {
                throw new Error(`Found ${occurrences} occurrences. Text must be unique.`);
            }

            // Perform replacement
            const index = normalizedContent.indexOf(normalizedOldText);
            const normalizedNewContent =
                normalizedContent.substring(0, index) +
                normalizedNewText +
                normalizedContent.substring(index + normalizedOldText.length);

            // Write back with original line endings
            const finalContent = bom + restoreLineEndings(normalizedNewContent, originalEnding);
            await ops.writeFile(absolutePath, finalContent);

            // Generate diff for UI
            const diffResult = generateDiffString(normalizedContent, normalizedNewContent);
            return {
                content: [{ type: "text", text: `Successfully replaced text in ${path}.` }],
                details: { diff: diffResult.diff, firstChangedLine: diffResult.firstChangedLine },
            };
        },
    };
}
```

Archer's simpler version (`edit.ts` lines 9-87):
```typescript
function generateDiffString(oldContent: string, newContent: string, contextLines = 4): string {
    const parts = Diff.diffLines(oldContent, newContent);
    const output: string[] = [];
    // ... unified diff generation with line numbers
    return output.join("\n");
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/tools/builtin/files.py` (add EditFileTool)
- `/home/dcramer/src/ash/src/ash/tools/builtin/__init__.py` (export new tool)

### Proposed Changes

```python
# In builtin/files.py, add EditFileTool:

import difflib
from dataclasses import dataclass


@dataclass
class EditDetails:
    """Details for edit tool results."""
    diff: str
    first_changed_line: int | None = None


def generate_unified_diff(
    old_content: str,
    new_content: str,
    file_path: str,
    context_lines: int = 3,
) -> tuple[str, int | None]:
    """Generate unified diff with line numbers.

    Returns:
        Tuple of (diff_string, first_changed_line).
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
        n=context_lines,
    )

    diff_lines = list(diff)
    if not diff_lines:
        return "", None

    # Find first changed line number from diff header
    first_line = None
    for line in diff_lines:
        if line.startswith("@@"):
            # Parse @@ -start,count +start,count @@
            import re
            match = re.match(r"@@ -(\d+)", line)
            if match:
                first_line = int(match.group(1))
                break

    return "\n".join(diff_lines), first_line


def normalize_line_endings(text: str) -> tuple[str, str]:
    """Normalize to LF, return (normalized, original_ending).

    Returns:
        Tuple of (normalized_text, original_line_ending).
        original_line_ending is 'crlf', 'cr', or 'lf'.
    """
    if "\r\n" in text:
        return text.replace("\r\n", "\n"), "crlf"
    elif "\r" in text:
        return text.replace("\r", "\n"), "cr"
    return text, "lf"


def restore_line_endings(text: str, ending: str) -> str:
    """Restore original line ending style."""
    if ending == "crlf":
        return text.replace("\n", "\r\n")
    elif ending == "cr":
        return text.replace("\n", "\r")
    return text


class EditFileTool(Tool):
    """Edit a file by replacing exact text.

    Safer than write_file because:
    - Only changes specific text, not entire file
    - Validates text exists and is unique
    - Returns diff showing exactly what changed
    - Preserves line endings
    """

    def __init__(self, executor: SandboxExecutor) -> None:
        """Initialize edit file tool.

        Args:
            executor: Shared sandbox executor.
        """
        self._executor = executor

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing exact text. "
            "The old_text must match exactly (including whitespace). "
            "Use this for precise, surgical edits. "
            "Returns a diff showing what changed."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative to workspace).",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find and replace (must match exactly).",
                },
                "new_text": {
                    "type": "string",
                    "description": "New text to replace the old text with.",
                },
            },
            "required": ["file_path", "old_text", "new_text"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult[EditDetails]:
        """Execute text replacement."""
        file_path = input_data.get("file_path")
        old_text = input_data.get("old_text")
        new_text = input_data.get("new_text")

        if not file_path:
            return ToolResult.error("Missing required parameter: file_path")
        if old_text is None:
            return ToolResult.error("Missing required parameter: old_text")
        if new_text is None:
            return ToolResult.error("Missing required parameter: new_text")

        # Read current content
        safe_path = shlex.quote(file_path)
        read_result = await self._executor.execute(f"cat {safe_path}")

        if not read_result.success:
            return ToolResult.error(f"File not found: {file_path}")

        content = read_result.stdout

        # Normalize line endings for comparison
        normalized_content, original_ending = normalize_line_endings(content)
        normalized_old, _ = normalize_line_endings(old_text)
        normalized_new, _ = normalize_line_endings(new_text)

        # Check if old text exists
        if normalized_old not in normalized_content:
            return ToolResult.error(
                f"Could not find the exact text in {file_path}. "
                "The old_text must match exactly including all whitespace and newlines."
            )

        # Check for multiple occurrences
        occurrences = normalized_content.count(normalized_old)
        if occurrences > 1:
            return ToolResult.error(
                f"Found {occurrences} occurrences of the text in {file_path}. "
                "The text must be unique. Please provide more context to make it unique."
            )

        # Perform replacement (using indexOf to avoid regex special chars)
        index = normalized_content.find(normalized_old)
        new_content = (
            normalized_content[:index] +
            normalized_new +
            normalized_content[index + len(normalized_old):]
        )

        # Verify change was made
        if normalized_content == new_content:
            return ToolResult.error(
                f"No changes made to {file_path}. "
                "The replacement produced identical content."
            )

        # Restore original line endings
        final_content = restore_line_endings(new_content, original_ending)

        # Write back
        write_result = await self._executor.write_file(file_path, final_content)
        if not write_result.success:
            return ToolResult.error(f"Failed to write file: {write_result.stderr}")

        # Generate diff for UI
        diff_str, first_line = generate_unified_diff(
            normalized_content,
            new_content,
            file_path,
        )

        return ToolResult.success(
            content=f"Successfully edited {file_path}",
            details=EditDetails(diff=diff_str, first_changed_line=first_line),
            old_length=len(old_text),
            new_length=len(new_text),
        )
```

### Effort

**M** (half day) - Straightforward implementation, mostly port from reference.

### Priority

**High** - Much safer than write_file for code edits. Shows agent exactly what changed.

---

## Gap 5: Attach/Share Tool

### What Ash is Missing

Archer has an `attach` tool that allows the agent to share files with users via Telegram. This enables:
- Sending generated files (images, documents, code)
- Sharing analysis results
- Providing downloadable artifacts

Ash has no mechanism for the agent to send files to users - only text responses.

### Reference

**Best implementation:** archer attach tool (`attach.ts`)
```typescript
export const attachTool: AgentTool<typeof attachSchema> = {
    name: "attach",
    label: "attach",
    description: `Attach a file to your response. Use this to share files with the user.

Supported file types with native Telegram rendering:
- **Images** (jpg, jpeg, png, webp): Displayed inline as photos
- **GIFs** (gif): Animated inline preview
- **Videos** (mp4, mov, avi, mkv, webm): Inline video player
- **Audio** (mp3, m4a, wav, ogg, flac): Inline audio player
- **Documents** (pdf, etc.): File attachment with preview`,
    parameters: attachSchema,
    execute: async (
        _toolCallId: string,
        { path, title }: { label: string; path: string; title?: string },
        signal?: AbortSignal,
    ) => {
        if (!uploadFn) {
            throw new Error("Upload function not configured");
        }

        const absolutePath = resolvePath(path);
        const fileName = title || basename(absolutePath);

        await uploadFn(absolutePath, fileName);

        return {
            content: [{ type: "text" as const, text: `Attached file: ${fileName}` }],
            details: undefined,
        };
    },
};
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/tools/builtin/attach.py` (new file)
- `/home/dcramer/src/ash/src/ash/tools/builtin/__init__.py`
- `/home/dcramer/src/ash/src/ash/tools/base.py` (add attachment callback to context)

### Proposed Changes

```python
# In base.py, add attachment callback type:

from pathlib import Path
from typing import Callable, Awaitable


# Callback for sending file attachments to users
AttachmentCallback = Callable[[Path, str | None], Awaitable[None]]


@dataclass
class ToolContext:
    """Context passed to tool execution."""

    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    cancellation: CancellationToken | None = None
    on_progress: ProgressCallback | None = None

    # Callback for attaching files to send to user
    # Provider-specific (e.g., Telegram sends as document/photo)
    attach_file: AttachmentCallback | None = None
```

```python
# New file: builtin/attach.py

"""File attachment tool for sending files to users."""

from pathlib import Path
from typing import Any

from ash.tools.base import Tool, ToolContext, ToolResult


# Supported file types for inline rendering hints
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".oga"}
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}


class AttachFileTool(Tool):
    """Attach a file to send to the user.

    Use this to share generated files, images, documents, or any
    file the user might want to download.

    The file will be sent through the conversation provider
    (e.g., as a Telegram attachment).
    """

    @property
    def name(self) -> str:
        return "attach_file"

    @property
    def description(self) -> str:
        return (
            "Attach a file to share with the user. "
            "Use this to send generated files, images, documents, or code. "
            "The file will be delivered through the chat interface. "
            "\n\nSupported with special rendering:\n"
            "- Images (jpg, png, webp, gif): Displayed inline\n"
            "- Videos (mp4, mov, webm): Inline player\n"
            "- Audio (mp3, wav, ogg): Inline player\n"
            "- Documents (pdf): Preview with download\n"
            "- Other files: Download link"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to attach (in workspace).",
                },
                "title": {
                    "type": "string",
                    "description": "Optional title/caption for the file. Defaults to filename.",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what you're sharing (for context).",
                },
            },
            "required": ["file_path", "description"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Attach file for sending to user."""
        file_path = input_data.get("file_path")
        title = input_data.get("title")
        description = input_data.get("description", "")

        if not file_path:
            return ToolResult.error("Missing required parameter: file_path")

        # Check if attachment is supported in this context
        if not context.attach_file:
            return ToolResult.error(
                "File attachment not available in this context. "
                "The conversation provider may not support file uploads."
            )

        # Resolve path (would go through sandbox in real implementation)
        path = Path(file_path)

        # Use filename as title if not provided
        display_title = title or path.name

        try:
            await context.attach_file(path, display_title)
        except FileNotFoundError:
            return ToolResult.error(f"File not found: {file_path}")
        except PermissionError:
            return ToolResult.error(f"Permission denied: {file_path}")
        except Exception as e:
            return ToolResult.error(f"Failed to attach file: {e}")

        # Determine file type for response
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            file_type = "image"
        elif suffix in VIDEO_EXTENSIONS:
            file_type = "video"
        elif suffix in AUDIO_EXTENSIONS:
            file_type = "audio"
        elif suffix in DOCUMENT_EXTENSIONS:
            file_type = "document"
        else:
            file_type = "file"

        return ToolResult.success(
            content=f"Attached {file_type}: {display_title}",
            file_path=str(path),
            file_type=file_type,
            title=display_title,
        )
```

```python
# In providers/telegram.py (example integration):

from ash.tools.base import ToolContext


async def create_tool_context(
    self,
    session_id: str,
    chat_id: int,
    user_id: int,
) -> ToolContext:
    """Create tool context with Telegram-specific capabilities."""

    async def send_attachment(path: Path, title: str | None) -> None:
        """Send file as Telegram attachment."""
        # Resolve path through sandbox (get actual host path)
        host_path = await self._sandbox.resolve_path(path)

        if not host_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Determine send method based on file type
        suffix = host_path.suffix.lower()

        if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            await self._bot.send_photo(
                chat_id=chat_id,
                photo=host_path,
                caption=title,
            )
        elif suffix == ".gif":
            await self._bot.send_animation(
                chat_id=chat_id,
                animation=host_path,
                caption=title,
            )
        elif suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            await self._bot.send_video(
                chat_id=chat_id,
                video=host_path,
                caption=title,
            )
        elif suffix in {".mp3", ".m4a", ".wav", ".ogg", ".flac"}:
            await self._bot.send_audio(
                chat_id=chat_id,
                audio=host_path,
                caption=title,
            )
        else:
            await self._bot.send_document(
                chat_id=chat_id,
                document=host_path,
                caption=title,
            )

    return ToolContext(
        session_id=session_id,
        user_id=str(user_id),
        chat_id=str(chat_id),
        provider="telegram",
        attach_file=send_attachment,
    )
```

### Effort

**M** (1 day) - Tool is simple but requires provider integration.

### Priority

**Medium** - Valuable for Telegram use case. Less important for CLI where files are already accessible.

---

## Gap 6: Grep/Find Tools

### What Ash is Missing

Pi-mono has dedicated read-only `grep` and `find` tools that provide:
- Structured search results with context
- Respects .gitignore
- Match limits with truncation
- Line length truncation for display
- Uses optimized tools (ripgrep, fd) with auto-download

Ash relies on bash commands in the sandbox, which works but:
- No structured output (just raw text)
- Requires knowing ripgrep/fd command syntax
- No automatic tool installation
- Easy to accidentally search huge directories

### Reference

**Pi-mono grep tool** (`grep.ts` lines 61-341) - uses ripgrep with JSON output:
```typescript
export function createGrepTool(cwd: string, options?: GrepToolOptions): AgentTool<typeof grepSchema> {
    return {
        name: "grep",
        label: "grep",
        description: `Search file contents for a pattern. Returns matching lines with file paths
                      and line numbers. Respects .gitignore. Output truncated to ${DEFAULT_LIMIT}
                      matches or ${DEFAULT_MAX_BYTES / 1024}KB.`,
        parameters: grepSchema,
        execute: async (...) => {
            const rgPath = await ensureTool("rg", true);
            // ... structured search with ripgrep --json
        },
    };
}
```

**Pi-mono find tool** (`find.ts`) - uses fd:
```typescript
export function createFindTool(cwd: string, options?: FindToolOptions): AgentTool<typeof findSchema> {
    return {
        name: "find",
        label: "find",
        description: `Search for files by glob pattern. Returns matching file paths.
                      Respects .gitignore. Output truncated to ${DEFAULT_LIMIT} results.`,
        // ...
    };
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/tools/builtin/search.py` (new file)
- `/home/dcramer/src/ash/src/ash/tools/builtin/__init__.py`

### Proposed Changes

```python
# New file: builtin/search.py

"""Read-only search tools - grep and find."""

import json
import shlex
from typing import Any

from ash.sandbox import SandboxExecutor
from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.truncation import truncate_head


# Limits
DEFAULT_MATCH_LIMIT = 100
DEFAULT_FILE_LIMIT = 1000
MAX_OUTPUT_BYTES = 50 * 1024  # 50KB
MAX_LINE_LENGTH = 500  # Truncate long lines in output


class GrepTool(Tool):
    """Search file contents for a pattern.

    Uses ripgrep (rg) for fast, .gitignore-respecting search.
    Returns matching lines with file paths and line numbers.
    """

    def __init__(self, executor: SandboxExecutor) -> None:
        self._executor = executor

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents for a pattern (regex or literal). "
            "Returns matching lines with file paths and line numbers. "
            "Respects .gitignore. "
            f"Output limited to {DEFAULT_MATCH_LIMIT} matches by default."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex or literal string).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: workspace root).",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.py'.",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false).",
                    "default": False,
                },
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as literal string instead of regex.",
                    "default": False,
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context before and after each match.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum matches to return (default: {DEFAULT_MATCH_LIMIT}).",
                    "default": DEFAULT_MATCH_LIMIT,
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute grep search."""
        pattern = input_data.get("pattern")
        if not pattern:
            return ToolResult.error("Missing required parameter: pattern")

        path = input_data.get("path", ".")
        glob_pattern = input_data.get("glob")
        ignore_case = input_data.get("ignore_case", False)
        literal = input_data.get("literal", False)
        ctx_lines = input_data.get("context", 0)
        limit = input_data.get("limit", DEFAULT_MATCH_LIMIT)

        # Build ripgrep command
        args = ["rg", "--json", "--line-number", "--color=never", "--hidden"]

        if ignore_case:
            args.append("--ignore-case")
        if literal:
            args.append("--fixed-strings")
        if glob_pattern:
            args.extend(["--glob", shlex.quote(glob_pattern)])
        if ctx_lines > 0:
            args.extend(["-C", str(ctx_lines)])

        # Max results (rg doesn't have --max-count for matches, use head)
        args.append(shlex.quote(pattern))
        args.append(shlex.quote(path))

        cmd = " ".join(args) + f" | head -n {limit * 10}"  # JSON lines, estimate

        result = await self._executor.execute(cmd, timeout=30)

        # Parse JSON output
        if not result.output.strip():
            return ToolResult.success("No matches found", match_count=0)

        matches = []
        for line in result.output.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "match":
                    data = event.get("data", {})
                    file_path = data.get("path", {}).get("text", "?")
                    line_num = data.get("line_number", 0)
                    lines = data.get("lines", {}).get("text", "").rstrip()

                    # Truncate long lines
                    if len(lines) > MAX_LINE_LENGTH:
                        lines = lines[:MAX_LINE_LENGTH] + "..."

                    matches.append(f"{file_path}:{line_num}: {lines}")

                    if len(matches) >= limit:
                        break
            except json.JSONDecodeError:
                continue

        if not matches:
            return ToolResult.success("No matches found", match_count=0)

        output = "\n".join(matches)
        truncation = truncate_head(output, max_bytes=MAX_OUTPUT_BYTES)

        limit_notice = ""
        if len(matches) >= limit:
            limit_notice = f"\n\n[Showing first {limit} matches. Use limit={limit*2} for more.]"

        return ToolResult.success(
            content=truncation.content + limit_notice,
            match_count=len(matches),
            **truncation.to_metadata(),
        )


class FindTool(Tool):
    """Search for files by glob pattern.

    Uses fd for fast, .gitignore-respecting file search.
    Returns matching file paths relative to search directory.
    """

    def __init__(self, executor: SandboxExecutor) -> None:
        self._executor = executor

    @property
    def name(self) -> str:
        return "find"

    @property
    def description(self) -> str:
        return (
            "Search for files by glob pattern. "
            "Returns matching file paths relative to search directory. "
            "Respects .gitignore. "
            f"Output limited to {DEFAULT_FILE_LIMIT} results by default."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json'.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: workspace root).",
                    "default": ".",
                },
                "type": {
                    "type": "string",
                    "enum": ["file", "directory", "any"],
                    "description": "Filter by type: file, directory, or any.",
                    "default": "file",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum results (default: {DEFAULT_FILE_LIMIT}).",
                    "default": DEFAULT_FILE_LIMIT,
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute file search."""
        pattern = input_data.get("pattern")
        if not pattern:
            return ToolResult.error("Missing required parameter: pattern")

        path = input_data.get("path", ".")
        file_type = input_data.get("type", "file")
        limit = input_data.get("limit", DEFAULT_FILE_LIMIT)

        # Build fd command
        args = ["fd", "--glob", "--color=never", "--hidden"]
        args.extend(["--max-results", str(limit)])

        if file_type == "file":
            args.append("--type=f")
        elif file_type == "directory":
            args.append("--type=d")

        args.append(shlex.quote(pattern))
        args.append(shlex.quote(path))

        cmd = " ".join(args)

        result = await self._executor.execute(cmd, timeout=30)

        if not result.output.strip():
            return ToolResult.success("No files found", file_count=0)

        files = result.output.strip().split("\n")
        files = [f.strip() for f in files if f.strip()]

        truncation = truncate_head("\n".join(files), max_bytes=MAX_OUTPUT_BYTES)

        limit_notice = ""
        if len(files) >= limit:
            limit_notice = f"\n\n[Showing first {limit} results. Use limit={limit*2} for more.]"

        return ToolResult.success(
            content=truncation.content + limit_notice,
            file_count=len(files),
            **truncation.to_metadata(),
        )
```

### Effort

**M** (half day) - Straightforward implementation using existing sandbox.

### Priority

**Medium** - Improves developer experience. Agent can currently use bash with rg/fd commands, but dedicated tools provide better UX.

---

## Gap 7: Tool Policies/Filtering

### What Ash is Missing

Pi-mono and clawdbot can restrict which tools are available per-agent or per-session:
- Read-only tools for exploration agents
- Full coding tools for implementation agents
- Custom tool sets for specialized agents

Ash tools are all-or-nothing - all registered tools are available to all agents.

Current ash tool registry (`registry.py`):
```python
class ToolRegistry:
    """Registry for tool instances. Manages tool registration and lookup."""

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions for LLM."""
        return [tool.to_definition() for tool in self._tools.values()]
```

### Reference

**Pi-mono tool sets** (`tools/index.ts` lines 37-52):
```typescript
// Default tools for full access mode
export const codingTools: Tool[] = [readTool, bashTool, editTool, writeTool];

// Read-only tools for exploration without modification
export const readOnlyTools: Tool[] = [readTool, grepTool, findTool, lsTool];

// All available tools
export const allTools = {
    read: readTool,
    bash: bashTool,
    edit: editTool,
    write: writeTool,
    grep: grepTool,
    find: findTool,
    ls: lsTool,
};
```

**Pi-mono tool creation with filtering** (`tools/index.ts` lines 64-88):
```typescript
export function createCodingTools(cwd: string, options?: ToolsOptions): Tool[] {
    return [createReadTool(cwd), createBashTool(cwd), createEditTool(cwd), createWriteTool(cwd)];
}

export function createReadOnlyTools(cwd: string, options?: ToolsOptions): Tool[] {
    return [createReadTool(cwd), createGrepTool(cwd), createFindTool(cwd), createLsTool(cwd)];
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/tools/registry.py`
- `/home/dcramer/src/ash/src/ash/tools/executor.py`

### Proposed Changes

```python
# In registry.py, add tool filtering:

from typing import Literal


# Predefined tool sets
ToolSet = Literal["all", "coding", "read_only", "none"]

# Tool categorization
CODING_TOOLS = {"bash", "read_file", "write_file", "edit_file"}
READ_ONLY_TOOLS = {"read_file", "grep", "find", "web_search", "web_fetch"}


class ToolRegistry:
    """Registry for tool instances with filtering support."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, Tool] = {}
        self._allowed_tools: set[str] | None = None  # None = all allowed
        self._blocked_tools: set[str] = set()

    def set_policy(
        self,
        tool_set: ToolSet = "all",
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
    ) -> None:
        """Set tool access policy.

        Args:
            tool_set: Predefined set ("all", "coding", "read_only", "none").
            allowed: Explicit allow list (overrides tool_set if provided).
            blocked: Tools to block (applied after allow list).
        """
        # Determine allowed tools
        if allowed is not None:
            self._allowed_tools = set(allowed)
        elif tool_set == "all":
            self._allowed_tools = None  # All allowed
        elif tool_set == "coding":
            self._allowed_tools = CODING_TOOLS.copy()
        elif tool_set == "read_only":
            self._allowed_tools = READ_ONLY_TOOLS.copy()
        elif tool_set == "none":
            self._allowed_tools = set()

        # Apply blocked tools
        self._blocked_tools = set(blocked) if blocked else set()

    def is_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed by current policy."""
        if tool_name in self._blocked_tools:
            return False
        if self._allowed_tools is None:
            return True
        return tool_name in self._allowed_tools

    def get(self, name: str) -> Tool:
        """Get a tool by name (respects policy)."""
        if not self.is_allowed(name):
            raise KeyError(f"Tool '{name}' not allowed by policy")
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]

    def get_definitions(self, respect_policy: bool = True) -> list[dict[str, Any]]:
        """Get tool definitions for LLM.

        Args:
            respect_policy: If True, only return allowed tools.

        Returns:
            List of tool definitions.
        """
        if respect_policy:
            return [
                tool.to_definition()
                for tool in self._tools.values()
                if self.is_allowed(tool.name)
            ]
        return [tool.to_definition() for tool in self._tools.values()]

    @property
    def available_names(self) -> list[str]:
        """Get list of available tool names (respects policy)."""
        return [name for name in self._tools.keys() if self.is_allowed(name)]
```

```python
# In executor.py, respect policy:

class ToolExecutor:
    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """Execute a tool by name."""
        context = context or ToolContext()

        # Check policy before execution
        if not self._registry.is_allowed(tool_name):
            logger.warning(f"Tool blocked by policy: {tool_name}")
            return ToolResult.error(f"Tool '{tool_name}' not available")

        # ... rest unchanged ...
```

```python
# Example usage for different agents:

# Exploration agent - read-only access
exploration_registry = ToolRegistry()
exploration_registry.register_defaults()
exploration_registry.set_policy(tool_set="read_only")

# Implementation agent - full coding access
coding_registry = ToolRegistry()
coding_registry.register_defaults()
coding_registry.set_policy(tool_set="coding")

# Custom agent - specific tools only
custom_registry = ToolRegistry()
custom_registry.register_defaults()
custom_registry.set_policy(
    allowed=["read_file", "web_search", "web_fetch"],
    blocked=["bash"],  # Explicitly block bash even if in allowed
)
```

### Effort

**S** (half day) - Simple set-based filtering.

### Priority

**Medium** - Useful for agent specialization and security. Becomes more important with sub-agents.

---

## Summary Table

| Gap | Description | Effort | Priority | Main Benefit |
|-----|-------------|--------|----------|--------------|
| 1 | AbortSignal/cancellation | M | **High** | Responsive UX, can stop runaway commands |
| 2 | Streaming progress | M | Medium | Live output, better feedback |
| 3 | Tool result details | S | Medium | Rich UI without wasting tokens |
| 4 | Edit tool with diff | M | **High** | Safe edits, visible changes |
| 5 | Attach/share tool | M | Medium | Send files to users (Telegram) |
| 6 | Grep/find tools | M | Medium | Structured search, better than raw bash |
| 7 | Tool policies/filtering | S | Medium | Agent specialization, security |

## Recommended Implementation Order

1. **Gap 4: Edit tool** (High priority, safer than write_file, shows diffs)
2. **Gap 1: Cancellation** (High priority, essential for responsive UX)
3. **Gap 3: Tool result details** (Medium, enables rich UI - do with edit tool)
4. **Gap 7: Tool policies** (Medium, quick win for agent specialization)
5. **Gap 6: Grep/find tools** (Medium, improves search UX)
6. **Gap 2: Streaming progress** (Medium, nice UX but more complex)
7. **Gap 5: Attach tool** (Medium, Telegram-specific, can defer)
