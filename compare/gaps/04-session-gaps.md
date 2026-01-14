# Session Management Gap Analysis

This document analyzes gaps between Ash's session management implementation and the reference implementations in pi-mono and Clawdbot.

**Files Analyzed:**
- Ash: `/home/dcramer/src/ash/src/ash/sessions/manager.py`, `types.py`, `writer.py`, `reader.py`
- Pi-mono: `/home/dcramer/src/pi-mono/packages/coding-agent/src/core/session-manager.ts`
- Clawdbot: `/home/dcramer/src/clawdbot/src/config/sessions.ts`

---

## Gap 1: Tree-based Session History

### What Ash is Missing

Ash sessions are linear-only. Entries have no parent/child relationships (lines 128-141, 233-241 in `types.py`):

```python
# types.py lines 128-141 - MessageEntry has no parent tracking
@dataclass
class MessageEntry:
    """Message entry - user or assistant message."""

    id: str
    role: Literal["user", "assistant", "system"]
    content: str | list[dict[str, Any]]
    created_at: datetime
    token_count: int | None = None
    user_id: str | None = None
    username: str | None = None
    display_name: str | None = None
    metadata: dict[str, Any] | None = None
    type: Literal["message"] = "message"
```

```python
# types.py lines 233-241 - ToolUseEntry has message_id but no parent_id
@dataclass
class ToolUseEntry:
    """Tool use entry - request to execute a tool."""

    id: str
    message_id: str  # Links to parent message, not tree structure
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"
```

### Reference Implementation (Pi-mono)

Pi-mono has full tree structure with `id` and `parentId` on every entry (lines 42-47, 129-138 in `session-manager.ts`):

```typescript
// session-manager.ts lines 42-47
export interface SessionEntryBase {
    type: string;
    id: string;
    parentId: string | null;  // Tree structure - parent pointer
    timestamp: string;
}

// session-manager.ts lines 129-138 - All entry types inherit tree structure
export type SessionEntry =
    | SessionMessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | CustomMessageEntry
    | LabelEntry;
```

The tree allows navigation via `getBranch()` (lines 913-922) and `getTree()` (lines 954-991):

```typescript
// session-manager.ts lines 913-922
getBranch(fromId?: string): SessionEntry[] {
    const path: SessionEntry[] = [];
    const startId = fromId ?? this.leafId;
    let current = startId ? this.byId.get(startId) : undefined;
    while (current) {
        path.unshift(current);
        current = current.parentId ? this.byId.get(current.parentId) : undefined;
    }
    return path;
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sessions/types.py`
- `/home/dcramer/src/ash/src/ash/sessions/manager.py`
- `/home/dcramer/src/ash/src/ash/sessions/writer.py`
- `/home/dcramer/src/ash/src/ash/sessions/reader.py`

### Concrete Python Code Changes

```python
# types.py - Add EntryBase with tree structure
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EntryBase:
    """Base class for all session entries with tree structure."""

    id: str
    parent_id: str | None  # None for root entries
    created_at: datetime

    @classmethod
    def generate_id(cls) -> str:
        """Generate unique entry ID."""
        return str(uuid.uuid4())[:8]  # Short IDs like pi-mono


@dataclass
class MessageEntry(EntryBase):
    """Message entry - user or assistant message."""

    role: Literal["user", "assistant", "system"]
    content: str | list[dict[str, Any]]
    token_count: int | None = None
    user_id: str | None = None
    username: str | None = None
    display_name: str | None = None
    metadata: dict[str, Any] | None = None
    type: Literal["message"] = "message"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "id": self.id,
            "parent_id": self.parent_id,  # Add parent tracking
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "token_count": self.token_count,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageEntry:
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            id=data["id"],
            parent_id=data.get("parent_id"),  # Parse parent_id
            role=data["role"],
            content=data["content"],
            created_at=created_at,
            token_count=data.get("token_count"),
            user_id=data.get("user_id"),
            username=data.get("username"),
            display_name=data.get("display_name"),
            metadata=data.get("metadata"),
        )

    @classmethod
    def create(
        cls,
        role: Literal["user", "assistant", "system"],
        content: str | list[dict[str, Any]],
        parent_id: str | None = None,  # Add parent_id parameter
        token_count: int | None = None,
        user_id: str | None = None,
        username: str | None = None,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageEntry:
        return cls(
            id=cls.generate_id(),
            parent_id=parent_id,
            role=role,
            content=content,
            created_at=now_utc(),
            token_count=token_count,
            user_id=user_id,
            username=username,
            display_name=display_name,
            metadata=metadata,
        )


# manager.py - Add tree navigation
class SessionManager:
    """Manages session lifecycle and persistence."""

    def __init__(self, ...):
        # ... existing init ...
        self._leaf_id: str | None = None  # Track current position in tree
        self._entries_by_id: dict[str, Entry] = {}  # Index for tree traversal

    async def get_branch(self, from_id: str | None = None) -> list[Entry]:
        """Walk from entry to root, returning path in chronological order.

        Args:
            from_id: Entry ID to start from (default: current leaf).

        Returns:
            List of entries from root to specified entry.
        """
        entries = await self._reader.load_entries()

        # Build index
        by_id: dict[str, Entry] = {}
        for entry in entries:
            if hasattr(entry, 'id'):
                by_id[entry.id] = entry

        # Walk from leaf to root
        path: list[Entry] = []
        start_id = from_id or self._leaf_id
        current = by_id.get(start_id) if start_id else None

        while current:
            path.insert(0, current)  # Prepend to get root-first order
            parent_id = getattr(current, 'parent_id', None)
            current = by_id.get(parent_id) if parent_id else None

        return path

    async def get_tree(self) -> list[SessionTreeNode]:
        """Get session as tree structure for navigation UI.

        Returns:
            List of root nodes (normally one, but orphans become roots).
        """
        entries = await self._reader.load_entries()

        # Skip header
        session_entries = [e for e in entries if not isinstance(e, SessionHeader)]

        # Build nodes
        node_map: dict[str, SessionTreeNode] = {}
        for entry in session_entries:
            node_map[entry.id] = SessionTreeNode(entry=entry, children=[])

        # Build tree
        roots: list[SessionTreeNode] = []
        for entry in session_entries:
            node = node_map[entry.id]
            parent_id = getattr(entry, 'parent_id', None)
            if parent_id is None:
                roots.append(node)
            elif parent_id in node_map:
                node_map[parent_id].children.append(node)
            else:
                roots.append(node)  # Orphan becomes root

        # Sort children by timestamp
        for node in node_map.values():
            node.children.sort(key=lambda n: n.entry.created_at)

        return roots


@dataclass
class SessionTreeNode:
    """Tree node for session navigation."""
    entry: Entry
    children: list[SessionTreeNode]
    label: str | None = None
```

### Effort: Large

Tree structure touches all entry types and changes the fundamental data model.

### Priority: High

Required for branching, proper compaction, and undo/redo functionality.

---

## Gap 2: Session Branching

### What Ash is Missing

Ash has no branching capability. Messages are appended linearly without any way to fork conversations (lines 136-172 in `manager.py`):

```python
# manager.py lines 136-172 - Linear append only
async def add_user_message(
    self,
    content: str,
    token_count: int | None = None,
    metadata: dict[str, Any] | None = None,
    user_id: str | None = None,
    username: str | None = None,
    display_name: str | None = None,
) -> str:
    """Add a user message to the session."""
    await self.ensure_session()

    entry = MessageEntry.create(
        role="user",
        content=content,
        token_count=token_count,
        # ... no parent_id tracking
    )
    await self._writer.write_message(entry)
    self._current_message_id = entry.id
    return entry.id
```

### Reference Implementation (Pi-mono)

Pi-mono has `branch()` and `branchWithSummary()` for forking conversations (lines 997-1041 in `session-manager.ts`):

```typescript
// session-manager.ts lines 997-1008
/**
 * Start a new branch from an earlier entry.
 * Moves the leaf pointer to the specified entry. The next appendXXX() call
 * will create a child of that entry, forming a new branch.
 */
branch(branchFromId: string): void {
    if (!this.byId.has(branchFromId)) {
        throw new Error(`Entry ${branchFromId} not found`);
    }
    this.leafId = branchFromId;
}

// session-manager.ts lines 1024-1041
/**
 * Start a new branch with a summary of the abandoned path.
 * Same as branch(), but also appends a branch_summary entry.
 */
branchWithSummary(branchFromId: string | null, summary: string, details?: unknown, fromHook?: boolean): string {
    if (branchFromId !== null && !this.byId.has(branchFromId)) {
        throw new Error(`Entry ${branchFromId} not found`);
    }
    this.leafId = branchFromId;
    const entry: BranchSummaryEntry = {
        type: "branch_summary",
        id: generateId(this.byId),
        parentId: branchFromId,
        timestamp: new Date().toISOString(),
        fromId: branchFromId ?? "root",
        summary,
        details,
        fromHook,
    };
    this._appendEntry(entry);
    return entry.id;
}
```

Pi-mono also has `BranchSummaryEntry` type (lines 76-84):

```typescript
// session-manager.ts lines 76-84
export interface BranchSummaryEntry<T = unknown> extends SessionEntryBase {
    type: "branch_summary";
    fromId: string;
    summary: string;
    /** Extension-specific data (not sent to LLM) */
    details?: T;
    /** True if generated by an extension, false if pi-generated */
    fromHook?: boolean;
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sessions/types.py`
- `/home/dcramer/src/ash/src/ash/sessions/manager.py`

### Concrete Python Code Changes

```python
# types.py - Add BranchSummaryEntry
@dataclass
class BranchSummaryEntry(EntryBase):
    """Branch summary entry - context from abandoned conversation path."""

    from_id: str  # Entry ID where branch started
    summary: str
    details: dict[str, Any] | None = None
    from_hook: bool = False  # True if generated by extension
    type: Literal["branch_summary"] = "branch_summary"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "id": self.id,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "from_id": self.from_id,
            "summary": self.summary,
        }
        if self.details:
            result["details"] = self.details
        if self.from_hook:
            result["from_hook"] = self.from_hook
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BranchSummaryEntry:
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = now_utc()
        return cls(
            id=data["id"],
            parent_id=data.get("parent_id"),
            created_at=created_at,
            from_id=data["from_id"],
            summary=data["summary"],
            details=data.get("details"),
            from_hook=data.get("from_hook", False),
        )

    @classmethod
    def create(
        cls,
        from_id: str,
        summary: str,
        parent_id: str | None = None,
        details: dict[str, Any] | None = None,
        from_hook: bool = False,
    ) -> BranchSummaryEntry:
        return cls(
            id=generate_id(),
            parent_id=parent_id,
            created_at=now_utc(),
            from_id=from_id,
            summary=summary,
            details=details,
            from_hook=from_hook,
        )


# manager.py - Add branching methods
class SessionManager:

    def __init__(self, ...):
        # ... existing init ...
        self._leaf_id: str | None = None

    def branch(self, branch_from_id: str) -> None:
        """Start a new branch from an earlier entry.

        Moves the leaf pointer to the specified entry. The next add_*
        call will create a child of that entry, forming a new branch.
        Existing entries are not modified.

        Args:
            branch_from_id: Entry ID to branch from.

        Raises:
            ValueError: If entry ID not found.
        """
        if branch_from_id not in self._entries_by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        """Reset the leaf pointer to null (before any entries).

        The next add_* call will create a new root entry (parent_id = None).
        Use this when re-editing the first user message.
        """
        self._leaf_id = None

    async def branch_with_summary(
        self,
        branch_from_id: str | None,
        summary: str,
        details: dict[str, Any] | None = None,
        from_hook: bool = False,
    ) -> str:
        """Start a new branch with a summary of the abandoned path.

        Same as branch(), but also appends a branch_summary entry that
        captures context from the abandoned conversation path.

        Args:
            branch_from_id: Entry ID to branch from (None for root).
            summary: Summary of the abandoned path.
            details: Optional extension-specific data.
            from_hook: True if generated by extension.

        Returns:
            ID of the branch_summary entry.

        Raises:
            ValueError: If branch_from_id not found.
        """
        await self.ensure_session()

        if branch_from_id is not None and branch_from_id not in self._entries_by_id:
            raise ValueError(f"Entry {branch_from_id} not found")

        self._leaf_id = branch_from_id

        entry = BranchSummaryEntry.create(
            from_id=branch_from_id or "root",
            summary=summary,
            parent_id=branch_from_id,
            details=details,
            from_hook=from_hook,
        )
        await self._writer.write_branch_summary(entry)

        self._entries_by_id[entry.id] = entry
        self._leaf_id = entry.id

        return entry.id

    async def create_branched_session(self, leaf_id: str) -> Path | None:
        """Create a new session file containing only the path to specified leaf.

        Useful for extracting a single conversation path from a branched session.

        Args:
            leaf_id: Entry ID of the leaf to extract.

        Returns:
            Path to new session file, or None if not persisting.

        Raises:
            ValueError: If leaf_id not found.
        """
        path = await self.get_branch(leaf_id)
        if not path:
            raise ValueError(f"Entry {leaf_id} not found")

        # Create new session with just this path
        new_header = SessionHeader.create(
            provider=self.provider,
            user_id=self.user_id,
            chat_id=self.chat_id,
        )
        # Add parent_session reference
        new_header.parent_session = str(self._session_dir)

        new_session_dir = self._session_dir.parent / new_header.id
        new_writer = SessionWriter(new_session_dir)
        await new_writer.write_header(new_header)

        # Re-parent entries to form linear path
        prev_id: str | None = None
        for entry in path:
            if isinstance(entry, SessionHeader):
                continue
            # Clone entry with updated parent
            entry_dict = entry.to_dict()
            entry_dict["parent_id"] = prev_id
            # Write to new session
            await new_writer._append_context(entry_dict)
            prev_id = entry.id

        return new_session_dir


# writer.py - Add branch summary writer
class SessionWriter:

    async def write_branch_summary(self, entry: BranchSummaryEntry) -> None:
        """Write a branch summary entry to context.jsonl only.

        Args:
            entry: Branch summary entry to write.
        """
        if not self._initialized:
            await self.ensure_directory()
        await self._append_context(entry.to_dict())
```

### Effort: Large

Requires tree structure (Gap 1) and new entry type, plus changes to context building.

### Priority: High

Critical for undo/redo, edit-and-regenerate, and conversation exploration.

---

## Gap 3: File Tracking in Compaction

### What Ash is Missing

Ash compaction only stores summary and token counts (lines 329-385 in `types.py`):

```python
# types.py lines 329-385
@dataclass
class CompactionEntry:
    """Compaction entry - marks context window compression."""

    id: str
    summary: str
    tokens_before: int
    tokens_after: int
    first_kept_entry_id: str
    created_at: datetime = field(default_factory=now_utc)
    type: Literal["compaction"] = "compaction"
    # No file tracking!
```

When compaction happens, Ash loses information about which files were read or modified before the compaction boundary.

### Reference Implementation (Pi-mono)

Pi-mono tracks `readFiles` and `modifiedFiles` in compaction (lines 27-31 and 696-708 in `compaction.ts`):

```typescript
// compaction.ts lines 27-31
/** Details stored in CompactionEntry.details for file tracking */
export interface CompactionDetails {
    readFiles: string[];
    modifiedFiles: string[];
}

// compaction.ts lines 696-708 - Store file tracking in compaction result
// Compute file lists and append to summary
const { readFiles, modifiedFiles } = computeFileLists(fileOps);
summary += formatFileOperations(readFiles, modifiedFiles);

return {
    summary,
    firstKeptEntryId,
    tokensBefore,
    details: { readFiles, modifiedFiles } as CompactionDetails,
};
```

File operations are extracted from tool calls (lines 28-56 in `utils.ts`):

```typescript
// utils.ts lines 28-56
export function extractFileOpsFromMessage(message: AgentMessage, fileOps: FileOperations): void {
    if (message.role !== "assistant") return;
    // ...
    for (const block of message.content) {
        // ...
        switch (block.name) {
            case "read":
                fileOps.read.add(path);
                break;
            case "write":
                fileOps.written.add(path);
                break;
            case "edit":
                fileOps.edited.add(path);
                break;
        }
    }
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sessions/types.py`
- `/home/dcramer/src/ash/src/ash/core/compaction.py` (or create new)

### Concrete Python Code Changes

```python
# types.py - Add file tracking to CompactionEntry
from dataclasses import dataclass, field


@dataclass
class CompactionDetails:
    """Details stored in CompactionEntry for context preservation."""

    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_files": self.read_files,
            "modified_files": self.modified_files,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompactionDetails:
        return cls(
            read_files=data.get("read_files", []),
            modified_files=data.get("modified_files", []),
        )


@dataclass
class CompactionEntry:
    """Compaction entry - marks context window compression."""

    id: str
    summary: str
    tokens_before: int
    tokens_after: int
    first_kept_entry_id: str
    created_at: datetime = field(default_factory=now_utc)
    details: CompactionDetails | None = None  # File tracking
    from_hook: bool = False  # True if generated by extension
    type: Literal["compaction"] = "compaction"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "type": self.type,
            "id": self.id,
            "summary": self.summary,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "first_kept_entry_id": self.first_kept_entry_id,
            "created_at": self.created_at.isoformat(),
        }
        if self.details:
            result["details"] = self.details.to_dict()
        if self.from_hook:
            result["from_hook"] = self.from_hook
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompactionEntry:
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = now_utc()

        details = None
        if data.get("details"):
            details = CompactionDetails.from_dict(data["details"])

        return cls(
            id=data["id"],
            summary=data["summary"],
            tokens_before=data["tokens_before"],
            tokens_after=data["tokens_after"],
            first_kept_entry_id=data["first_kept_entry_id"],
            created_at=created_at,
            details=details,
            from_hook=data.get("from_hook", False),
        )


# compaction.py - New file for file tracking
from dataclasses import dataclass, field
from typing import Any

from ash.sessions.types import (
    CompactionDetails,
    Entry,
    MessageEntry,
    ToolUseEntry,
)


@dataclass
class FileOperations:
    """Tracks file operations during session."""

    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)


def extract_file_ops_from_tool_use(tool_use: ToolUseEntry, file_ops: FileOperations) -> None:
    """Extract file operations from a tool use entry.

    Args:
        tool_use: Tool use entry to extract from.
        file_ops: FileOperations to update.
    """
    path = tool_use.input.get("path") or tool_use.input.get("file_path")
    if not path or not isinstance(path, str):
        return

    tool_name = tool_use.name.lower()
    if tool_name in ("read", "read_file", "cat"):
        file_ops.read.add(path)
    elif tool_name in ("write", "write_file"):
        file_ops.written.add(path)
    elif tool_name in ("edit", "edit_file", "patch"):
        file_ops.edited.add(path)


def extract_file_ops_from_entries(
    entries: list[Entry],
    prev_compaction_details: CompactionDetails | None = None,
) -> FileOperations:
    """Extract all file operations from session entries.

    Args:
        entries: List of session entries to scan.
        prev_compaction_details: Details from previous compaction to merge.

    Returns:
        Combined FileOperations.
    """
    file_ops = FileOperations()

    # Seed from previous compaction
    if prev_compaction_details:
        for f in prev_compaction_details.read_files:
            file_ops.read.add(f)
        for f in prev_compaction_details.modified_files:
            file_ops.edited.add(f)

    # Extract from tool uses
    for entry in entries:
        if isinstance(entry, ToolUseEntry):
            extract_file_ops_from_tool_use(entry, file_ops)

    return file_ops


def compute_file_lists(file_ops: FileOperations) -> CompactionDetails:
    """Compute final file lists from file operations.

    Files that were both read and modified are only listed in modified.

    Args:
        file_ops: Accumulated file operations.

    Returns:
        CompactionDetails with read_files and modified_files.
    """
    modified = file_ops.written | file_ops.edited
    read_only = sorted(file_ops.read - modified)
    modified_sorted = sorted(modified)

    return CompactionDetails(
        read_files=read_only,
        modified_files=modified_sorted,
    )


def format_file_operations(details: CompactionDetails) -> str:
    """Format file operations as XML tags for summary.

    Args:
        details: CompactionDetails to format.

    Returns:
        XML-formatted string to append to summary.
    """
    sections: list[str] = []

    if details.read_files:
        files_str = "\n".join(details.read_files)
        sections.append(f"<read-files>\n{files_str}\n</read-files>")

    if details.modified_files:
        files_str = "\n".join(details.modified_files)
        sections.append(f"<modified-files>\n{files_str}\n</modified-files>")

    if not sections:
        return ""

    return "\n\n" + "\n\n".join(sections)


# manager.py - Update add_compaction to include file tracking
async def add_compaction(
    self,
    summary: str,
    tokens_before: int,
    tokens_after: int,
    first_kept_entry_id: str,
    details: CompactionDetails | None = None,  # Add file tracking
    from_hook: bool = False,
) -> None:
    """Record a compaction event with file tracking.

    Args:
        summary: Summary of compacted content.
        tokens_before: Token count before compaction.
        tokens_after: Token count after compaction.
        first_kept_entry_id: ID of first entry kept after compaction.
        details: File tracking details (read/modified files).
        from_hook: True if compaction was triggered by extension.
    """
    await self.ensure_session()

    entry = CompactionEntry.create(
        summary=summary,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        first_kept_entry_id=first_kept_entry_id,
        details=details,
        from_hook=from_hook,
    )
    await self._writer.write_compaction(entry)
```

### Effort: Medium

Requires new data structures but doesn't change fundamental architecture.

### Priority: High

Critical for maintaining context across long sessions. Without file tracking, the agent loses awareness of previously read/modified files after compaction.

---

## Gap 4: Custom Entry Types

### What Ash is Missing

Ash has a fixed set of entry types (lines 388-389 in `types.py`):

```python
# types.py lines 388-389
# Union type for all entry types - fixed set
Entry = SessionHeader | MessageEntry | ToolUseEntry | ToolResultEntry | CompactionEntry
```

Extensions cannot add custom entry types to store extension-specific data.

### Reference Implementation (Pi-mono)

Pi-mono supports custom entries for extension data (lines 96-100 in `session-manager.ts`):

```typescript
// session-manager.ts lines 96-100
/**
 * Custom entry for extensions to store extension-specific data in the session.
 * Use customType to identify your extension's entries.
 *
 * Does NOT participate in LLM context (ignored by buildSessionContext).
 */
export interface CustomEntry<T = unknown> extends SessionEntryBase {
    type: "custom";
    customType: string;
    data?: T;
}
```

And custom message entries that DO participate in LLM context (lines 121-127):

```typescript
// session-manager.ts lines 121-127
export interface CustomMessageEntry<T = unknown> extends SessionEntryBase {
    type: "custom_message";
    customType: string;
    content: string | (TextContent | ImageContent)[];
    details?: T;
    display: boolean;  // Controls TUI rendering
}
```

With methods to append them (lines 804-844):

```typescript
// session-manager.ts lines 804-816
appendCustomEntry(customType: string, data?: unknown): string {
    const entry: CustomEntry = {
        type: "custom",
        customType,
        data,
        id: generateId(this.byId),
        parentId: this.leafId,
        timestamp: new Date().toISOString(),
    };
    this._appendEntry(entry);
    return entry.id;
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sessions/types.py`
- `/home/dcramer/src/ash/src/ash/sessions/manager.py`
- `/home/dcramer/src/ash/src/ash/sessions/writer.py`
- `/home/dcramer/src/ash/src/ash/sessions/reader.py`

### Concrete Python Code Changes

```python
# types.py - Add custom entry types
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class CustomEntry(EntryBase, Generic[T]):
    """Custom entry for extensions to store extension-specific data.

    Does NOT participate in LLM context (ignored by load_messages_for_llm).
    Use custom_type to identify your extension's entries.
    """

    custom_type: str  # Extension identifier (e.g., "artifact-index")
    data: T | None = None  # Extension-specific payload
    type: Literal["custom"] = "custom"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "id": self.id,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "custom_type": self.custom_type,
        }
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomEntry:
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = now_utc()
        return cls(
            id=data["id"],
            parent_id=data.get("parent_id"),
            created_at=created_at,
            custom_type=data["custom_type"],
            data=data.get("data"),
        )

    @classmethod
    def create(
        cls,
        custom_type: str,
        data: T | None = None,
        parent_id: str | None = None,
    ) -> CustomEntry[T]:
        return cls(
            id=generate_id(),
            parent_id=parent_id,
            created_at=now_utc(),
            custom_type=custom_type,
            data=data,
        )


@dataclass
class CustomMessageEntry(EntryBase, Generic[T]):
    """Custom message entry for extensions to inject messages into LLM context.

    Unlike CustomEntry, this DOES participate in LLM context.
    The content is converted to a user message in load_messages_for_llm().
    """

    custom_type: str  # Extension identifier
    content: str | list[dict[str, Any]]  # Message content
    details: T | None = None  # Extension-specific metadata (not sent to LLM)
    display: bool = True  # Whether to show in TUI
    type: Literal["custom_message"] = "custom_message"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "id": self.id,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "custom_type": self.custom_type,
            "content": self.content,
            "display": self.display,
        }
        if self.details is not None:
            result["details"] = self.details
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomMessageEntry:
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = now_utc()
        return cls(
            id=data["id"],
            parent_id=data.get("parent_id"),
            created_at=created_at,
            custom_type=data["custom_type"],
            content=data["content"],
            details=data.get("details"),
            display=data.get("display", True),
        )

    @classmethod
    def create(
        cls,
        custom_type: str,
        content: str | list[dict[str, Any]],
        display: bool = True,
        details: T | None = None,
        parent_id: str | None = None,
    ) -> CustomMessageEntry[T]:
        return cls(
            id=generate_id(),
            parent_id=parent_id,
            created_at=now_utc(),
            custom_type=custom_type,
            content=content,
            details=details,
            display=display,
        )


# Update Entry union type
Entry = (
    SessionHeader
    | MessageEntry
    | ToolUseEntry
    | ToolResultEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | CustomMessageEntry
    | LabelEntry
)


# Update parse_entry
def parse_entry(data: dict[str, Any]) -> Entry:
    match data.get("type"):
        case "session":
            return SessionHeader.from_dict(data)
        case "message":
            return MessageEntry.from_dict(data)
        case "tool_use":
            return ToolUseEntry.from_dict(data)
        case "tool_result":
            return ToolResultEntry.from_dict(data)
        case "compaction":
            return CompactionEntry.from_dict(data)
        case "branch_summary":
            return BranchSummaryEntry.from_dict(data)
        case "custom":
            return CustomEntry.from_dict(data)
        case "custom_message":
            return CustomMessageEntry.from_dict(data)
        case "label":
            return LabelEntry.from_dict(data)
        case unknown:
            raise ValueError(f"Unknown entry type: {unknown}")


# manager.py - Add custom entry methods
class SessionManager:

    async def add_custom_entry(
        self,
        custom_type: str,
        data: Any = None,
    ) -> str:
        """Add a custom entry for extension data.

        Custom entries do NOT participate in LLM context.
        Use this to persist extension state across session reloads.

        Args:
            custom_type: Extension identifier for filtering on reload.
            data: Extension-specific data payload.

        Returns:
            Entry ID.
        """
        await self.ensure_session()

        entry = CustomEntry.create(
            custom_type=custom_type,
            data=data,
            parent_id=self._leaf_id,
        )
        await self._writer.write_custom(entry)

        self._entries_by_id[entry.id] = entry
        self._leaf_id = entry.id

        return entry.id

    async def add_custom_message(
        self,
        custom_type: str,
        content: str | list[dict[str, Any]],
        display: bool = True,
        details: Any = None,
    ) -> str:
        """Add a custom message that participates in LLM context.

        Args:
            custom_type: Extension identifier.
            content: Message content (string or content blocks).
            display: Whether to show in TUI.
            details: Extension-specific metadata (not sent to LLM).

        Returns:
            Entry ID.
        """
        await self.ensure_session()

        entry = CustomMessageEntry.create(
            custom_type=custom_type,
            content=content,
            display=display,
            details=details,
            parent_id=self._leaf_id,
        )
        await self._writer.write_custom_message(entry)

        self._entries_by_id[entry.id] = entry
        self._leaf_id = entry.id

        return entry.id

    async def get_custom_entries(self, custom_type: str) -> list[CustomEntry]:
        """Get all custom entries of a specific type.

        Useful for extensions to reconstruct state on session reload.

        Args:
            custom_type: Extension identifier to filter by.

        Returns:
            List of matching CustomEntry objects.
        """
        entries = await self._reader.load_entries()
        return [
            e for e in entries
            if isinstance(e, CustomEntry) and e.custom_type == custom_type
        ]
```

### Effort: Medium

New entry types but follows existing patterns.

### Priority: Medium

Enables extensibility but not required for core functionality.

---

## Gap 5: Label Entries

### What Ash is Missing

Ash has no way to bookmark or label points in a conversation. There's no equivalent to pi-mono's `LabelEntry`.

### Reference Implementation (Pi-mono)

Pi-mono has label entries for user-defined bookmarks (lines 103-107 in `session-manager.ts`):

```typescript
// session-manager.ts lines 103-107
/** Label entry for user-defined bookmarks/markers on entries. */
export interface LabelEntry extends SessionEntryBase {
    type: "label";
    targetId: string;
    label: string | undefined;
}
```

With methods to set/clear labels (lines 887-906):

```typescript
// session-manager.ts lines 887-906
/**
 * Set or clear a label on an entry.
 * Pass undefined or empty string to clear the label.
 */
appendLabelChange(targetId: string, label: string | undefined): string {
    if (!this.byId.has(targetId)) {
        throw new Error(`Entry ${targetId} not found`);
    }
    const entry: LabelEntry = {
        type: "label",
        id: generateId(this.byId),
        parentId: this.leafId,
        timestamp: new Date().toISOString(),
        targetId,
        label,
    };
    this._appendEntry(entry);
    if (label) {
        this.labelsById.set(targetId, label);
    } else {
        this.labelsById.delete(targetId);
    }
    return entry.id;
}
```

Labels are resolved and included in tree nodes (lines 959-962, 877-880):

```typescript
// session-manager.ts lines 959-962
// Create nodes with resolved labels
for (const entry of entries) {
    const label = this.labelsById.get(entry.id);
    nodeMap.set(entry.id, { entry, children: [], label });
}

// session-manager.ts lines 877-880
getLabel(id: string): string | undefined {
    return this.labelsById.get(id);
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sessions/types.py`
- `/home/dcramer/src/ash/src/ash/sessions/manager.py`

### Concrete Python Code Changes

```python
# types.py - Add LabelEntry
@dataclass
class LabelEntry(EntryBase):
    """Label entry for user-defined bookmarks/markers on entries."""

    target_id: str  # Entry being labeled
    label: str | None  # Label text (None to clear)
    type: Literal["label"] = "label"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "target_id": self.target_id,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelEntry:
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = now_utc()
        return cls(
            id=data["id"],
            parent_id=data.get("parent_id"),
            created_at=created_at,
            target_id=data["target_id"],
            label=data.get("label"),
        )

    @classmethod
    def create(
        cls,
        target_id: str,
        label: str | None,
        parent_id: str | None = None,
    ) -> LabelEntry:
        return cls(
            id=generate_id(),
            parent_id=parent_id,
            created_at=now_utc(),
            target_id=target_id,
            label=label,
        )


# manager.py - Add label methods
class SessionManager:

    def __init__(self, ...):
        # ... existing init ...
        self._labels_by_id: dict[str, str] = {}  # target_id -> label

    async def set_label(self, target_id: str, label: str | None) -> str:
        """Set or clear a label on an entry.

        Labels are user-defined markers for bookmarking/navigation.

        Args:
            target_id: Entry ID to label.
            label: Label text, or None/empty to clear.

        Returns:
            ID of the label entry.

        Raises:
            ValueError: If target_id not found.
        """
        await self.ensure_session()

        if target_id not in self._entries_by_id:
            raise ValueError(f"Entry {target_id} not found")

        entry = LabelEntry.create(
            target_id=target_id,
            label=label,
            parent_id=self._leaf_id,
        )
        await self._writer.write_label(entry)

        # Update label index
        if label:
            self._labels_by_id[target_id] = label
        else:
            self._labels_by_id.pop(target_id, None)

        self._entries_by_id[entry.id] = entry
        self._leaf_id = entry.id

        return entry.id

    def get_label(self, entry_id: str) -> str | None:
        """Get the label for an entry, if any.

        Args:
            entry_id: Entry ID to look up.

        Returns:
            Label text or None.
        """
        return self._labels_by_id.get(entry_id)

    async def get_labeled_entries(self) -> dict[str, str]:
        """Get all labeled entries.

        Returns:
            Dict mapping entry IDs to their labels.
        """
        return dict(self._labels_by_id)

    async def _build_label_index(self) -> None:
        """Build label index from entries.

        Called on session load to reconstruct label state.
        """
        entries = await self._reader.load_entries()
        self._labels_by_id.clear()

        for entry in entries:
            if isinstance(entry, LabelEntry):
                if entry.label:
                    self._labels_by_id[entry.target_id] = entry.label
                else:
                    self._labels_by_id.pop(entry.target_id, None)


# writer.py - Add label writer
class SessionWriter:

    async def write_label(self, entry: LabelEntry) -> None:
        """Write a label entry to context.jsonl only.

        Args:
            entry: Label entry to write.
        """
        if not self._initialized:
            await self.ensure_directory()
        await self._append_context(entry.to_dict())
```

### Effort: Small

Simple data structure and index.

### Priority: Low

Nice-to-have for navigation but not required for core functionality.

---

## Gap 6: Session Version Migrations

### What Ash is Missing

Ash has a version constant but no migration logic (lines 12-13 in `types.py`):

```python
# types.py lines 12-13
# Session format version - increment when breaking format changes
SESSION_VERSION = "1"
```

Old sessions cannot be migrated to new formats. Changes to entry structure would break existing sessions.

### Reference Implementation (Pi-mono)

Pi-mono has structured migrations from v1 -> v2 -> v3 (lines 27, 195-256 in `session-manager.ts`):

```typescript
// session-manager.ts line 27
export const CURRENT_SESSION_VERSION = 3;

// session-manager.ts lines 195-222
/** Migrate v1 -> v2: add id/parentId tree structure. Mutates in place. */
function migrateV1ToV2(entries: FileEntry[]): void {
    const ids = new Set<string>();
    let prevId: string | null = null;

    for (const entry of entries) {
        if (entry.type === "session") {
            entry.version = 2;
            continue;
        }

        entry.id = generateId(ids);
        entry.parentId = prevId;
        prevId = entry.id;

        // Convert firstKeptEntryIndex to firstKeptEntryId for compaction
        if (entry.type === "compaction") {
            const comp = entry as CompactionEntry & { firstKeptEntryIndex?: number };
            if (typeof comp.firstKeptEntryIndex === "number") {
                const targetEntry = entries[comp.firstKeptEntryIndex];
                if (targetEntry && targetEntry.type !== "session") {
                    comp.firstKeptEntryId = targetEntry.id;
                }
                delete comp.firstKeptEntryIndex;
            }
        }
    }
}

// session-manager.ts lines 224-240
/** Migrate v2 -> v3: rename hookMessage role to custom. Mutates in place. */
function migrateV2ToV3(entries: FileEntry[]): void {
    for (const entry of entries) {
        if (entry.type === "session") {
            entry.version = 3;
            continue;
        }

        if (entry.type === "message") {
            const msgEntry = entry as SessionMessageEntry;
            if (msgEntry.message && (msgEntry.message as { role: string }).role === "hookMessage") {
                (msgEntry.message as { role: string }).role = "custom";
            }
        }
    }
}

// session-manager.ts lines 246-256
function migrateToCurrentVersion(entries: FileEntry[]): boolean {
    const header = entries.find((e) => e.type === "session") as SessionHeader | undefined;
    const version = header?.version ?? 1;

    if (version >= CURRENT_SESSION_VERSION) return false;

    if (version < 2) migrateV1ToV2(entries);
    if (version < 3) migrateV2ToV3(entries);

    return true;
}
```

Migrations are applied on session load and the file is rewritten (lines 627-639):

```typescript
// session-manager.ts lines 627-639
setSessionFile(sessionFile: string): void {
    this.sessionFile = resolve(sessionFile);
    if (existsSync(this.sessionFile)) {
        this.fileEntries = loadEntriesFromFile(this.sessionFile);
        // ...
        if (migrateToCurrentVersion(this.fileEntries)) {
            this._rewriteFile();  // Persist migrated data
        }
        this._buildIndex();
        // ...
    }
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sessions/types.py`
- `/home/dcramer/src/ash/src/ash/sessions/reader.py`
- `/home/dcramer/src/ash/src/ash/sessions/migrations.py` (new)

### Concrete Python Code Changes

```python
# types.py - Update version tracking
SESSION_VERSION = 2  # Bump when adding tree structure


@dataclass
class SessionHeader:
    """Session header entry - first line in context.jsonl."""

    id: str
    created_at: datetime
    provider: str
    user_id: str | None = None
    chat_id: str | None = None
    version: int = SESSION_VERSION  # Change to int for easier comparison
    parent_session: str | None = None  # For branched sessions
    type: Literal["session"] = "session"


# migrations.py - New file for version migrations
"""Session format migrations."""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

CURRENT_VERSION = 2


def migrate_to_current(entries: list[dict[str, Any]]) -> bool:
    """Migrate entries to current version.

    Mutates entries in place.

    Args:
        entries: List of entry dicts from JSONL.

    Returns:
        True if any migration was applied.
    """
    # Find header
    header = next((e for e in entries if e.get("type") == "session"), None)
    if not header:
        return False

    # Parse version (handle both string and int)
    version_raw = header.get("version", "1")
    version = int(version_raw) if isinstance(version_raw, str) else version_raw

    if version >= CURRENT_VERSION:
        return False

    logger.info("Migrating session from v%d to v%d", version, CURRENT_VERSION)

    # Apply migrations in order
    if version < 2:
        _migrate_v1_to_v2(entries)

    return True


def _migrate_v1_to_v2(entries: list[dict[str, Any]]) -> None:
    """Migrate v1 -> v2: add id/parent_id tree structure.

    V1 entries don't have IDs or parent tracking.
    V2 adds tree structure with id/parent_id on all entries.
    """
    existing_ids: set[str] = set()
    prev_id: str | None = None

    for entry in entries:
        entry_type = entry.get("type")

        if entry_type == "session":
            entry["version"] = 2
            continue

        # Generate short ID if missing
        if "id" not in entry:
            new_id = _generate_short_id(existing_ids)
            entry["id"] = new_id
            existing_ids.add(new_id)

        # Add parent_id
        if "parent_id" not in entry:
            entry["parent_id"] = prev_id

        prev_id = entry["id"]

        # Migrate compaction entry format
        if entry_type == "compaction":
            # Convert index-based reference to ID-based
            if "first_kept_entry_index" in entry:
                index = entry.pop("first_kept_entry_index")
                # Find the entry at that index (skip header)
                non_header = [e for e in entries if e.get("type") != "session"]
                if 0 <= index < len(non_header):
                    entry["first_kept_entry_id"] = non_header[index].get("id", "")


def _generate_short_id(existing: set[str]) -> str:
    """Generate unique 8-char ID."""
    for _ in range(100):
        candidate = str(uuid.uuid4())[:8]
        if candidate not in existing:
            return candidate
    # Fallback to full UUID
    return str(uuid.uuid4())


# reader.py - Apply migrations on load
import json
import logging
from pathlib import Path

import aiofiles

from ash.sessions.migrations import migrate_to_current
from ash.sessions.types import parse_entry

logger = logging.getLogger(__name__)


class SessionReader:
    """Reads session entries from JSONL files."""

    async def load_entries(self) -> list[Entry]:
        """Load all entries from context.jsonl, applying migrations if needed."""
        if not self.context_file.exists():
            return []

        # Load raw dicts first
        raw_entries: list[dict[str, Any]] = []
        async with aiofiles.open(self.context_file, encoding="utf-8") as f:
            line_num = 0
            async for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Failed to parse line %d in %s: %s",
                        line_num,
                        self.context_file,
                        e,
                    )

        # Apply migrations
        if migrate_to_current(raw_entries):
            await self._rewrite_file(raw_entries)

        # Parse into typed entries
        entries: list[Entry] = []
        for data in raw_entries:
            try:
                entries.append(parse_entry(data))
            except ValueError as e:
                logger.warning("Failed to parse entry: %s", e)

        return entries

    async def _rewrite_file(self, entries: list[dict[str, Any]]) -> None:
        """Rewrite session file with migrated entries.

        Args:
            entries: Migrated entry dicts.
        """
        logger.info("Rewriting session file after migration: %s", self.context_file)

        # Write to temp file then rename
        temp_file = self.context_file.with_suffix(".jsonl.tmp")
        async with aiofiles.open(temp_file, "w", encoding="utf-8") as f:
            for entry in entries:
                line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                await f.write(line + "\n")

        # Atomic rename
        temp_file.rename(self.context_file)
```

### Effort: Medium

Migration framework is straightforward, but each migration needs careful testing.

### Priority: High

Required before making breaking changes to session format (like adding tree structure).

---

## Gap 7: Idle Timeout

### What Ash is Missing

Ash sessions never expire. There's no idle timeout configuration (lines 46-75 in `manager.py`):

```python
# manager.py lines 46-75 - No timeout handling
def __init__(
    self,
    provider: str,
    chat_id: str | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
    sessions_path: Path | None = None,
) -> None:
    """Initialize session manager.

    # No idle timeout parameter
    """
    self.provider = provider
    self.chat_id = chat_id
    # ... no timeout tracking
```

### Reference Implementation (Clawdbot)

Clawdbot has configurable idle timeout (lines 55-69, 112-114, 188-204 in `sessions.ts`):

```typescript
// sessions.ts (via types.ts lines 55-69)
export type SessionConfig = {
    scope?: SessionScope;
    resetTriggers?: string[];
    idleMinutes?: number;  // Session idle timeout
    heartbeatIdleMinutes?: number;
    store?: string;
    // ...
};

// auto-reply/reply/session.ts lines 112-114
const idleMinutes = Math.max(
    sessionCfg?.idleMinutes ?? DEFAULT_IDLE_MINUTES,
    1,
);

// auto-reply/reply/session.ts lines 188-204
const idleMs = idleMinutes * 60_000;
const freshEntry = entry && Date.now() - entry.updatedAt <= idleMs;

if (!isNewSession && freshEntry) {
    sessionId = entry.sessionId;
    systemSent = entry.systemSent ?? false;
    // ... continue existing session
} else {
    sessionId = crypto.randomUUID();
    isNewSession = true;
    systemSent = false;
    // ... start new session
}
```

Default timeout is 60 minutes (line 194 in `sessions.ts`):

```typescript
// sessions.ts line 194
export const DEFAULT_IDLE_MINUTES = 60;
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sessions/manager.py`
- `/home/dcramer/src/ash/src/ash/config/models.py`

### Concrete Python Code Changes

```python
# config/models.py - Add session config
from pydantic import BaseModel, Field


class SessionConfig(BaseModel):
    """Session configuration."""

    idle_minutes: int = Field(
        default=60,
        ge=1,
        description="Session idle timeout in minutes. Sessions older than this are considered expired."
    )
    reset_triggers: list[str] = Field(
        default=["/new", "/reset"],
        description="Commands that trigger a new session."
    )


# manager.py - Add idle timeout support
from datetime import datetime, timedelta
from typing import Any

from ash.config.models import SessionConfig

DEFAULT_IDLE_MINUTES = 60


class SessionManager:
    """Manages session lifecycle and persistence."""

    def __init__(
        self,
        provider: str,
        chat_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        sessions_path: Path | None = None,
        idle_minutes: int = DEFAULT_IDLE_MINUTES,  # Add timeout config
    ) -> None:
        """Initialize session manager.

        Args:
            provider: Provider name (e.g., "cli", "telegram", "api").
            chat_id: Optional chat/conversation ID.
            user_id: Optional user ID.
            thread_id: Optional thread ID (for forum-style chats).
            sessions_path: Override sessions directory (for testing).
            idle_minutes: Session idle timeout in minutes (default 60).
        """
        self.provider = provider
        self.chat_id = chat_id
        self.user_id = user_id
        self.thread_id = thread_id
        self.idle_minutes = max(idle_minutes, 1)  # Minimum 1 minute

        # ... rest of init ...

    def is_session_fresh(self) -> bool:
        """Check if current session is within idle timeout.

        Returns:
            True if session is fresh (not expired), False otherwise.
        """
        if self._header is None:
            return False

        # Check last message time
        return self._is_timestamp_fresh(self._header.created_at)

    async def is_session_active(self) -> bool:
        """Check if session is active (has recent activity).

        Returns:
            True if session has activity within idle timeout.
        """
        last_time = await self.get_last_message_time()
        if last_time is None:
            return False
        return self._is_timestamp_fresh(last_time)

    def _is_timestamp_fresh(self, timestamp: datetime) -> bool:
        """Check if timestamp is within idle timeout.

        Args:
            timestamp: Timestamp to check.

        Returns:
            True if within idle timeout.
        """
        idle_delta = timedelta(minutes=self.idle_minutes)
        return datetime.now(timestamp.tzinfo) - timestamp <= idle_delta

    async def ensure_session(self) -> SessionHeader:
        """Ensure session exists and is fresh, creating if needed.

        If existing session has expired (idle timeout), creates a new one.

        Returns:
            Session header.
        """
        if self._header is not None:
            # Check if session is still fresh
            if await self.is_session_active():
                return self._header
            else:
                # Session expired, create new one
                logger.info("Session expired after %d minutes idle, creating new", self.idle_minutes)
                self._header = None

        # Try to load existing
        self._header = await self._reader.load_header()
        if self._header is not None:
            # Check freshness of loaded session
            if await self.is_session_active():
                return self._header
            else:
                logger.info("Loaded session expired, creating new")
                self._header = None

        # Create new session
        self._header = SessionHeader.create(
            provider=self.provider,
            user_id=self.user_id,
            chat_id=self.chat_id,
        )
        await self._writer.write_header(self._header)
        logger.info("Created new session: %s", self._key)

        return self._header

    async def force_new_session(self) -> SessionHeader:
        """Force creation of a new session regardless of freshness.

        Use this for explicit /new or /reset commands.

        Returns:
            New session header.
        """
        self._header = SessionHeader.create(
            provider=self.provider,
            user_id=self.user_id,
            chat_id=self.chat_id,
        )
        # Create new session directory with unique ID
        self._key = session_key(self.provider, self.chat_id, self.user_id, self.thread_id)
        self._session_dir = (self._session_dir.parent / self._header.id)

        self._reader = SessionReader(self._session_dir)
        self._writer = SessionWriter(self._session_dir)

        await self._writer.write_header(self._header)
        logger.info("Forced new session: %s", self._key)

        return self._header


# Example usage in provider handlers
async def handle_message(message: str, session_config: SessionConfig) -> str:
    """Handle incoming message with session management."""

    # Check for reset triggers
    if any(message.strip().lower().startswith(t) for t in session_config.reset_triggers):
        session = SessionManager(
            provider="telegram",
            chat_id="123",
            idle_minutes=session_config.idle_minutes,
        )
        await session.force_new_session()
        return "Started new session"

    # Normal message handling with automatic expiry
    session = SessionManager(
        provider="telegram",
        chat_id="123",
        idle_minutes=session_config.idle_minutes,
    )
    await session.ensure_session()  # Auto-creates new if expired

    # ... process message ...
```

### Effort: Small

Straightforward time comparison logic.

### Priority: Medium

Important for chat providers (Telegram) where users expect conversations to reset after inactivity.

---

## Summary

| Gap | Description | Effort | Priority |
|-----|-------------|--------|----------|
| 1 | Tree-based session history | Large | High |
| 2 | Session branching | Large | High |
| 3 | File tracking in compaction | Medium | High |
| 4 | Custom entry types | Medium | Medium |
| 5 | Label entries | Small | Low |
| 6 | Session version migrations | Medium | High |
| 7 | Idle timeout | Small | Medium |

### Implementation Order

1. **Session version migrations** (Gap 6) - Required foundation for all other changes
2. **Tree-based session history** (Gap 1) - Core data model change
3. **Session branching** (Gap 2) - Builds on tree structure
4. **File tracking in compaction** (Gap 3) - Critical for long sessions
5. **Idle timeout** (Gap 7) - Quick win for chat providers
6. **Custom entry types** (Gap 4) - Enables extensibility
7. **Label entries** (Gap 5) - Nice-to-have for navigation
