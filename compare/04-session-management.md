# Session Management Comparison

This document compares session management implementations across four related AI assistant codebases: **ash**, **archer**, **clawdbot**, and **pi-mono**. All share heritage from pi-coding-agent's session architecture but have evolved differently based on their use cases.

## Overview

Session management in AI assistants handles:
- **Context persistence**: Storing conversation history for LLM context reconstruction
- **Session scoping**: Determining session boundaries (per-user, per-chat, global)
- **History compaction**: Summarizing older context when approaching token limits
- **Session lifecycle**: Creation, loading, timeout/expiration, reset

## Comparison Table

| Feature | ash | archer | clawdbot | pi-mono |
|---------|-----|--------|----------|---------|
| **Language** | Python | TypeScript | TypeScript | TypeScript |
| **Format** | JSONL (dual-file) | JSONL (dual-file) | JSON store + JSONL | JSONL |
| **Branching** | No | Yes (via parentId) | Yes (forks from parent) | Full tree |
| **Compaction** | LLM summary | LLM summary | Via pi-coding-agent | LLM summary + file tracking |
| **Session Scope** | provider_chatId_userId | Channel-based | per-sender / global | cwd-based |
| **Entry Types** | 5 | Inherited from pi | Inherited from pi | 8 |
| **Idle Timeout** | No | No | Yes (idleMinutes) | No |
| **Version Migrations** | No (v1 only) | No | Via pi-coding-agent | Yes (v1->v2->v3) |

## Detailed Analysis

### 1. ash (Python)

**Core files:**
- `/home/dcramer/src/ash/src/ash/sessions/manager.py`
- `/home/dcramer/src/ash/src/ash/sessions/types.py`
- `/home/dcramer/src/ash/src/ash/sessions/writer.py`

**Architecture:**

Ash uses a dual-file JSONL approach with linear history (no branching):

```python
# Session key generation
def session_key(
    provider: str,
    chat_id: str | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
) -> str:
    parts = [_sanitize(provider)]
    if chat_id:
        parts.append(_sanitize(chat_id))
        if thread_id:
            parts.append(_sanitize(thread_id))
    elif user_id:
        parts.append(_sanitize(user_id))
    return "_".join(parts)
```

**Entry Types:**

```python
Entry = SessionHeader | MessageEntry | ToolUseEntry | ToolResultEntry | CompactionEntry
```

1. `session` - Header with metadata (id, provider, user_id, chat_id)
2. `message` - User/assistant messages with optional token counts
3. `tool_use` - Tool invocation records
4. `tool_result` - Tool execution results
5. `compaction` - Context compression markers

**Dual-File Storage:**

```python
class SessionWriter:
    """Writes session entries to JSONL files.

    Maintains two files:
    - context.jsonl: Full LLM context (all entry types)
    - history.jsonl: Human-readable conversation (messages only)
    """
```

**Compaction:**

```python
async def add_compaction(
    self,
    summary: str,           # LLM-generated summary
    tokens_before: int,
    tokens_after: int,
    first_kept_entry_id: str,  # Entry ID where context resumes
) -> None:
```

**Key Characteristics:**
- Simple linear history (no tree structure)
- Async-first design with aiofiles
- Provider-aware session scoping (supports Telegram threads)
- External ID tracking for deduplication
- Separate history file for grep-friendly searching

---

### 2. archer (TypeScript)

**Core file:** `/home/dcramer/src/archer/src/context.ts`

**Architecture:**

Archer adapts pi-coding-agent's session manager for Telegram channel-based storage:

```typescript
export class MomSessionManager {
    private sessionId: string;
    private contextFile: string;
    private logFile: string;
    private channelDir: string;
    private flushed: boolean = false;
    private inMemoryEntries: FileEntry[] = [];
    private leafId: string | null = null;
```

**Tree Structure (inherited):**

```typescript
private _createEntryBase(): Omit<SessionEntryBase, "type"> {
    const id = uuidv4();
    const base = {
        id,
        parentId: this.leafId,  // Links to parent entry
        timestamp: new Date().toISOString(),
    };
    this.leafId = id;
    return base;
}
```

**Log-to-Context Sync:**

A unique feature that syncs user messages from `log.jsonl` to `context.jsonl`:

```typescript
/**
 * Sync user messages from log.jsonl that aren't in context.jsonl.
 * Handles:
 * - Messages that arrived while archer was offline
 * - Messages that arrived while archer was processing a previous turn
 */
syncFromLog(excludeTs?: string): void {
    // Build set of timestamps already in context
    const contextTimestamps = new Set<string>();
    const contextMessageTexts = new Set<string>();

    // ... deduplication logic ...

    // Add missing messages to context
    for (const { timestamp, message } of newMessages) {
        const entry: SessionMessageEntry = {
            type: "message",
            id,
            parentId: this.leafId,
            timestamp,
            message,
        };
        this.leafId = id;
        this.inMemoryEntries.push(entry);
        appendFileSync(this.contextFile, `${JSON.stringify(entry)}\n`);
    }
}
```

**Branching (Disabled):**

```typescript
/** Not used by mom but required by AgentSession interface */
createBranchedSession(_leafId: string): string | null {
    return null; // Mom doesn't support branching
}
```

**Key Characteristics:**
- Inherits tree structure but doesn't use branching
- Per-channel context persistence (single file per chat)
- Offline message sync from log file
- Deferred flush (only persists after first assistant response)

---

### 3. clawdbot (TypeScript)

**Core files:**
- `/home/dcramer/src/clawdbot/src/config/sessions.ts`
- `/home/dcramer/src/clawdbot/src/auto-reply/reply/session.ts`

**Architecture:**

Clawdbot uses a JSON store for session metadata with JSONL transcripts:

```typescript
export type SessionEntry = {
    sessionId: string;
    updatedAt: number;
    sessionFile?: string;
    spawnedBy?: string;          // Parent session for sub-agents
    systemSent?: boolean;
    abortedLastRun?: boolean;
    chatType?: SessionChatType;  // "direct" | "group" | "room"
    thinkingLevel?: string;
    // ... many more fields for runtime state
};
```

**Session Scoping:**

```typescript
export type SessionScope = "per-sender" | "global";

export function resolveSessionKey(
    scope: SessionScope,
    ctx: MsgContext,
    mainKey?: string,
) {
    const explicit = ctx.SessionKey?.trim();
    if (explicit) return explicit;
    // ... group resolution logic ...
    const isGroup = raw.startsWith("group:") || ...;
    if (!isGroup) return canonical;  // Direct chats collapse to main key
    return `agent:${DEFAULT_AGENT_ID}:${raw}`;
}
```

**Idle Timeout:**

```typescript
const idleMinutes = Math.max(
    sessionCfg?.idleMinutes ?? DEFAULT_IDLE_MINUTES,  // Default: 60
    1,
);
const idleMs = idleMinutes * 60_000;
const freshEntry = entry && Date.now() - entry.updatedAt <= idleMs;

if (!isNewSession && freshEntry) {
    sessionId = entry.sessionId;
    systemSent = entry.systemSent ?? false;
} else {
    sessionId = crypto.randomUUID();
    isNewSession = true;
}
```

**Session Forking:**

```typescript
function forkSessionFromParent(params: {
    parentEntry: SessionEntry;
}): { sessionId: string; sessionFile: string } | null {
    const manager = SessionManager.open(parentSessionFile);
    const leafId = manager.getLeafId();
    if (leafId) {
        const sessionFile = manager.createBranchedSession(leafId)
            ?? manager.getSessionFile();
        return { sessionId, sessionFile };
    }
}
```

**Group Session Resolution:**

```typescript
export function resolveGroupSessionKey(ctx: MsgContext): GroupKeyResolution | null {
    // Handles: group:, @g.us (WhatsApp), :group:, :channel:
    // Normalizes to: ${provider}:${kind}:${id}
    // Returns legacy key for migration
}
```

**Key Characteristics:**
- Separate session metadata store (sessions.json) from transcripts
- Rich session state (thinking level, model overrides, queue settings)
- Idle-based session expiration
- Multi-provider support with provider-specific group handling
- Session forking for sub-agent spawning
- Reset triggers ("/new", "/reset")

---

### 4. pi-mono (TypeScript)

**Core file:** `/home/dcramer/src/pi-mono/packages/coding-agent/src/core/session-manager.ts`

**Architecture:**

Pi-mono has the most sophisticated session system with full tree branching:

```typescript
export class SessionManager {
    private sessionId: string = "";
    private sessionFile: string | undefined;
    private sessionDir: string;
    private cwd: string;
    private persist: boolean;
    private flushed: boolean = false;
    private fileEntries: FileEntry[] = [];
    private byId: Map<string, SessionEntry> = new Map();
    private labelsById: Map<string, string> = new Map();
    private leafId: string | null = null;
```

**Entry Types:**

```typescript
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

1. `message` - User/assistant/custom messages
2. `thinking_level_change` - Thinking mode changes
3. `model_change` - Model switches
4. `compaction` - Context compression with file tracking
5. `branch_summary` - Summary of abandoned branches
6. `custom` - Extension data storage
7. `custom_message` - Extension messages in LLM context
8. `label` - User bookmarks on entries

**Tree Traversal:**

```typescript
/**
 * Build the session context from entries using tree traversal.
 * If leafId is provided, walks from that entry to root.
 */
export function buildSessionContext(
    entries: SessionEntry[],
    leafId?: string | null,
    byId?: Map<string, SessionEntry>,
): SessionContext {
    // Walk from leaf to root, collecting path
    const path: SessionEntry[] = [];
    let current: SessionEntry | undefined = leaf;
    while (current) {
        path.unshift(current);
        current = current.parentId ? byId.get(current.parentId) : undefined;
    }
    // ... compaction handling ...
}
```

**Branching:**

```typescript
/**
 * Start a new branch from an earlier entry.
 * Moves the leaf pointer to the specified entry.
 */
branch(branchFromId: string): void {
    if (!this.byId.has(branchFromId)) {
        throw new Error(`Entry ${branchFromId} not found`);
    }
    this.leafId = branchFromId;
}

/**
 * Start a new branch with a summary of the abandoned path.
 */
branchWithSummary(branchFromId: string | null, summary: string, ...): string {
    this.leafId = branchFromId;
    const entry: BranchSummaryEntry = {
        type: "branch_summary",
        fromId: branchFromId ?? "root",
        summary,
        // ...
    };
    this._appendEntry(entry);
    return entry.id;
}
```

**Compaction with File Tracking:**

```typescript
export interface CompactionEntry<T = unknown> extends SessionEntryBase {
    type: "compaction";
    summary: string;
    firstKeptEntryId: string;
    tokensBefore: number;
    details?: T;           // e.g., { readFiles: [], modifiedFiles: [] }
    fromHook?: boolean;    // Extension-generated vs pi-generated
}
```

**Version Migrations:**

```typescript
export const CURRENT_SESSION_VERSION = 3;

function migrateToCurrentVersion(entries: FileEntry[]): boolean {
    const version = header?.version ?? 1;
    if (version >= CURRENT_SESSION_VERSION) return false;

    if (version < 2) migrateV1ToV2(entries);  // Add id/parentId tree
    if (version < 3) migrateV2ToV3(entries);  // Rename hookMessage -> custom

    return true;
}
```

**Key Characteristics:**
- Full tree structure with parent/child relationships
- Branch-and-summarize for conversation exploration
- Labels for user bookmarking
- Custom entries for extension state persistence
- Version migrations (v1->v2->v3)
- Session listing across all project directories
- cwd-based session directory encoding

---

## Key Differences

### 1. History Model

| Codebase | Model | Complexity |
|----------|-------|------------|
| ash | Linear | Simple append-only |
| archer | Linear (tree capable) | Has parentId but no branching UI |
| clawdbot | Linear with forks | Can fork for sub-agents |
| pi-mono | Full tree | Navigate/branch anywhere |

### 2. Session Scoping

| Codebase | Scope Strategy |
|----------|---------------|
| ash | `provider_chatId_userId_threadId` |
| archer | Per-Telegram channel directory |
| clawdbot | per-sender (default) or global, with group normalization |
| pi-mono | cwd-encoded directory (per-project) |

### 3. File Organization

| Codebase | Files per Session |
|----------|-------------------|
| ash | `context.jsonl` + `history.jsonl` |
| archer | `context.jsonl` + `log.jsonl` |
| clawdbot | `sessions.json` (metadata) + `{sessionId}.jsonl` |
| pi-mono | Single `{timestamp}_{sessionId}.jsonl` |

### 4. Compaction Details

| Codebase | Compaction Features |
|----------|-------------------|
| ash | summary, tokens_before, tokens_after, first_kept_entry_id |
| archer | Inherited from pi (summary, firstKeptEntryId, tokensBefore, details) |
| clawdbot | Delegated to pi-coding-agent |
| pi-mono | summary, firstKeptEntryId, tokensBefore, details (readFiles, modifiedFiles), fromHook |

---

## Recommendations for ash

### 1. Consider Tree Structure for Future Features

Pi-mono's tree structure enables powerful features like:
- Re-editing earlier messages without losing history
- Exploring alternative conversation paths
- Summarizing abandoned branches

If ash needs to support conversation exploration or multi-turn editing, adopting `parentId` linking would be valuable.

### 2. Add File Tracking to Compaction

Pi-mono tracks `readFiles` and `modifiedFiles` in compaction entries:

```python
# Potential enhancement for ash
@dataclass
class CompactionEntry:
    # ... existing fields ...
    read_files: list[str] | None = None      # Files referenced before compaction
    modified_files: list[str] | None = None  # Files modified before compaction
```

This helps the LLM understand project context after compaction.

### 3. Consider Idle Timeout

Clawdbot's idle timeout (`idleMinutes`, default 60) provides natural session boundaries:

```python
# Potential enhancement for ash
async def should_start_new_session(self, idle_minutes: int = 60) -> bool:
    last_message_time = await self.get_last_message_time()
    if last_message_time is None:
        return True
    idle_threshold = timedelta(minutes=idle_minutes)
    return datetime.now(UTC) - last_message_time > idle_threshold
```

### 4. Session Metadata Store

Clawdbot's separate metadata store enables:
- Fast session listing without parsing all JSONL files
- Rich session state (thinking level, model overrides)
- Session-level configuration persistence

Consider if ash would benefit from similar metadata caching.

### 5. Version Migration Support

Pi-mono's migration system ensures forward compatibility:

```python
# Potential pattern for ash
SESSION_VERSION = "1"

def migrate_to_current_version(entries: list[Entry]) -> bool:
    header = next((e for e in entries if isinstance(e, SessionHeader)), None)
    version = int(header.version) if header else 1

    if version >= int(SESSION_VERSION):
        return False

    # Add migrations as format evolves
    # if version < 2: migrate_v1_to_v2(entries)

    return True
```

### 6. Log Sync Pattern

Archer's `syncFromLog()` pattern is valuable for:
- Catching up on messages received while offline
- Handling messages received during processing

This could be useful for ash's Telegram integration where messages can arrive while the bot is processing.
