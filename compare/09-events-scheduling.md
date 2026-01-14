# Events and Scheduling Systems Comparison

This document compares the events and scheduling implementations across four codebases: **ash** (Python), **archer** (TypeScript), **clawdbot** (TypeScript), and **pi-mono** (TypeScript).

## Overview

All four systems implement file-based scheduling for agent-driven task execution. The core responsibility is the same: watch for schedule definitions, compute when jobs are due, execute them at the appropriate time, and manage their lifecycle.

| Aspect | ash | archer | clawdbot | pi-mono |
|--------|-----|--------|----------|---------|
| **Language** | Python | TypeScript | TypeScript | TypeScript |
| **Storage Format** | JSONL (single file) | JSON (one file per event) | JSON (single file) | JSON (one file per event) |
| **Cron Library** | croniter | croner | croner | croner |
| **File Watching** | Polling (5s) | fs.watch + debounce | Timer-based | fs.watch + debounce |
| **Event Types** | one-shot, periodic | immediate, one-shot, periodic | at, every, cron | immediate, one-shot, periodic |
| **Execution Tracking** | last_run only | None | Full history + status | None |
| **Job Control** | Cancel by ID | Delete file | Enable/disable, force run | Delete file |
| **Concurrency** | fcntl file locking | None | Async lock chain | None |

---

## 1. ash (Python)

**Core File:** `/home/dcramer/src/ash/src/ash/events/schedule.py`

### Architecture

Ash uses a single JSONL file (`~/.ash/schedule.jsonl`) with polling-based watching. Each line is a complete JSON object representing one schedule entry.

### Event Types

```python
@dataclass
class ScheduleEntry:
    message: str
    id: str | None = None              # Stable identifier (8-char hex)
    trigger_at: datetime | None = None  # One-shot
    cron: str | None = None             # Periodic
    last_run: datetime | None = None    # For periodic
    chat_id: str | None = None
    user_id: str | None = None
    username: str | None = None
    provider: str | None = None
    created_at: datetime | None = None
```

**Supported Types:**
- **One-shot (`trigger_at`)**: Executes once at specified time, deleted after execution
- **Periodic (`cron`)**: Cron expression, `last_run` updated after each execution

### File Format

```jsonl
{"id": "a1b2c3d4", "trigger_at": "2026-01-12T09:00:00Z", "message": "Check the build", "chat_id": "123"}
{"id": "e5f6g7h8", "cron": "0 8 * * *", "message": "Daily summary", "chat_id": "456", "last_run": "2026-01-11T08:00:00Z"}
```

### Key Implementation Details

**Polling Loop:**
```python
async def _poll_loop(self) -> None:
    while self._running:
        try:
            await self._check_schedule()
        except Exception as e:
            logger.error(f"Error checking schedule: {e}")
        await asyncio.sleep(self._poll_interval)  # Default: 5 seconds
```

**File Locking:**
```python
@contextmanager
def _file_lock(self, file: IO) -> Iterator[None]:
    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
```

**Due Detection:**
```python
def is_due(self) -> bool:
    now = datetime.now(UTC)
    if self.trigger_at:
        return now >= self.trigger_at
    if self.cron:
        next_run = self._next_run_time()
        if next_run is None:
            return False
        return now >= next_run
    return False
```

**Cron Next Run Calculation:**
```python
def _next_run_time(self) -> datetime | None:
    if self.last_run:
        return croniter(self.cron, self.last_run).get_next(datetime)
    else:
        # Never run: get most recent cron occurrence (immediately due)
        return croniter(self.cron, datetime.now(UTC)).get_prev(datetime)
```

### Strengths
- Simple JSONL format is human-readable and `grep`-able
- File locking prevents race conditions
- Extra fields preserved through `_extra` dict
- Statistics via `get_stats()`

### Limitations
- Polling (5s) introduces latency
- No execution history beyond `last_run`
- No enable/disable without removing entries

---

## 2. archer (TypeScript)

**Core File:** `/home/dcramer/src/archer/src/events.ts`

### Architecture

Archer uses one JSON file per event in a `data/events/` directory. File watching with `fs.watch` provides instant detection of changes.

### Event Types

```typescript
export interface ImmediateEvent {
    type: "immediate";
    channelId: string;
    text: string;
}

export interface OneShotEvent {
    type: "one-shot";
    channelId: string;
    text: string;
    at: string; // ISO 8601 with timezone offset
}

export interface PeriodicEvent {
    type: "periodic";
    channelId: string;
    text: string;
    schedule: string; // cron syntax
    timezone: string; // IANA timezone
}
```

**Supported Types:**
- **Immediate**: Execute now (stale detection prevents old events from running)
- **One-shot**: Execute at specific time, file deleted after
- **Periodic**: Cron expression with timezone, file persists

### File Watching

```typescript
start(): void {
    if (!existsSync(this.eventsDir)) {
        mkdirSync(this.eventsDir, { recursive: true });
    }

    this.scanExisting();

    this.watcher = watch(this.eventsDir, (_eventType, filename) => {
        if (!filename || !filename.endsWith(".json")) return;
        this.debounce(filename, () => this.handleFileChange(filename));
    });
}
```

**Debouncing:**
```typescript
private debounce(filename: string, fn: () => void): void {
    const existing = this.debounceTimers.get(filename);
    if (existing) {
        clearTimeout(existing);
    }
    this.debounceTimers.set(
        filename,
        setTimeout(() => {
            this.debounceTimers.delete(filename);
            fn();
        }, DEBOUNCE_MS),  // 100ms
    );
}
```

**Retry Logic:**
```typescript
for (let i = 0; i < MAX_RETRIES; i++) {  // 3 retries
    try {
        const content = await readFile(filePath, "utf-8");
        event = this.parseEvent(content, filename);
        break;
    } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
        if (i < MAX_RETRIES - 1) {
            await this.sleep(RETRY_BASE_MS * 2 ** i);  // Exponential backoff
        }
    }
}
```

**Stale Detection for Immediate Events:**
```typescript
private handleImmediate(filename: string, event: ImmediateEvent): void {
    const stat = statSync(filePath);
    if (stat.mtimeMs < this.startTime) {
        log.logInfo(`Stale immediate event, deleting: ${filename}`);
        this.deleteFile(filename);
        return;
    }
    this.execute(filename, event);
}
```

### Strengths
- Instant file change detection via `fs.watch`
- Per-file isolation (delete file = cancel event)
- Timezone support for periodic events
- Exponential backoff on parse failures

### Limitations
- No execution history
- No job state persistence (lost on restart for in-memory timers)
- One-shot events in the past are silently deleted

---

## 3. clawdbot (TypeScript)

**Core Files:**
- `/home/dcramer/src/clawdbot/src/cron/service.ts`
- `/home/dcramer/src/clawdbot/src/cron/types.ts`
- `/home/dcramer/src/clawdbot/src/cron/schedule.ts`
- `/home/dcramer/src/clawdbot/src/cron/store.ts`

### Architecture

Clawdbot has the most sophisticated scheduling system. It uses a single JSON file with rich job metadata, execution tracking, and support for both main session and isolated agent execution.

### Type Definitions

```typescript
export type CronSchedule =
  | { kind: "at"; atMs: number }
  | { kind: "every"; everyMs: number; anchorMs?: number }
  | { kind: "cron"; expr: string; tz?: string };

export type CronSessionTarget = "main" | "isolated";
export type CronWakeMode = "next-heartbeat" | "now";

export type CronPayload =
  | { kind: "systemEvent"; text: string }
  | {
      kind: "agentTurn";
      message: string;
      model?: string;
      thinking?: string;
      timeoutSeconds?: number;
      deliver?: boolean;
      provider?: "last" | "whatsapp" | "telegram" | ...;
      to?: string;
      bestEffortDeliver?: boolean;
    };

export type CronJobState = {
  nextRunAtMs?: number;
  runningAtMs?: number;
  lastRunAtMs?: number;
  lastStatus?: "ok" | "error" | "skipped";
  lastError?: string;
  lastDurationMs?: number;
};

export type CronJob = {
  id: string;
  name: string;
  description?: string;
  enabled: boolean;
  createdAtMs: number;
  updatedAtMs: number;
  schedule: CronSchedule;
  sessionTarget: CronSessionTarget;
  wakeMode: CronWakeMode;
  payload: CronPayload;
  isolation?: CronIsolation;
  state: CronJobState;
};
```

**Supported Schedule Types:**
- **at**: One-shot at specific timestamp (milliseconds)
- **every**: Interval-based with optional anchor time
- **cron**: Standard cron expression with timezone

### Schedule Computation

```typescript
export function computeNextRunAtMs(
  schedule: CronSchedule,
  nowMs: number,
): number | undefined {
  if (schedule.kind === "at") {
    return schedule.atMs > nowMs ? schedule.atMs : undefined;
  }

  if (schedule.kind === "every") {
    const everyMs = Math.max(1, Math.floor(schedule.everyMs));
    const anchor = Math.max(0, Math.floor(schedule.anchorMs ?? nowMs));
    if (nowMs < anchor) return anchor;
    const elapsed = nowMs - anchor;
    const steps = Math.max(1, Math.floor((elapsed + everyMs - 1) / everyMs));
    return anchor + steps * everyMs;
  }

  // cron expression
  const cron = new Cron(expr, { timezone: schedule.tz, catch: false });
  const next = cron.nextRun(new Date(nowMs));
  return next ? next.getTime() : undefined;
}
```

### Stuck Job Detection

```typescript
const STUCK_RUN_MS = 2 * 60 * 60 * 1000;  // 2 hours

private recomputeNextRuns() {
    const now = this.deps.nowMs();
    for (const job of this.store.jobs) {
        const runningAt = job.state.runningAtMs;
        if (typeof runningAt === "number" && now - runningAt > STUCK_RUN_MS) {
            this.deps.log.warn({ jobId: job.id }, "cron: clearing stuck running marker");
            job.state.runningAtMs = undefined;
        }
        job.state.nextRunAtMs = this.computeJobNextRunAtMs(job, now);
    }
}
```

### Async Lock Chain

```typescript
private async locked<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.op.then(fn, fn);
    this.op = next.then(() => undefined, () => undefined);
    return (await next) as T;
}
```

### Timer-Based Execution

```typescript
private armTimer() {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (!this.deps.cronEnabled) return;

    const nextAt = this.nextWakeAtMs();
    if (!nextAt) return;

    const delay = Math.max(nextAt - this.deps.nowMs(), 0);
    const clampedDelay = Math.min(delay, MAX_TIMEOUT_MS);  // 2^31-1 to avoid overflow

    this.timer = setTimeout(() => void this.onTimer(), clampedDelay);
    this.timer.unref?.();
}
```

### File Storage with Atomic Writes

```typescript
export async function saveCronStore(storePath: string, store: CronStoreFile) {
    await fs.promises.mkdir(path.dirname(storePath), { recursive: true });
    const tmp = `${storePath}.${process.pid}.${Math.random().toString(16).slice(2)}.tmp`;
    const json = JSON.stringify(store, null, 2);
    await fs.promises.writeFile(tmp, json, "utf-8");
    await fs.promises.rename(tmp, storePath);
    try {
        await fs.promises.copyFile(storePath, `${storePath}.bak`);
    } catch { /* best-effort */ }
}
```

### Strengths
- Full execution history (status, duration, errors)
- Enable/disable jobs without deletion
- Isolated vs main session targets
- Stuck job detection and recovery
- Atomic file writes with backup
- Interval scheduling with anchors
- Event emission for monitoring

### Limitations
- More complex configuration
- Single-file storage (no per-job isolation)
- No file watching (relies on timer wake)

---

## 4. pi-mono (TypeScript)

**Core File:** `/home/dcramer/src/pi-mono/packages/mom/src/events.ts`

### Architecture

Pi-mono shares the same pattern as archer (shared heritage). Uses one JSON file per event with `fs.watch` detection.

### Event Types

Identical to archer:
```typescript
export interface ImmediateEvent {
    type: "immediate";
    channelId: string;
    text: string;
}

export interface OneShotEvent {
    type: "one-shot";
    channelId: string;
    text: string;
    at: string;
}

export interface PeriodicEvent {
    type: "periodic";
    channelId: string;
    text: string;
    schedule: string;
    timezone: string;
}
```

### Key Difference from Archer

The main difference is the bot integration:
- **archer**: `TelegramBotWrapper` integration
- **pi-mono**: `SlackBot` integration

```typescript
// pi-mono
private execute(filename: string, event: MomEvent, deleteAfter: boolean = true): void {
    const syntheticEvent: SlackEvent = {
        type: "mention",
        channel: event.channelId,
        user: "EVENT",
        text: message,
        ts: Date.now().toString(),
    };
    const enqueued = this.slack.enqueueEvent(syntheticEvent);
    // ...
}

// archer
private execute(filename: string, event: MomEvent, deleteAfter: boolean = true): void {
    const syntheticEvent: TelegramEvent = {
        type: "message",
        chatId: event.channelId,
        messageId: 0,
        user: "EVENT",
        text: message,
        ts: Math.floor(Date.now() / 1000).toString(),
    };
    const enqueued = this.bot.enqueueEvent(syntheticEvent);
    // ...
}
```

---

## Key Differences

### Storage Strategy

| Codebase | Strategy | Pros | Cons |
|----------|----------|------|------|
| **ash** | Single JSONL | Atomic, grep-able, ordered | Line-based updates needed |
| **archer/pi-mono** | One file per event | Simple lifecycle, easy cancellation | Directory scanning overhead |
| **clawdbot** | Single JSON | Rich structure, atomic | Must rewrite entire file |

### File Change Detection

| Codebase | Method | Latency | Reliability |
|----------|--------|---------|-------------|
| **ash** | Polling (5s) | Up to 5s | High (no OS dependencies) |
| **archer/pi-mono** | fs.watch + debounce | ~100ms | Platform-dependent |
| **clawdbot** | Timer-based | Precise | High |

### Execution Model

| Codebase | Post-Execution | Error Handling |
|----------|----------------|----------------|
| **ash** | Delete one-shot, update `last_run` for periodic | Log and continue, mark as processed |
| **archer/pi-mono** | Delete file (one-shot/immediate) | Delete invalid files |
| **clawdbot** | Update state, disable successful one-shots | Track `lastError`, emit events |

### Concurrency Control

| Codebase | Method |
|----------|--------|
| **ash** | `fcntl.flock()` on file |
| **archer/pi-mono** | None (single-process assumed) |
| **clawdbot** | Promise chain lock (`this.op.then()`) |

---

## Recommendations

### For Simple Agents
**archer/pi-mono pattern** works well:
- One file per event is intuitive
- Delete file = cancel event
- `fs.watch` provides fast detection

### For Production Systems
**clawdbot pattern** is more robust:
- Execution history aids debugging
- Enable/disable without losing configuration
- Stuck job detection prevents runaway states
- Atomic writes prevent corruption

### For Unix-First Design
**ash pattern** aligns with filesystem philosophy:
- JSONL is `tail -f` friendly
- File locking handles concurrent access
- Polling is simple and reliable
- Works across all platforms

### Hybrid Approach
Consider combining:
1. **Single file** (ash/clawdbot) for atomic updates and simpler backup
2. **Execution tracking** (clawdbot) for observability
3. **fs.watch** (archer) for lower latency
4. **File locking** (ash) for concurrency safety

---

## Summary

All four systems successfully implement file-based scheduling, but with different trade-offs:

- **ash**: Unix-first simplicity with JSONL and polling
- **archer/pi-mono**: Event-per-file with instant detection via fs.watch
- **clawdbot**: Enterprise-grade with full state tracking and job control

The choice depends on requirements: ash for simplicity, archer/pi-mono for per-event isolation, clawdbot for production observability.
