# Events/Scheduling Gap Analysis

This document analyzes 7 specific gaps between ash's scheduling implementation and the reference implementations in clawdbot and archer.

## Files Analyzed

**Ash:**
- `/home/dcramer/src/ash/src/ash/events/schedule.py` - Schedule watcher and entry management
- `/home/dcramer/src/ash/src/ash/sandbox/cli/commands/schedule.py` - Sandboxed CLI for creating schedules

**References:**
- `/home/dcramer/src/clawdbot/src/cron/service.ts` - Full-featured cron service with execution tracking
- `/home/dcramer/src/clawdbot/src/cron/types.ts` - Rich type definitions for cron jobs
- `/home/dcramer/src/archer/src/events.ts` - File-watching event system with immediate events

---

## Gap 1: Immediate Events

### What Ash is Missing

Ash only supports two event types: one-shot (`trigger_at`) and periodic (`cron`). There's no way to create an event that triggers immediately (ASAP). Archer has an "immediate" event type that fires as soon as it's detected.

Use cases for immediate events:
- Agent creating a follow-up task during execution
- Deferred actions that shouldn't block current processing
- Cross-session communication

Current ash code (`schedule.py` lines 59-72):
```python
def is_due(self) -> bool:
    """Check if this entry is due for execution."""
    now = datetime.now(UTC)

    if self.trigger_at:
        return now >= self.trigger_at

    if self.cron:
        next_run = self._next_run_time()
        if next_run is None:
            return False
        return now >= next_run

    return False  # No type -> never due
```

### Reference

**Best implementation:** archer (`events.ts` lines 12-16, 208-219)
```typescript
export interface ImmediateEvent {
    type: "immediate";
    channelId: string;
    text: string;
}

private handleImmediate(filename: string, event: ImmediateEvent): void {
    const filePath = join(this.eventsDir, filename);

    // Check if stale (created before harness started)
    try {
        const stat = statSync(filePath);
        if (stat.mtimeMs < this.startTime) {
            log.logInfo(`Stale immediate event, deleting: ${filename}`);
            this.deleteFile(filename);
            return;
        }
    } catch {
        return;
    }

    log.logInfo(`Executing immediate event: ${filename}`);
    this.execute(filename, event);
}
```

Note: Archer tracks `startTime` to avoid executing stale immediate events that were created before the current process started.

### Files to Modify

- `/home/dcramer/src/ash/src/ash/events/schedule.py`
- `/home/dcramer/src/ash/src/ash/sandbox/cli/commands/schedule.py`

### Proposed Changes

```python
# In schedule.py, modify ScheduleEntry:

@dataclass
class ScheduleEntry:
    """A schedule entry from the JSONL file."""

    message: str
    id: str | None = None
    trigger_at: datetime | None = None  # One-shot at specific time
    cron: str | None = None  # Periodic cron expression
    immediate: bool = False  # Execute ASAP (NEW)
    last_run: datetime | None = None
    # Context fields...
    chat_id: str | None = None
    user_id: str | None = None
    username: str | None = None
    provider: str | None = None
    created_at: datetime | None = None
    line_number: int = 0
    _extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_periodic(self) -> bool:
        return self.cron is not None

    @property
    def is_immediate(self) -> bool:
        return self.immediate

    def is_due(self) -> bool:
        """Check if this entry is due for execution."""
        now = datetime.now(UTC)

        # Immediate events are always due
        if self.immediate:
            return True

        if self.trigger_at:
            return now >= self.trigger_at

        if self.cron:
            next_run = self._next_run_time()
            if next_run is None:
                return False
            return now >= next_run

        return False

    def to_json_line(self) -> str:
        """Serialize entry back to JSON line."""
        data: dict[str, Any] = dict(self._extra)
        data["message"] = self.message

        if self.id:
            data["id"] = self.id

        if self.immediate:
            data["immediate"] = True

        if self.trigger_at:
            data["trigger_at"] = self.trigger_at.isoformat()

        # ... rest unchanged ...

    @classmethod
    def from_line(cls, line: str, line_number: int = 0) -> "ScheduleEntry | None":
        """Parse entry from JSONL line."""
        # ... existing parsing ...

        immediate = data.get("immediate", False)

        # Require at least one trigger type
        if not trigger_at and not cron and not immediate:
            return None

        return cls(
            message=message,
            id=data.get("id"),
            trigger_at=trigger_at,
            cron=cron,
            immediate=immediate,  # NEW
            last_run=last_run,
            # ... rest unchanged ...
        )
```

```python
# In schedule.py, modify ScheduleWatcher to track start time for stale detection:

class ScheduleWatcher:
    def __init__(self, schedule_file: Path, poll_interval: float = 5.0):
        self._schedule_file = schedule_file
        self._poll_interval = poll_interval
        self._handlers: list[ScheduleHandler] = []
        self._running = False
        self._task: asyncio.Task | None = None
        self._start_time: datetime | None = None  # NEW: Track when watcher started

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = datetime.now(UTC)  # NEW
        logger.info(f"Starting schedule watcher: {self._schedule_file}")
        self._task = asyncio.create_task(self._poll_loop())

    async def _check_schedule(self) -> None:
        """Check the schedule file and trigger due entries."""
        # ... existing code to read and parse ...

        due = [e for e in entries if e.is_due()]
        if not due:
            return

        triggered_one_shot: set[int] = set()
        triggered_immediate: set[int] = set()  # NEW
        updated_periodic: dict[int, ScheduleEntry] = {}

        for entry in due:
            # Skip stale immediate events (created before watcher started)
            if entry.is_immediate:
                if entry.created_at and self._start_time:
                    if entry.created_at < self._start_time:
                        logger.debug(
                            f"Skipping stale immediate event: {entry.message[:50]}..."
                        )
                        triggered_immediate.add(entry.line_number)
                        continue

            logger.info(
                f"Triggering scheduled task: {entry.message[:50]}... "
                f"(chat_id={entry.chat_id}, provider={entry.provider})"
            )
            try:
                for handler in self._handlers:
                    await handler(entry)
            except Exception as e:
                logger.error(f"Handler error for scheduled task: {e}")

            # Always mark entry as processed
            if entry.is_immediate:
                triggered_immediate.add(entry.line_number)  # Always remove immediate
            elif entry.is_periodic:
                entry.last_run = datetime.now(UTC)
                updated_periodic[entry.line_number] = entry
            else:
                triggered_one_shot.add(entry.line_number)

        # Rewrite file: remove one-shots and immediates, update periodic
        to_remove = triggered_one_shot | triggered_immediate
        if to_remove or updated_periodic:
            new_lines = []
            for i, line in enumerate(lines):
                if i in to_remove:
                    continue
                if i in updated_periodic:
                    new_lines.append(updated_periodic[i].to_json_line())
                else:
                    new_lines.append(line)

            self._write_lines(new_lines)
```

```python
# In sandbox/cli/commands/schedule.py, add --immediate flag:

@app.command()
def create(
    message: Annotated[str, typer.Argument(help="The task message/prompt to execute")],
    at: Annotated[
        str | None,
        typer.Option("--at", help="ISO 8601 UTC timestamp for one-time execution"),
    ] = None,
    cron: Annotated[
        str | None,
        typer.Option("--cron", help="Cron expression for recurring execution"),
    ] = None,
    immediate: Annotated[
        bool,
        typer.Option("--immediate", "-i", help="Execute as soon as possible"),
    ] = False,
) -> None:
    """Create a scheduled task.

    Examples:
        ash schedule create "Check the build" --at 2026-01-12T10:00:00Z
        ash schedule create "Daily status" --cron "0 8 * * *"
        ash schedule create "Follow up on this" --immediate
    """
    ctx = _require_routing_context()

    # Validate exactly one trigger type
    trigger_count = sum([bool(at), bool(cron), immediate])
    if trigger_count == 0:
        typer.echo(
            "Error: Must specify --at (one-time), --cron (recurring), or --immediate",
            err=True,
        )
        raise typer.Exit(1)

    if trigger_count > 1:
        typer.echo("Error: Specify only one of --at, --cron, or --immediate", err=True)
        raise typer.Exit(1)

    # ... existing validation for --at and --cron ...

    entry: dict = {
        "id": _generate_id(),
        "message": message,
    }

    if immediate:
        entry["immediate"] = True
    elif at:
        entry["trigger_at"] = at
    elif cron:
        entry["cron"] = cron

    # ... rest unchanged ...

    # Confirmation
    if immediate:
        typer.echo(f"Scheduled immediate task (id={entry_id}): {preview}")
    elif at:
        typer.echo(f"Scheduled one-time task (id={entry_id}) for {at}: {preview}")
    else:
        typer.echo(f"Scheduled recurring task (id={entry_id}) ({cron}): {preview}")
```

### Effort

**S** (2-3 hours) - Simple boolean flag with stale detection logic.

### Priority

**High** - Enables deferred actions and cross-session communication. Very useful for agent autonomy.

---

## Gap 2: "every" Interval Type

### What Ash is Missing

Ash only supports cron expressions for periodic tasks. Clawdbot also supports an "every" interval type with a simple millisecond duration. This is more intuitive for simple intervals like "every 2 hours" without needing to know cron syntax.

Current ash code only handles cron:
```python
if self.cron:
    next_run = self._next_run_time()
```

### Reference

**Best implementation:** clawdbot (`types.ts` lines 1-4, `schedule.ts`)
```typescript
export type CronSchedule =
  | { kind: "at"; atMs: number }
  | { kind: "every"; everyMs: number; anchorMs?: number }
  | { kind: "cron"; expr: string; tz?: string };
```

The `every` type computes next run as:
```typescript
function computeNextRunAtMs(schedule: CronSchedule, nowMs: number): number {
  if (schedule.kind === "every") {
    const { everyMs, anchorMs = 0 } = schedule;
    // Compute next occurrence after nowMs based on interval from anchor
    const elapsed = nowMs - anchorMs;
    const periods = Math.ceil(elapsed / everyMs);
    return anchorMs + periods * everyMs;
  }
  // ... cron handling
}
```

The `anchorMs` allows scheduling "every 6 hours starting at midnight" vs "every 6 hours starting now".

### Files to Modify

- `/home/dcramer/src/ash/src/ash/events/schedule.py`
- `/home/dcramer/src/ash/src/ash/sandbox/cli/commands/schedule.py`

### Proposed Changes

```python
# In schedule.py, modify ScheduleEntry:

from datetime import timedelta

# Duration parsing (simple format: "1h", "30m", "1d", "2h30m")
DURATION_PATTERN = re.compile(
    r"^(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.IGNORECASE
)


def parse_duration(duration_str: str) -> timedelta | None:
    """Parse a duration string like '1h30m' or '2d' into timedelta."""
    match = DURATION_PATTERN.match(duration_str.strip())
    if not match:
        return None

    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)

    if days == hours == minutes == seconds == 0:
        return None

    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


@dataclass
class ScheduleEntry:
    """A schedule entry from the JSONL file."""

    message: str
    id: str | None = None
    trigger_at: datetime | None = None  # One-shot at specific time
    cron: str | None = None  # Periodic cron expression
    every: str | None = None  # Interval like "1h" or "30m" (NEW)
    anchor: datetime | None = None  # Anchor time for interval (NEW)
    immediate: bool = False
    last_run: datetime | None = None
    # ... rest unchanged ...

    @property
    def is_periodic(self) -> bool:
        return self.cron is not None or self.every is not None

    def is_due(self) -> bool:
        """Check if this entry is due for execution."""
        now = datetime.now(UTC)

        if self.immediate:
            return True

        if self.trigger_at:
            return now >= self.trigger_at

        if self.every:
            next_run = self._next_interval_time()
            if next_run is None:
                return False
            return now >= next_run

        if self.cron:
            next_run = self._next_cron_time()
            if next_run is None:
                return False
            return now >= next_run

        return False

    def _next_interval_time(self) -> datetime | None:
        """Calculate next run time from interval and last_run/anchor."""
        if not self.every:
            return None

        duration = parse_duration(self.every)
        if duration is None:
            logger.warning(f"Failed to parse interval '{self.every}' for entry {self.id}")
            return None

        # Use anchor or created_at as base, defaulting to epoch if neither set
        anchor = self.anchor or self.created_at or datetime(1970, 1, 1, tzinfo=UTC)

        if self.last_run:
            # Next run after last_run
            return self.last_run + duration
        else:
            # First run: find the next interval occurrence after anchor
            now = datetime.now(UTC)
            elapsed = now - anchor
            periods = int(elapsed / duration)
            next_run = anchor + duration * (periods + 1)
            # But if we've never run and the interval has passed, run now
            if next_run > now:
                return anchor + duration * periods
            return next_run

    def _next_cron_time(self) -> datetime | None:
        """Calculate next run time from cron and last_run."""
        # ... existing _next_run_time() logic, renamed ...

    def to_json_line(self) -> str:
        """Serialize entry back to JSON line."""
        data: dict[str, Any] = dict(self._extra)
        data["message"] = self.message

        # ... existing fields ...

        if self.every:
            data["every"] = self.every
            if self.anchor:
                data["anchor"] = self.anchor.isoformat()

        # ... rest unchanged ...

    @classmethod
    def from_line(cls, line: str, line_number: int = 0) -> "ScheduleEntry | None":
        """Parse entry from JSONL line."""
        # ... existing parsing ...

        every = data.get("every")
        anchor = parse_datetime("anchor")

        # Validate interval format if present
        if every and parse_duration(every) is None:
            logger.warning(f"Invalid interval format: {every}")
            return None

        # Require at least one trigger type
        if not trigger_at and not cron and not every and not immediate:
            return None

        return cls(
            # ... existing fields ...
            every=every,
            anchor=anchor,
            # ... rest ...
        )
```

```python
# In sandbox/cli/commands/schedule.py:

@app.command()
def create(
    message: Annotated[str, typer.Argument(help="The task message/prompt to execute")],
    at: Annotated[str | None, typer.Option("--at", help="ISO 8601 UTC timestamp")] = None,
    cron: Annotated[str | None, typer.Option("--cron", help="Cron expression")] = None,
    every: Annotated[
        str | None,
        typer.Option("--every", help="Interval like '1h', '30m', '2d' (simpler than cron)"),
    ] = None,
    immediate: Annotated[bool, typer.Option("--immediate", "-i")] = False,
) -> None:
    """Create a scheduled task.

    Examples:
        ash schedule create "Check build" --at 2026-01-12T10:00:00Z
        ash schedule create "Daily status" --cron "0 8 * * *"
        ash schedule create "Hourly check" --every 1h
        ash schedule create "Every 30 minutes" --every 30m
        ash schedule create "Follow up" --immediate
    """
    ctx = _require_routing_context()

    # Validate exactly one trigger type
    trigger_count = sum([bool(at), bool(cron), bool(every), immediate])
    if trigger_count == 0:
        typer.echo(
            "Error: Must specify --at, --cron, --every, or --immediate",
            err=True,
        )
        raise typer.Exit(1)

    if trigger_count > 1:
        typer.echo("Error: Specify only one trigger type", err=True)
        raise typer.Exit(1)

    # Validate --every format
    if every:
        from ash.events.schedule import parse_duration
        if parse_duration(every) is None:
            typer.echo(
                f"Error: Invalid interval format '{every}'. "
                "Use format like '1h', '30m', '2d', '1h30m'",
                err=True,
            )
            raise typer.Exit(1)

    # ... rest of validation ...

    entry: dict = {"id": _generate_id(), "message": message}

    if immediate:
        entry["immediate"] = True
    elif at:
        entry["trigger_at"] = at
    elif cron:
        entry["cron"] = cron
    elif every:
        entry["every"] = every
        entry["anchor"] = datetime.now(UTC).isoformat()  # Anchor to creation time

    # ... rest unchanged ...

    # Confirmation
    if every:
        typer.echo(f"Scheduled recurring task (id={entry_id}) every {every}: {preview}")
```

### Effort

**M** (half day) - Duration parsing, interval calculation, and CLI changes.

### Priority

**Medium** - Nice UX improvement. Cron works but "every 1h" is much friendlier.

---

## Gap 3: Execution History Tracking

### What Ash is Missing

Ash doesn't track any execution history for scheduled tasks. When a periodic task runs, it only updates `last_run`. Clawdbot tracks comprehensive execution state:

- `lastRunAtMs` - When the job last ran
- `lastStatus` - "ok" | "error" | "skipped"
- `lastDurationMs` - How long the execution took
- `lastError` - Error message if failed
- `runningAtMs` - When currently running job started

This enables:
- Debugging failed jobs
- Understanding performance characteristics
- Detecting stuck jobs
- Retry logic based on last status

Current ash code (`schedule.py` lines 306-311):
```python
if entry.is_periodic:
    entry.last_run = datetime.now(UTC)  # Only tracks timestamp
    updated_periodic[entry.line_number] = entry
```

### Reference

**Best implementation:** clawdbot (`types.ts` lines 36-43)
```typescript
export type CronJobState = {
  nextRunAtMs?: number;
  runningAtMs?: number;
  lastRunAtMs?: number;
  lastStatus?: "ok" | "error" | "skipped";
  lastError?: string;
  lastDurationMs?: number;
};
```

And the execution tracking (`service.ts` lines 461-492):
```typescript
private async executeJob(job: CronJob, nowMs: number, opts: { forced: boolean }) {
  const startedAt = this.deps.nowMs();
  job.state.runningAtMs = startedAt;
  job.state.lastError = undefined;
  this.emit({ jobId: job.id, action: "started", runAtMs: startedAt });

  const finish = async (
    status: "ok" | "error" | "skipped",
    err?: string,
    summary?: string,
  ) => {
    const endedAt = this.deps.nowMs();
    job.state.runningAtMs = undefined;
    job.state.lastRunAtMs = startedAt;
    job.state.lastStatus = status;
    job.state.lastDurationMs = Math.max(0, endedAt - startedAt);
    job.state.lastError = err;
    // ...
  };
  // ...
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/events/schedule.py`

### Proposed Changes

```python
# In schedule.py, add execution state fields to ScheduleEntry:

from typing import Literal

ExecutionStatus = Literal["ok", "error", "skipped"]


@dataclass
class ScheduleEntry:
    """A schedule entry from the JSONL file."""

    message: str
    id: str | None = None
    trigger_at: datetime | None = None
    cron: str | None = None
    every: str | None = None
    anchor: datetime | None = None
    immediate: bool = False
    last_run: datetime | None = None  # When last executed (kept for backwards compat)

    # Execution state (NEW)
    running_at: datetime | None = None  # When currently running job started
    last_status: ExecutionStatus | None = None  # ok, error, skipped
    last_duration_ms: int | None = None  # Execution duration in milliseconds
    last_error: str | None = None  # Error message if failed

    # Context fields
    chat_id: str | None = None
    user_id: str | None = None
    username: str | None = None
    provider: str | None = None
    created_at: datetime | None = None
    line_number: int = 0
    _extra: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        """Serialize entry back to JSON line."""
        data: dict[str, Any] = dict(self._extra)
        data["message"] = self.message

        # ... existing fields ...

        # Execution state
        if self.running_at:
            data["running_at"] = self.running_at.isoformat()
        if self.last_status:
            data["last_status"] = self.last_status
        if self.last_duration_ms is not None:
            data["last_duration_ms"] = self.last_duration_ms
        if self.last_error:
            data["last_error"] = self.last_error

        # ... rest unchanged ...

    @classmethod
    def from_line(cls, line: str, line_number: int = 0) -> "ScheduleEntry | None":
        """Parse entry from JSONL line."""
        # ... existing parsing ...

        running_at = parse_datetime("running_at")
        last_status = data.get("last_status")
        if last_status and last_status not in ("ok", "error", "skipped"):
            last_status = None
        last_duration_ms = data.get("last_duration_ms")
        if last_duration_ms is not None:
            try:
                last_duration_ms = int(last_duration_ms)
            except (TypeError, ValueError):
                last_duration_ms = None
        last_error = data.get("last_error")

        return cls(
            # ... existing fields ...
            running_at=running_at,
            last_status=last_status,
            last_duration_ms=last_duration_ms,
            last_error=last_error,
            # ... rest ...
        )


# In ScheduleWatcher, track execution state:

async def _check_schedule(self) -> None:
    """Check the schedule file and trigger due entries."""
    # ... existing read and parse ...

    due = [e for e in entries if e.is_due()]
    if not due:
        return

    triggered_one_shot: set[int] = set()
    triggered_immediate: set[int] = set()
    updated_periodic: dict[int, ScheduleEntry] = {}

    for entry in due:
        # Skip if already running (shouldn't happen with proper locking)
        if entry.running_at:
            logger.warning(f"Skipping entry {entry.id}: already running since {entry.running_at}")
            continue

        # Mark as running
        start_time = datetime.now(UTC)
        entry.running_at = start_time

        # Track result
        status: ExecutionStatus = "ok"
        error_msg: str | None = None

        logger.info(
            f"Triggering scheduled task: {entry.message[:50]}... "
            f"(chat_id={entry.chat_id}, provider={entry.provider})"
        )

        try:
            for handler in self._handlers:
                await handler(entry)
        except Exception as e:
            logger.error(f"Handler error for scheduled task: {e}")
            status = "error"
            error_msg = str(e)[:500]  # Truncate long errors

        # Calculate duration
        end_time = datetime.now(UTC)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Update state
        entry.running_at = None
        entry.last_run = start_time
        entry.last_status = status
        entry.last_duration_ms = duration_ms
        if error_msg:
            entry.last_error = error_msg
        else:
            entry.last_error = None  # Clear previous error on success

        # Determine what to do with the entry
        if entry.is_immediate:
            triggered_immediate.add(entry.line_number)
        elif entry.is_periodic:
            updated_periodic[entry.line_number] = entry
        else:
            triggered_one_shot.add(entry.line_number)

    # ... rest unchanged (rewrite file) ...


# Add method to get execution stats:

def get_stats(self) -> dict[str, Any]:
    entries = self.get_entries()
    periodic_count = sum(1 for e in entries if e.is_periodic)
    due_count = sum(1 for e in entries if e.is_due())
    error_count = sum(1 for e in entries if e.last_status == "error")
    running_count = sum(1 for e in entries if e.running_at is not None)

    return {
        "running": self._running,
        "schedule_file": str(self._schedule_file),
        "total": len(entries),
        "one_shot": len(entries) - periodic_count,
        "periodic": periodic_count,
        "due": due_count,
        "errors": error_count,  # NEW
        "currently_running": running_count,  # NEW
    }
```

### Effort

**M** (half day) - Adding fields and updating execution logic.

### Priority

**High** - Essential for debugging and reliability. Without this, failures are invisible.

---

## Gap 4: Job Enable/Disable

### What Ash is Missing

Ash has no way to pause a scheduled job without deleting it. Clawdbot has an `enabled` boolean that allows temporarily disabling jobs while preserving their configuration.

Use cases:
- Pause a noisy job during debugging
- Disable a job during maintenance
- Keep job config but stop execution

Current ash: the only way to stop a job is to cancel (delete) it entirely.

### Reference

**Best implementation:** clawdbot (`types.ts` lines 45-58)
```typescript
export type CronJob = {
  id: string;
  name: string;
  description?: string;
  enabled: boolean;  // <-- Enable/disable without deletion
  createdAtMs: number;
  updatedAtMs: number;
  schedule: CronSchedule;
  // ...
};
```

And the filtering logic (`service.ts` lines 163-173):
```typescript
async list(opts?: { includeDisabled?: boolean }) {
  return await this.locked(async () => {
    await this.ensureLoaded();
    const includeDisabled = opts?.includeDisabled === true;
    const jobs = (this.store?.jobs ?? []).filter(
      (j) => includeDisabled || j.enabled,
    );
    return jobs.sort(
      (a, b) => (a.state.nextRunAtMs ?? 0) - (b.state.nextRunAtMs ?? 0),
    );
  });
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/events/schedule.py`
- `/home/dcramer/src/ash/src/ash/sandbox/cli/commands/schedule.py`

### Proposed Changes

```python
# In schedule.py, add enabled field to ScheduleEntry:

@dataclass
class ScheduleEntry:
    """A schedule entry from the JSONL file."""

    message: str
    id: str | None = None
    enabled: bool = True  # NEW: False to pause without deleting
    trigger_at: datetime | None = None
    cron: str | None = None
    every: str | None = None
    # ... rest unchanged ...

    def is_due(self) -> bool:
        """Check if this entry is due for execution."""
        # Disabled entries are never due
        if not self.enabled:
            return False

        # ... rest of existing logic ...

    def to_json_line(self) -> str:
        """Serialize entry back to JSON line."""
        data: dict[str, Any] = dict(self._extra)
        data["message"] = self.message

        if self.id:
            data["id"] = self.id

        # Only include enabled field if False (compact format)
        if not self.enabled:
            data["enabled"] = False

        # ... rest unchanged ...

    @classmethod
    def from_line(cls, line: str, line_number: int = 0) -> "ScheduleEntry | None":
        """Parse entry from JSONL line."""
        # ... existing parsing ...

        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            enabled = True

        return cls(
            message=message,
            id=data.get("id"),
            enabled=enabled,  # NEW
            # ... rest ...
        )


# In ScheduleWatcher, add enable/disable methods:

def enable_entry(self, entry_id: str) -> bool:
    """Enable a disabled entry.

    Returns True if entry was found and enabled, False otherwise.
    """
    return self._set_entry_enabled(entry_id, True)

def disable_entry(self, entry_id: str) -> bool:
    """Disable an entry without removing it.

    Returns True if entry was found and disabled, False otherwise.
    """
    return self._set_entry_enabled(entry_id, False)

def _set_entry_enabled(self, entry_id: str, enabled: bool) -> bool:
    """Set enabled state for an entry."""
    if not self._schedule_file.exists():
        return False

    lines = self._schedule_file.read_text().splitlines()
    new_lines = []
    found = False

    for line in lines:
        entry = ScheduleEntry.from_line(line)
        if entry and entry.id == entry_id:
            entry.enabled = enabled
            new_lines.append(entry.to_json_line())
            found = True
        else:
            new_lines.append(line)

    if not found:
        return False

    self._write_lines(new_lines)
    return True

def get_entries(self, include_disabled: bool = True) -> list[ScheduleEntry]:
    """Get schedule entries.

    Args:
        include_disabled: If False, only return enabled entries.
    """
    if not self._schedule_file.exists():
        return []
    lines = self._schedule_file.read_text().splitlines()
    entries = [
        entry
        for i, line in enumerate(lines)
        if (entry := ScheduleEntry.from_line(line, i)) is not None
    ]
    if not include_disabled:
        entries = [e for e in entries if e.enabled]
    return entries
```

```python
# In sandbox/cli/commands/schedule.py, add enable/disable commands:

@app.command()
def enable(
    entry_id: Annotated[
        str, typer.Option("--id", "-i", help="Entry ID to enable")
    ],
) -> None:
    """Enable a disabled scheduled task."""
    entries = _read_entries()

    found = None
    for entry in entries:
        if entry.get("id") == entry_id:
            found = entry
            break

    if not found:
        typer.echo(f"Error: No task found with ID {entry_id}", err=True)
        raise typer.Exit(1)

    # Check ownership
    user_id = os.environ.get("ASH_USER_ID")
    if user_id and found.get("user_id") != user_id:
        typer.echo(f"Error: Task {entry_id} does not belong to you", err=True)
        raise typer.Exit(1)

    if found.get("enabled", True):
        typer.echo(f"Task {entry_id} is already enabled")
        return

    found["enabled"] = True
    _write_entries(entries)

    message = found.get("message", "")
    preview = f"{message[:50]}..." if len(message) > 50 else message
    typer.echo(f"Enabled: {preview}")


@app.command()
def disable(
    entry_id: Annotated[
        str, typer.Option("--id", "-i", help="Entry ID to disable")
    ],
) -> None:
    """Disable a scheduled task without deleting it."""
    entries = _read_entries()

    found = None
    for entry in entries:
        if entry.get("id") == entry_id:
            found = entry
            break

    if not found:
        typer.echo(f"Error: No task found with ID {entry_id}", err=True)
        raise typer.Exit(1)

    # Check ownership
    user_id = os.environ.get("ASH_USER_ID")
    if user_id and found.get("user_id") != user_id:
        typer.echo(f"Error: Task {entry_id} does not belong to you", err=True)
        raise typer.Exit(1)

    if not found.get("enabled", True):
        typer.echo(f"Task {entry_id} is already disabled")
        return

    found["enabled"] = False
    _write_entries(entries)

    message = found.get("message", "")
    preview = f"{message[:50]}..." if len(message) > 50 else message
    typer.echo(f"Disabled: {preview}")


# Update list command to show enabled status:

@app.command("list")
def list_tasks(
    include_disabled: Annotated[
        bool, typer.Option("--all", "-a", help="Include disabled tasks")
    ] = False,
) -> None:
    """List scheduled tasks for the current user."""
    entries = _filter_by_user(_read_entries())

    if not include_disabled:
        entries = [e for e in entries if e.get("enabled", True)]

    if not entries:
        typer.echo("No scheduled tasks found.")
        return

    # Table with status column
    typer.echo(f"{'ID':<10} {'Status':<10} {'Type':<10} {'Schedule':<20} {'Message'}")
    typer.echo("-" * 90)

    for entry in entries:
        entry_id = entry.get("id", "?")
        status = "enabled" if entry.get("enabled", True) else "disabled"
        task_type = "periodic" if "cron" in entry or "every" in entry else "one-shot"
        message = entry.get("message", "")
        message_preview = f"{message[:30]}..." if len(message) > 30 else message

        if "cron" in entry:
            schedule = entry["cron"][:18]
        elif "every" in entry:
            schedule = f"every {entry['every']}"
        elif "trigger_at" in entry:
            schedule = entry["trigger_at"][:16]
        elif entry.get("immediate"):
            schedule = "immediate"
        else:
            schedule = "?"

        typer.echo(f"{entry_id:<10} {status:<10} {task_type:<10} {schedule:<20} {message_preview}")

    typer.echo(f"\nTotal: {len(entries)} task(s)")
```

### Effort

**S** (2-3 hours) - Simple boolean field with CLI commands.

### Priority

**Medium** - Useful for operations and debugging, but not critical.

---

## Gap 5: Stuck Job Detection

### What Ash is Missing

Ash has no mechanism to detect or recover from stuck jobs. If a handler hangs forever, the job appears to be running indefinitely. Clawdbot detects jobs that have been running for more than 2 hours and clears their running state.

This prevents:
- Jobs that are stuck forever
- Cascading failures when the watcher restarts
- Misleading "running" status for dead jobs

### Reference

**Best implementation:** clawdbot (`service.ts` lines 56, 371-391)
```typescript
const STUCK_RUN_MS = 2 * 60 * 60 * 1000;  // 2 hours

private recomputeNextRuns() {
  if (!this.store) return;
  const now = this.deps.nowMs();
  for (const job of this.store.jobs) {
    if (!job.state) job.state = {};
    if (!job.enabled) {
      job.state.nextRunAtMs = undefined;
      job.state.runningAtMs = undefined;
      continue;
    }
    const runningAt = job.state.runningAtMs;
    if (typeof runningAt === "number" && now - runningAt > STUCK_RUN_MS) {
      this.deps.log.warn(
        { jobId: job.id, runningAtMs: runningAt },
        "cron: clearing stuck running marker",
      );
      job.state.runningAtMs = undefined;
    }
    job.state.nextRunAtMs = this.computeJobNextRunAtMs(job, now);
  }
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/events/schedule.py`

### Proposed Changes

```python
# In schedule.py, add stuck detection:

from datetime import timedelta

# Maximum time a job can be "running" before we consider it stuck
STUCK_JOB_THRESHOLD = timedelta(hours=2)


class ScheduleWatcher:
    """Watches a schedule.jsonl file and triggers handlers when entries are due."""

    def __init__(self, schedule_file: Path, poll_interval: float = 5.0):
        self._schedule_file = schedule_file
        self._poll_interval = poll_interval
        self._handlers: list[ScheduleHandler] = []
        self._running = False
        self._task: asyncio.Task | None = None
        self._start_time: datetime | None = None
        self._stuck_threshold = STUCK_JOB_THRESHOLD

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = datetime.now(UTC)

        # Clean up stuck jobs on startup
        await self._clear_stuck_jobs()

        logger.info(f"Starting schedule watcher: {self._schedule_file}")
        self._task = asyncio.create_task(self._poll_loop())

    async def _clear_stuck_jobs(self) -> None:
        """Clear running state from jobs that appear stuck.

        Called on startup and periodically to recover from crashed handlers.
        """
        if not self._schedule_file.exists():
            return

        now = datetime.now(UTC)

        with self._schedule_file.open("r") as f:
            with self._file_lock(f):
                lines = f.read().splitlines()

        entries = []
        modified = False

        for i, line in enumerate(lines):
            entry = ScheduleEntry.from_line(line, i)
            if entry is None:
                entries.append(line)  # Keep unparseable lines
                continue

            # Check for stuck jobs
            if entry.running_at:
                running_duration = now - entry.running_at
                if running_duration > self._stuck_threshold:
                    logger.warning(
                        f"Clearing stuck job {entry.id}: "
                        f"running since {entry.running_at} ({running_duration})"
                    )
                    entry.running_at = None
                    entry.last_status = "error"
                    entry.last_error = f"Job stuck (running > {self._stuck_threshold})"
                    modified = True

            entries.append(entry.to_json_line() if isinstance(entry, ScheduleEntry) else line)

        if modified:
            self._write_lines([e if isinstance(e, str) else e for e in entries])

    async def _poll_loop(self) -> None:
        check_count = 0
        while self._running:
            try:
                await self._check_schedule()

                # Periodically check for stuck jobs (every 60 checks = ~5 minutes with 5s interval)
                check_count += 1
                if check_count >= 60:
                    await self._clear_stuck_jobs()
                    check_count = 0

            except Exception as e:
                logger.error(f"Error checking schedule: {e}")
            await asyncio.sleep(self._poll_interval)

    def get_stats(self) -> dict[str, Any]:
        entries = self.get_entries()
        now = datetime.now(UTC)

        periodic_count = sum(1 for e in entries if e.is_periodic)
        due_count = sum(1 for e in entries if e.is_due())
        error_count = sum(1 for e in entries if e.last_status == "error")
        running_count = sum(1 for e in entries if e.running_at is not None)

        # Check for potentially stuck jobs
        stuck_count = sum(
            1 for e in entries
            if e.running_at and (now - e.running_at) > self._stuck_threshold
        )

        return {
            "running": self._running,
            "schedule_file": str(self._schedule_file),
            "total": len(entries),
            "one_shot": len(entries) - periodic_count,
            "periodic": periodic_count,
            "due": due_count,
            "errors": error_count,
            "currently_running": running_count,
            "stuck": stuck_count,  # NEW
        }
```

### Effort

**S** (1-2 hours) - Simple threshold check on running_at.

### Priority

**Medium** - Important for reliability but rare in practice. Most handlers don't hang forever.

---

## Gap 6: File Watching vs Polling

### What Ash is Missing

Ash uses a 5-second polling interval to check for schedule changes. Archer uses `fs.watch` with debouncing for near-instant detection of new events.

Tradeoffs:
- **Polling (Ash)**: Simple, reliable across all filesystems, but up to 5s latency
- **fs.watch (Archer)**: Instant detection, but can have issues on some filesystems (network mounts, WSL, etc.)

### Reference

**Best implementation:** archer (`events.ts` lines 39-79)
```typescript
const DEBOUNCE_MS = 100;

export class EventsWatcher {
    private watcher: FSWatcher | null = null;
    private debounceTimers: Map<string, NodeJS.Timeout> = new Map();

    start(): void {
        // Ensure events directory exists
        if (!existsSync(this.eventsDir)) {
            mkdirSync(this.eventsDir, { recursive: true });
        }

        // Scan existing files
        this.scanExisting();

        // Watch for changes
        this.watcher = watch(this.eventsDir, (_eventType, filename) => {
            if (!filename || !filename.endsWith(".json")) return;
            this.debounce(filename, () => this.handleFileChange(filename));
        });
    }

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
            }, DEBOUNCE_MS),
        );
    }
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/events/schedule.py`

### Proposed Changes

```python
# In schedule.py, add optional file watching:

import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent


class ScheduleFileHandler(FileSystemEventHandler):
    """Watchdog handler for schedule file changes."""

    def __init__(self, callback: Callable[[], None], debounce_ms: int = 100):
        self._callback = callback
        self._debounce_ms = debounce_ms
        self._debounce_timer: asyncio.TimerHandle | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return
        if self._loop is None:
            return

        # Debounce: cancel previous timer and set new one
        if self._debounce_timer:
            self._debounce_timer.cancel()

        self._debounce_timer = self._loop.call_later(
            self._debounce_ms / 1000.0,
            lambda: asyncio.create_task(self._run_callback()),
        )

    async def _run_callback(self) -> None:
        try:
            self._callback()
        except Exception as e:
            logger.error(f"File watch callback error: {e}")


class ScheduleWatcher:
    """Watches a schedule.jsonl file and triggers handlers when entries are due.

    By default uses polling. Set use_file_watch=True for near-instant detection
    (requires watchdog package and may not work on all filesystems).
    """

    def __init__(
        self,
        schedule_file: Path,
        poll_interval: float = 5.0,
        use_file_watch: bool = False,
    ):
        self._schedule_file = schedule_file
        self._poll_interval = poll_interval
        self._use_file_watch = use_file_watch
        self._handlers: list[ScheduleHandler] = []
        self._running = False
        self._task: asyncio.Task | None = None
        self._start_time: datetime | None = None
        self._stuck_threshold = STUCK_JOB_THRESHOLD

        # File watching components
        self._observer: Observer | None = None
        self._file_handler: ScheduleFileHandler | None = None
        self._pending_check = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = datetime.now(UTC)

        await self._clear_stuck_jobs()

        logger.info(
            f"Starting schedule watcher: {self._schedule_file} "
            f"(file_watch={self._use_file_watch})"
        )

        if self._use_file_watch:
            self._start_file_watcher()

        self._task = asyncio.create_task(self._poll_loop())

    def _start_file_watcher(self) -> None:
        """Start the file system watcher for immediate change detection."""
        try:
            from watchdog.observers import Observer

            self._file_handler = ScheduleFileHandler(
                callback=self._on_file_changed,
                debounce_ms=100,
            )
            self._file_handler.set_loop(asyncio.get_event_loop())

            self._observer = Observer()
            self._observer.schedule(
                self._file_handler,
                str(self._schedule_file.parent),
                recursive=False,
            )
            self._observer.start()
            logger.debug(f"File watcher started for {self._schedule_file.parent}")
        except ImportError:
            logger.warning(
                "watchdog not installed, falling back to polling. "
                "Install with: pip install watchdog"
            )
            self._use_file_watch = False
        except Exception as e:
            logger.warning(f"Failed to start file watcher: {e}, falling back to polling")
            self._use_file_watch = False

    def _on_file_changed(self) -> None:
        """Called when the schedule file changes."""
        self._pending_check = True

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        # Stop file watcher
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=1.0)
            self._observer = None

        # Stop poll task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        check_count = 0
        while self._running:
            try:
                # Check if file watcher triggered a pending check
                if self._pending_check or not self._use_file_watch:
                    self._pending_check = False
                    await self._check_schedule()

                check_count += 1
                if check_count >= 60:
                    await self._clear_stuck_jobs()
                    check_count = 0

            except Exception as e:
                logger.error(f"Error checking schedule: {e}")

            # With file watching, use longer poll interval (fallback only)
            interval = self._poll_interval * 6 if self._use_file_watch else self._poll_interval
            await asyncio.sleep(interval)
```

### Effort

**M** (half day) - Requires watchdog integration and fallback logic.

### Priority

**Low** - 5-second latency is acceptable for most use cases. File watching adds complexity and potential reliability issues.

---

## Gap 7: Event Naming/Descriptions

### What Ash is Missing

Ash schedule entries only have a `message` field. Clawdbot has rich naming:

- `name` - Human-readable job name (required)
- `description` - Longer description (optional)
- `label` - Short label for UI display

This makes job listings more useful and enables better organization.

Current ash:
```json
{"message": "Check the build status and report any failures", ...}
```

Clawdbot:
```json
{
  "name": "Build Status Check",
  "description": "Monitor CI/CD pipeline and report failures to dev channel",
  "payload": {"kind": "agentTurn", "message": "Check the build..."},
  ...
}
```

### Reference

**Best implementation:** clawdbot (`types.ts` lines 45-52, `service.ts` lines 59-102)
```typescript
export type CronJob = {
  id: string;
  name: string;          // Required, human-readable name
  description?: string;  // Optional longer description
  enabled: boolean;
  // ...
};

// Name inference for legacy jobs
function inferLegacyName(job: {...}) {
  const text = job?.payload?.message ?? "";
  const firstLine = text.split("\n").map((l) => l.trim()).find(Boolean) ?? "";
  if (firstLine) return truncateText(firstLine, 60);
  // Fallback to schedule type
  if (kind === "cron") return `Cron: ${job.schedule.expr}`;
  if (kind === "every") return `Every: ${job.schedule.everyMs}ms`;
  if (kind === "at") return "One-shot";
  return "Cron job";
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/events/schedule.py`
- `/home/dcramer/src/ash/src/ash/sandbox/cli/commands/schedule.py`

### Proposed Changes

```python
# In schedule.py, add name/description fields:

@dataclass
class ScheduleEntry:
    """A schedule entry from the JSONL file."""

    message: str
    id: str | None = None
    name: str | None = None  # NEW: Human-readable name
    description: str | None = None  # NEW: Longer description
    enabled: bool = True
    # ... rest unchanged ...

    def get_display_name(self) -> str:
        """Get a display name for this entry.

        Uses name if set, otherwise infers from message or schedule.
        """
        if self.name:
            return self.name

        # Infer from message: use first line, truncated
        first_line = self.message.split("\n")[0].strip()
        if len(first_line) > 60:
            return first_line[:57] + "..."
        if first_line:
            return first_line

        # Fallback to schedule type
        if self.cron:
            return f"Cron: {self.cron}"
        if self.every:
            return f"Every {self.every}"
        if self.trigger_at:
            return f"At {self.trigger_at.isoformat()[:16]}"
        if self.immediate:
            return "Immediate"

        return "Scheduled task"

    def to_json_line(self) -> str:
        """Serialize entry back to JSON line."""
        data: dict[str, Any] = dict(self._extra)
        data["message"] = self.message

        if self.id:
            data["id"] = self.id

        # Optional naming
        if self.name:
            data["name"] = self.name
        if self.description:
            data["description"] = self.description

        # ... rest unchanged ...

    @classmethod
    def from_line(cls, line: str, line_number: int = 0) -> "ScheduleEntry | None":
        """Parse entry from JSONL line."""
        # ... existing parsing ...

        name = data.get("name")
        if name and not isinstance(name, str):
            name = None
        description = data.get("description")
        if description and not isinstance(description, str):
            description = None

        return cls(
            message=message,
            id=data.get("id"),
            name=name,
            description=description,
            # ... rest ...
        )
```

```python
# In sandbox/cli/commands/schedule.py:

@app.command()
def create(
    message: Annotated[str, typer.Argument(help="The task message/prompt to execute")],
    at: Annotated[str | None, typer.Option("--at")] = None,
    cron: Annotated[str | None, typer.Option("--cron")] = None,
    every: Annotated[str | None, typer.Option("--every")] = None,
    immediate: Annotated[bool, typer.Option("--immediate", "-i")] = False,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Human-readable name for the task"),
    ] = None,
    description: Annotated[
        str | None,
        typer.Option("--desc", help="Longer description of the task"),
    ] = None,
) -> None:
    """Create a scheduled task.

    Examples:
        ash schedule create "Check build" --at 2026-01-12T10:00:00Z
        ash schedule create "Check build" --cron "0 8 * * *" --name "Daily Build Check"
        ash schedule create "Status update" --every 1h --name "Hourly Status" --desc "Post status to channel"
    """
    # ... existing validation ...

    entry: dict = {
        "id": _generate_id(),
        "message": message,
    }

    # Optional naming
    if name:
        entry["name"] = name
    if description:
        entry["description"] = description

    # ... rest of trigger fields ...


# Update list command to show names:

@app.command("list")
def list_tasks(include_disabled: bool = False) -> None:
    """List scheduled tasks for the current user."""
    entries = _filter_by_user(_read_entries())

    if not include_disabled:
        entries = [e for e in entries if e.get("enabled", True)]

    if not entries:
        typer.echo("No scheduled tasks found.")
        return

    typer.echo(f"{'ID':<10} {'Name':<25} {'Schedule':<20} {'Status'}")
    typer.echo("-" * 80)

    for entry in entries:
        entry_id = entry.get("id", "?")

        # Get display name (from name field or infer from message)
        name = entry.get("name")
        if not name:
            msg = entry.get("message", "")
            name = msg.split("\n")[0][:22] + "..." if len(msg.split("\n")[0]) > 22 else msg.split("\n")[0]
        else:
            name = name[:22] + "..." if len(name) > 22 else name

        if "cron" in entry:
            schedule = entry["cron"][:18]
        elif "every" in entry:
            schedule = f"every {entry['every']}"
        elif "trigger_at" in entry:
            schedule = entry["trigger_at"][:16]
        elif entry.get("immediate"):
            schedule = "immediate"
        else:
            schedule = "?"

        status = "enabled" if entry.get("enabled", True) else "disabled"
        if entry.get("last_status") == "error":
            status = "error"

        typer.echo(f"{entry_id:<10} {name:<25} {schedule:<20} {status}")

    typer.echo(f"\nTotal: {len(entries)} task(s)")


# Add show command for full details:

@app.command()
def show(
    entry_id: Annotated[
        str, typer.Option("--id", "-i", help="Entry ID to show")
    ],
) -> None:
    """Show full details of a scheduled task."""
    entries = _read_entries()

    found = None
    for entry in entries:
        if entry.get("id") == entry_id:
            found = entry
            break

    if not found:
        typer.echo(f"Error: No task found with ID {entry_id}", err=True)
        raise typer.Exit(1)

    # Check ownership
    user_id = os.environ.get("ASH_USER_ID")
    if user_id and found.get("user_id") != user_id:
        typer.echo(f"Error: Task {entry_id} does not belong to you", err=True)
        raise typer.Exit(1)

    typer.echo(f"ID:          {found.get('id', '?')}")
    if found.get("name"):
        typer.echo(f"Name:        {found['name']}")
    if found.get("description"):
        typer.echo(f"Description: {found['description']}")
    typer.echo(f"Status:      {'enabled' if found.get('enabled', True) else 'disabled'}")

    if "cron" in found:
        typer.echo(f"Schedule:    cron: {found['cron']}")
    elif "every" in found:
        typer.echo(f"Schedule:    every {found['every']}")
    elif "trigger_at" in found:
        typer.echo(f"Schedule:    at {found['trigger_at']}")
    elif found.get("immediate"):
        typer.echo("Schedule:    immediate")

    typer.echo(f"\nMessage:\n{found.get('message', '')}")

    if found.get("last_run"):
        typer.echo(f"\nLast run:    {found['last_run']}")
        if found.get("last_status"):
            typer.echo(f"Last status: {found['last_status']}")
        if found.get("last_duration_ms"):
            typer.echo(f"Duration:    {found['last_duration_ms']}ms")
        if found.get("last_error"):
            typer.echo(f"Last error:  {found['last_error']}")
```

### Effort

**S** (2-3 hours) - Simple string fields with inference fallback.

### Priority

**Low** - Nice for organization but not essential. Message field works fine for most cases.

---

## Summary Table

| Gap | Description | Effort | Priority | Main Benefit |
|-----|-------------|--------|----------|--------------|
| 1 | Immediate events | S | **High** | Deferred actions, cross-session comms |
| 2 | "every" interval type | M | Medium | Simpler than cron for basic intervals |
| 3 | Execution history tracking | M | **High** | Debug failures, understand performance |
| 4 | Job enable/disable | S | Medium | Pause without delete |
| 5 | Stuck job detection | S | Medium | Recover from hung handlers |
| 6 | File watching vs polling | M | Low | Faster detection (but more complex) |
| 7 | Event naming/descriptions | S | Low | Better organization |

## Recommended Implementation Order

1. **Gap 1: Immediate events** (High priority, enables deferred actions)
2. **Gap 3: Execution history tracking** (High priority, essential for debugging)
3. **Gap 4: Job enable/disable** (Medium, quick win)
4. **Gap 5: Stuck job detection** (Medium, builds on Gap 3's running_at)
5. **Gap 2: "every" interval type** (Medium, nice UX)
6. **Gap 7: Event naming/descriptions** (Low, polish)
7. **Gap 6: File watching** (Low, complexity vs benefit tradeoff)

## Implementation Notes

### Dependencies

- Gaps 3 and 5 work together (stuck detection requires running_at tracking)
- Gap 6 requires `watchdog` package as optional dependency
- All gaps are backwards compatible with existing JSONL format

### Migration Path

All proposed changes preserve the existing JSONL format. New fields are optional and entries without them continue to work:

```json
// Old format still works:
{"trigger_at": "2026-01-12T09:00:00Z", "message": "..."}

// New format adds optional fields:
{"trigger_at": "2026-01-12T09:00:00Z", "message": "...", "name": "Build Check", "enabled": true, "last_status": "ok"}
```

### Testing Considerations

- Immediate events: test stale event detection with mocked start time
- Execution history: test error capture and duration tracking
- Stuck detection: test with mocked timestamps
- File watching: test fallback to polling when watchdog unavailable
