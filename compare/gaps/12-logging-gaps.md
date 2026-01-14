# Logging and Observability Gap Analysis

## Summary

Ash has a solid foundation for logging with JSONL file output and Rich console integration. However, it lacks several features present in the reference implementations (Clawdbot and Archer) that would improve operational visibility, security, and developer experience.

**Key Gaps:**
1. Secret redaction in logs (HIGH priority - security)
2. Subsystem color coding (MEDIUM - developer UX)
3. Console capture/interception (LOW - completeness)
4. Log file auto-pruning (MEDIUM - operational)
5. Usage summary formatting (LOW - visibility)
6. Multiple output styles (LOW - flexibility)
7. Configurable console vs file levels (MEDIUM - operational)

---

## Gap 1: Secret Redaction Patterns

### Description

Clawdbot automatically redacts sensitive information (API keys, tokens, passwords, PEM blocks) from logs before they're written. Ash has no redaction mechanism, meaning secrets can appear in JSONL logs and console output.

### Why It Matters

- Log files may be shared for debugging or stored in less secure locations
- Sentry breadcrumbs could capture sensitive data
- JSONL session transcripts include tool outputs that may contain secrets

### Reference Implementation

**File:** `/home/dcramer/src/clawdbot/src/logging/redact.ts`

Clawdbot's approach:
- Configurable redaction mode (`off`, `tools`)
- Default patterns for common secret formats (API keys, tokens, passwords)
- Pattern matching via regex with configurable patterns
- Partial masking to preserve debuggability (e.g., `sk-abc1...xyz9`)
- Special handling for PEM blocks

Key patterns matched:
```typescript
// ENV-style: API_KEY=secret123
// JSON fields: "apiKey": "secret"
// CLI flags: --token secret
// Bearer tokens
// Provider-specific prefixes: sk-, ghp_, xox[baprs]-, etc.
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/ash/logging.py` | Add `SecretRedactor` class, integrate into handlers |
| `src/ash/config/models.py` | Add `LoggingConfig.redact_patterns` option |
| `tests/test_logging.py` | Test redaction patterns |

### Implementation

```python
# src/ash/logging.py

import re
from dataclasses import dataclass, field

# Default patterns for secret detection
DEFAULT_REDACT_PATTERNS: list[str] = [
    # ENV-style assignments: API_KEY=secret
    r'\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)\b\s*[=:]\s*(["\']?)([^\s"\'\\]+)\1',
    # JSON fields
    r'"(?:apiKey|token|secret|password|passwd|accessToken|refreshToken)"\s*:\s*"([^"]+)"',
    # CLI flags
    r'--(?:api[-_]?key|token|secret|password|passwd)\s+(["\']?)([^\s"\']+)\1',
    # Authorization headers
    r'Authorization\s*[:=]\s*Bearer\s+([A-Za-z0-9._\-+=]+)',
    r'\bBearer\s+([A-Za-z0-9._\-+=]{18,})\b',
    # PEM blocks
    r'-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----',
    # Common token prefixes
    r'\b(sk-[A-Za-z0-9_-]{8,})\b',           # OpenAI/Anthropic
    r'\b(ghp_[A-Za-z0-9]{20,})\b',           # GitHub PAT
    r'\b(github_pat_[A-Za-z0-9_]{20,})\b',   # GitHub fine-grained PAT
    r'\b(xox[baprs]-[A-Za-z0-9-]{10,})\b',   # Slack
    r'\b(xapp-[A-Za-z0-9-]{10,})\b',         # Slack app
    r'\b(gsk_[A-Za-z0-9_-]{10,})\b',         # Groq
    r'\b(AIza[0-9A-Za-z\-_]{20,})\b',        # Google API
    r'\b(npm_[A-Za-z0-9]{10,})\b',           # npm
    r'\b(\d{6,}:[A-Za-z0-9_-]{20,})\b',      # Telegram bot tokens
]

REDACT_MIN_LENGTH = 18
REDACT_KEEP_START = 6
REDACT_KEEP_END = 4


@dataclass
class SecretRedactor:
    """Redacts sensitive information from log messages."""

    patterns: list[re.Pattern[str]] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.patterns:
            self.patterns = [
                re.compile(p, re.IGNORECASE) for p in DEFAULT_REDACT_PATTERNS
            ]

    def _mask_token(self, token: str) -> str:
        """Mask a token, preserving start/end for debugging."""
        if len(token) < REDACT_MIN_LENGTH:
            return "***"
        start = token[:REDACT_KEEP_START]
        end = token[-REDACT_KEEP_END:]
        return f"{start}...{end}"

    def _redact_pem(self, block: str) -> str:
        """Redact PEM block content."""
        lines = block.strip().split("\n")
        if len(lines) < 2:
            return "***"
        return f"{lines[0]}\n...redacted...\n{lines[-1]}"

    def redact(self, text: str) -> str:
        """Redact sensitive information from text."""
        if not self.enabled or not text:
            return text

        result = text
        for pattern in self.patterns:
            def replacer(match: re.Match[str]) -> str:
                full = match.group(0)
                if "PRIVATE KEY" in full:
                    return self._redact_pem(full)
                # Get the last capturing group (the actual secret)
                groups = [g for g in match.groups() if g]
                token = groups[-1] if groups else full
                masked = self._mask_token(token)
                if token == full:
                    return masked
                return full.replace(token, masked)

            result = pattern.sub(replacer, result)

        return result


# Module-level redactor instance
_redactor: SecretRedactor | None = None


def get_redactor() -> SecretRedactor:
    """Get or create the global redactor instance."""
    global _redactor
    if _redactor is None:
        _redactor = SecretRedactor()
    return _redactor


def configure_redactor(enabled: bool = True, patterns: list[str] | None = None) -> None:
    """Configure the global redactor."""
    global _redactor
    compiled = [re.compile(p, re.IGNORECASE) for p in (patterns or DEFAULT_REDACT_PATTERNS)]
    _redactor = SecretRedactor(patterns=compiled, enabled=enabled)
```

Update `JSONLHandler.emit()` to use redaction:

```python
def emit(self, record: logging.LogRecord) -> None:
    """Write a log record as JSON."""
    try:
        redactor = get_redactor()
        message = redactor.redact(record.getMessage())

        # ... rest of emit logic using redacted message
```

### Effort

Medium (2-3 hours)

### Priority

**HIGH** - Security concern. Secrets in logs are a real risk.

---

## Gap 2: Subsystem Color Coding

### Description

Clawdbot assigns distinct colors to different subsystems (memory, tools, providers, etc.) for visual differentiation in console output. Ash's Rich handler uses monotone formatting.

### Why It Matters

- Quickly identify which subsystem generated a log entry
- Scan logs visually for specific components during debugging
- Reduce cognitive load when monitoring server output

### Reference Implementation

**File:** `/home/dcramer/src/clawdbot/src/logging.ts` (lines 416-446)

```typescript
const SUBSYSTEM_COLORS = [
  "cyan",
  "green",
  "yellow",
  "blue",
  "magenta",
  "red",
] as const;

const SUBSYSTEM_COLOR_OVERRIDES: Record<string, typeof SUBSYSTEM_COLORS[number]> = {
  "gmail-watcher": "blue",
};

function pickSubsystemColor(color: ChalkInstance, subsystem: string): ChalkInstance {
  const override = SUBSYSTEM_COLOR_OVERRIDES[subsystem];
  if (override) return color[override];
  // Hash-based color selection for consistency
  let hash = 0;
  for (let i = 0; i < subsystem.length; i += 1) {
    hash = (hash * 31 + subsystem.charCodeAt(i)) | 0;
  }
  const idx = Math.abs(hash) % SUBSYSTEM_COLORS.length;
  return color[SUBSYSTEM_COLORS[idx]];
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/ash/logging.py` | Add `SubsystemColorFormatter`, color mapping |

### Implementation

```python
# src/ash/logging.py

from rich.text import Text

SUBSYSTEM_COLORS = ["cyan", "green", "yellow", "blue", "magenta", "red"]

# Override specific subsystems for consistency
SUBSYSTEM_COLOR_OVERRIDES: dict[str, str] = {
    "providers": "blue",
    "tools": "yellow",
    "memory": "green",
    "core": "cyan",
    "sessions": "magenta",
    "scheduler": "red",
}


def _hash_subsystem(name: str) -> int:
    """Generate consistent hash for subsystem name."""
    h = 0
    for char in name:
        h = (h * 31 + ord(char)) & 0xFFFFFFFF
    return h


def get_subsystem_color(subsystem: str) -> str:
    """Get consistent color for a subsystem."""
    if subsystem in SUBSYSTEM_COLOR_OVERRIDES:
        return SUBSYSTEM_COLOR_OVERRIDES[subsystem]
    idx = _hash_subsystem(subsystem) % len(SUBSYSTEM_COLORS)
    return SUBSYSTEM_COLORS[idx]


class ColoredComponentFormatter(logging.Formatter):
    """Formatter that colorizes component names."""

    def format(self, record: logging.LogRecord) -> str:
        # Extract component from logger name
        parts = record.name.split(".")
        if len(parts) >= 2 and parts[0] == "ash":
            component = parts[1]
        else:
            component = parts[0]

        record.component = component
        record.component_color = get_subsystem_color(component)
        return super().format(record)
```

Update Rich handler configuration:

```python
if use_rich:
    from rich.logging import RichHandler

    console_handler = RichHandler(
        rich_tracebacks=False,
        show_path=False,
        show_time=True,
        markup=True,
        highlighter=None,  # Disable default highlighting
    )
    # Use color formatter
    console_handler.setFormatter(
        ColoredComponentFormatter("[%(component_color)s]%(component)s[/] | %(message)s")
    )
```

### Effort

Low (1-2 hours)

### Priority

**MEDIUM** - Developer experience improvement.

---

## Gap 3: Console Capture

### Description

Clawdbot intercepts all `console.*` calls to ensure they're captured in log files. This catches output from third-party libraries or stray print statements. Ash relies on standard logging which may miss some output.

### Why It Matters

- Third-party libraries may use print() or console output
- Ensures complete capture of all output for debugging
- Prevents information from being lost when running in server mode

### Reference Implementation

**File:** `/home/dcramer/src/clawdbot/src/logging.ts` (lines 305-376)

```typescript
export function enableConsoleCapture(): void {
  if (consolePatched) return;
  consolePatched = true;

  const logger = getLogger();
  const original = {
    log: console.log,
    info: console.info,
    warn: console.warn,
    error: console.error,
  };

  const forward = (level: Level, orig: (...args: unknown[]) => void) =>
    (...args: unknown[]) => {
      const formatted = util.format(...args);
      // Log to file via pino
      logger[level](formatted);
      // Also emit to original console
      orig.apply(console, args);
    };

  console.log = forward("info", original.log);
  console.info = forward("info", original.info);
  console.warn = forward("warn", original.warn);
  console.error = forward("error", original.error);
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/ash/logging.py` | Add `capture_print_statements()` function |

### Implementation

```python
# src/ash/logging.py

import builtins
import sys
from io import StringIO

_original_print: Any = None
_print_captured = False


class PrintCapture:
    """Captures print() calls and logs them."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        # Capture output
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        file = kwargs.get("file", sys.stdout)

        # Build message
        message = sep.join(str(arg) for arg in args)

        # Log it (without end newline)
        if message.strip():
            self.logger.log(self.level, message.rstrip())

        # Also emit to original destination
        if _original_print:
            _original_print(*args, **kwargs)


def capture_print_statements(logger_name: str = "ash.print") -> None:
    """Redirect print() calls to logging.

    Call this early in server startup to ensure all print() calls
    are captured in log files.
    """
    global _original_print, _print_captured

    if _print_captured:
        return

    _original_print = builtins.print
    _print_captured = True

    logger = logging.getLogger(logger_name)
    builtins.print = PrintCapture(logger)


def restore_print() -> None:
    """Restore original print() function."""
    global _original_print, _print_captured

    if _original_print:
        builtins.print = _original_print
        _original_print = None
    _print_captured = False
```

### Effort

Low (1 hour)

### Priority

**LOW** - Python's logging is generally well-adopted. Most libraries use it correctly.

---

## Gap 4: Log File Auto-Pruning

### Description

Clawdbot automatically deletes log files older than 24 hours to prevent unbounded disk usage. Ash's JSONL logs in `~/.ash/logs/` grow indefinitely.

### Why It Matters

- Long-running servers accumulate significant log data
- Users shouldn't need to manually manage log rotation
- Prevents disk space issues in production deployments

### Reference Implementation

**File:** `/home/dcramer/src/clawdbot/src/logging.ts` (lines 635-659)

```typescript
const MAX_LOG_AGE_MS = 24 * 60 * 60 * 1000; // 24h

function pruneOldRollingLogs(dir: string): void {
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    const cutoff = Date.now() - MAX_LOG_AGE_MS;
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      if (!entry.name.startsWith(`${LOG_PREFIX}-`) || !entry.name.endsWith(LOG_SUFFIX))
        continue;
      const fullPath = path.join(dir, entry.name);
      try {
        const stat = fs.statSync(fullPath);
        if (stat.mtimeMs < cutoff) {
          fs.rmSync(fullPath, { force: true });
        }
      } catch {
        // ignore errors during pruning
      }
    }
  } catch {
    // ignore missing dir or read errors
  }
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/ash/logging.py` | Add `prune_old_logs()`, call from `JSONLHandler` |
| `src/ash/config/models.py` | Add `LoggingConfig.retention_days` option |

### Implementation

```python
# src/ash/logging.py

from datetime import timedelta

DEFAULT_LOG_RETENTION_DAYS = 7


def prune_old_logs(
    logs_dir: Path,
    retention_days: int = DEFAULT_LOG_RETENTION_DAYS,
    prefix: str = "",
    suffix: str = ".jsonl",
) -> int:
    """Delete log files older than retention period.

    Args:
        logs_dir: Directory containing log files.
        retention_days: Days to keep logs.
        prefix: File prefix to match (empty = all).
        suffix: File suffix to match.

    Returns:
        Number of files deleted.
    """
    if not logs_dir.exists():
        return 0

    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = 0

    try:
        for entry in logs_dir.iterdir():
            if not entry.is_file():
                continue
            if prefix and not entry.name.startswith(prefix):
                continue
            if suffix and not entry.name.endswith(suffix):
                continue

            try:
                # Use mtime for age check
                mtime = datetime.fromtimestamp(entry.stat().st_mtime, UTC)
                if mtime < cutoff:
                    entry.unlink()
                    deleted += 1
            except OSError:
                # Ignore errors on individual files
                pass
    except OSError:
        # Ignore errors reading directory
        pass

    return deleted


class JSONLHandler(logging.Handler):
    """Handler that writes structured log entries to a JSONL file."""

    def __init__(
        self,
        logs_dir: Path,
        retention_days: int = DEFAULT_LOG_RETENTION_DAYS,
    ):
        super().__init__()
        self._logs_dir = logs_dir
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days
        self._current_date: str | None = None
        self._file: TextIO | None = None
        self._pruned_today = False

    def _get_log_file(self) -> TextIO:
        """Get the current log file, rotating daily."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._current_date != today or self._file is None:
            if self._file:
                self._file.close()
            self._current_date = today
            log_path = self._logs_dir / f"{today}.jsonl"
            self._file = log_path.open("a", encoding="utf-8")

            # Prune old logs once per day (on rotation)
            if not self._pruned_today:
                prune_old_logs(self._logs_dir, self._retention_days)
                self._pruned_today = True

        return self._file
```

### Effort

Low (1-2 hours)

### Priority

**MEDIUM** - Important for long-running deployments.

---

## Gap 5: Usage Summary Formatting

### Description

Archer provides a nicely formatted usage summary with token counts, cache stats, and cost breakdown. Ash tracks usage but doesn't display it prominently.

### Why It Matters

- Cost visibility during development and production
- Cache effectiveness monitoring
- Token budget awareness for long conversations

### Reference Implementation

**File:** `/home/dcramer/src/archer/src/log.ts` (lines 184-236)

```typescript
export function logUsageSummary(
  ctx: LogContext,
  usage: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    cost: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number };
  },
  contextTokens?: number,
  contextWindow?: number,
): string {
  const formatTokens = (count: number): string => {
    if (count < 1000) return count.toString();
    if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
    if (count < 1000000) return `${Math.round(count / 1000)}k`;
    return `${(count / 1000000).toFixed(1)}M`;
  };

  const lines: string[] = [];
  lines.push("<b>Usage Summary</b>");
  lines.push(`Tokens: ${usage.input.toLocaleString()} in, ${usage.output.toLocaleString()} out`);
  if (usage.cacheRead > 0 || usage.cacheWrite > 0) {
    lines.push(`Cache: ${usage.cacheRead.toLocaleString()} read, ${usage.cacheWrite.toLocaleString()} write`);
  }
  if (contextTokens && contextWindow) {
    const contextPercent = ((contextTokens / contextWindow) * 100).toFixed(1);
    lines.push(`Context: ${formatTokens(contextTokens)} / ${formatTokens(contextWindow)} (${contextPercent}%)`);
  }
  lines.push(
    `Cost: $${usage.cost.input.toFixed(4)} in, $${usage.cost.output.toFixed(4)} out`
  );
  lines.push(`<b>Total: $${usage.cost.total.toFixed(4)}</b>`);
  return lines.join("\n");
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/ash/llm/types.py` | Extend `Usage` dataclass with cache fields |
| `src/ash/core/formatting.py` | New file for usage formatting |
| `src/ash/cli/commands/chat.py` | Display usage after responses |

### Implementation

```python
# src/ash/core/formatting.py
"""Formatting utilities for display output."""

from dataclasses import dataclass


@dataclass
class UsageStats:
    """Extended usage statistics with cost calculation."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Per-token costs (can be model-specific)
    input_cost_per_1k: float = 0.003
    output_cost_per_1k: float = 0.015
    cache_read_cost_per_1k: float = 0.0003
    cache_write_cost_per_1k: float = 0.00375

    @property
    def input_cost(self) -> float:
        return (self.input_tokens / 1000) * self.input_cost_per_1k

    @property
    def output_cost(self) -> float:
        return (self.output_tokens / 1000) * self.output_cost_per_1k

    @property
    def cache_read_cost(self) -> float:
        return (self.cache_read_tokens / 1000) * self.cache_read_cost_per_1k

    @property
    def cache_write_cost(self) -> float:
        return (self.cache_write_tokens / 1000) * self.cache_write_cost_per_1k

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost + self.cache_read_cost + self.cache_write_cost


def format_token_count(count: int) -> str:
    """Format token count with K/M suffix."""
    if count < 1000:
        return str(count)
    if count < 10000:
        return f"{count / 1000:.1f}k"
    if count < 1_000_000:
        return f"{count // 1000}k"
    return f"{count / 1_000_000:.1f}M"


def format_usage_summary(
    usage: UsageStats,
    context_tokens: int | None = None,
    context_window: int | None = None,
    compact: bool = False,
) -> str:
    """Format usage statistics for display.

    Args:
        usage: Usage statistics.
        context_tokens: Current context size.
        context_window: Maximum context window.
        compact: Use compact single-line format.

    Returns:
        Formatted usage string.
    """
    if compact:
        parts = [
            f"{usage.input_tokens:,} in",
            f"{usage.output_tokens:,} out",
        ]
        if usage.cache_read_tokens or usage.cache_write_tokens:
            parts.append(f"({usage.cache_read_tokens:,} cache)")
        parts.append(f"${usage.total_cost:.4f}")
        return " | ".join(parts)

    lines = []
    lines.append(f"Tokens: {usage.input_tokens:,} in, {usage.output_tokens:,} out")

    if usage.cache_read_tokens or usage.cache_write_tokens:
        lines.append(
            f"Cache: {usage.cache_read_tokens:,} read, "
            f"{usage.cache_write_tokens:,} write"
        )

    if context_tokens and context_window:
        pct = (context_tokens / context_window) * 100
        lines.append(
            f"Context: {format_token_count(context_tokens)} / "
            f"{format_token_count(context_window)} ({pct:.1f}%)"
        )

    lines.append(
        f"Cost: ${usage.input_cost:.4f} in, ${usage.output_cost:.4f} out"
    )
    if usage.cache_read_tokens or usage.cache_write_tokens:
        lines[-1] += (
            f", ${usage.cache_read_cost:.4f} cache read, "
            f"${usage.cache_write_cost:.4f} cache write"
        )

    lines.append(f"Total: ${usage.total_cost:.4f}")

    return "\n".join(lines)


def format_usage_rich(usage: UsageStats) -> str:
    """Format usage for Rich console with markup."""
    return (
        f"[dim]{usage.input_tokens:,} in + {usage.output_tokens:,} out"
        + (
            f" ({usage.cache_read_tokens:,} cache read)"
            if usage.cache_read_tokens
            else ""
        )
        + f" = [bold]${usage.total_cost:.4f}[/bold][/dim]"
    )
```

### Effort

Medium (2-3 hours) - Requires extending Usage tracking in LLM clients.

### Priority

**LOW** - Nice to have for cost awareness.

---

## Gap 6: Multiple Output Styles

### Description

Clawdbot supports three console output styles: `pretty` (colorful with timestamps), `compact` (minimal), and `json` (structured). Ash has one format controlled by the `use_rich` flag.

### Why It Matters

- Different contexts need different output formats
- JSON output for log aggregation/analysis
- Compact output for resource-constrained environments

### Reference Implementation

**File:** `/home/dcramer/src/clawdbot/src/logging.ts` (lines 47-52, 129-135)

```typescript
export type ConsoleStyle = "pretty" | "compact" | "json";

function normalizeConsoleStyle(style?: string): ConsoleStyle {
  if (style === "compact" || style === "json" || style === "pretty") {
    return style;
  }
  if (!process.stdout.isTTY) return "compact";
  return "pretty";
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/ash/logging.py` | Add `ConsoleStyle` enum, multiple formatters |
| `src/ash/config/models.py` | Add `LoggingConfig.console_style` option |

### Implementation

```python
# src/ash/logging.py

from enum import Enum


class ConsoleStyle(str, Enum):
    """Console output style."""

    PRETTY = "pretty"   # Rich formatting with colors
    COMPACT = "compact"  # Minimal output
    JSON = "json"        # Structured JSON


def _detect_console_style() -> ConsoleStyle:
    """Detect appropriate console style based on environment."""
    import sys

    # Check for explicit setting
    style = os.environ.get("ASH_LOG_STYLE", "").lower()
    if style in ("pretty", "compact", "json"):
        return ConsoleStyle(style)

    # Auto-detect: JSON for non-TTY, pretty for TTY
    if not sys.stdout.isatty():
        return ConsoleStyle.COMPACT

    return ConsoleStyle.PRETTY


class CompactFormatter(logging.Formatter):
    """Minimal formatter for compact output."""

    def format(self, record: logging.LogRecord) -> str:
        level_char = record.levelname[0]  # E, W, I, D
        return f"{level_char} {record.getMessage()}"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured output."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        parts = record.name.split(".")
        component = parts[1] if len(parts) >= 2 and parts[0] == "ash" else parts[0]

        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "component": component,
            "message": record.getMessage(),
        }

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry)


def configure_logging(
    level: str | None = None,
    use_rich: bool = False,
    log_to_file: bool = False,
    console_style: ConsoleStyle | None = None,
) -> None:
    """Configure logging for Ash."""
    # ... existing level resolution ...

    # Resolve console style
    style = console_style or _detect_console_style()

    handlers: list[logging.Handler] = []

    # Configure console handler based on style
    console_handler = logging.StreamHandler()

    if style == ConsoleStyle.PRETTY and use_rich:
        from rich.logging import RichHandler
        console_handler = RichHandler(
            rich_tracebacks=False,
            show_path=False,
            show_time=True,
            markup=True,
        )
        console_handler.setFormatter(
            ColoredComponentFormatter("%(component)s | %(message)s")
        )
    elif style == ConsoleStyle.JSON:
        console_handler.setFormatter(JSONFormatter())
    elif style == ConsoleStyle.COMPACT:
        console_handler.setFormatter(CompactFormatter())
    else:
        # Default plain format
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    handlers.append(console_handler)
    # ... rest of configuration ...
```

### Effort

Medium (2-3 hours)

### Priority

**LOW** - Current Rich-based output works well for most cases.

---

## Gap 7: Configurable Console vs File Levels

### Description

Clawdbot allows separate log levels for console and file output. This lets you keep console clean (INFO) while capturing DEBUG to files for troubleshooting. Ash uses a single level for both.

### Why It Matters

- Console can stay clean while files capture verbose debugging
- Different operational contexts need different visibility
- Troubleshooting without restarting with different flags

### Reference Implementation

**File:** `/home/dcramer/src/clawdbot/src/logging.ts` (lines 32-37, 83-89)

```typescript
export type LoggerSettings = {
  level?: Level;          // File level
  file?: string;
  consoleLevel?: Level;   // Separate console level
  consoleStyle?: ConsoleStyle;
};

function resolveConsoleSettings(): ConsoleSettings {
  const cfg = loadConfig().logging;
  const level = normalizeConsoleLevel(cfg?.consoleLevel);
  const style = normalizeConsoleStyle(cfg?.consoleStyle);
  return { level, style };
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/ash/logging.py` | Add `console_level` parameter to `configure_logging()` |
| `src/ash/config/models.py` | Add `LoggingConfig.console_level` option |

### Implementation

```python
# src/ash/logging.py

def configure_logging(
    level: str | None = None,
    console_level: str | None = None,  # NEW
    use_rich: bool = False,
    log_to_file: bool = False,
    console_style: ConsoleStyle | None = None,
) -> None:
    """Configure logging for Ash.

    Args:
        level: Log level for file output (DEBUG, INFO, WARNING, ERROR).
            If None, uses ASH_LOG_LEVEL env var or INFO.
        console_level: Log level for console output.
            If None, uses ASH_CONSOLE_LOG_LEVEL env var or same as level.
        use_rich: Use Rich handler for colorful output (server mode).
        log_to_file: Also write logs to JSONL files in ~/.ash/logs/.
        console_style: Console output style (pretty, compact, json).
    """
    # Resolve file level
    if level is None:
        level = os.environ.get("ASH_LOG_LEVEL", "INFO").upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            level = "INFO"

    file_level = getattr(logging, level)

    # Resolve console level (default to file level)
    if console_level is None:
        console_level = os.environ.get("ASH_CONSOLE_LOG_LEVEL", level).upper()
        if console_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            console_level = level

    console_log_level = getattr(logging, console_level)

    handlers: list[logging.Handler] = []

    # Configure console handler
    # ... existing console handler setup ...
    console_handler.setLevel(console_log_level)  # Set console-specific level
    handlers.append(console_handler)

    # Configure file handler with its own level
    if log_to_file:
        file_handler = JSONLHandler(get_logs_path())
        file_handler.setLevel(file_level)  # File uses separate level
        handlers.append(file_handler)

    # Set root logger to minimum of both levels to allow all messages through
    root_level = min(file_level, console_log_level)

    logging.basicConfig(
        level=root_level,
        handlers=handlers,
        force=True,
    )
    # ... rest of configuration ...
```

Config model addition:

```python
# src/ash/config/models.py

class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    console_level: str | None = None  # Defaults to level if not set
    console_style: str = "pretty"
    retention_days: int = 7
    redact_sensitive: bool = True
    redact_patterns: list[str] | None = None
```

### Effort

Low (1-2 hours)

### Priority

**MEDIUM** - Useful for operational flexibility.

---

## Implementation Priority Matrix

| Gap | Effort | Priority | Security | UX Impact |
|-----|--------|----------|----------|-----------|
| 1. Secret redaction | Medium | HIGH | Yes | Low |
| 2. Subsystem colors | Low | MEDIUM | No | High |
| 3. Console capture | Low | LOW | No | Low |
| 4. Log auto-pruning | Low | MEDIUM | No | Medium |
| 5. Usage formatting | Medium | LOW | No | Medium |
| 6. Multiple styles | Medium | LOW | No | Low |
| 7. Console vs file levels | Low | MEDIUM | No | Medium |

### Recommended Implementation Order

1. **Secret redaction** - Security issue, implement first
2. **Log auto-pruning** - Quick win, prevents operational issues
3. **Console vs file levels** - Low effort, immediate value
4. **Subsystem colors** - Quick UX improvement
5. **Usage formatting** - When extending cost tracking
6. **Multiple styles** - When needed for log aggregation
7. **Console capture** - Only if missing output becomes an issue

---

## Summary

Ash's logging foundation is solid with JSONL file output and Rich integration. The most critical gap is **secret redaction** which is a security concern. The other gaps are quality-of-life improvements that would bring Ash to parity with Clawdbot's logging capabilities.

Total estimated effort: ~12-15 hours for all gaps.
