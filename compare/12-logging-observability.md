# Logging and Observability Comparison

This document compares logging and observability implementations across four agent codebases: ash, archer, clawdbot, and pi-mono.

## Overview

Each codebase takes a different approach to logging based on its deployment context:

| Codebase | Language | Primary Approach | File Logging | Error Tracking |
|----------|----------|------------------|--------------|----------------|
| ash | Python | Structured JSONL + Rich console | Daily rolling files | Sentry integration |
| archer | TypeScript | Colorized console only | None | None |
| clawdbot | TypeScript | tslog with JSONL transport | Daily rolling with auto-pruning | None |
| pi-mono | TypeScript | Minimal debug dump | On-demand debug file | None |

## Comparison Table

| Feature | ash | archer | clawdbot | pi-mono |
|---------|-----|--------|----------|---------|
| **Log Format** | JSONL | Plain text | JSONL | Plain text dump |
| **Console Output** | Rich or plain | Chalk colors | Colored prefixes | TUI only |
| **File Logging** | `~/.ash/logs/YYYY-MM-DD.jsonl` | None | `/tmp/clawdbot/clawdbot-YYYY-MM-DD.log` | `~/.pi/agent/pi-debug.log` |
| **Structured Logging** | Yes (JSON fields) | No (formatted strings) | Yes (JSON fields) | No |
| **Log Levels** | DEBUG, INFO, WARNING, ERROR | Implicit via function names | trace, debug, info, warn, error, fatal | N/A |
| **Secret Redaction** | None | None | Pattern-based | None |
| **Third-party Suppression** | Yes (httpx, aiogram, etc.) | N/A | Yes (Baileys noise) | N/A |
| **Subsystem Loggers** | Via logger name hierarchy | Via LogContext | Hierarchical subsystem loggers | N/A |
| **Error Tracking** | Sentry SDK | None | None | None |
| **Console Capture** | No | No | Yes (console.* to file) | No |
| **Retention Policy** | Manual cleanup | N/A | 24-hour auto-pruning | Manual cleanup |

---

## ash (Python)

### Architecture

Ash uses Python's standard `logging` module with custom handlers for JSONL file output and optional Rich console formatting. Observability is enhanced through optional Sentry integration.

**Core files:**
- `/home/dcramer/src/ash/src/ash/logging.py` - Centralized logging configuration
- `/home/dcramer/src/ash/src/ash/observability/__init__.py` - Sentry integration

### Log Levels and Guidelines

```python
# Logging Levels:
# - DEBUG: Development details, API slot acquisition, cache hits
# - INFO: User-facing operations, skill/tool completion summaries
# - WARNING: Recoverable issues, retries, missing optional config
# - ERROR: Failures that affect operation

# Guidelines:
# - Tools: Log at INFO only in executor.py (single source of truth)
# - LLM calls: Log at DEBUG level (too noisy for INFO)
# - User messages: Log at INFO in providers (telegram, etc.)
# - Retries: Log at INFO on retry attempt, WARNING on exhaustion
```

### JSONLHandler - File Logging

```python
class JSONLHandler(logging.Handler):
    """Handler that writes structured log entries to a JSONL file.

    Logs are written to ~/.ash/logs/YYYY-MM-DD.jsonl with one JSON object per line.
    This format is inspectable with standard tools (cat, grep, jq) and can be
    mounted read-only in the sandbox for debugging.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Extract component from logger name
        parts = record.name.split(".")
        if len(parts) >= 2 and parts[0] == "ash":
            component = parts[1]
        else:
            component = parts[0]

        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "component": component,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            formatter = self.formatter or logging.Formatter()
            entry["exception"] = formatter.formatException(record.exc_info)

        log_file = self._get_log_file()
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()
```

### ComponentFormatter - Console Output

```python
class ComponentFormatter(logging.Formatter):
    """Formatter that extracts component name from logger path.

    Converts full module paths to short component names:
    - ash.providers.telegram.handlers -> providers
    - ash.tools.executor -> tools
    - ash.core.agent -> core
    """

    def format(self, record: logging.LogRecord) -> str:
        parts = record.name.split(".")
        if len(parts) >= 2 and parts[0] == "ash":
            record.component = parts[1]
        else:
            record.component = parts[0]
        return super().format(record)
```

### Third-Party Logger Suppression

```python
# Third-party loggers that are too noisy at INFO level
NOISY_LOGGERS = [
    "httpx",        # HTTP client used by Anthropic/OpenAI
    "httpcore",     # httpx dependency
    "uvicorn.access",  # Request logging
    "aiogram",      # Telegram library
    "aiogram.event",
    "anthropic",    # Anthropic SDK
    "openai",       # OpenAI SDK
]

# Suppress noisy third-party loggers
for logger_name in NOISY_LOGGERS:
    lib_logger = logging.getLogger(logger_name)
    lib_logger.setLevel(logging.WARNING)
```

### Configuration Entry Point

```python
def configure_logging(
    level: str | None = None,
    use_rich: bool = False,
    log_to_file: bool = False,
) -> None:
    """Configure logging for Ash.

    Call this once at application startup (CLI or server).

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
            If None, uses ASH_LOG_LEVEL env var or INFO.
        use_rich: Use Rich handler for colorful output (server mode).
        log_to_file: Also write logs to JSONL files in ~/.ash/logs/.
    """
```

### Sentry Integration

```python
def init_sentry(config: "SentryConfig", server_mode: bool = False) -> bool:
    """Initialize Sentry if configured."""
    integrations = [
        AsyncioIntegration(),
        LoggingIntegration(
            level=logging.INFO,   # Capture INFO+ as breadcrumbs
            event_level=logging.ERROR,  # Create events for ERROR+
        ),
    ]

    if server_mode:
        integrations.append(FastApiIntegration())

    sentry_sdk.init(
        dsn=config.dsn.get_secret_value(),
        environment=config.environment,
        release=config.release,
        traces_sample_rate=config.traces_sample_rate,
        profiles_sample_rate=config.profiles_sample_rate,
        send_default_pii=config.send_default_pii,
        debug=config.debug,
        integrations=integrations,
    )
```

### Key Design Decisions

1. **Single source of truth**: Tools only logged in `executor.py` with timing
2. **Environment-based configuration**: `ASH_LOG_LEVEL` for dynamic control
3. **Mode-specific output**: Rich for server, plain for CLI
4. **JSONL format**: Compatible with `cat`, `grep`, `jq` for debugging

---

## archer (TypeScript)

### Architecture

Archer uses a simple function-based logging approach with Chalk for colorization. No file logging - all output goes to console. Each log function is purpose-specific.

**Core file:**
- `/home/dcramer/src/archer/src/log.ts`

### LogContext for User/Channel Tracking

```typescript
export interface LogContext {
    channelId: string;    // chatId for Telegram
    userName?: string;
    channelName?: string; // Chat name for display
}

function formatContext(ctx: LogContext): string {
    const chat = ctx.channelName || ctx.channelId;
    const user = ctx.userName || "unknown";
    return `[${chat}:${user}]`;
}
```

### Timestamp Format

```typescript
function timestamp(): string {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, "0");
    const mm = String(now.getMinutes()).padStart(2, "0");
    const ss = String(now.getSeconds()).padStart(2, "0");
    return `[${hh}:${mm}:${ss}]`;
}
```

### Purpose-Specific Log Functions

```typescript
// User messages - green
export function logUserMessage(ctx: LogContext, text: string): void {
    console.log(chalk.green(`${timestamp()} ${formatContext(ctx)} ${text}`));
}

// Tool execution - yellow with success/error indicators
export function logToolStart(ctx: LogContext, toolName: string, label: string, args: Record<string, unknown>): void {
    const formattedArgs = formatToolArgs(args);
    console.log(chalk.yellow(`${timestamp()} ${formatContext(ctx)} ↳ ${toolName}: ${label}`));
    if (formattedArgs) {
        const indented = formattedArgs.split("\n").map((line) => `           ${line}`).join("\n");
        console.log(chalk.dim(indented));
    }
}

export function logToolSuccess(ctx: LogContext, toolName: string, durationMs: number, result: string): void {
    const duration = (durationMs / 1000).toFixed(1);
    console.log(chalk.yellow(`${timestamp()} ${formatContext(ctx)} ✓ ${toolName} (${duration}s)`));
    // Result output truncated and indented
}

export function logToolError(ctx: LogContext, toolName: string, durationMs: number, error: string): void {
    const duration = (durationMs / 1000).toFixed(1);
    console.log(chalk.yellow(`${timestamp()} ${formatContext(ctx)} ✗ ${toolName} (${duration}s)`));
}
```

### Usage Summary with Cost Breakdown

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
    // Formats token counts (1k, 100k, 1.5M) and costs
    // Returns HTML-formatted summary for Telegram
    // Also logs to console with emoji: 💰 Usage
}
```

### System-Level Logging

```typescript
export function logInfo(message: string): void {
    console.log(chalk.blue(`${timestamp()} [system] ${message}`));
}

export function logWarning(message: string, details?: string): void {
    console.log(chalk.yellow(`${timestamp()} [system] ⚠ ${message}`));
    // Optional indented details
}

export function logAgentError(ctx: LogContext | "system", error: string): void {
    const context = ctx === "system" ? "[system]" : formatContext(ctx);
    console.log(chalk.yellow(`${timestamp()} ${context} ✗ Agent error`));
}
```

### Key Design Decisions

1. **Console-only**: No file persistence, immediate feedback
2. **Color semantics**: Green=user, Yellow=agent/tools, Blue=system, Dim=details
3. **Context threading**: All logs include channel/user for multi-tenant debugging
4. **Truncation**: Output truncated at 1000 chars to prevent flood
5. **Visual indicators**: Unicode symbols (✓, ✗, ↳, 💰, 💭) for scan-ability

---

## clawdbot (TypeScript)

### Architecture

Clawdbot uses the tslog library with custom transports for JSONL file logging. Features subsystem-based loggers with colored prefixes and comprehensive secret redaction.

**Core files:**
- `/home/dcramer/src/clawdbot/src/logging.ts` - Main logging infrastructure
- `/home/dcramer/src/clawdbot/src/logging/redact.ts` - Pattern-based secret redaction

### Log Levels and Configuration

```typescript
const ALLOWED_LEVELS = [
    "silent", "fatal", "error", "warn", "info", "debug", "trace"
] as const;

export type LoggerSettings = {
    level?: Level;
    file?: string;
    consoleLevel?: Level;
    consoleStyle?: ConsoleStyle;  // "pretty" | "compact" | "json"
};
```

### Rolling File Logs with Auto-Pruning

```typescript
export const DEFAULT_LOG_DIR = "/tmp/clawdbot";
const LOG_PREFIX = "clawdbot";
const LOG_SUFFIX = ".log";
const MAX_LOG_AGE_MS = 24 * 60 * 60 * 1000; // 24h

function defaultRollingPathForToday(): string {
    const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
    return path.join(DEFAULT_LOG_DIR, `${LOG_PREFIX}-${today}${LOG_SUFFIX}`);
}

function pruneOldRollingLogs(dir: string): void {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    const cutoff = Date.now() - MAX_LOG_AGE_MS;
    for (const entry of entries) {
        if (!entry.isFile()) continue;
        if (!entry.name.startsWith(`${LOG_PREFIX}-`)) continue;
        const fullPath = path.join(dir, entry.name);
        const stat = fs.statSync(fullPath);
        if (stat.mtimeMs < cutoff) {
            fs.rmSync(fullPath, { force: true });
        }
    }
}
```

### JSONL Transport

```typescript
function buildLogger(settings: ResolvedSettings): TsLogger<LogObj> {
    fs.mkdirSync(path.dirname(settings.file), { recursive: true });
    if (isRollingPath(settings.file)) {
        pruneOldRollingLogs(path.dirname(settings.file));
    }

    const logger = new TsLogger<LogObj>({
        name: "clawdbot",
        minLevel: levelToMinLevel(settings.level),
        type: "hidden",  // no ansi formatting
    });

    logger.attachTransport((logObj: LogObj) => {
        const time = logObj.date?.toISOString?.() ?? new Date().toISOString();
        const line = JSON.stringify({ ...logObj, time });
        fs.appendFileSync(settings.file, `${line}\n`, { encoding: "utf8" });
    });

    return logger;
}
```

### Subsystem Loggers with Color Prefixes

```typescript
const SUBSYSTEM_COLORS = ["cyan", "green", "yellow", "blue", "magenta", "red"] as const;

function pickSubsystemColor(color: ChalkInstance, subsystem: string): ChalkInstance {
    const override = SUBSYSTEM_COLOR_OVERRIDES[subsystem];
    if (override) return color[override];
    let hash = 0;
    for (let i = 0; i < subsystem.length; i += 1) {
        hash = (hash * 31 + subsystem.charCodeAt(i)) | 0;
    }
    const idx = Math.abs(hash) % SUBSYSTEM_COLORS.length;
    return color[SUBSYSTEM_COLORS[idx]];
}

export function createSubsystemLogger(subsystem: string): SubsystemLogger {
    const logger: SubsystemLogger = {
        subsystem,
        trace: (message, meta) => emit("trace", message, meta),
        debug: (message, meta) => emit("debug", message, meta),
        info: (message, meta) => emit("info", message, meta),
        warn: (message, meta) => emit("warn", message, meta),
        error: (message, meta) => emit("error", message, meta),
        fatal: (message, meta) => emit("fatal", message, meta),
        raw: (message) => { /* direct output */ },
        child: (name) => createSubsystemLogger(`${subsystem}/${name}`),
    };
    return logger;
}
```

### Console Capture

```typescript
/**
 * Route console.* calls through pino while still emitting to stdout/stderr.
 * This keeps user-facing output unchanged but guarantees every console call
 * is captured in log files.
 */
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
            if (shouldSuppressConsoleMessage(formatted)) return;
            // Log to file
            logger[level](formatted);
            // Forward to original console method
            orig.apply(console, args);
        };

    console.log = forward("info", original.log);
    console.info = forward("info", original.info);
    console.warn = forward("warn", original.warn);
    console.error = forward("error", original.error);
}
```

### Pattern-Based Secret Redaction

```typescript
const DEFAULT_REDACT_PATTERNS: string[] = [
    // ENV-style assignments
    String.raw`\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)\b\s*[=:]\s*(["']?)([^\s"'\\]+)\1`,
    // JSON fields
    String.raw`"(?:apiKey|token|secret|password|passwd|accessToken|refreshToken)"\s*:\s*"([^"]+)"`,
    // CLI flags
    String.raw`--(?:api[-_]?key|token|secret|password|passwd)\s+(["']?)([^\s"']+)\1`,
    // Authorization headers
    String.raw`Authorization\s*[:=]\s*Bearer\s+([A-Za-z0-9._\-+=]+)`,
    // PEM blocks
    String.raw`-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----`,
    // Common token prefixes
    String.raw`\b(sk-[A-Za-z0-9_-]{8,})\b`,        // OpenAI
    String.raw`\b(ghp_[A-Za-z0-9]{20,})\b`,        // GitHub PAT
    String.raw`\b(github_pat_[A-Za-z0-9_]{20,})\b`, // GitHub fine-grained PAT
    String.raw`\b(xox[baprs]-[A-Za-z0-9-]{10,})\b`, // Slack
    String.raw`\b(AIza[0-9A-Za-z\-_]{20,})\b`,     // Google API
    String.raw`\b(npm_[A-Za-z0-9]{10,})\b`,        // npm
];

function maskToken(token: string): string {
    if (token.length < DEFAULT_REDACT_MIN_LENGTH) return "***";
    const start = token.slice(0, DEFAULT_REDACT_KEEP_START);  // 6 chars
    const end = token.slice(-DEFAULT_REDACT_KEEP_END);        // 4 chars
    return `${start}…${end}`;
}

export function redactSensitiveText(text: string, options?: RedactOptions): string {
    if (!text) return text;
    const resolved = options ?? resolveConfigRedaction();
    if (normalizeMode(resolved.mode) === "off") return text;
    const patterns = resolvePatterns(resolved.patterns);
    return redactText(text, patterns);
}
```

### Console Output Styles

```typescript
function formatConsoleLine(opts: {
    level: Level;
    subsystem: string;
    message: string;
    style: ConsoleStyle;
    meta?: Record<string, unknown>;
}): string {
    if (opts.style === "json") {
        return JSON.stringify({
            time: new Date().toISOString(),
            level: opts.level,
            subsystem: opts.subsystem,
            message: opts.message,
            ...opts.meta,
        });
    }
    // Pretty or compact formatting with colors
    const color = getColorForConsole();
    const prefix = `[${displaySubsystem}]`;
    const time = opts.style === "pretty"
        ? color.gray(new Date().toISOString().slice(11, 19))
        : "";
    return `${time} ${prefixColor(prefix)} ${levelColor(opts.message)}`;
}
```

### Key Design Decisions

1. **tslog library**: Mature logging framework with transport support
2. **Dual output**: Both file (JSONL) and console (formatted)
3. **Auto-pruning**: 24-hour retention prevents disk bloat
4. **Console capture**: All `console.*` calls also go to file
5. **Secret redaction**: Configurable patterns mask sensitive data
6. **Subsystem hierarchy**: Child loggers inherit parent context

---

## pi-mono (TypeScript)

### Architecture

Pi-mono takes a minimal approach to logging. It has a debug dump feature for troubleshooting but no continuous logging infrastructure.

**Core file:**
- `/home/dcramer/src/pi-mono/packages/coding-agent/src/config.ts` - Path definitions
- `/home/dcramer/src/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts` - Debug dump

### Debug Log Path

```typescript
/** Get path to debug log file */
export function getDebugLogPath(): string {
    return join(getAgentDir(), `${APP_NAME}-debug.log`);
}
// Results in: ~/.pi/agent/pi-debug.log
```

### On-Demand Debug Dump

```typescript
private handleDebugCommand(): void {
    const width = this.ui.terminal.columns;
    const allLines = this.ui.render(width);

    const debugLogPath = getDebugLogPath();
    const debugData = [
        `Debug output at ${new Date().toISOString()}`,
        `Terminal width: ${width}`,
        `Total lines: ${allLines.length}`,
        "",
        "=== All rendered lines with visible widths ===",
        ...allLines.map((line, idx) => {
            const vw = visibleWidth(line);
            const escaped = JSON.stringify(line);
            return `[${idx}] (w=${vw}) ${escaped}`;
        }),
        "",
        "=== Agent messages (JSONL) ===",
        ...this.session.messages.map((msg) => JSON.stringify(msg)),
        "",
    ].join("\n");

    fs.mkdirSync(path.dirname(debugLogPath), { recursive: true });
    fs.writeFileSync(debugLogPath, debugData);
}
```

### Key Design Decisions

1. **TUI-focused**: UI handles all user-facing output
2. **On-demand debugging**: No continuous logging overhead
3. **Session dump**: Messages exported as JSONL when needed
4. **Minimal dependencies**: No logging library required

---

## Key Differences

### 1. Logging Philosophy

| Codebase | Philosophy |
|----------|------------|
| ash | "Logs are for humans to inspect with Unix tools" |
| archer | "Console is for real-time operator monitoring" |
| clawdbot | "Capture everything, redact secrets, auto-cleanup" |
| pi-mono | "TUI is the interface, dump debug only when needed" |

### 2. File Logging Strategy

- **ash**: Always-available JSONL files, daily rotation, no auto-cleanup
- **archer**: No file logging - ephemeral console output only
- **clawdbot**: JSONL files with 24-hour auto-pruning
- **pi-mono**: Single debug dump file, overwritten on each dump

### 3. Structured vs. Formatted

- **ash, clawdbot**: Structured JSON for machine parsing + formatted console for humans
- **archer**: Formatted strings only - human-optimized
- **pi-mono**: Plain text dump with manual structure

### 4. Secret Handling

Only **clawdbot** has comprehensive secret redaction with:
- Regex patterns for common secret formats (API keys, tokens, passwords)
- Partial masking to preserve debugging value (`sk-1234...abcd`)
- PEM block redaction
- Configurable patterns

### 5. Third-Party Noise

- **ash**: Suppresses httpx, aiogram, anthropic SDK logs
- **clawdbot**: Suppresses Baileys session management noise
- **archer, pi-mono**: No third-party library integration

### 6. Error Tracking

Only **ash** integrates with an external error tracking service (Sentry):
- Automatic breadcrumb capture from INFO+ logs
- Error events for ERROR+ logs
- FastAPI integration for server mode
- Tracing and profiling support

---

## Recommendations

### For New Projects

1. **Start with structured logging**: JSONL format is universally parseable
2. **Add secret redaction early**: Patterns are easier to add than remove secrets later
3. **Consider retention**: Auto-pruning prevents disk issues in long-running deployments
4. **Separate concerns**: File for persistence, console for operators

### Potential Improvements

| Codebase | Suggestion |
|----------|------------|
| ash | Add secret redaction patterns |
| ash | Consider auto-pruning old log files |
| archer | Add optional file logging for debugging deployed instances |
| archer | Consider secret redaction for logged tool args |
| clawdbot | Consider error tracking integration (Sentry) |
| pi-mono | Add optional continuous logging mode for debugging |

### Best Practices Observed

1. **ash**: Clear logging guidelines in code comments, single source of truth for tool logging
2. **archer**: Context threading for multi-tenant debugging
3. **clawdbot**: Comprehensive redaction patterns, console capture, pino-compatible adapter
4. **pi-mono**: Minimal overhead when logging not needed

### Common Patterns Worth Adopting

1. **Daily rolling files** (ash, clawdbot): Predictable, easy to find
2. **Component/subsystem extraction** (ash, clawdbot): Filter logs by module
3. **Truncation for console** (archer): Prevent output flood
4. **JSONL format** (ash, clawdbot): `grep`, `jq`, `tail -f` friendly
