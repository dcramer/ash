# CLI Interface Comparison

Comparing command-line interface designs across four AI assistant codebases.

## Overview

CLI interfaces in AI assistants serve multiple purposes: configuration management, interactive chat modes, background services, and administrative operations. The four codebases take distinctly different approaches based on their deployment models and target users.

| Codebase | Language | Framework | Philosophy |
|----------|----------|-----------|------------|
| **ash** | Python | Typer | Modular subcommands with functional organization |
| **archer** | TypeScript | Manual argparse | Single-purpose entry point, minimal CLI |
| **clawdbot** | TypeScript | Commander.js | Feature-rich with wizards and extensive subcommands |
| **pi-mono** | TypeScript | Manual + TUI | Per-package CLIs focused on interactive modes |

## Comparison Table

| Feature | ash | archer | clawdbot | pi-mono |
|---------|-----|--------|----------|---------|
| **Framework** | Typer | Manual | Commander.js | Manual |
| **Top-level Commands** | 12 | 1 | 30+ | 2 packages |
| **Subcommand Depth** | 1 level | None | 2-3 levels | 1 level |
| **Interactive Wizard** | No | No | Yes (onboard, configure) | No |
| **JSON Output** | No | No | Yes (`--json`) | Yes (`--mode json`) |
| **Dry-run Support** | No | No | Yes (`--dry-run`) | No |
| **Help System** | Auto-generated | Manual | Themed + examples | Manual |
| **Config Management** | Yes | File-based | Yes + wizard | File-based |
| **Service Management** | Yes (systemd) | No | Yes (daemon) | No |
| **Shell Completion** | Via Typer | No | No | No |

## Detailed Command Inventory

### ash (Python/Typer)

**Entry point:** `src/ash/cli/app.py`

Commands are organized as separate modules in `src/ash/cli/commands/`:

| Command | Module | Description |
|---------|--------|-------------|
| `init` | `init.py` | Initialize configuration file |
| `serve` | `serve.py` | Start server with Telegram polling/webhook |
| `chat` | `chat.py` | Interactive CLI chat or single prompt |
| `config` | `config.py` | Configuration management (show, validate) |
| `database` | `database.py` | Database operations |
| `memory` | `memory.py` | Memory management (list, add, remove, clear) |
| `schedule` | `schedule.py` | Scheduled tasks (list, cancel, clear) |
| `sessions` | `sessions.py` | Session management (list, view, search, clear) |
| `upgrade` | `upgrade.py` | Run migrations, check sandbox |
| `sandbox` | `sandbox.py` | Build sandbox Docker image |
| `service` | `service.py` | Systemd service management |
| `skill` | `skill.py` | Skill validation and listing |

**Pattern:** Each command module exports a `register(app)` function that attaches commands to the Typer app. Commands use action-based subcommand patterns (e.g., `ash memory list`, `ash memory add`).

**Example command signature:**
```python
@app.command()
def memory(
    action: Annotated[str, typer.Argument(help="Action: list, add, remove, clear")],
    query: Annotated[str | None, typer.Option("--query", "-q")] = None,
    # ... more options
) -> None:
```

---

### archer (TypeScript/Manual)

**Entry point:** `src/main.ts`

Minimal CLI with only startup options:

| Argument | Description |
|----------|-------------|
| `<working-directory>` | Required positional argument |
| `--sandbox=host\|docker:<name>` | Sandbox execution mode |

**Design philosophy:** Archer is a pure Telegram bot - it has no interactive CLI mode, no administrative commands, and no configuration wizard. All configuration is file-based (`~/.archer/telegram.json`) or environment variables.

**Usage:**
```
archer [--sandbox=host|docker:<name>] <working-directory>
```

This is the simplest approach: the CLI exists solely to start the service with appropriate sandbox configuration.

---

### clawdbot (TypeScript/Commander.js)

**Entry point:** `src/cli/program.ts`

The most extensive CLI with 30+ commands organized hierarchically:

#### Top-Level Commands

| Command | Description |
|---------|-------------|
| `setup` | Initialize config and workspace |
| `onboard` | Interactive setup wizard |
| `configure` | Update models, providers, skills |
| `doctor` | Health checks and quick fixes |
| `message` | Send messages (deeply nested) |
| `agent` | Run agent turn via Gateway |
| `agents` | Manage isolated agents |
| `daemon` | Background service control |
| `gateway` | WebSocket gateway management |
| `logs` | View service logs |
| `models` | Model configuration |
| `nodes` | Node management |
| `sandbox` | Container management |
| `tui` | Terminal UI |
| `cron` | Scheduled task management |
| `dns` | DNS configuration |
| `docs` | Documentation commands |
| `hooks` | Lifecycle hooks |
| `pairing` | Device pairing |
| `providers` | Provider management |
| `skills` | Skill management |
| `update` | Self-update |
| `status` | Show local status |
| `health` | Gateway health check |
| `sessions` | Session listing |
| `browser` | Browser automation |

#### Nested Command Structure

**message** subcommands (deeply nested):
- `send`, `poll`, `react`, `reactions`, `read`, `edit`, `delete`, `pin`, `unpin`, `pins`, `permissions`, `search`, `timeout`, `kick`, `ban`
- `thread`: `create`, `list`, `reply`
- `emoji`: `list`, `upload`
- `sticker`: `send`, `upload`
- `role`: `info`, `add`, `remove`
- `channel`: `info`, `list`
- `member`: `info`
- `voice`: `status`
- `event`: `list`, `create`

**agents** subcommands: `list`, `add`, `delete`

**sandbox** subcommands: `list`, `recreate`, `explain`

#### Notable Features

**Interactive Wizards:**
```typescript
program
  .command("onboard")
  .option("--non-interactive", "Run without prompts")
  .option("--flow <flow>", "Wizard flow: quickstart|advanced")
  .option("--auth-choice <choice>", "Auth: setup-token|claude-cli|...")
  // ... 40+ options for programmatic setup
```

**Themed Output:**
```typescript
program.configureHelp({
  optionTerm: (option) => theme.option(option.flags),
  subcommandTerm: (cmd) => theme.command(cmd.name()),
});
```

**Examples in Help:**
```typescript
program.addHelpText("afterAll", () => {
  return `\n${theme.heading("Examples:")}\n${fmtExamples}\n`;
});
```

**Global Options:**
- `--dev` - Dev profile isolation
- `--profile <name>` - Named profile
- `--no-color` - Disable ANSI colors

---

### pi-mono (TypeScript/Manual)

**Structure:** Monorepo with per-package CLIs

#### packages/coding-agent

**Entry point:** `src/cli.ts` -> `src/main.ts`

| Option | Description |
|--------|-------------|
| `--provider <name>` | Model provider |
| `--model <id>` | Model ID |
| `--api-key <key>` | API key override |
| `--system-prompt <text>` | Custom system prompt |
| `--mode <mode>` | Output: text, json, rpc |
| `--print, -p` | Non-interactive mode |
| `--continue, -c` | Continue previous session |
| `--resume, -r` | Session picker UI |
| `--session <path>` | Specific session file |
| `--models <patterns>` | Model patterns for cycling |
| `--thinking <level>` | Thinking level |
| `--extension, -e <path>` | Load extension |
| `--list-models [search]` | List available models |
| `--export <file>` | Export session to HTML |
| `--tools <tools>` | Enable specific tools |
| `--no-tools` | Disable all tools |

**Focus:** Interactive TUI mode with model cycling (Ctrl+P), session management, and extension support.

#### packages/pods

**Entry point:** `src/cli.ts`

GPU pod management for vLLM deployments:

| Command | Description |
|---------|-------------|
| `pods` | List pods |
| `pods setup <name> "<ssh>"` | Setup pod |
| `pods active <name>` | Switch active pod |
| `pods remove <name>` | Remove pod |
| `shell [<name>]` | SSH shell |
| `ssh [<name>] "<cmd>"` | Run SSH command |
| `start <model>` | Start model |
| `stop [<name>]` | Stop model(s) |
| `list` | List running models |
| `logs <name>` | Stream logs |
| `agent <name>` | Chat with model |

---

## Key Differences

### 1. Organizational Philosophy

**ash:** Flat command structure with action arguments
```bash
ash memory list
ash memory add -q "fact"
ash sessions view -q "key"
```

**clawdbot:** Deep nesting with subcommands
```bash
clawdbot message thread create --thread-name "Topic"
clawdbot sandbox recreate --agent mybot
clawdbot providers login --provider telegram
```

**pi-mono:** Flags over subcommands
```bash
pi-agent --continue --thinking high
pi-agent --list-models sonnet
```

**archer:** No CLI management - pure service startup
```bash
archer --sandbox=docker:sandbox ~/work
```

### 2. Configuration Approach

| Codebase | Method |
|----------|--------|
| **ash** | TOML file, `ash init` creates default |
| **archer** | JSON file + environment variables |
| **clawdbot** | JSON file + interactive wizards + migration |
| **pi-mono** | JSON file + environment variables |

### 3. Interactive Features

| Feature | ash | archer | clawdbot | pi-mono |
|---------|-----|--------|----------|---------|
| Interactive chat | Yes | No | Via TUI | Yes (default) |
| Setup wizard | No | No | Yes | No |
| Session picker | No | No | No | Yes (--resume) |
| Model cycling | No | No | No | Yes (Ctrl+P) |
| Theme customization | No | No | Yes | Yes |

### 4. Output Formats

**ash:** Human-readable tables via Rich
```
Memory Entries
ID        Scope    Created      Source  Content
abc123    global   2024-01-15   cli     User prefers dark mode
```

**clawdbot:** Optional JSON for scripting
```bash
clawdbot status --json
clawdbot sandbox list --json
clawdbot agents list --json --bindings
```

**pi-mono:** Multiple modes
```bash
pi-agent --mode json  # JSONL streaming
pi-agent --mode rpc   # RPC protocol
pi-agent --mode text  # Default human-readable
```

### 5. Service Management

**ash:** Systemd integration
```bash
ash service install
ash service start
ash service status
ash service logs
```

**clawdbot:** Custom daemon management
```bash
clawdbot daemon install
clawdbot daemon start
clawdbot daemon stop
clawdbot daemon logs --follow
```

**archer/pi-mono:** No built-in service management

---

## Recommendations

### For ash

1. **Consider adding `--json` output** for scripted operations, especially for `sessions list`, `memory list`, and `schedule list`

2. **Add command aliases** for common operations:
   - `ash mem` -> `ash memory`
   - `ash sess` -> `ash sessions`

3. **Improve discoverability** with examples in help text for each command

### General Observations

1. **Typer provides good defaults** - ash's help system is clean and auto-generated

2. **Action-argument pattern works well** for CRUD operations (`memory list`, `memory add`)

3. **clawdbot's wizard approach** is powerful for onboarding but adds complexity

4. **pi-mono's flag-based design** is cleaner for single-purpose tools

5. **archer's minimal approach** is appropriate for single-service deployments

---

## Code Examples

### ash: Typer Command Registration

```python
# src/ash/cli/commands/memory.py
def register(app: typer.Typer) -> None:
    @app.command()
    def memory(
        action: Annotated[str, typer.Argument(help="Action: list, add, remove, clear")],
        query: Annotated[str | None, typer.Option("--query", "-q")] = None,
        entry_id: Annotated[str | None, typer.Option("--id")] = None,
        force: Annotated[bool, typer.Option("--force", "-f")] = False,
    ) -> None:
        """Manage memory entries."""
        asyncio.run(_run_memory_action(action, query, entry_id, force))
```

### clawdbot: Commander.js with Nested Commands

```typescript
// src/cli/sandbox-cli.ts
export function registerSandboxCli(program: Command) {
  const sandbox = program
    .command("sandbox")
    .description("Manage sandbox containers")
    .action(() => sandbox.help({ error: true }));

  sandbox
    .command("list")
    .option("--json", "Output as JSON", false)
    .option("--browser", "Browser containers only", false)
    .action(async (opts) => {
      await sandboxListCommand(opts, defaultRuntime);
    });
}
```

### pi-mono: Manual Argument Parsing

```typescript
// packages/coding-agent/src/cli/args.ts
export function parseArgs(args: string[]): Args {
  const result: Args = { messages: [], fileArgs: [] };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--continue" || arg === "-c") {
      result.continue = true;
    } else if (arg === "--model" && i + 1 < args.length) {
      result.model = args[++i];
    } else if (!arg.startsWith("-")) {
      result.messages.push(arg);
    }
  }
  return result;
}
```

### archer: Minimal Startup

```typescript
// src/main.ts
function parseArgs(): ParsedArgs {
  const args = process.argv.slice(2);
  let sandbox: SandboxConfig = { type: "host" };
  let workingDir: string | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--sandbox=")) {
      sandbox = parseSandboxArg(args[i].slice("--sandbox=".length));
    } else if (!args[i].startsWith("-")) {
      workingDir = resolve(args[i]);
    }
  }
  return { workingDir, sandbox };
}
```
