# Sandbox Execution Gap Analysis

This document analyzes potential improvements to Ash's sandbox system, comparing it with approaches in Clawdbot and Archer.

**Note:** Ash is **ahead** on security with read-only rootfs, all capabilities dropped, optional gVisor runtime (runsc), pids limits, tmpfs with noexec, and non-root user execution. The reference implementations have simpler security models. This analysis focuses on **operational gaps** - features that would improve usability, maintainability, and flexibility.

**Files Analyzed:**
- Ash: `/home/dcramer/src/ash/src/ash/sandbox/manager.py`
- Ash: `/home/dcramer/src/ash/src/ash/sandbox/executor.py`
- Ash: `/home/dcramer/src/ash/src/ash/cli/commands/sandbox.py`
- Ash: `/home/dcramer/src/ash/src/ash/sandbox/verify.py`
- Clawdbot: `/home/dcramer/src/clawdbot/src/agents/sandbox.ts`
- Archer: `/home/dcramer/src/archer/src/sandbox.ts`

---

## Gap 1: Host Mode Fallback

### What Ash is Missing

Ash requires Docker for sandbox execution - there's no fallback for development or simple deployments. Archer provides a `host` mode that executes commands directly on the host machine:

```typescript
// archer/src/sandbox.ts lines 3-18
export type SandboxConfig = { type: "host" } | { type: "docker"; container: string };

export function parseSandboxArg(value: string): SandboxConfig {
    if (value === "host") {
        return { type: "host" };
    }
    if (value.startsWith("docker:")) {
        const container = value.slice("docker:".length);
        if (!container) {
            console.error("Error: docker sandbox requires container name");
            process.exit(1);
        }
        return { type: "docker", container };
    }
    console.error(`Error: Invalid sandbox type '${value}'`);
    process.exit(1);
}
```

```typescript
// archer/src/sandbox.ts lines 72-77
export function createExecutor(config: SandboxConfig): Executor {
    if (config.type === "host") {
        return new HostExecutor();
    }
    return new DockerExecutor(config.container);
}
```

### Why It Matters

- **Development convenience**: Developers can test without Docker running
- **Simple deployments**: Some environments don't need sandboxing (trusted agents, local use)
- **CI testing**: Easier to run tests without Docker-in-Docker
- **Graceful degradation**: Can warn and continue when Docker unavailable

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sandbox/executor.py` - Add HostExecutor
- `/home/dcramer/src/ash/src/ash/sandbox/manager.py` - Add host mode support
- `/home/dcramer/src/ash/src/ash/config/models.py` - Add sandbox mode config

### Concrete Python Code Changes

```python
# Modify: src/ash/sandbox/executor.py

"""High-level command execution with host fallback."""

import asyncio
import logging
import shlex
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class SandboxMode(Enum):
    """Sandbox execution mode."""

    DOCKER = "docker"  # Full Docker isolation (default)
    HOST = "host"      # Direct host execution (no isolation)


class Executor(Protocol):
    """Protocol for command executors."""

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        environment: dict[str, str] | None = None,
    ) -> "ExecutionResult": ...

    async def cleanup(self) -> None: ...


@dataclass
class ExecutionResult:
    """Result of command execution."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        parts = [p for p in (self.stdout, self.stderr) if p]
        return "\n".join(parts)


class HostExecutor:
    """Execute commands directly on the host (no sandboxing).

    WARNING: This provides NO security isolation. Only use for:
    - Development/testing
    - Trusted agent deployments
    - Environments where Docker is unavailable
    """

    def __init__(
        self,
        work_dir: Path | None = None,
        environment: dict[str, str] | None = None,
    ):
        self._work_dir = work_dir or Path.cwd()
        self._environment = environment or {}

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        environment: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute command on host."""
        merged_env = {**self._environment, **(environment or {})}
        timeout = timeout or 60

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._work_dir,
                env={**dict(subprocess.os.environ), **merged_env} if merged_env else None,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                return ExecutionResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    timed_out=False,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {timeout} seconds",
                    timed_out=True,
                )

        except Exception as e:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                timed_out=False,
            )

    async def cleanup(self) -> None:
        """No cleanup needed for host executor."""
        pass


def create_executor(
    mode: SandboxMode = SandboxMode.DOCKER,
    **kwargs,
) -> Executor:
    """Create an executor based on the configured mode.

    Args:
        mode: Execution mode (docker or host)
        **kwargs: Additional arguments passed to executor

    Returns:
        Executor instance
    """
    if mode == SandboxMode.HOST:
        logger.warning(
            "Using HOST mode - commands execute without sandboxing. "
            "Only use for development or trusted environments."
        )
        return HostExecutor(
            work_dir=kwargs.get("work_dir"),
            environment=kwargs.get("environment"),
        )

    # Default to Docker sandbox
    from ash.sandbox.executor import SandboxExecutor
    return SandboxExecutor(**kwargs)
```

```python
# Modify: src/ash/config/models.py - add to SandboxConfig

class SandboxConfig(BaseModel):
    """Sandbox configuration."""

    mode: Literal["docker", "host"] = "docker"
    # ... existing fields ...
```

### Effort Estimate

**Small** (1-2 hours)
- Add HostExecutor class
- Add mode config option
- Add factory function
- Update agent to use factory

### Priority

**Medium** - Improves developer experience but Docker is a reasonable requirement for production.

---

## Gap 2: Tool Allow/Deny Lists

### What Ash is Missing

Ash runs all tools through the same sandbox without per-tool filtering. Clawdbot has granular control over which tools can run in the sandbox, with both global and per-agent configuration:

```typescript
// clawdbot/src/agents/sandbox.ts lines 47-68
export type SandboxToolPolicy = {
  allow?: string[];
  deny?: string[];
};

export type SandboxToolPolicySource = {
  source: "agent" | "global" | "default";
  key: string;
};

export type SandboxToolPolicyResolved = {
  allow: string[];
  deny: string[];
  sources: {
    allow: SandboxToolPolicySource;
    deny: SandboxToolPolicySource;
  };
};
```

```typescript
// clawdbot/src/agents/sandbox.ts lines 159-178
const DEFAULT_TOOL_ALLOW = [
  "bash",
  "process",
  "read",
  "write",
  "edit",
  "sessions_list",
  "sessions_history",
  "sessions_send",
  "sessions_spawn",
  "session_status",
];
const DEFAULT_TOOL_DENY = [
  "browser",
  "canvas",
  "nodes",
  "cron",
  "discord",
  "gateway",
];
```

```typescript
// clawdbot/src/agents/sandbox.ts lines 229-243
function normalizeToolList(values?: string[]) {
  if (!values) return [];
  return values
    .map((value) => value.trim())
    .filter(Boolean)
    .map((value) => value.toLowerCase());
}

function isToolAllowed(policy: SandboxToolPolicy, name: string) {
  const deny = new Set(normalizeToolList(policy.deny));
  if (deny.has(name.toLowerCase())) return false;
  const allow = normalizeToolList(policy.allow);
  if (allow.length === 0) return true;
  return allow.includes(name.toLowerCase());
}
```

### Why It Matters

- **Fine-grained control**: Some tools shouldn't run in sandbox (browser automation, API calls)
- **Agent-specific policies**: Different agents have different trust levels
- **Security layering**: Explicit allow/deny lists as defense in depth
- **Error messages**: Users understand why a tool was blocked and how to fix it

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sandbox/policy.py` - New file for tool policies
- `/home/dcramer/src/ash/src/ash/config/models.py` - Add tool policy config
- `/home/dcramer/src/ash/src/ash/tools/executor.py` - Check policy before sandbox execution

### Concrete Python Code Changes

```python
# New file: src/ash/sandbox/policy.py
"""Sandbox tool execution policies.

Controls which tools can execute in the sandbox environment.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PolicySource(Enum):
    """Source of the policy setting."""

    DEFAULT = "default"
    GLOBAL = "global"
    AGENT = "agent"


@dataclass
class ToolPolicy:
    """Tool allow/deny policy."""

    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)


@dataclass
class ResolvedToolPolicy:
    """Resolved tool policy with source tracking."""

    allow: list[str]
    deny: list[str]
    allow_source: PolicySource
    deny_source: PolicySource


# Default policies - can be overridden in config
DEFAULT_TOOL_ALLOW: list[str] = [
    "bash",
    "read_file",
    "write_file",
    "list_directory",
    "search_files",
    "git",
    "python",
]

DEFAULT_TOOL_DENY: list[str] = [
    "browser",
    "http_request",  # Should go through proxy or be explicitly allowed
    "send_message",  # Keep messaging out of sandbox
]


def normalize_tool_list(values: list[str] | None) -> list[str]:
    """Normalize tool names to lowercase, trimmed."""
    if not values:
        return []
    return [v.strip().lower() for v in values if v.strip()]


def is_tool_allowed(policy: ToolPolicy, tool_name: str) -> bool:
    """Check if a tool is allowed by the policy.

    Logic:
    1. If tool is in deny list, reject
    2. If allow list is empty, allow all (except denied)
    3. If allow list is non-empty, only allow listed tools

    Args:
        policy: The tool policy to check against
        tool_name: Name of the tool to check

    Returns:
        True if tool is allowed
    """
    name = tool_name.strip().lower()
    deny = set(normalize_tool_list(policy.deny))

    # Deny list takes precedence
    if name in deny:
        return False

    allow = normalize_tool_list(policy.allow)

    # Empty allow list = allow all (except denied)
    if not allow:
        return True

    # Non-empty allow list = whitelist mode
    return name in allow


def resolve_tool_policy(
    global_policy: ToolPolicy | None = None,
    agent_policy: ToolPolicy | None = None,
) -> ResolvedToolPolicy:
    """Resolve tool policy from global and agent-specific settings.

    Agent policy overrides global policy for each list independently.

    Args:
        global_policy: Global sandbox tool policy
        agent_policy: Agent-specific tool policy

    Returns:
        Resolved policy with source tracking
    """
    # Determine allow list
    if agent_policy and agent_policy.allow:
        allow = agent_policy.allow
        allow_source = PolicySource.AGENT
    elif global_policy and global_policy.allow:
        allow = global_policy.allow
        allow_source = PolicySource.GLOBAL
    else:
        allow = DEFAULT_TOOL_ALLOW
        allow_source = PolicySource.DEFAULT

    # Determine deny list
    if agent_policy and agent_policy.deny:
        deny = agent_policy.deny
        deny_source = PolicySource.AGENT
    elif global_policy and global_policy.deny:
        deny = global_policy.deny
        deny_source = PolicySource.GLOBAL
    else:
        deny = DEFAULT_TOOL_DENY
        deny_source = PolicySource.DEFAULT

    return ResolvedToolPolicy(
        allow=allow,
        deny=deny,
        allow_source=allow_source,
        deny_source=deny_source,
    )


def format_blocked_message(
    tool_name: str,
    policy: ResolvedToolPolicy,
) -> str:
    """Format a helpful error message when a tool is blocked.

    Args:
        tool_name: Name of the blocked tool
        policy: The resolved policy that blocked it

    Returns:
        Human-readable explanation with fix suggestions
    """
    name = tool_name.strip().lower()
    deny = set(normalize_tool_list(policy.deny))
    allow = normalize_tool_list(policy.allow)

    lines = [f"Tool '{tool_name}' blocked by sandbox policy."]

    if name in deny:
        lines.append(f"Reason: Tool is in deny list (source: {policy.deny_source.value})")
        lines.append("Fix: Remove from sandbox.tools.deny in config")
    elif allow and name not in allow:
        lines.append(f"Reason: Tool not in allow list (source: {policy.allow_source.value})")
        lines.append("Fix: Add to sandbox.tools.allow in config, or set allow=[] to allow all")

    return "\n".join(lines)
```

```python
# Modify: src/ash/config/models.py

class SandboxToolPolicy(BaseModel):
    """Sandbox tool execution policy."""

    allow: list[str] = Field(
        default_factory=list,
        description="Tools allowed in sandbox. Empty = allow all except denied.",
    )
    deny: list[str] = Field(
        default_factory=list,
        description="Tools denied in sandbox. Takes precedence over allow.",
    )


class SandboxConfig(BaseModel):
    """Sandbox configuration."""

    # ... existing fields ...
    tools: SandboxToolPolicy = Field(
        default_factory=SandboxToolPolicy,
        description="Tool execution policies for the sandbox",
    )
```

```python
# Modify: src/ash/tools/executor.py - add policy check

from ash.sandbox.policy import is_tool_allowed, format_blocked_message

async def execute_tool(
    self,
    tool_name: str,
    args: dict[str, Any],
    sandbox_policy: ResolvedToolPolicy | None = None,
) -> ToolResult:
    """Execute a tool, respecting sandbox policy."""

    # Check sandbox policy if running in sandbox mode
    if sandbox_policy and self._sandbox_enabled:
        policy = ToolPolicy(
            allow=sandbox_policy.allow,
            deny=sandbox_policy.deny,
        )
        if not is_tool_allowed(policy, tool_name):
            return ToolResult(
                success=False,
                output="",
                error=format_blocked_message(tool_name, sandbox_policy),
            )

    # ... existing execution logic ...
```

### Effort Estimate

**Medium** (2-4 hours)
- Create policy module
- Add config schema
- Integrate with tool executor
- Add tests for policy resolution

### Priority

**Medium** - Useful for multi-agent deployments with different trust levels.

---

## Gap 3: Container Pruning/Cleanup

### What Ash is Missing

Ash creates containers per-session but doesn't automatically clean up old/idle containers. Clawdbot has automatic pruning based on idle time and max age:

```typescript
// clawdbot/src/agents/sandbox.ts lines 107-110
export type SandboxPruneConfig = {
  idleHours: number;
  maxAgeDays: number;
};
```

```typescript
// clawdbot/src/agents/sandbox.ts lines 157-158
const DEFAULT_SANDBOX_IDLE_HOURS = 24;
const DEFAULT_SANDBOX_MAX_AGE_DAYS = 7;
```

```typescript
// clawdbot/src/agents/sandbox.ts lines 1176-1200
async function pruneSandboxContainers(cfg: SandboxConfig) {
  const now = Date.now();
  const idleHours = cfg.prune.idleHours;
  const maxAgeDays = cfg.prune.maxAgeDays;
  if (idleHours === 0 && maxAgeDays === 0) return;
  const registry = await readRegistry();
  for (const entry of registry.entries) {
    const idleMs = now - entry.lastUsedAtMs;
    const ageMs = now - entry.createdAtMs;
    if (
      (idleHours > 0 && idleMs > idleHours * 60 * 60 * 1000) ||
      (maxAgeDays > 0 && ageMs > maxAgeDays * 24 * 60 * 60 * 1000)
    ) {
      try {
        await execDocker(["rm", "-f", entry.containerName], {
          allowFailure: true,
        });
      } catch {
        // ignore prune failures
      } finally {
        await removeRegistryEntry(entry.containerName);
      }
    }
  }
}
```

Clawdbot also maintains a registry of containers with usage timestamps:

```typescript
// clawdbot/src/agents/sandbox.ts lines 197-207
type SandboxRegistryEntry = {
  containerName: string;
  sessionKey: string;
  createdAtMs: number;
  lastUsedAtMs: number;
  image: string;
};

type SandboxRegistry = {
  entries: SandboxRegistryEntry[];
};
```

### Why It Matters

- **Resource cleanup**: Containers accumulate over time, consuming disk and memory
- **Disk space**: Old containers can fill up `/var/lib/docker`
- **Stale state**: Old containers may have outdated images or state
- **Operational hygiene**: Automatic cleanup reduces manual maintenance

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sandbox/registry.py` - New container registry
- `/home/dcramer/src/ash/src/ash/sandbox/manager.py` - Add pruning logic
- `/home/dcramer/src/ash/src/ash/config/models.py` - Add prune config
- `/home/dcramer/src/ash/src/ash/cli/commands/sandbox.py` - Add prune command

### Concrete Python Code Changes

```python
# New file: src/ash/sandbox/registry.py
"""Container registry for tracking sandbox containers."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ContainerEntry:
    """Registry entry for a sandbox container."""

    container_id: str
    container_name: str
    session_id: str | None
    image: str
    created_at: datetime
    last_used_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "container_id": self.container_id,
            "container_name": self.container_name,
            "session_id": self.session_id,
            "image": self.image,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContainerEntry":
        """Create from dict."""
        return cls(
            container_id=data["container_id"],
            container_name=data.get("container_name", data["container_id"][:12]),
            session_id=data.get("session_id"),
            image=data["image"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used_at=datetime.fromisoformat(data["last_used_at"]),
        )


class ContainerRegistry:
    """Track sandbox containers for lifecycle management."""

    def __init__(self, registry_path: Path):
        """Initialize registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self._path = registry_path
        self._entries: dict[str, ContainerEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for entry_data in data.get("entries", []):
                entry = ContainerEntry.from_dict(entry_data)
                self._entries[entry.container_id] = entry
        except Exception:
            logger.warning("Failed to load container registry", exc_info=True)

    def _save(self) -> None:
        """Save registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [e.to_dict() for e in self._entries.values()]
        }
        self._path.write_text(json.dumps(data, indent=2))

    def register(
        self,
        container_id: str,
        image: str,
        session_id: str | None = None,
        container_name: str | None = None,
    ) -> ContainerEntry:
        """Register a new container or update existing."""
        now = datetime.now(UTC)

        if container_id in self._entries:
            # Update existing
            entry = self._entries[container_id]
            entry.last_used_at = now
        else:
            # Create new
            entry = ContainerEntry(
                container_id=container_id,
                container_name=container_name or container_id[:12],
                session_id=session_id,
                image=image,
                created_at=now,
                last_used_at=now,
            )
            self._entries[container_id] = entry

        self._save()
        return entry

    def touch(self, container_id: str) -> None:
        """Update last_used_at timestamp."""
        if container_id in self._entries:
            self._entries[container_id].last_used_at = datetime.now(UTC)
            self._save()

    def remove(self, container_id: str) -> None:
        """Remove container from registry."""
        if container_id in self._entries:
            del self._entries[container_id]
            self._save()

    def list_all(self) -> list[ContainerEntry]:
        """List all registered containers."""
        return list(self._entries.values())

    def get_stale(
        self,
        idle_hours: int = 24,
        max_age_days: int = 7,
    ) -> list[ContainerEntry]:
        """Get containers that should be pruned.

        Args:
            idle_hours: Remove if idle longer than this (0 = disabled)
            max_age_days: Remove if older than this (0 = disabled)

        Returns:
            List of containers to prune
        """
        now = datetime.now(UTC)
        stale = []

        for entry in self._entries.values():
            idle_seconds = (now - entry.last_used_at).total_seconds()
            age_seconds = (now - entry.created_at).total_seconds()

            if idle_hours > 0 and idle_seconds > idle_hours * 3600:
                stale.append(entry)
            elif max_age_days > 0 and age_seconds > max_age_days * 86400:
                stale.append(entry)

        return stale
```

```python
# Modify: src/ash/sandbox/manager.py - add pruning

from ash.sandbox.registry import ContainerRegistry

class SandboxManager:
    """Manage Docker containers for sandboxed code execution."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        registry_path: Path | None = None,
    ):
        self._config = config or SandboxConfig()
        self._client: docker.DockerClient | None = None
        self._containers: dict[str, Container] = {}

        # Initialize registry
        default_registry = Path.home() / ".ash" / "sandbox" / "containers.json"
        self._registry = ContainerRegistry(registry_path or default_registry)

    async def create_container(
        self,
        name: str | None = None,
        environment: dict[str, str] | None = None,
        session_id: str | None = None,  # Add session tracking
    ) -> str:
        """Create a new sandbox container with security hardening."""
        # ... existing creation logic ...

        # Register container
        self._registry.register(
            container_id=container.id,
            image=self._config.image,
            session_id=session_id,
            container_name=name,
        )

        return container.id

    async def exec_command(self, container_id: str, ...) -> tuple[int, str, str]:
        """Execute a command in a container."""
        # Touch registry to update last_used_at
        self._registry.touch(container_id)

        # ... existing execution logic ...

    async def prune_containers(
        self,
        idle_hours: int = 24,
        max_age_days: int = 7,
    ) -> list[str]:
        """Remove stale containers.

        Args:
            idle_hours: Remove if idle longer than this
            max_age_days: Remove if older than this

        Returns:
            List of removed container IDs
        """
        stale = self._registry.get_stale(idle_hours, max_age_days)
        removed = []

        for entry in stale:
            try:
                await self.remove_container(entry.container_id, force=True)
                self._registry.remove(entry.container_id)
                removed.append(entry.container_id)
                logger.info(
                    "Pruned container %s (idle: %s, age: %s)",
                    entry.container_id[:12],
                    entry.last_used_at,
                    entry.created_at,
                )
            except Exception:
                logger.warning(
                    "Failed to prune container %s",
                    entry.container_id[:12],
                    exc_info=True,
                )

        return removed
```

```python
# Modify: src/ash/cli/commands/sandbox.py - add prune command

def _sandbox_prune(idle_hours: int, max_age_days: int, dry_run: bool) -> None:
    """Prune stale sandbox containers."""
    import asyncio
    from ash.sandbox.manager import SandboxManager
    from ash.sandbox.registry import ContainerRegistry

    registry_path = Path.home() / ".ash" / "sandbox" / "containers.json"
    registry = ContainerRegistry(registry_path)

    stale = registry.get_stale(idle_hours, max_age_days)

    if not stale:
        console.print("[green]No stale containers to prune[/green]")
        return

    console.print(f"Found {len(stale)} stale container(s):")
    for entry in stale:
        idle = (datetime.now(UTC) - entry.last_used_at).total_seconds() / 3600
        age = (datetime.now(UTC) - entry.created_at).total_seconds() / 86400
        console.print(
            f"  {entry.container_id[:12]} - idle: {idle:.1f}h, age: {age:.1f}d"
        )

    if dry_run:
        console.print("\n[yellow]Dry run - no containers removed[/yellow]")
        return

    manager = SandboxManager()
    removed = asyncio.run(manager.prune_containers(idle_hours, max_age_days))

    success(f"Pruned {len(removed)} container(s)")
```

### Effort Estimate

**Medium** (3-4 hours)
- Create registry module
- Integrate with manager
- Add CLI command
- Add background pruning on startup

### Priority

**High** - Prevents resource exhaustion in long-running deployments.

---

## Gap 4: Symlink Escape Detection

### What Ash is Missing

Ash relies solely on Docker mount isolation for path security. Clawdbot includes explicit symlink traversal detection (`assertSandboxPath()`) as defense in depth. While Docker mounts prevent actual escapes, explicit checks provide:

1. Better error messages
2. Detection of escape attempts for logging/alerting
3. Protection against mount misconfiguration

```typescript
// Example symlink check pattern (not directly in sandbox.ts but used in tools)
function assertSandboxPath(path: string, workspaceDir: string): void {
    const resolved = realpathSync(path);
    if (!resolved.startsWith(workspaceDir)) {
        throw new Error(`Path traversal attempt: ${path} resolves to ${resolved}`);
    }
}
```

### Why It Matters

- **Defense in depth**: Multiple layers of protection
- **Better errors**: "Symlink escape blocked" vs generic Docker errors
- **Audit trail**: Log escape attempts for security monitoring
- **Future-proofing**: Protection if mount config changes

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sandbox/security.py` - New security utilities
- `/home/dcramer/src/ash/src/ash/sandbox/executor.py` - Add path validation

### Concrete Python Code Changes

```python
# New file: src/ash/sandbox/security.py
"""Sandbox security utilities.

Provides defense-in-depth path validation beyond Docker mount isolation.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class PathEscapeError(Exception):
    """Raised when a path escape attempt is detected."""

    def __init__(self, path: str, resolved: str, allowed_root: str):
        self.path = path
        self.resolved = resolved
        self.allowed_root = allowed_root
        super().__init__(
            f"Path escape blocked: '{path}' resolves to '{resolved}' "
            f"which is outside allowed root '{allowed_root}'"
        )


def resolve_sandbox_path(
    path: str | Path,
    workspace_root: Path,
    follow_symlinks: bool = True,
) -> Path:
    """Resolve and validate a path within the sandbox workspace.

    Args:
        path: Path to validate (may be relative or absolute)
        workspace_root: The allowed root directory
        follow_symlinks: Whether to resolve symlinks

    Returns:
        Resolved absolute path

    Raises:
        PathEscapeError: If path escapes workspace
    """
    path = Path(path)
    workspace_root = workspace_root.resolve()

    # Handle relative paths
    if not path.is_absolute():
        path = workspace_root / path

    # Resolve symlinks if requested
    if follow_symlinks:
        try:
            resolved = path.resolve()
        except OSError as e:
            # Path doesn't exist - resolve parent and append filename
            parent = path.parent.resolve() if path.parent.exists() else workspace_root
            resolved = parent / path.name
    else:
        resolved = path

    # Check if resolved path is within workspace
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        logger.warning(
            "Path escape attempt detected: %s -> %s (root: %s)",
            path,
            resolved,
            workspace_root,
        )
        raise PathEscapeError(str(path), str(resolved), str(workspace_root))

    return resolved


def validate_command_paths(
    command: str,
    workspace_root: Path,
) -> list[str]:
    """Extract and validate file paths from a shell command.

    This is a best-effort check - shell parsing is complex.
    Returns list of warnings for suspicious paths.

    Args:
        command: Shell command to analyze
        workspace_root: Allowed workspace root

    Returns:
        List of warning messages for suspicious paths
    """
    import shlex

    warnings = []

    try:
        tokens = shlex.split(command)
    except ValueError:
        # Malformed command - let it fail in execution
        return warnings

    for token in tokens:
        # Skip flags
        if token.startswith("-"):
            continue

        # Check for path-like tokens
        if "/" in token or token.startswith("~"):
            # Expand ~ and environment variables
            expanded = os.path.expanduser(os.path.expandvars(token))

            # Check for obvious escape patterns
            suspicious_patterns = [
                "..",
                "/etc/",
                "/var/",
                "/usr/",
                "/root",
                "/home/",
                "/proc/",
                "/sys/",
            ]

            for pattern in suspicious_patterns:
                if pattern in expanded and not expanded.startswith(str(workspace_root)):
                    warnings.append(
                        f"Suspicious path pattern '{pattern}' in: {token}"
                    )

    return warnings


def is_safe_filename(filename: str) -> bool:
    """Check if a filename is safe to use.

    Rejects:
    - Empty filenames
    - Filenames with path separators
    - Special names like . and ..
    - Names starting with -

    Args:
        filename: Filename to check

    Returns:
        True if filename is safe
    """
    if not filename or not filename.strip():
        return False

    # No path separators
    if "/" in filename or "\\" in filename:
        return False

    # No special names
    if filename in (".", ".."):
        return False

    # No leading dash (could be interpreted as flag)
    if filename.startswith("-"):
        return False

    # No null bytes
    if "\x00" in filename:
        return False

    return True
```

```python
# Modify: src/ash/sandbox/executor.py - add path validation

from ash.sandbox.security import validate_command_paths

async def execute(
    self,
    command: str,
    timeout: int | None = None,
    validate_paths: bool = True,
    **kwargs,
) -> ExecutionResult:
    """Execute a command in the sandbox.

    Args:
        command: Shell command to execute
        timeout: Execution timeout
        validate_paths: Whether to check for suspicious paths
        **kwargs: Additional arguments
    """
    # Validate paths in command (defense in depth)
    if validate_paths and self._config.workspace_path:
        warnings = validate_command_paths(command, self._config.workspace_path)
        for warning in warnings:
            logger.warning("Command path warning: %s", warning)

    # ... existing execution logic ...
```

### Effort Estimate

**Small** (1-2 hours)
- Create security module
- Add path validation
- Integrate with executor
- Add tests

### Priority

**Low** - Docker mounts already provide isolation. This is defense-in-depth.

---

## Gap 5: Container Reuse Optimization

### What Ash is Missing

Ash creates containers per-session, which works but could be optimized. Clawdbot supports multiple scoping strategies:

```typescript
// clawdbot/src/agents/sandbox.ts lines 112
export type SandboxScope = "session" | "agent" | "shared";
```

```typescript
// clawdbot/src/agents/sandbox.ts lines 245-254
export function resolveSandboxScope(params: {
  scope?: SandboxScope;
  perSession?: boolean;
}): SandboxScope {
  if (params.scope) return params.scope;
  if (typeof params.perSession === "boolean") {
    return params.perSession ? "session" : "shared";
  }
  return "agent";
}
```

- **session**: One container per session (Ash's current behavior)
- **agent**: One container per agent (shared across sessions)
- **shared**: One container for all agents

### Why It Matters

- **Startup latency**: Reusing containers is faster than creating new ones
- **Resource efficiency**: Fewer containers running simultaneously
- **State sharing**: Some use cases benefit from shared workspace state
- **Flexibility**: Different deployments have different needs

### Files to Modify

- `/home/dcramer/src/ash/src/ash/sandbox/manager.py` - Add scoping logic
- `/home/dcramer/src/ash/src/ash/config/models.py` - Add scope config

### Concrete Python Code Changes

```python
# Modify: src/ash/config/models.py

from enum import Enum

class SandboxScope(str, Enum):
    """Container scoping strategy."""

    SESSION = "session"  # One container per session (isolated)
    USER = "user"        # One container per user (shared across sessions)
    SHARED = "shared"    # One container for all (maximum reuse)


class SandboxConfig(BaseModel):
    """Sandbox configuration."""

    scope: SandboxScope = SandboxScope.SESSION
    # ... existing fields ...
```

```python
# Modify: src/ash/sandbox/manager.py

class SandboxManager:
    """Manage Docker containers for sandboxed code execution."""

    def __init__(self, config: SandboxConfig | None = None):
        self._config = config or SandboxConfig()
        self._client: docker.DockerClient | None = None
        self._containers: dict[str, Container] = {}

        # Scope-based container cache
        self._scope_containers: dict[str, str] = {}  # scope_key -> container_id

    def _get_scope_key(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Get the container scope key based on config.

        Args:
            session_id: Current session ID
            user_id: Current user ID

        Returns:
            Scope key for container lookup
        """
        scope = self._config.scope

        if scope == SandboxScope.SHARED:
            return "shared"
        elif scope == SandboxScope.USER:
            return f"user:{user_id or 'default'}"
        else:  # SESSION
            return f"session:{session_id or 'default'}"

    async def get_or_create_container(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> str:
        """Get existing container for scope or create new one.

        Args:
            session_id: Current session ID
            user_id: Current user ID
            environment: Environment variables

        Returns:
            Container ID
        """
        scope_key = self._get_scope_key(session_id, user_id)

        # Check for existing container in scope
        if scope_key in self._scope_containers:
            container_id = self._scope_containers[scope_key]
            try:
                container = self._get_container(container_id)
                # Verify container is still running
                container.reload()
                if container.status == "running":
                    logger.debug(
                        "Reusing container %s for scope %s",
                        container_id[:12],
                        scope_key,
                    )
                    return container_id
            except (KeyError, docker.errors.NotFound):
                # Container gone, remove from cache
                del self._scope_containers[scope_key]

        # Create new container
        container_id = await self.create_container(
            name=f"ash-sandbox-{scope_key.replace(':', '-')[:20]}",
            environment=environment,
        )
        await self.start_container(container_id)

        # Cache for reuse
        self._scope_containers[scope_key] = container_id

        return container_id
```

### Effort Estimate

**Small** (1-2 hours)
- Add scope config
- Add scope-based caching
- Update executor to use scoped containers

### Priority

**Medium** - Improves performance for high-throughput deployments.

---

## Gap 6: Sandbox Status/Health Command

### What Ash is Missing

Ash's `sandbox status` command shows basic info but lacks detailed health checks and resource usage. A comprehensive status command would help operators diagnose issues.

Current Ash status output:
- Docker running/not running
- Image built/not built
- Number of running containers

Missing:
- Individual container details
- Resource usage (CPU, memory, disk)
- Health check results
- Registry vs actual container reconciliation
- Image version mismatch warnings

### Why It Matters

- **Debugging**: Quick visibility into sandbox state
- **Operations**: Identify resource-hungry containers
- **Maintenance**: Find stale or unhealthy containers
- **Verification**: Confirm containers match expected configuration

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/commands/sandbox.py` - Enhance status command

### Concrete Python Code Changes

```python
# Modify: src/ash/cli/commands/sandbox.py

import asyncio
from datetime import datetime, UTC
from rich.table import Table
from rich.panel import Panel

def _sandbox_status_detailed() -> None:
    """Show detailed sandbox status with resource usage."""
    # Check Docker daemon
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error("Docker is not running or not accessible")
            return
        docker_info = json.loads(result.stdout)
    except FileNotFoundError:
        error("Docker is not installed")
        return

    # Docker daemon info
    console.print(Panel.fit(
        f"[bold]Docker Daemon[/bold]\n"
        f"Version: {docker_info.get('ServerVersion', 'unknown')}\n"
        f"Containers: {docker_info.get('Containers', 0)} "
        f"(running: {docker_info.get('ContainersRunning', 0)})\n"
        f"Images: {docker_info.get('Images', 0)}",
        title="Docker Status",
    ))

    # Sandbox image info
    result = subprocess.run(
        [
            "docker", "images", "ash-sandbox:latest",
            "--format", "{{.ID}}\t{{.CreatedAt}}\t{{.Size}}",
        ],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        parts = result.stdout.strip().split("\t")
        console.print(Panel.fit(
            f"[bold]Sandbox Image[/bold]\n"
            f"ID: {parts[0]}\n"
            f"Created: {parts[1] if len(parts) > 1 else 'unknown'}\n"
            f"Size: {parts[2] if len(parts) > 2 else 'unknown'}",
            title="Image Status",
        ))
    else:
        warning("Sandbox image not built. Run: ash sandbox build")

    # List running sandbox containers with details
    result = subprocess.run(
        [
            "docker", "ps",
            "--filter", "ancestor=ash-sandbox:latest",
            "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.RunningFor}}",
        ],
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        table = Table(title="Running Sandbox Containers")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Uptime", style="magenta")
        table.add_column("CPU %", justify="right")
        table.add_column("Memory", justify="right")

        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 4:
                container_id = parts[0]

                # Get resource usage
                stats_result = subprocess.run(
                    [
                        "docker", "stats", container_id,
                        "--no-stream",
                        "--format", "{{.CPUPerc}}\t{{.MemUsage}}",
                    ],
                    capture_output=True,
                    text=True,
                )

                cpu = "N/A"
                memory = "N/A"
                if stats_result.returncode == 0 and stats_result.stdout.strip():
                    stats = stats_result.stdout.strip().split("\t")
                    cpu = stats[0] if len(stats) > 0 else "N/A"
                    memory = stats[1] if len(stats) > 1 else "N/A"

                table.add_row(
                    parts[0][:12],
                    parts[1][:20],
                    parts[2],
                    parts[3],
                    cpu,
                    memory,
                )

        console.print(table)
    else:
        console.print("[dim]No running sandbox containers[/dim]")

    # Show all containers (including stopped)
    result = subprocess.run(
        [
            "docker", "ps", "-a",
            "--filter", "ancestor=ash-sandbox:latest",
            "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Size}}",
        ],
        capture_output=True,
        text=True,
    )

    stopped = []
    for line in result.stdout.strip().split("\n"):
        if line and "Exited" in line:
            stopped.append(line)

    if stopped:
        console.print(f"\n[yellow]Stopped containers: {len(stopped)}[/yellow]")
        console.print("[dim]Run 'ash sandbox clean' to remove stopped containers[/dim]")

    # Registry reconciliation
    registry_path = Path.home() / ".ash" / "sandbox" / "containers.json"
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
            entries = registry.get("entries", [])
            if entries:
                console.print(f"\n[dim]Registry entries: {len(entries)}[/dim]")

                # Check for orphaned entries
                for entry in entries:
                    check = subprocess.run(
                        ["docker", "inspect", entry.get("container_id", "")[:12]],
                        capture_output=True,
                    )
                    if check.returncode != 0:
                        warning(
                            f"Orphaned registry entry: {entry.get('container_id', '')[:12]}"
                        )
        except Exception:
            pass


def _sandbox_health_check() -> None:
    """Run health checks on sandbox infrastructure."""
    checks = []

    # Check 1: Docker daemon
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
    )
    checks.append(("Docker daemon", result.returncode == 0))

    # Check 2: Sandbox image
    result = subprocess.run(
        ["docker", "images", "-q", "ash-sandbox:latest"],
        capture_output=True,
        text=True,
    )
    checks.append(("Sandbox image", bool(result.stdout.strip())))

    # Check 3: Can create container
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "ash-sandbox:latest",
            "echo", "health-check",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    checks.append(("Container creation", result.returncode == 0))

    # Check 4: Security features
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "ash-sandbox:latest",
            "whoami",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    checks.append(("Security hardening", result.returncode == 0))

    # Check 5: gVisor (if configured)
    result = subprocess.run(
        ["docker", "info", "--format", "{{.Runtimes}}"],
        capture_output=True,
        text=True,
    )
    has_gvisor = "runsc" in result.stdout
    checks.append(("gVisor runtime", has_gvisor, "optional"))

    # Display results
    table = Table(title="Sandbox Health Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Notes", style="dim")

    all_passed = True
    for check in checks:
        name = check[0]
        passed = check[1]
        notes = check[2] if len(check) > 2 else ""

        if passed:
            status = "[green]PASS[/green]"
        elif notes == "optional":
            status = "[yellow]SKIP[/yellow]"
        else:
            status = "[red]FAIL[/red]"
            all_passed = False

        table.add_row(name, status, notes)

    console.print(table)

    if all_passed:
        success("All health checks passed")
    else:
        error("Some health checks failed")
        raise typer.Exit(1)


# Update the main sandbox command to support new actions
@app.command()
def sandbox(
    action: Annotated[
        str,
        typer.Argument(help="Action: build, status, health, clean, prune"),
    ],
    ...
) -> None:
    """Manage the Docker sandbox environment."""

    if action == "status":
        _sandbox_status_detailed()
    elif action == "health":
        _sandbox_health_check()
    elif action == "prune":
        _sandbox_prune(idle_hours=24, max_age_days=7, dry_run=False)
    # ... existing actions ...
```

### Effort Estimate

**Small** (1-2 hours)
- Enhance status command
- Add health check command
- Add prune command integration

### Priority

**Medium** - Improves operational visibility for production deployments.

---

## Summary

| Gap | Description | Effort | Priority |
|-----|-------------|--------|----------|
| 1. Host Mode Fallback | Execute commands without Docker for dev/simple deployments | Small | Medium |
| 2. Tool Allow/Deny Lists | Granular control over which tools can run in sandbox | Medium | Medium |
| 3. Container Pruning | Automatic cleanup of idle/old containers | Medium | High |
| 4. Symlink Escape Detection | Defense-in-depth path validation | Small | Low |
| 5. Container Reuse | Scope-based container sharing for performance | Small | Medium |
| 6. Status/Health Command | Enhanced CLI for sandbox inspection and diagnostics | Small | Medium |

### Recommended Implementation Order

1. **Container Pruning** (Gap 3) - Prevents resource exhaustion, essential for production
2. **Status/Health Command** (Gap 6) - Helps diagnose issues, quick win
3. **Container Reuse** (Gap 5) - Performance improvement, simple to implement
4. **Tool Allow/Deny Lists** (Gap 2) - Security enhancement for multi-agent setups
5. **Host Mode Fallback** (Gap 1) - Developer convenience, not critical for production
6. **Symlink Escape Detection** (Gap 4) - Defense in depth, Docker already provides isolation
