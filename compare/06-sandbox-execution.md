# Sandbox Execution Comparison

This document compares sandbox execution implementations across four codebases: **ash**, **archer**, **clawdbot**, and **pi-mono**.

## Overview

Sandbox execution isolates agent-executed commands from the host system, preventing accidental or malicious damage. All four codebases use Docker as the sandboxing mechanism, but with varying levels of sophistication and security hardening.

| Codebase | Language | Docker Mode | Security Level | Complexity |
|----------|----------|-------------|----------------|------------|
| ash | Python | Mandatory | High | Medium |
| archer | TypeScript | Optional | Low | Low |
| clawdbot | TypeScript | Configurable | High | High |
| pi-mono | TypeScript | Optional | Low | Low |

## Comparison Table

| Feature | ash | archer | clawdbot | pi-mono |
|---------|-----|--------|----------|---------|
| **Execution Mode** | Docker only | Host or Docker | Docker with modes | Host or Docker |
| **Security Hardening** | Comprehensive | Basic | Comprehensive | Basic |
| **Read-only rootfs** | Yes | No | Yes | No |
| **Capabilities dropped** | ALL | No | ALL | No |
| **pids_limit** | 100 | No | Configurable | No |
| **Memory limit** | 512MB default | No | Configurable | No |
| **CPU limit** | 1.0 default | No | Configurable | No |
| **no-new-privileges** | Yes | No | Yes | No |
| **Workspace mount** | none/ro/rw | rw | none/ro/rw | rw |
| **Network isolation** | none/bridge | Host | none/configurable | Host |
| **gVisor support** | Yes (runsc) | No | No | No |
| **Tool policies** | No | No | Yes (allow/deny) | No |
| **Scope modes** | N/A | N/A | session/agent/shared | N/A |
| **Container pruning** | Manual | Manual | Automatic | Manual |
| **Output limits** | No | 10MB | No | 10MB |
| **tmpfs mounts** | Yes (with limits) | No | Yes | No |
| **Timeout handling** | Yes | Yes | Yes | Yes |
| **Docker context aware** | Yes (colima etc.) | No | No | No |
| **seccomp/AppArmor** | No | No | Configurable | No |
| **Custom DNS** | Yes | No | Yes | No |

## Detailed Analysis

### 1. ash (Python)

**Files:** `src/ash/sandbox/manager.py`, `src/ash/sandbox/executor.py`

ash implements the most security-focused approach with Docker as mandatory for all bash execution. It provides comprehensive container hardening out of the box.

#### Architecture

```
SandboxExecutor (high-level API)
    └── SandboxManager (Docker container lifecycle)
            └── docker-py client
```

#### Key Security Features

```python
# From manager.py - Container security configuration
container_config: dict[str, Any] = {
    "read_only": True,  # Immutable root filesystem
    "security_opt": ["no-new-privileges:true"],
    "cap_drop": ["ALL"],  # Drop all capabilities
    "pids_limit": 100,  # Fork bomb protection
    "tmpfs": {
        "/tmp": "size=64m,noexec,nosuid,nodev,uid=1000,gid=1000",
        "/home/sandbox": "size=64m,noexec,nosuid,nodev,uid=1000,gid=1000",
        "/var/tmp": "size=32m,noexec,nosuid,nodev,uid=1000,gid=1000",
        "/run": "size=16m,noexec,nosuid,nodev,uid=1000,gid=1000",
    },
}
```

#### Configuration Options

```python
@dataclass
class SandboxConfig:
    image: str = "ash-sandbox:latest"
    timeout: int = 60
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    runtime: Literal["runc", "runsc"] = "runc"  # runsc = gVisor
    network_mode: Literal["none", "bridge"] = "none"
    workspace_access: Literal["none", "ro", "rw"] = "rw"
    # Additional mounts for sessions, logs, RPC socket
```

#### gVisor Support

ash is unique in supporting gVisor (runsc) for enhanced syscall isolation:

```python
# Use gVisor runtime if configured
if self._config.runtime == "runsc":
    container_config["runtime"] = "runsc"
```

#### Docker Context Awareness

ash detects and respects Docker contexts (colima, Docker Desktop, etc.):

```python
async def _get_docker_host_async() -> str | None:
    """Get the Docker host URL, respecting the current Docker context."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "context", "inspect",
        stdout=asyncio.subprocess.PIPE,
    )
    # ... parse context endpoint
```

#### Custom Dockerfile

The sandbox image includes security hardening:

```dockerfile
# Non-root user
RUN useradd -m -s /bin/bash -u 1000 sandbox

# Remove setuid/setgid binaries
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

# Restricted PATH
ENV PATH=/usr/local/bin:/usr/bin:/bin

# Security aliases in .bashrc
alias sudo='echo "sudo: permission denied"'
alias docker='echo "docker: permission denied"'
```

---

### 2. archer (TypeScript)

**File:** `src/sandbox.ts`

archer provides a simple, optional sandboxing approach with host or Docker execution modes.

#### Architecture

```
createExecutor(config) -> Executor
    ├── HostExecutor (direct execution)
    └── DockerExecutor (container execution)
```

#### Execution Mode Selection

```typescript
export type SandboxConfig =
  | { type: "host" }
  | { type: "docker"; container: string };

export function parseSandboxArg(value: string): SandboxConfig {
    if (value === "host") return { type: "host" };
    if (value.startsWith("docker:")) {
        const container = value.slice("docker:".length);
        return { type: "docker", container };
    }
}
```

#### Host Executor

```typescript
class HostExecutor implements Executor {
    async exec(command: string, options?: ExecOptions): Promise<ExecResult> {
        const shell = process.platform === "win32" ? "cmd" : "sh";
        const shellArgs = process.platform === "win32" ? ["/c"] : ["-c"];

        const child = spawn(shell, [...shellArgs, command], {
            detached: true,
            stdio: ["ignore", "pipe", "pipe"],
        });

        // Output limiting (10MB)
        child.stdout?.on("data", (data) => {
            stdout += data.toString();
            if (stdout.length > 10 * 1024 * 1024) {
                stdout = stdout.slice(0, 10 * 1024 * 1024);
            }
        });
    }
}
```

#### Docker Executor

Docker execution is minimal - wraps commands in `docker exec`:

```typescript
class DockerExecutor implements Executor {
    async exec(command: string, options?: ExecOptions): Promise<ExecResult> {
        const dockerCmd = `docker exec ${this.container} sh -c ${shellEscape(command)}`;
        const hostExecutor = new HostExecutor();
        return hostExecutor.exec(dockerCmd, options);
    }
}
```

**Notable:** No security hardening in container configuration - relies entirely on the pre-created container's settings.

---

### 3. clawdbot (TypeScript)

**Files:** `src/agents/sandbox.ts`, `src/agents/sandbox-paths.ts`

clawdbot has the most sophisticated sandbox implementation with per-agent configuration, tool policies, scope modes, and automatic container pruning.

#### Architecture

```
resolveSandboxContext(config, sessionKey)
    ├── resolveSandboxConfigForAgent() -> SandboxConfig
    ├── ensureSandboxContainer() -> containerName
    ├── ensureSandboxBrowser() -> browser context (optional)
    └── SandboxRegistry (persistent tracking)
```

#### Scope Modes

```typescript
export type SandboxScope = "session" | "agent" | "shared";

// session: One container per session (most isolated)
// agent: One container per agent (moderate)
// shared: Single container for all (least isolated)
```

#### Tool Policies

clawdbot implements allow/deny lists for tools:

```typescript
export type SandboxToolPolicy = {
    allow?: string[];  // Whitelist
    deny?: string[];   // Blacklist
};

const DEFAULT_TOOL_ALLOW = [
    "bash", "process", "read", "write", "edit",
    "sessions_list", "sessions_history", "sessions_send",
    "sessions_spawn", "session_status",
];

const DEFAULT_TOOL_DENY = [
    "browser", "canvas", "nodes", "cron", "discord", "gateway",
];
```

#### Container Creation with Security

```typescript
export function buildSandboxCreateArgs(params: {
    name: string;
    cfg: SandboxDockerConfig;
    scopeKey: string;
}) {
    const args = ["create", "--name", params.name];

    // Security hardening
    if (params.cfg.readOnlyRoot) args.push("--read-only");
    for (const entry of params.cfg.tmpfs) {
        args.push("--tmpfs", entry);
    }
    if (params.cfg.network) args.push("--network", params.cfg.network);
    for (const cap of params.cfg.capDrop) {
        args.push("--cap-drop", cap);
    }
    args.push("--security-opt", "no-new-privileges");

    // Optional seccomp/AppArmor profiles
    if (params.cfg.seccompProfile) {
        args.push("--security-opt", `seccomp=${params.cfg.seccompProfile}`);
    }
    if (params.cfg.apparmorProfile) {
        args.push("--security-opt", `apparmor=${params.cfg.apparmorProfile}`);
    }

    // Resource limits
    if (params.cfg.pidsLimit > 0) {
        args.push("--pids-limit", String(params.cfg.pidsLimit));
    }
    if (memory) args.push("--memory", memory);
    if (cpus > 0) args.push("--cpus", String(cpus));
}
```

#### Path Security

clawdbot includes symlink escape detection:

```typescript
// sandbox-paths.ts
async function assertNoSymlink(relative: string, root: string) {
    const parts = relative.split(path.sep).filter(Boolean);
    let current = root;
    for (const part of parts) {
        current = path.join(current, part);
        const stat = await fs.lstat(current);
        if (stat.isSymbolicLink()) {
            throw new Error(`Symlink not allowed in sandbox path: ${current}`);
        }
    }
}
```

#### Automatic Container Pruning

```typescript
async function pruneSandboxContainers(cfg: SandboxConfig) {
    const idleHours = cfg.prune.idleHours;  // Default: 24
    const maxAgeDays = cfg.prune.maxAgeDays;  // Default: 7

    for (const entry of registry.entries) {
        const idleMs = now - entry.lastUsedAtMs;
        const ageMs = now - entry.createdAtMs;
        if (
            (idleHours > 0 && idleMs > idleHours * 60 * 60 * 1000) ||
            (maxAgeDays > 0 && ageMs > maxAgeDays * 24 * 60 * 60 * 1000)
        ) {
            await execDocker(["rm", "-f", entry.containerName]);
            await removeRegistryEntry(entry.containerName);
        }
    }
}
```

#### Browser Sandbox Integration

clawdbot uniquely supports sandboxed browser containers:

```typescript
export type SandboxBrowserConfig = {
    enabled: boolean;
    image: string;
    containerPrefix: string;
    cdpPort: number;
    vncPort: number;
    noVncPort: number;
    headless: boolean;
    enableNoVnc: boolean;
    autoStart: boolean;
};
```

---

### 4. pi-mono (TypeScript)

**File:** `packages/mom/src/sandbox.ts`

pi-mono shares the same codebase pattern as archer (likely common heritage). The implementation is nearly identical.

#### Key Differences from archer

None significant - the code is functionally identical:

```typescript
// Identical type definitions
export type SandboxConfig =
  | { type: "host" }
  | { type: "docker"; container: string };

// Identical executor pattern
export function createExecutor(config: SandboxConfig): Executor {
    if (config.type === "host") return new HostExecutor();
    return new DockerExecutor(config.container);
}
```

#### Docker Shell Script

pi-mono includes a management script for simple container lifecycle:

```bash
# docker.sh
docker run -d \
    --name "$CONTAINER_NAME" \
    -v "${DATA_DIR}:/workspace" \
    "$IMAGE" \
    tail -f /dev/null
```

**Notable:** No security hardening - uses Alpine with default settings.

---

## Key Differences

### Security Model

| Aspect | ash | archer | clawdbot | pi-mono |
|--------|-----|--------|----------|---------|
| **Philosophy** | Security-first, Docker mandatory | Convenience-first, optional sandbox | Configurable security per agent | Convenience-first, optional sandbox |
| **Root filesystem** | Read-only | Writable | Read-only (configurable) | Writable |
| **Capabilities** | All dropped | All retained | All dropped (configurable) | All retained |
| **User** | Non-root (sandbox:1000) | Root | Configurable | Root |

### Feature Complexity

1. **ash**: Medium complexity, focused on security
   - Single container per executor
   - gVisor support for enhanced isolation
   - Verification test suite included

2. **archer/pi-mono**: Low complexity, simple wrapper
   - Basic host/docker toggle
   - No container management
   - Output limiting only

3. **clawdbot**: High complexity, full-featured
   - Per-agent configuration
   - Tool policy enforcement
   - Scope modes (session/agent/shared)
   - Container registry and pruning
   - Browser sandbox integration

### Container Lifecycle

| Codebase | Creation | Cleanup | Tracking |
|----------|----------|---------|----------|
| ash | On-demand, reused | Manual | In-memory |
| archer | Pre-existing required | Manual | None |
| clawdbot | On-demand, per-scope | Automatic pruning | Persistent registry |
| pi-mono | Pre-existing required | Manual | None |

---

## Recommendations

### For Security-Critical Deployments

**Use ash or clawdbot patterns:**

```python
# Minimum security settings
container_config = {
    "read_only": True,
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges:true"],
    "pids_limit": 100,
    "mem_limit": "512m",
    "network_disabled": True,  # or "none"
}
```

### For Multi-Agent Systems

**Adopt clawdbot's scope model:**

- `session` scope for untrusted/external users
- `agent` scope for internal agents
- `shared` scope only for trusted, controlled environments

### For Simple Use Cases

**archer/pi-mono pattern is acceptable when:**

- Agent is trusted (internal tooling)
- Host isolation not required
- Quick prototyping

### Universal Best Practices

1. **Always drop capabilities** - `--cap-drop=ALL`
2. **Use read-only rootfs** - `--read-only`
3. **Limit resources** - memory, CPU, pids
4. **Run as non-root** - create dedicated user
5. **Isolate network** - `--network=none` by default
6. **Limit tmpfs sizes** - prevent disk exhaustion
7. **Set timeouts** - prevent runaway processes

### Missing Features to Consider

1. **seccomp profiles** - clawdbot supports this, others don't
2. **Resource accounting** - none track actual resource usage
3. **Audit logging** - commands executed should be logged
4. **Rate limiting** - prevent rapid container creation
5. **Image verification** - ensure trusted base images

---

## Summary

| Codebase | Best For | Avoid For |
|----------|----------|-----------|
| **ash** | Production with untrusted input | N/A |
| **archer** | Development/prototyping | Production with untrusted users |
| **clawdbot** | Multi-agent production systems | Simple single-agent setups |
| **pi-mono** | Development/prototyping | Production with untrusted users |

The ideal sandbox implementation combines:
- ash's security hardening and gVisor support
- clawdbot's scope modes and tool policies
- Automatic container lifecycle management
- Comprehensive audit logging
