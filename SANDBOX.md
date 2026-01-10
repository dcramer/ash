# Sandbox Specification

This document specifies the security model and expected behavior of Ash's Docker sandbox for executing untrusted bash commands.

## Overview

All bash commands from the LLM run in an isolated Docker container. The sandbox is **mandatory** - there is no option to run commands directly on the host. This protects against malicious or accidental damage from AI-generated commands.

## Security Model

### Threat Model

The sandbox protects against:
- **Malicious commands** - LLM generating harmful commands (intentional or via prompt injection)
- **Accidental damage** - Commands that could damage the host system
- **Resource exhaustion** - Fork bombs, memory exhaustion, disk filling
- **Data exfiltration** - Unauthorized access to host files or secrets
- **Privilege escalation** - Attempts to gain root or host access

### Trust Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                     HOST SYSTEM                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   Ash Agent                             │ │
│  │  - Runs on host                                         │ │
│  │  - Has access to config (~/.ash/)                       │ │
│  │  - Has access to SQLite database                        │ │
│  │  - Communicates with LLM API                            │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                    Tool Execution                            │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Docker Container (Sandbox)                 │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  Bash commands execute here                       │  │ │
│  │  │  - Isolated filesystem                            │  │ │
│  │  │  - Limited resources                              │  │ │
│  │  │  - Unprivileged user                              │  │ │
│  │  │  - Optional network access                        │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Security Controls

### Container Isolation

| Control | Implementation | Purpose |
|---------|----------------|---------|
| Read-only root filesystem | `--read-only` | Prevent persistent changes |
| Dropped capabilities | `cap_drop: ALL` | Remove Linux capabilities |
| No privilege escalation | `no-new-privileges` | Prevent setuid exploitation |
| Process limit | `pids_limit: 100` | Fork bomb protection |
| Memory limit | `mem_limit: 512m` | Memory exhaustion protection |
| CPU limit | `cpu_limit: 1.0` | CPU exhaustion protection |
| Non-root user | `USER sandbox` | Reduced privilege |
| Removed setuid binaries | Dockerfile cleanup | Prevent privilege escalation |

### Filesystem Access

| Path | Access | Notes |
|------|--------|-------|
| `/` (root) | Read-only | Immutable base system |
| `/etc`, `/usr`, `/bin` | Read-only | System directories protected |
| `/workspace` | Configurable (none/ro/rw) | Mounted from host workspace |
| `/tmp` | Read-write (tmpfs, 64MB) | Temporary files, noexec |
| `/home/sandbox` | Read-write (tmpfs, 64MB) | User home, noexec |
| `/var/tmp` | Read-write (tmpfs, 32MB) | Temporary files, noexec |
| `/run` | Read-write (tmpfs, 16MB) | Runtime files, noexec |
| `/root` | No access | Root home inaccessible |

### Network Access

| Mode | Behavior |
|------|----------|
| `none` | Completely isolated, no network |
| `bridge` | Standard Docker networking, can reach internet |

Optional controls when network enabled:
- `dns_servers` - Custom DNS for filtering (e.g., Pi-hole)
- `http_proxy` - Route traffic through proxy for monitoring

### Runtime Options

| Runtime | Security Level | Trade-off |
|---------|---------------|-----------|
| `runc` (default) | High | Standard container isolation |
| `runsc` (gVisor) | Very High | Syscall interception, slight performance overhead |

## Expected Behaviors

### MUST Allow

1. **Command execution** - Bash commands run and return output
2. **Python execution** - `python3` available for scripting
3. **Common tools** - `git`, `curl`, `jq`, `vim`, `less`, `tree` available
4. **Workspace access** - Read/write to `/workspace` when configured
5. **Temp file creation** - Write to `/tmp` for temporary files
6. **Network requests** - HTTP/HTTPS when `network_mode: bridge`
7. **Exit codes** - Non-zero exit codes preserved and reported
8. **Stderr capture** - Error output captured and returned

### MUST Block

1. **System modification** - Writing to `/etc`, `/usr`, `/bin`, etc.
2. **Privilege escalation** - `sudo`, `su`, setuid binaries
3. **Container escape** - Access to host filesystem outside mounts
4. **Resource exhaustion** - Fork bombs, memory bombs limited
5. **Persistent malware** - Read-only filesystem prevents persistence
6. **Host secret access** - No access to host environment variables
7. **Unlimited execution** - Commands timeout after configured limit

### SHOULD Behave

1. **Timeout handling** - Long-running commands killed after timeout
2. **Output truncation** - Very long output truncated to prevent memory issues
3. **Graceful errors** - Clear error messages for blocked operations
4. **Clean environment** - No leaked state between command executions

## Configuration

```toml
[sandbox]
# Container image (build with: ash sandbox build)
image = "ash-sandbox:latest"

# Execution limits
timeout = 60          # seconds
memory_limit = "512m"
cpu_limit = 1.0

# Runtime: "runc" (default) or "runsc" (gVisor)
runtime = "runc"

# Network: "none" (isolated) or "bridge" (has network)
network_mode = "bridge"

# Optional: Custom DNS servers for filtering
# dns_servers = ["1.1.1.1", "8.8.8.8"]

# Optional: HTTP proxy for monitoring traffic
# http_proxy = "http://localhost:8888"

# Workspace mounting: "none", "ro" (read-only), "rw" (read-write)
workspace_access = "rw"
```

## Verification

### Automated Tests

Run the automated verification suite:

```bash
ash sandbox verify
```

This runs 31 tests across 5 categories:
- **SECURITY** (10 tests) - User isolation, filesystem restrictions
- **RESOURCES** (4 tests) - Timeouts, tmpfs, noexec
- **NETWORK** (3 tests) - DNS, HTTP, HTTPS connectivity
- **FUNCTIONAL** (8 tests) - Available tools and utilities
- **EDGE_CASES** (6 tests) - Special characters, output handling

### Manual Prompt Tests

View manual test cases for prompt evaluation:

```bash
ash sandbox prompts
```

Key scenarios to test:
1. `rm -rf /` → "Read-only file system"
2. `sudo whoami` → "command not found" or "permission denied"
3. Fork bomb `:(){ :|:& };:` → Contained by pids limit
4. Memory bomb → Killed by memory limit

## Incident Response

If a sandbox escape or security issue is discovered:

1. **Stop the service** - `ash sandbox clean` removes all containers
2. **Review logs** - Check what commands were executed
3. **Update image** - `ash sandbox build --force` rebuilds with fixes
4. **Report issue** - File security issue in repository

## Future Enhancements

Potential improvements under consideration:
- [ ] Seccomp profile customization
- [ ] AppArmor profile support
- [ ] Network allowlist (specific hosts only)
- [ ] Per-command resource limits
- [ ] Audit logging of all commands
- [ ] Container image signing
