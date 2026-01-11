# Sandbox

> Isolated Docker container for executing untrusted commands

Files: src/ash/sandbox/manager.py, src/ash/sandbox/executor.py, src/ash/sandbox/verify.py, docker/Dockerfile.sandbox

## Requirements

### MUST

- Execute commands in isolated Docker container
- Run as unprivileged user (not root)
- Read-only root filesystem
- Drop all Linux capabilities
- Block privilege escalation (no sudo, su, setuid)
- Enforce process limits (fork bomb protection)
- Enforce memory limits
- Enforce execution timeout
- Provide writable /tmp and /home/sandbox via tmpfs
- Return exit code, stdout, stderr

### SHOULD

- Support gVisor runtime for enhanced isolation
- Support network modes (none, bridge)
- Support workspace mounting with access control (none, ro, rw)
- Support custom DNS and HTTP proxy
- Support environment variable injection (for API keys)

### MAY

- Seccomp profile customization
- AppArmor profile support
- Per-command resource limits

## Interface

```python
@dataclass
class SandboxConfig:
    image: str = "ash-sandbox:latest"
    timeout: int = 60
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    runtime: Literal["runc", "runsc"] = "runc"
    network_mode: Literal["none", "bridge"] = "none"
    workspace_access: Literal["none", "ro", "rw"] = "rw"

class SandboxExecutor:
    def __init__(
        config: SandboxConfig,
        environment: dict[str, str] | None = None,  # Injected env vars
    ): ...

    async def execute(command: str, timeout: int = None) -> ExecutionResult
    async def cleanup() -> None

@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
```

```bash
ash sandbox build [--force]  # Build sandbox image
ash sandbox status           # Show sandbox status
ash sandbox clean            # Remove containers
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| `echo hello` | exit_code=0, stdout="hello\n" | Basic execution |
| `exit 42` | exit_code=42 | Exit code preserved |
| `sleep 120` | timed_out=True | Killed after timeout |
| `rm -rf /` | exit_code!=0, "Read-only" | Filesystem protected |
| `sudo whoami` | exit_code!=0 | No sudo available |

## Errors

| Condition | Response |
|-----------|----------|
| Docker not running | SandboxError: "Docker is not running" |
| Image not found | SandboxError: "Image not found" |
| Timeout exceeded | ExecutionResult with timed_out=True |
| Container creation fails | SandboxError with details |

## Verification

```bash
uv run pytest tests/test_sandbox_verify.py -v
```

Security tests verify:
- user_is_sandbox - Commands run as 'sandbox' user
- user_not_root - UID != 0
- sudo_blocked - sudo unavailable
- etc_readonly - Cannot write to /etc
- usr_readonly - Cannot write to /usr
- timeout_enforced - Commands timeout after limit
- tmp_writable - /tmp is writable
- python_available - Python 3 works
- bash_available - Bash works
- exit_code_preserved - Non-zero exits reported
