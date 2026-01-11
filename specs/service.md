# Service

> Background process management with OS-native integration

Files: src/ash/service/manager.py, src/ash/service/backends/*.py, src/ash/cli/app.py

## Requirements

### MUST

- Start/stop/restart the server as a background process
- Report service status with PID, uptime, resource usage
- Support systemd user services on Linux (~/.config/systemd/user/)
- Support launchd user agents on macOS (~/Library/LaunchAgents/)
- Provide fallback daemonization for unsupported systems
- Handle SIGTERM for graceful shutdown
- Store PID file in $ASH_HOME/run/ash.pid
- Write logs to $ASH_HOME/logs/service.log (non-journald backends)
- Auto-detect the best backend for the current system

### SHOULD

- Provide install/uninstall for auto-start on login
- Allow viewing logs via unified interface
- Support log following with -f flag
- Return idempotent results (stop when stopped = success)

### MAY

- Support SIGHUP for configuration reload
- Support Windows services in future
- Provide health check endpoint integration

## Interface

### CLI Commands

```bash
ash service start           # Start background service
ash service start -f        # Run in foreground (no daemonize)
ash service stop            # Stop service gracefully
ash service restart         # Restart service
ash service status          # Show status
ash service logs            # View last 50 lines
ash service logs -f         # Follow logs
ash service logs -n 100     # View last 100 lines
ash service install         # Enable auto-start
ash service uninstall       # Disable auto-start
```

### Python API

```python
class ServiceState(Enum):
    RUNNING, STOPPED, STARTING, STOPPING, FAILED, UNKNOWN

@dataclass
class ServiceStatus:
    state: ServiceState
    pid: int | None
    uptime_seconds: float | None
    memory_mb: float | None
    message: str | None

class ServiceBackend(ABC):
    name: str
    is_available: bool
    async def start() -> bool
    async def stop() -> bool
    async def restart() -> bool
    async def status() -> ServiceStatus
    async def install() -> bool
    async def uninstall() -> bool
    def get_log_source() -> str | Path

class ServiceManager:
    def __init__(backend: ServiceBackend | None = None)
    async def start() -> tuple[bool, str]
    async def stop() -> tuple[bool, str]
    async def restart() -> tuple[bool, str]
    async def status() -> ServiceStatus
    async def install() -> tuple[bool, str]
    async def uninstall() -> tuple[bool, str]
    async def logs(follow: bool, lines: int) -> AsyncIterator[str]
```

## Configuration

```toml
[service]
log_level = "info"           # debug, info, warning, error
max_log_size = 10485760      # 10MB
log_backup_count = 5
```

## Paths

```python
get_pid_path() -> Path      # $ASH_HOME/run/ash.pid
get_service_log_path() -> Path  # $ASH_HOME/logs/service.log
```

## Backends

### Systemd (Linux)

- Unit file: ~/.config/systemd/user/ash.service
- Commands: systemctl --user start/stop/enable/disable ash
- Logs: journalctl --user -u ash
- Detection: systemctl --user status succeeds

### Launchd (macOS)

- Plist: ~/Library/LaunchAgents/com.ash.agent.plist
- Commands: launchctl load/unload
- Logs: $ASH_HOME/logs/service.log
- Detection: sys.platform == "darwin" and launchctl exists

### Generic (Fallback)

- PID file: $ASH_HOME/run/ash.pid
- Start: Fork subprocess with start_new_session
- Stop: SIGTERM, wait, SIGKILL if needed
- Logs: $ASH_HOME/logs/service.log
- Install: Not supported (returns error)

## Behaviors

| Action | Already Running | Not Running |
|--------|-----------------|-------------|
| start | Error (show PID) | Start, return success |
| stop | SIGTERM → wait → success | Success (idempotent) |
| restart | Stop then start | Start |
| status | Show PID, uptime, memory | Show "stopped" |
| install | Create unit/plist, enable | Same |
| uninstall | Stop, remove files | Remove files |

## Errors

| Condition | Response |
|-----------|----------|
| No service manager | Fall back to generic |
| Service already running | Error with PID |
| Permission denied | Error with suggestion |
| Install on generic backend | Error: "Auto-start requires systemd/launchd" |
| Stop timeout | SIGKILL after 3 seconds |

## Verification

```bash
# Start/stop cycle
ash service start && ash service status
ash service stop && ash service status

# Logs
ash service start
ash service logs -n 10
ash service logs -f &
ash service stop

# Install/uninstall (Linux/macOS only)
ash service install
ash service status  # After reboot/relogin
ash service uninstall

# Unit tests
uv run pytest tests/test_service.py -v
```
