# Sentry

> Optional error tracking, logging, and performance monitoring

Files: src/ash/observability/__init__.py, src/ash/config/models.py

## Requirements

### MUST

- Be an optional dependency (not required for core functionality)
- Initialize before any async operations
- Capture unhandled exceptions automatically
- Forward Python logging at ERROR+ level to Sentry events
- Support DSN configuration via config file or SENTRY_DSN env var
- Support environment and release tags
- Skip initialization gracefully when sentry-sdk not installed
- Skip initialization gracefully when DSN not configured

### SHOULD

- Enable AsyncIO integration for proper async context
- Enable FastAPI integration when running in server mode
- Collect breadcrumbs from INFO+ log messages
- Support configurable traces sample rate
- Not send PII by default (GDPR compliance)

### MAY

- Support profiling (profiles_sample_rate)
- Support custom tags per request

## Interface

```python
def init_sentry(config: SentryConfig, server_mode: bool = False) -> bool:
    """Initialize Sentry if available and configured.

    Returns True if initialized, False if skipped.
    """

class SentryConfig(BaseModel):
    dsn: SecretStr | None = None
    environment: str | None = None
    release: str | None = None
    traces_sample_rate: float = 0.1  # 0.0-1.0
    profiles_sample_rate: float = 0.0  # 0.0-1.0
    send_default_pii: bool = False
    debug: bool = False
```

## Configuration

```toml
[sentry]
dsn = "https://..."  # or SENTRY_DSN env var (required to enable)
environment = "production"  # optional
release = "ash@0.1.0"  # optional
traces_sample_rate = 0.1  # 0.0-1.0, default 0.1
profiles_sample_rate = 0.0  # 0.0-1.0, default 0.0
send_default_pii = false  # default false
debug = false  # SDK debug logging
```

## Behaviors

| Scenario | Behavior |
|----------|----------|
| sentry-sdk not installed | Skip initialization, return False |
| DSN not configured | Skip initialization, return False |
| DSN configured | Initialize with integrations, return True |
| Server mode | Include FastAPI + AsyncIO + Logging integrations |
| CLI mode | Include AsyncIO + Logging integrations |
| Unhandled exception | Captured and sent to Sentry |
| logger.error() call | Creates Sentry event |
| logger.info() call | Added as breadcrumb |

## Errors

| Condition | Response |
|-----------|----------|
| Invalid DSN format | sentry_sdk raises on init |
| Network unavailable | Events queued, no crash |
| Invalid sample rate | Pydantic ValidationError (must be 0.0-1.0) |

## Verification

```bash
# Install optional dep
uv sync --extra sentry

# Verify config loads with sentry section
uv run ash config validate

# Run tests
uv run pytest tests/ -v
```

- Sentry skipped when not installed
- Sentry skipped when DSN not configured
- Sentry initializes with valid DSN
- FastAPI integration only in server mode
- Sample rate validation works (0.0-1.0 range)
