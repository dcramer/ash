# Server

> FastAPI application for webhooks and API endpoints

Files: src/ash/server/app.py, src/ash/server/routes/health.py, src/ash/server/routes/webhooks.py

## Requirements

### MUST

- FastAPI application with lifespan management
- Health check endpoint at /health
- Telegram webhook endpoint at /webhook/telegram
- Connect database on startup
- Disconnect database on shutdown
- Stop providers on shutdown

### SHOULD

- Store components in app.state for dependency injection
- Return 200 for webhook errors (prevent Telegram retries)
- Support streaming responses via Telegram provider

### MAY

- Add authentication for API endpoints
- Add rate limiting
- Add metrics endpoint

## Interface

```python
class AshServer:
    def __init__(
        self,
        database: Database,
        agent: Agent,
        telegram_provider: TelegramProvider | None = None,
    ): ...

    @property
    def app(self) -> FastAPI

    async def get_telegram_handler(self) -> TelegramMessageHandler | None

def create_app(
    database: Database,
    agent: Agent,
    telegram_provider: TelegramProvider | None = None,
) -> FastAPI: ...
```

### Routes

```
GET  /health          -> {"status": "ok"}
POST /webhook/telegram -> 200 OK (empty)
```

### App State

```python
app.state.server: AshServer
app.state.database: Database
app.state.agent: Agent
app.state.telegram_provider: TelegramProvider | None
```

## Configuration

```toml
[server]
host = "0.0.0.0"
port = 8000

[telegram]
webhook_url = "https://example.com"  # Enables webhook mode
```

## CLI

```bash
uv run ash serve              # Start server
uv run ash serve --host 0.0.0.0 --port 8080
```

## Behaviors

| Scenario | Behavior |
|----------|----------|
| Startup | Connect DB, init Telegram handler |
| Shutdown | Stop Telegram, disconnect DB |
| Health check | Return {"status": "ok"} |
| Telegram webhook | Parse JSON, process update, return 200 |
| Webhook error | Log exception, return 200 |
| No Telegram config | Skip Telegram routes |

## Errors

| Condition | Response |
|-----------|----------|
| Database connection failed | Server fails to start |
| Telegram not configured | 500 on webhook (shouldn't happen) |
| Webhook processing error | 200 OK (logged, prevents retry) |
| Invalid webhook JSON | Logged, 200 OK |

## Verification

```bash
uv run pytest tests/test_server.py -v
uv run ash serve &
curl http://localhost:8000/health
```

- Server starts and responds
- Health endpoint returns OK
- Webhook endpoint receives updates
- Clean shutdown on SIGTERM
