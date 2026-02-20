# Ash - Personal Assistant Agent Implementation Plan

## Overview

**Ash** is a Python-based personal assistant agent with customizable personality (SOUL), memory, sandboxed tool execution, and Telegram integration.

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.12+ | Latest features, best async support |
| Package Manager | uv | 10-100x faster than pip/poetry, Rust-based |
| Type Checker | ty | Astral's type checker, 10-60x faster than mypy |
| Linter/Formatter | ruff | Replaces flake8, black, isort in one tool |
| Async | asyncio | Native, works with all chosen libs |
| CLI | Typer | Type hints, auto-help, great DX |
| HTTP Server | FastAPI | Async-native, OpenAPI docs |
| Telegram | aiogram 3.x | Fully async, modern Python |
| Config | TOML + Markdown | TOML for settings, MD for identity |
| Database | SQLite + sqlite-vec | Embedded, vector search for memory |
| Vector Search | sqlite-vec via SemanticRetriever | See decision below |
| ORM | SQLAlchemy 2.0 | Async support, industry standard |
| Migrations | Alembic | SQLAlchemy's migration tool, batch mode for SQLite |
| LLM | anthropic + openai SDKs | Official async SDKs |
| Sandbox | docker-py | Official Python SDK |
| Web Search | Brave Search API | Good free tier, privacy-focused |
| Testing | pytest + pytest-asyncio | Industry standard, async support |

## Architectural Decisions

### Vector Search: sqlite-vec

We use sqlite-vec for vector embeddings rather than alternatives like LanceDB.

**Why sqlite-vec:**
- **Right-sized for our scale** - We have ~1000s of vectors (memories), not millions/billions
- **Single file storage** - Everything in one inspectable `.db` file
- **Standard SQL inserts** - Each memory addition is a single insert, not batch
- **Immediate deletes** - Memory supersession creates frequent deletes; sqlite-vec handles this natively
- **Mature foundation** - SQLite is decades old with proven reliability

**Why not LanceDB:**
- Designed for millions/billions of vectors with batch operations
- Single inserts create fragmentation (needs batching 10-100k rows)
- Soft deletes require manual compaction in OSS version
- Storage is a directory with multiple `.lance` files
- Young project (v1.0.0 Dec 2024) compared to SQLite

**Abstraction Layer:**
The `SemanticRetriever` class in `src/ash/memory/retrieval.py` abstracts vector operations:
- `index_memory(memory_id, content)` - Store embedding
- `search_memories(query, ...)` - Vector similarity search
- `delete_memory_embedding(memory_id)` - Remove embedding
- `initialize_vector_tables()` - Setup

This abstraction allows future replacement of the vector backend if needed.

### Skill State: File-Based Storage

Skill state uses simple JSON files at `~/.ash/data/skills/<skill-name>.json` rather than SQLite.

**Rationale:**
- Simple key-value data doesn't need a database
- Files are directly inspectable (`cat`, `jq`)
- Aligns with "filesystem first" principle from CLAUDE.md
- Currently unused infrastructure - no existing data to migrate

## Complete Toolchain

### Development Tools
| Tool | Version | Purpose |
|------|---------|---------|
| **uv** | latest | Package management, virtual environments, lockfile |
| **ruff** | >=0.8.0 | Linting (replaces flake8, pylint) + formatting (replaces black, isort) |
| **ty** | beta | Type checking (Astral's mypy replacement, 10-60x faster) |
| **prek** | latest | Git hooks for automated quality checks (Rust-based pre-commit) |
| **pytest** | >=8.0.0 | Testing framework |
| **pytest-asyncio** | >=0.24.0 | Async test support |
| **pytest-cov** | >=5.0.0 | Code coverage |

### Runtime Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| **typer** | >=0.12.0 | CLI framework with type hints |
| **fastapi** | >=0.115.0 | Async HTTP server |
| **uvicorn** | >=0.32.0 | ASGI server |
| **aiogram** | >=3.15.0 | Telegram Bot API (async) |
| **anthropic** | >=0.40.0 | Claude API SDK |
| **openai** | >=1.50.0 | OpenAI API SDK |
| **sqlalchemy** | >=2.0.0 | Async ORM |
| **alembic** | >=1.14.0 | Database migrations |
| **aiosqlite** | >=0.20.0 | Async SQLite driver |
| **sqlite-vec** | >=0.1.0 | Vector search extension |
| **pydantic** | >=2.9.0 | Data validation |
| **pydantic-settings** | >=2.5.0 | Settings management |
| **docker** | >=7.0.0 | Docker SDK for sandboxing |
| **httpx** | >=0.27.0 | Async HTTP client |
| **rich** | >=13.0.0 | Terminal formatting |

### Build & Packaging
| Tool | Purpose |
|------|---------|
| **hatchling** | PEP 517 build backend |
| **uv.lock** | Reproducible dependency lockfile |

## Directory Structure

```
ash/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI
├── .pre-commit-config.yaml         # Prek/pre-commit hooks config
├── .python-version                 # Python 3.12
├── .gitignore
├── LICENSE
├── README.md
├── SPEC.md                         # This file
├── pyproject.toml                  # All config consolidated
├── uv.lock                         # Lock file (commit this!)
├── alembic.ini                     # Alembic configuration
├── config.example.toml             # Example user config
│
├── migrations/                     # Alembic migrations
│   ├── env.py                      # Migration environment
│   ├── script.py.mako              # Migration template
│   └── versions/                   # Migration files
│       └── 001_initial_schema.py
│
├── src/
│   └── ash/
│       ├── __init__.py
│       ├── __main__.py             # python -m ash
│       ├── py.typed                # PEP 561 marker
│       │
│       ├── cli/                    # Typer CLI
│       │   ├── __init__.py         # Export app
│       │   ├── app.py              # Main Typer app
│       │   └── commands/
│       │       ├── __init__.py
│       │       ├── serve.py        # ash serve
│       │       ├── config.py       # ash config
│       │       ├── db.py           # ash db (migrate, rollback, status)
│       │       └── memory.py       # ash memory
│       │
│       ├── core/                   # Core abstractions
│       │   ├── __init__.py
│       │   ├── agent.py            # Main orchestrator
│       │   ├── session.py          # Session management
│       │   └── types.py            # Shared types
│       │
│       ├── config/                 # Configuration
│       │   ├── __init__.py
│       │   ├── loader.py           # TOML + env loading
│       │   ├── models.py           # Pydantic models
│       │   └── workspace.py        # SOUL.md/USER.md loading
│       │
│       ├── llm/                    # LLM abstraction
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract interface
│       │   ├── anthropic.py        # Claude provider
│       │   ├── openai.py           # OpenAI provider
│       │   ├── registry.py         # Provider registry
│       │   └── types.py            # Message types
│       │
│       ├── providers/              # Communication providers
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract interface
│       │   ├── registry.py         # Provider registry
│       │   └── telegram/
│       │       ├── __init__.py
│       │       ├── provider.py     # Telegram implementation
│       │       └── handlers.py     # Message handlers
│       │
│       ├── tools/                  # Tool system
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract interface
│       │   ├── registry.py         # Discovery + registration
│       │   ├── executor.py         # Tool execution
│       │   └── builtin/
│       │       ├── __init__.py
│       │       ├── bash.py         # Sandboxed bash
│       │       └── web_search.py   # Brave Search
│       │
│       ├── sandbox/                # Docker sandboxing
│       │   ├── __init__.py
│       │   ├── manager.py          # Container lifecycle
│       │   └── executor.py         # Command execution
│       │
│       ├── db/                     # Database layer
│       │   ├── __init__.py
│       │   ├── engine.py           # Async SQLAlchemy engine
│       │   └── models.py           # SQLAlchemy ORM models
│       │
│       ├── memory/                 # Memory + retrieval
│       │   ├── __init__.py
│       │   ├── store.py            # Memory store (uses db layer)
│       │   ├── embeddings.py       # Embedding generation
│       │   └── retrieval.py        # Semantic search
│       │
│       └── server/                 # HTTP server
│           ├── __init__.py
│           ├── app.py              # FastAPI app
│           └── routes/
│               ├── __init__.py
│               ├── webhooks.py     # Provider webhooks
│               └── health.py       # Health checks
│
├── workspace/                      # Default workspace template
│   ├── SOUL.md                     # Agent personality
│   ├── USER.md                     # User profile template
│   └── TOOLS.md                    # Tool documentation
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_llm.py
│   │   ├── test_memory.py
│   │   └── test_tools.py
│   └── integration/
│       ├── __init__.py
│       └── test_agent.py
│
└── docker/
    ├── Dockerfile                  # Main application
    ├── Dockerfile.sandbox          # Sandbox base image
    └── docker-compose.yml          # Development stack
```

## Implementation Phases

### Phase 1: Project Foundation ✅
1. Initialize with `uv init`
2. Set up pyproject.toml with all dependencies and tool config
3. Create directory structure (src layout)
4. Set up prek hooks
5. Create .gitignore, README.md, LICENSE
6. Implement configuration loading (`config/loader.py`, `config/models.py`)
7. Create example config file (`config.example.toml`)
8. Set up basic CLI with Typer (`cli/app.py`)
9. Add `py.typed` marker for PEP 561

### Phase 2: LLM Abstraction Layer
1. Define message types (`llm/types.py`)
   - Message, ContentBlock, ToolUse, ToolResult
   - StreamChunk for streaming responses
2. Define LLM provider interface (`llm/base.py`)
   - `complete()` and `stream()` methods
   - `embed()` for embeddings
3. Implement Anthropic provider (`llm/anthropic.py`)
4. Implement OpenAI provider (`llm/openai.py`)
5. Create provider registry (`llm/registry.py`)

### Phase 3: Database & Memory System
1. Set up async SQLAlchemy engine (`db/engine.py`)
2. Define SQLAlchemy ORM models (`db/models.py`)
3. Initialize Alembic with async support (`migrations/env.py`)
4. Create initial migration (`migrations/versions/001_initial_schema.py`)
5. Implement memory store (`memory/store.py`)
6. Implement embedding generation (`memory/embeddings.py`)
7. Implement semantic search with sqlite-vec (`memory/retrieval.py`)
8. Add `ash db` CLI commands (migrate, rollback, status)

### Phase 4: Docker Sandbox
1. Create sandbox Dockerfile (`docker/Dockerfile.sandbox`)
2. Implement sandbox manager (`sandbox/manager.py`)
3. Implement command executor (`sandbox/executor.py`)

### Phase 5: Tool System
1. Define tool interface (`tools/base.py`)
2. Create tool registry with discovery (`tools/registry.py`)
3. Implement bash tool (`tools/builtin/bash.py`)
4. Implement web search tool (`tools/builtin/web_search.py`)

### Phase 6: Agent Core
1. Implement session management (`core/session.py`)
2. Create workspace loader for SOUL.md/USER.md (`config/workspace.py`)
3. Implement agent orchestrator with agentic loop (`core/agent.py`)

### Phase 7: Telegram Provider
1. Define provider interface (`providers/base.py`)
2. Implement Telegram provider with aiogram (`providers/telegram/`)
3. Support both polling and webhook modes
4. Implement streaming responses (edit message as content arrives)

### Phase 8: Server & CLI Commands
1. Create FastAPI app with webhook routes (`server/app.py`)
2. Implement `ash serve` command
3. Implement `ash config` commands
4. Implement `ash memory` commands

### Phase 9: Integration & Polish
1. Create default workspace files (SOUL.md, USER.md)
2. Write docker-compose.yml for development
3. Add tests for core components
4. Set up GitHub Actions CI
5. Documentation and README

## Key Interfaces

### LLM Provider
```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from ash.llm.types import Message, StreamChunk, ToolDefinition

class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Message: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]: ...

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]: ...
```

### Communication Provider
```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Awaitable
from ash.providers.types import IncomingMessage, OutgoingMessage

MessageHandler = Callable[[IncomingMessage], Awaitable[None]]

class Provider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def start(self, handler: MessageHandler) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send(self, message: OutgoingMessage) -> str: ...

    @abstractmethod
    async def send_streaming(
        self,
        chat_id: str,
        stream: AsyncIterator[str],
        *,
        reply_to: str | None = None,
    ) -> str: ...
```

### Tool
```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from ash.tools.types import ToolResult, ToolContext

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]: ...

    @abstractmethod
    async def execute(
        self,
        input: BaseModel,
        context: ToolContext,
    ) -> ToolResult: ...
```

## Database & Migrations

### Alembic Configuration (alembic.ini)

```ini
[alembic]
script_location = migrations
sqlalchemy.url = sqlite+aiosqlite:///%(here)s/data/ash.db

[post_write_hooks]
hooks = ruff
ruff.type = exec
ruff.executable = uv
ruff.options = run ruff format REVISION_SCRIPT_FILENAME
```

### Async Migration Environment (migrations/env.py)

```python
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from ash.db.models import Base
from ash.config import get_settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # Required for SQLite ALTER TABLE
    )
    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # Required for SQLite ALTER TABLE
    )
    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

### SQLAlchemy Models (src/ash/db/models.py)

```python
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, relationship

class Base(DeclarativeBase):
    pass

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    provider = Column(String, nullable=False)
    chat_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_ = Column("metadata", JSON)

    messages = relationship("Message", back_populates="session")

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    token_count = Column(Integer)
    metadata_ = Column("metadata", JSON)

    session = relationship("Session", back_populates="messages")
```

## Memory Schema

```sql
-- Sessions/Conversations
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    UNIQUE(provider, chat_id)
);

-- Messages
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_count INTEGER,
    metadata JSON
);
CREATE INDEX idx_messages_session ON messages(session_id, created_at);

-- Vector embeddings (sqlite-vec)
CREATE VIRTUAL TABLE message_embeddings USING vec0(
    message_id TEXT PRIMARY KEY,
    embedding FLOAT[1536]
);

-- Memory entries
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    owner_user_id TEXT,
    subject_person_id TEXT,
    metadata JSON
);

CREATE VIRTUAL TABLE memory_embeddings USING vec0(
    memory_id TEXT PRIMARY KEY,
    embedding FLOAT[1536]
);

-- User profiles
CREATE TABLE user_profiles (
    user_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    username TEXT,
    display_name TEXT,
    profile_data JSON,
    notes TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tool execution history
CREATE TABLE tool_executions (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    tool_name TEXT NOT NULL,
    input JSON NOT NULL,
    output TEXT,
    success BOOLEAN NOT NULL,
    duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_tool_exec_session ON tool_executions(session_id, created_at);
```

## Configuration Structure

```toml
# ~/.ash/config.toml
workspace = "~/.ash/workspace"

[default_llm]
provider = "anthropic"
model = "claude-sonnet-4-6"
temperature = 0.7
max_tokens = 4096

[fallback_llm]
provider = "openai"
model = "gpt-4o"

[telegram]
# bot_token loaded from TELEGRAM_BOT_TOKEN env var
allowed_users = ["@username", "123456789"]
webhook_url = "https://..."  # optional, uses polling if omitted

[sandbox]
image = "ash-sandbox:latest"
timeout = 60
memory_limit = "512m"
cpu_limit = 1.0
network_disabled = true

[server]
host = "127.0.0.1"
port = 8080
webhook_path = "/webhook"

[memory]
database_path = "~/.ash/memory.db"
embedding_model = "text-embedding-3-small"
max_context_messages = 20

[brave_search]
# api_key loaded from BRAVE_SEARCH_API_KEY env var
```

## Developer Workflow

```bash
# Initial setup
git clone <repo>
cd ash
uv sync --all-groups
prek install

# Database migrations
uv run ash upgrade                  # Apply migrations + check sandbox
uv run ash db rollback              # Rollback last migration
uv run ash db status                # Show migration status
uv run alembic revision --autogenerate -m "description"  # Create new migration

# Development
uv run ash serve                    # Start server
uv run pytest                       # Run tests
uv run ruff check --fix .           # Lint
uv run ruff format .                # Format

# Type checking (when ty is stable)
uvx ty check

# Add dependency
uv add <package>
uv add --dev <package>

# Docker development
docker compose up -d
```

## Verification Plan

1. **Linting & Formatting**: `uv run ruff check . && uv run ruff format --check .`
2. **Type Checking**: `uvx ty check` (when stable) or `uv run pyright`
3. **Unit tests**: `uv run pytest tests/unit`
4. **Integration tests**: `uv run pytest tests/integration`
5. **Coverage**: `uv run pytest --cov-report=html` (target 80%+)
6. **Manual testing**:
   - Send message via Telegram, verify response
   - Test bash tool execution in sandbox
   - Test web search tool
   - Verify memory retrieval works
   - Test streaming responses
7. **Docker**: `docker compose up` and test full stack

## CI Pipeline (.github/workflows/ci.yml)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --all-groups
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest --cov-report=xml
      - uses: codecov/codecov-action@v4
```

## Critical Files

- `pyproject.toml` - All project configuration and dependencies
- `alembic.ini` - Database migration configuration
- `migrations/env.py` - Async migration environment
- `src/ash/db/models.py` - SQLAlchemy ORM models
- `src/ash/db/engine.py` - Async database engine
- `src/ash/core/agent.py` - Agentic loop orchestrator
- `src/ash/llm/base.py` - LLM provider interface
- `src/ash/tools/base.py` - Tool interface
- `src/ash/providers/base.py` - Communication provider interface
- `src/ash/memory/store.py` - Memory store with retrieval
- `src/ash/sandbox/manager.py` - Docker container management
