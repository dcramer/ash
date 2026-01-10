"""Configuration models using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr


class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider: Literal["anthropic", "openai"]
    model: str
    api_key: SecretStr | None = None
    temperature: float = 0.7
    max_tokens: int = 4096


class TelegramConfig(BaseModel):
    """Configuration for Telegram provider."""

    bot_token: SecretStr | None = None
    allowed_users: list[str] = []
    webhook_url: str | None = None


class SandboxConfig(BaseModel):
    """Configuration for Docker sandbox."""

    image: str = "ash-sandbox:latest"
    timeout: int = 60
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    network_disabled: bool = True


class ServerConfig(BaseModel):
    """Configuration for HTTP server."""

    host: str = "127.0.0.1"
    port: int = 8080
    webhook_path: str = "/webhook"


class MemoryConfig(BaseModel):
    """Configuration for memory system."""

    database_path: Path = Path("~/.ash/memory.db")
    embedding_model: str = "text-embedding-3-small"
    max_context_messages: int = 20


class BraveSearchConfig(BaseModel):
    """Configuration for Brave Search API."""

    api_key: SecretStr | None = None


class AshConfig(BaseModel):
    """Root configuration model."""

    workspace: Path = Path("~/.ash/workspace")
    default_llm: LLMConfig
    fallback_llm: LLMConfig | None = None
    telegram: TelegramConfig | None = None
    sandbox: SandboxConfig = SandboxConfig()
    server: ServerConfig = ServerConfig()
    memory: MemoryConfig = MemoryConfig()
    brave_search: BraveSearchConfig | None = None
