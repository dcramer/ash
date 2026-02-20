"""Shared test fixtures and factories."""

from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

import pytest

from ash.config.models import AshConfig, LLMConfig
from ash.graph.graph import KnowledgeGraph
from ash.graph.persistence import GraphPersistence
from ash.graph.vectors import NumpyVectorIndex
from ash.llm.base import LLMProvider
from ash.llm.types import (
    CompletionResponse,
    ContentBlock,
    Message,
    Role,
    StreamChunk,
    StreamEventType,
    TextContent,
    ToolDefinition,
    ToolUse,
    Usage,
)
from ash.store.store import Store
from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.registry import ToolRegistry

# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def minimal_config() -> AshConfig:
    """Minimal valid configuration."""
    return AshConfig(
        default_llm=LLMConfig(
            provider="openai",
            model="gpt-5.2",
        )
    )


@pytest.fixture
def full_config(tmp_path: Path) -> AshConfig:
    """Full configuration with all options."""
    return AshConfig(
        workspace=tmp_path / "workspace",
        default_llm=LLMConfig(
            provider="openai",
            model="gpt-5.2",
            temperature=0.5,
            max_tokens=2048,
        ),
        fallback_llm=LLMConfig(
            provider="openai",
            model="gpt-5-mini",
        ),
    )


@pytest.fixture
def config_toml_content() -> str:
    """Valid TOML config content."""
    return """
workspace = "/tmp/ash-workspace"

[default_llm]
provider = "openai"
model = "gpt-5.2"
temperature = 0.7
max_tokens = 4096
"""


@pytest.fixture
def config_file(tmp_path: Path, config_toml_content: str) -> Path:
    """Create a temporary config file."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_toml_content)
    return config_path


# =============================================================================
# Graph Store Fixtures
# =============================================================================


@pytest.fixture
def graph_dir(tmp_path: Path) -> Path:
    """Create a temporary graph directory."""
    gd = tmp_path / "graph"
    gd.mkdir()
    (gd / "embeddings").mkdir()
    return gd


@pytest.fixture
async def graph_store(
    graph_dir: Path, mock_llm: "MockLLMProvider"
) -> AsyncGenerator[Store, None]:
    """Create a Store backed by in-memory KnowledgeGraph."""
    graph = KnowledgeGraph()
    persistence = GraphPersistence(graph_dir)
    vector_index = NumpyVectorIndex()

    # Use a mock embedding generator
    embedding_gen = MockEmbeddingGenerator()

    store = Store(
        graph=graph,
        persistence=persistence,
        vector_index=vector_index,
        embedding_generator=embedding_gen,  # type: ignore[arg-type]
        llm=mock_llm,
    )
    store._llm_model = "mock-model"

    yield store


class MockEmbeddingGenerator:
    """Mock embedding generator for testing."""

    async def embed(self, text: str) -> list[float]:
        """Return a deterministic mock embedding."""
        # Simple hash-based embedding for deterministic results
        h = hash(text) % (2**32)
        import struct

        # Generate 1536 floats from the hash
        result = []
        for i in range(1536):
            seed = h ^ (i * 2654435761)
            val = struct.unpack("f", struct.pack("I", seed & 0x7F7FFFFF))[0]
            result.append(val)
        # Normalize
        import math

        norm = math.sqrt(sum(x * x for x in result))
        if norm > 0:
            result = [x / norm for x in result]
        return result


@pytest.fixture(autouse=True)
def _isolate_ash_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    """Safety net: ensure ALL tests use a temporary ASH_HOME, never ~/.ash."""
    home = tmp_path / ".ash"
    home.mkdir()
    monkeypatch.setenv("ASH_HOME", str(home))

    from ash.config.paths import get_ash_home

    get_ash_home.cache_clear()
    yield home
    get_ash_home.cache_clear()


@pytest.fixture
def ash_home(_isolate_ash_home: Path) -> Path:
    """Explicit reference to the temporary ASH_HOME directory."""
    return _isolate_ash_home


# =============================================================================
# LLM Fixtures and Mocks
# =============================================================================


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(
        self,
        responses: list[Message] | None = None,
        stream_chunks: list[StreamChunk] | None = None,
    ):
        self.responses = responses or []
        self.stream_chunks = stream_chunks or []
        self.complete_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self._response_index = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        thinking: Any = None,
        reasoning: str | None = None,
    ) -> CompletionResponse:
        self.complete_calls.append(
            {
                "messages": messages,
                "model": model,
                "tools": tools,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        if self._response_index < len(self.responses):
            message = self.responses[self._response_index]
            self._response_index += 1
        else:
            message = Message(role=Role.ASSISTANT, content="Mock response")

        return CompletionResponse(
            message=message,
            usage=Usage(input_tokens=100, output_tokens=50),
            stop_reason="end_turn",
            model=model or "mock-model",
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        thinking: Any = None,
        reasoning: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        self.stream_calls.append(
            {
                "messages": messages,
                "model": model,
                "tools": tools,
                "system": system,
            }
        )

        for chunk in self.stream_chunks:
            yield chunk

        # Default streaming response if none provided
        if not self.stream_chunks:
            yield StreamChunk(type=StreamEventType.MESSAGE_START)
            yield StreamChunk(type=StreamEventType.TEXT_DELTA, content="Mock ")
            yield StreamChunk(type=StreamEventType.TEXT_DELTA, content="response")
            yield StreamChunk(type=StreamEventType.MESSAGE_END)

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        # Return mock embeddings (1536 dimensions like OpenAI)
        return [[0.1] * 1536 for _ in texts]


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_llm_with_tool_use() -> MockLLMProvider:
    """Create a mock LLM that requests tool use."""
    tool_use_response = Message(
        role=Role.ASSISTANT,
        content=[
            ToolUse(
                id="tool_123",
                name="test_tool",
                input={"arg": "value"},
            )
        ],
    )
    final_response = Message(
        role=Role.ASSISTANT,
        content="Tool executed successfully.",
    )
    return MockLLMProvider(responses=[tool_use_response, final_response])


# =============================================================================
# Tool Fixtures
# =============================================================================


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool for testing",
        result: ToolResult | None = None,
    ):
        self._name = name
        self._description = description
        self._result = result or ToolResult.success("Mock tool executed")
        self.execute_calls: list[tuple[dict[str, Any], ToolContext]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "arg": {"type": "string", "description": "An argument"},
            },
            "required": ["arg"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        self.execute_calls.append((input_data, context))
        return self._result


@pytest.fixture
def mock_tool() -> MockTool:
    """Create a mock tool."""
    return MockTool()


@pytest.fixture
def tool_registry(mock_tool: MockTool) -> ToolRegistry:
    """Create a tool registry with a mock tool."""
    registry = ToolRegistry()
    registry.register(mock_tool)
    return registry


@pytest.fixture
def failing_tool() -> MockTool:
    """Create a tool that returns an error."""
    return MockTool(
        name="failing_tool",
        result=ToolResult.error("Tool execution failed"),
    )


# =============================================================================
# Message Factories
# =============================================================================


def make_message(
    role: Role = Role.USER,
    content: str | list[ContentBlock] = "Hello",
) -> Message:
    """Factory for creating messages."""
    return Message(role=role, content=content)


def make_text_content(text: str = "Hello") -> TextContent:
    """Factory for creating text content blocks."""
    return TextContent(text=text)


def make_tool_use(
    id: str = "tool_123",
    name: str = "test_tool",
    input: dict[str, Any] | None = None,
) -> ToolUse:
    """Factory for creating tool use blocks."""
    return ToolUse(id=id, name=name, input=input or {})


# =============================================================================
# CLI Test Helpers
# =============================================================================


@pytest.fixture
def cli_runner(request):
    """Create a Typer CLI test runner with colors disabled.

    Temporarily disables pytest live logging (log_cli) because it
    conflicts with Click's CliRunner stdout capture — the live log
    handler writes to the real stdout and can close Click's BytesIO
    wrapper, causing ``ValueError: I/O operation on closed file``.
    """
    import logging

    from typer.testing import CliRunner

    # Disable live log handler if active (log_cli = true in pyproject.toml)
    live_manager = request.config.pluginmanager.get_plugin("logging-plugin")
    if live_manager and hasattr(live_manager, "log_cli_handler"):
        handler = live_manager.log_cli_handler
        old_level = handler.level
        handler.setLevel(logging.CRITICAL + 1)  # Suppress all output
    else:
        handler = None
        old_level = None

    yield CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    if handler is not None and old_level is not None:
        handler.setLevel(old_level)


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "SOUL.md").write_text("# Test Soul\n\nYou are a test assistant.")
    return workspace
