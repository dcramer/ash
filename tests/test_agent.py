"""Tests for agent orchestration."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from ash.config import AshConfig
from ash.config.models import ModelConfig
from ash.config.workspace import Workspace
from ash.core.agent import Agent, AgentConfig, AgentResponse
from ash.core.prompt import SystemPromptBuilder
from ash.core.session import SessionState
from ash.llm.base import LLMProvider
from ash.llm.types import (
    CompletionResponse,
    Message,
    Role,
    StreamChunk,
    StreamEventType,
    TextContent,
    ToolDefinition,
    ToolUse,
    Usage,
)
from ash.skills.registry import SkillRegistry
from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.executor import ToolExecutor
from ash.tools.registry import ToolRegistry


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
    ) -> CompletionResponse:
        self.complete_calls.append(
            {
                "messages": messages,
                "model": model,
                "tools": tools,
                "system": system,
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
    ) -> AsyncGenerator[StreamChunk, None]:
        self.stream_calls.append({"messages": messages})

        for chunk in self.stream_chunks:
            yield chunk

        if not self.stream_chunks:
            yield StreamChunk(type=StreamEventType.MESSAGE_START)
            yield StreamChunk(type=StreamEventType.TEXT_DELTA, content="Mock response")
            yield StreamChunk(type=StreamEventType.MESSAGE_END)

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        return [[0.0] * 128 for _ in texts]


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool",
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
            "properties": {"arg": {"type": "string"}},
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
def workspace(tmp_path: Path) -> Workspace:
    """Create a test workspace."""
    return Workspace(
        path=tmp_path,
        soul="You are a test assistant.",
    )


@pytest.fixture
def skill_registry() -> SkillRegistry:
    """Create empty skill registry for testing."""
    return SkillRegistry()


@pytest.fixture
def config(tmp_path: Path) -> AshConfig:
    """Create test config."""
    return AshConfig(
        workspace=tmp_path,
        models={"default": ModelConfig(provider="anthropic", model="claude-test")},
    )


def create_test_prompt_builder(
    workspace: Workspace,
    tool_registry: ToolRegistry,
    skill_registry: SkillRegistry | None = None,
    config: AshConfig | None = None,
) -> SystemPromptBuilder:
    """Helper to create prompt builder for tests."""
    if skill_registry is None:
        skill_registry = SkillRegistry()
    if config is None:
        config = AshConfig(
            workspace=workspace.path,
            models={"default": ModelConfig(provider="anthropic", model="claude-test")},
        )
    return SystemPromptBuilder(
        workspace=workspace,
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        config=config,
    )


@pytest.fixture
def session() -> SessionState:
    """Create a test session."""
    return SessionState(
        session_id="test-session",
        provider="test",
        chat_id="chat-123",
        user_id="user-456",
    )


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_defaults(self):
        config = AgentConfig()
        assert config.model is None
        assert config.max_tokens == 4096
        assert config.temperature is None  # None = use provider default
        assert config.max_tool_iterations == 25  # MAX_TOOL_ITERATIONS constant
        assert config.context_token_budget == 100000
        assert config.recency_window == 10
        assert config.system_prompt_buffer == 8000

    def test_custom_values(self):
        config = AgentConfig(
            model="claude-3-opus",
            max_tokens=2048,
            temperature=0.5,
            max_tool_iterations=5,
        )
        assert config.model == "claude-3-opus"
        assert config.max_tokens == 2048


class TestAgentResponse:
    """Tests for AgentResponse."""

    def test_create_response(self):
        response = AgentResponse(
            text="Hello!",
            tool_calls=[{"name": "test", "result": "ok"}],
            iterations=2,
        )
        assert response.text == "Hello!"
        assert len(response.tool_calls) == 1
        assert response.iterations == 2


class TestAgent:
    """Tests for Agent orchestrator."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM that returns simple text."""
        return MockLLMProvider(
            responses=[Message(role=Role.ASSISTANT, content="Hello! How can I help?")]
        )

    @pytest.fixture
    def test_tool_registry(self):
        """Create tool registry with mock tool."""
        registry = ToolRegistry()
        registry.register(MockTool(name="test_tool"))
        return registry

    @pytest.fixture
    def agent(self, mock_llm, test_tool_registry, workspace):
        """Create agent for testing."""
        executor = ToolExecutor(test_tool_registry)
        prompt_builder = create_test_prompt_builder(workspace, test_tool_registry)
        return Agent(
            llm=mock_llm,
            tool_executor=executor,
            prompt_builder=prompt_builder,
        )

    async def test_process_simple_message(self, agent, session):
        response = await agent.process_message("Hello", session)

        assert response.text == "Hello! How can I help?"
        assert response.iterations == 1
        assert response.tool_calls == []

    async def test_process_message_adds_to_session(self, agent, session):
        await agent.process_message("Hello", session)

        messages = session.get_messages_for_llm()
        assert len(messages) == 2
        assert messages[0].role == Role.USER
        assert messages[0].content == "Hello"
        assert messages[1].role == Role.ASSISTANT

    async def test_process_message_with_tool_use(self, workspace):
        """Test agent handles tool use correctly."""
        # First response requests tool use
        tool_use_response = Message(
            role=Role.ASSISTANT,
            content=[
                ToolUse(id="tool-1", name="test_tool", input={"arg": "value"}),
            ],
        )
        # Second response is final text
        final_response = Message(
            role=Role.ASSISTANT,
            content="Tool executed, here's the result.",
        )

        mock_llm = MockLLMProvider(responses=[tool_use_response, final_response])
        registry = ToolRegistry()
        registry.register(MockTool(name="test_tool"))
        executor = ToolExecutor(registry)
        prompt_builder = create_test_prompt_builder(workspace, registry)

        agent = Agent(
            llm=mock_llm,
            tool_executor=executor,
            prompt_builder=prompt_builder,
        )

        session = SessionState(
            session_id="test",
            provider="test",
            chat_id="chat",
            user_id="user",
        )

        response = await agent.process_message("Use the tool", session)

        assert response.text == "Tool executed, here's the result."
        assert response.iterations == 2
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "test_tool"

    async def test_max_iterations_limit(self, workspace):
        """Test agent stops at max iterations."""
        # LLM always requests tool use
        tool_use_response = Message(
            role=Role.ASSISTANT,
            content=[
                ToolUse(id="tool-1", name="test_tool", input={"arg": "loop"}),
            ],
        )

        # Create LLM that always returns tool use
        mock_llm = MockLLMProvider(responses=[tool_use_response] * 20)
        registry = ToolRegistry()
        registry.register(MockTool(name="test_tool"))
        executor = ToolExecutor(registry)
        prompt_builder = create_test_prompt_builder(workspace, registry)

        config = AgentConfig(max_tool_iterations=3)
        agent = Agent(
            llm=mock_llm,
            tool_executor=executor,
            prompt_builder=prompt_builder,
            config=config,
        )

        session = SessionState(
            session_id="test",
            provider="test",
            chat_id="chat",
            user_id="user",
        )

        response = await agent.process_message("Loop forever", session)

        assert response.iterations == 3
        assert "maximum" in response.text.lower()

    async def test_system_prompt_from_workspace(self, agent):
        """Test that system prompt includes workspace content."""
        assert "test assistant" in agent.system_prompt.lower()

    async def test_tool_definitions_conversion(self, agent):
        definitions = agent._get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0].name == "test_tool"

    async def test_process_message_streaming(self, workspace):
        """Test streaming message processing."""
        from ash.llm.types import StreamChunk, StreamEventType

        mock_llm = MockLLMProvider(
            stream_chunks=[
                StreamChunk(type=StreamEventType.MESSAGE_START),
                StreamChunk(type=StreamEventType.TEXT_DELTA, content="Hello "),
                StreamChunk(type=StreamEventType.TEXT_DELTA, content="world!"),
                StreamChunk(type=StreamEventType.MESSAGE_END),
            ]
        )

        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        prompt_builder = create_test_prompt_builder(workspace, registry)

        agent = Agent(
            llm=mock_llm,
            tool_executor=executor,
            prompt_builder=prompt_builder,
        )

        session = SessionState(
            session_id="test",
            provider="test",
            chat_id="chat",
            user_id="user",
        )

        chunks = []
        async for chunk in agent.process_message_streaming("Hi", session):
            chunks.append(chunk)

        assert "Hello " in chunks
        assert "world!" in chunks


class TestSessionState:
    """Tests for SessionState."""

    def test_create_session(self):
        session = SessionState(
            session_id="sess-1",
            provider="telegram",
            chat_id="chat-123",
            user_id="user-456",
        )
        assert session.session_id == "sess-1"
        assert session.messages == []
        assert session._token_counts == []
        assert session._message_ids == []

    def test_add_user_message(self, session):
        msg = session.add_user_message("Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert len(session.messages) == 1

    def test_add_assistant_message(self, session):
        msg = session.add_assistant_message("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"

    def test_add_assistant_message_with_blocks(self, session):
        blocks = [
            TextContent(text="Let me help"),
            ToolUse(id="t1", name="bash", input={"cmd": "ls"}),
        ]
        msg = session.add_assistant_message(blocks)
        assert msg.role == Role.ASSISTANT
        assert len(msg.content) == 2

    def test_add_tool_result(self, session):
        msg = session.add_tool_result(
            tool_use_id="t1",
            content="file1.txt\nfile2.txt",
            is_error=False,
        )
        assert msg.role == Role.USER
        assert len(msg.content) == 1

    def test_get_messages_for_llm(self, session):
        session.add_user_message("Hello")
        session.add_assistant_message("Hi!")
        messages = session.get_messages_for_llm()
        assert len(messages) == 2
        # Should be a copy
        messages.clear()
        assert len(session.messages) == 2

    def test_get_pending_tool_uses(self, session):
        session.add_assistant_message(
            [
                TextContent(text="Running..."),
                ToolUse(id="t1", name="bash", input={}),
                ToolUse(id="t2", name="search", input={}),
            ]
        )
        pending = session.get_pending_tool_uses()
        assert len(pending) == 2
        assert pending[0].name == "bash"
        assert pending[1].name == "search"

    def test_get_pending_tool_uses_empty(self, session):
        session.add_user_message("Hello")
        assert session.get_pending_tool_uses() == []

    def test_get_pending_tool_uses_no_tools(self, session):
        session.add_assistant_message("Just text")
        assert session.get_pending_tool_uses() == []

    def test_get_last_text_response(self, session):
        session.add_user_message("Hello")
        session.add_assistant_message("Hi there!")
        assert session.get_last_text_response() == "Hi there!"

    def test_get_last_text_response_none(self, session):
        session.add_user_message("Hello")
        assert session.get_last_text_response() is None

    def test_clear_messages(self, session):
        session.add_user_message("Hello")
        session.add_assistant_message("Hi!")
        session.clear_messages()
        assert session.messages == []

    def test_to_dict_and_back(self, session):
        session.add_user_message("Hello")
        session.add_assistant_message(
            [
                TextContent(text="Let me help"),
                ToolUse(id="t1", name="bash", input={"cmd": "ls"}),
            ]
        )
        session.add_tool_result("t1", "output", is_error=False)

        data = session.to_dict()
        restored = SessionState.from_dict(data)

        assert restored.session_id == session.session_id
        assert len(restored.messages) == 3
        assert restored.messages[0].role == Role.USER

    def test_to_json_and_back(self, session):
        session.add_user_message("Test")
        json_str = session.to_json()
        restored = SessionState.from_json(json_str)
        assert restored.session_id == session.session_id
        assert len(restored.messages) == 1

    # Tests for smart pruning

    def test_get_messages_for_llm_no_budget(self, session):
        """Without budget, returns all messages."""
        session.add_user_message("Hello")
        session.add_assistant_message("Hi!")
        session.add_user_message("How are you?")
        session.add_assistant_message("I'm good!")

        messages = session.get_messages_for_llm()
        assert len(messages) == 4

    def test_get_messages_for_llm_with_large_budget(self, session):
        """With large budget, returns all messages."""
        session.add_user_message("Hello")
        session.add_assistant_message("Hi!")

        messages = session.get_messages_for_llm(token_budget=10000)
        assert len(messages) == 2

    def test_get_messages_for_llm_keeps_recency_window(self, session):
        """Recency window is always kept even when budget is tight."""
        # Add 15 messages with explicit token counts
        for i in range(15):
            if i % 2 == 0:
                session.add_user_message(f"Message {i}")
            else:
                session.add_assistant_message(f"Response {i}")

        # Set explicit token counts (100 tokens each message)
        session.set_token_counts([100] * 15)

        # Budget of 500 with recency_window=5 means:
        # - Recency window uses 5 * 100 = 500 tokens (exactly fits)
        # - No room for older messages
        messages = session.get_messages_for_llm(token_budget=500, recency_window=5)
        assert len(messages) == 5

        # Verify it's the last 5 messages
        assert messages[0].content == "Message 10"
        assert messages[-1].content == "Message 14"

    def test_get_messages_for_llm_prunes_old_messages(self, session):
        """Old messages are pruned when budget is tight."""
        # Add messages with known token counts
        session.add_user_message("a" * 100)  # ~26 tokens
        session.add_assistant_message("b" * 100)  # ~26 tokens
        session.add_user_message("c" * 100)  # ~26 tokens
        session.add_assistant_message("d" * 100)  # ~26 tokens

        # Set token counts (simulating DB load)
        session.set_token_counts([30, 30, 30, 30])

        # Budget of 70 with recency window of 2 = keep last 2 (60 tokens)
        # Then try to fit more from older = 0 more fit
        messages = session.get_messages_for_llm(token_budget=70, recency_window=2)
        assert len(messages) == 2  # Only recency window fits

    def test_get_messages_for_llm_adds_older_when_budget_allows(self, session):
        """Older messages included when budget allows."""
        session.add_user_message("a" * 40)  # ~11 tokens
        session.add_assistant_message("b" * 40)  # ~11 tokens
        session.add_user_message("c" * 40)  # ~11 tokens
        session.add_assistant_message("d" * 40)  # ~11 tokens

        session.set_token_counts([15, 15, 15, 15])

        # Budget of 100 with recency of 2 = 30 used, 70 remaining
        # Can fit both older messages (30 tokens)
        messages = session.get_messages_for_llm(token_budget=100, recency_window=2)
        assert len(messages) == 4

    def test_set_and_get_token_counts(self, session):
        """Token counts can be set and used."""
        session.add_user_message("Hello")
        session.add_assistant_message("Hi!")

        session.set_token_counts([10, 15])

        # _get_token_counts should return cached values
        counts = session._get_token_counts()
        assert counts == [10, 15]

    def test_set_and_get_message_ids(self, session):
        """Message IDs can be set and retrieved."""
        session.add_user_message("Hello")
        session.add_assistant_message("Hi!")

        session.set_message_ids(["msg-1", "msg-2"])

        recent = session.get_recent_message_ids(2)
        assert recent == {"msg-1", "msg-2"}

    def test_get_recent_message_ids_subset(self, session):
        """Only recent message IDs returned."""
        session.add_user_message("M1")
        session.add_user_message("M2")
        session.add_user_message("M3")
        session.add_user_message("M4")

        session.set_message_ids(["id-1", "id-2", "id-3", "id-4"])

        recent = session.get_recent_message_ids(2)
        assert recent == {"id-3", "id-4"}

    def test_get_recent_message_ids_empty(self, session):
        """Returns empty set when no IDs set."""
        recent = session.get_recent_message_ids(5)
        assert recent == set()

    def test_token_counts_estimated_when_not_cached(self, session):
        """Token counts are estimated for new messages."""
        session.add_user_message("Hello there!")
        session.add_assistant_message("Hi!")

        # No cached counts, so should estimate
        counts = session._get_token_counts()
        assert len(counts) == 2
        assert all(c > 0 for c in counts)


class TestWorkspace:
    """Tests for Workspace."""

    def test_soul_content(self, tmp_path):
        workspace = Workspace(
            path=tmp_path,
            soul="You are Ash.",
        )
        assert workspace.soul == "You are Ash."

    def test_custom_files(self, tmp_path):
        workspace = Workspace(
            path=tmp_path,
            soul="You are Ash.",
            custom_files={"extra.md": "Extra content"},
        )
        assert workspace.custom_files["extra.md"] == "Extra content"


class TestSystemPromptBuilder:
    """Tests for SystemPromptBuilder."""

    @pytest.fixture
    def prompt_builder(self, workspace, config) -> SystemPromptBuilder:
        """Create a prompt builder for testing."""
        registry = ToolRegistry()
        registry.register(MockTool(name="test_tool", description="A test tool"))
        skill_registry = SkillRegistry()
        return SystemPromptBuilder(
            workspace=workspace,
            tool_registry=registry,
            skill_registry=skill_registry,
            config=config,
        )

    def test_build_includes_soul(self, prompt_builder):
        """Test that build includes SOUL content."""
        prompt = prompt_builder.build()
        assert "test assistant" in prompt.lower()

    def test_build_includes_tools_section(self, prompt_builder):
        """Test that build includes tools section."""
        prompt = prompt_builder.build()
        assert "Available Tools" in prompt
        assert "test_tool" in prompt
        assert "A test tool" in prompt

    def test_build_includes_workspace_section(self, prompt_builder):
        """Test that build includes workspace info."""
        prompt = prompt_builder.build()
        assert "Workspace" in prompt
        assert "Working directory" in prompt

    def test_build_includes_sandbox_section(self, prompt_builder):
        """Test that build includes sandbox info."""
        prompt = prompt_builder.build()
        assert "Sandbox" in prompt
        assert "sandboxed environment" in prompt

    def test_build_with_runtime_info(self, prompt_builder):
        """Test that runtime info is included when provided.

        Note: os, arch, python are intentionally excluded to prevent
        host system awareness. Only model/provider/timezone/time are shown.
        """
        from ash.core.prompt import PromptContext, RuntimeInfo

        runtime = RuntimeInfo(
            model="claude-test",
            provider="anthropic",
            timezone="America/New_York",
            time="2024-01-15 10:30:00",
        )
        context = PromptContext(runtime=runtime)
        prompt = prompt_builder.build(context)

        assert "Runtime" in prompt
        assert "model=claude-test" in prompt
        assert "America/New_York" in prompt
        # Host system info (os, arch, python) should NOT be present
        assert "os=" not in prompt
        assert "python=" not in prompt
