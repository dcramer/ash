"""Tests for agent orchestration."""

from pathlib import Path
from typing import Any

import pytest

from ash.config.workspace import Workspace
from ash.core.agent import Agent, AgentConfig, AgentResponse
from ash.core.session import SessionState
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
from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.executor import ToolExecutor
from ash.tools.registry import ToolRegistry


class MockLLMProvider:
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

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
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
        temperature: float = 0.7,
    ):
        self.stream_calls.append({"messages": messages})

        for chunk in self.stream_chunks:
            yield chunk

        if not self.stream_chunks:
            yield StreamChunk(type=StreamEventType.MESSAGE_START)
            yield StreamChunk(type=StreamEventType.TEXT_DELTA, content="Mock response")
            yield StreamChunk(type=StreamEventType.MESSAGE_END)


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
        user="Test user profile.",
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
        assert config.temperature == 0.7
        assert config.max_tool_iterations == 10

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
    def tool_registry(self):
        """Create tool registry with mock tool."""
        registry = ToolRegistry()
        registry.register(MockTool(name="test_tool"))
        return registry

    @pytest.fixture
    def agent(self, mock_llm, tool_registry, workspace):
        """Create agent for testing."""
        executor = ToolExecutor(tool_registry)
        return Agent(
            llm=mock_llm,
            tool_executor=executor,
            workspace=workspace,
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

        agent = Agent(
            llm=mock_llm,
            tool_executor=executor,
            workspace=workspace,
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

        config = AgentConfig(max_tool_iterations=3)
        agent = Agent(
            llm=mock_llm,
            tool_executor=executor,
            workspace=workspace,
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

    async def test_system_prompt_from_workspace(self, agent, workspace):
        assert agent.system_prompt == workspace.system_prompt
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

        agent = Agent(
            llm=mock_llm,
            tool_executor=executor,
            workspace=workspace,
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


class TestWorkspace:
    """Tests for Workspace."""

    def test_system_prompt_with_soul(self, tmp_path):
        workspace = Workspace(
            path=tmp_path,
            soul="You are Ash.",
        )
        assert "You are Ash." in workspace.system_prompt

    def test_system_prompt_with_user(self, tmp_path):
        workspace = Workspace(
            path=tmp_path,
            soul="You are Ash.",
            user="User prefers formal language.",
        )
        prompt = workspace.system_prompt
        assert "You are Ash." in prompt
        assert "User Profile" in prompt
        assert "User prefers formal language." in prompt

    def test_system_prompt_with_tools(self, tmp_path):
        workspace = Workspace(
            path=tmp_path,
            soul="You are Ash.",
            tools="bash: run shell commands",
        )
        prompt = workspace.system_prompt
        assert "Available Tools" in prompt
        assert "bash" in prompt
