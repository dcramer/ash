"""Tests for LLM types and message handling."""

from ash.llm.types import (
    CompletionResponse,
    ContentBlockType,
    Message,
    Role,
    StreamChunk,
    StreamEventType,
    TextContent,
    ToolDefinition,
    ToolResult,
    ToolUse,
    Usage,
)


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self):
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.SYSTEM.value == "system"

    def test_role_is_string(self):
        # Role inherits from str, so comparisons work
        assert Role.USER == "user"
        assert Role.USER.value == "user"


class TestContentBlockType:
    """Tests for ContentBlockType enum."""

    def test_content_block_types(self):
        assert ContentBlockType.TEXT.value == "text"
        assert ContentBlockType.TOOL_USE.value == "tool_use"
        assert ContentBlockType.TOOL_RESULT.value == "tool_result"


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_stream_event_types(self):
        assert StreamEventType.TEXT_DELTA.value == "text_delta"
        assert StreamEventType.TOOL_USE_START.value == "tool_use_start"
        assert StreamEventType.MESSAGE_END.value == "message_end"


class TestTextContent:
    """Tests for TextContent dataclass."""

    def test_create_text_content(self):
        content = TextContent(text="Hello, world!")
        assert content.text == "Hello, world!"
        assert content.type == ContentBlockType.TEXT

    def test_text_content_type_default(self):
        content = TextContent(text="Test")
        assert content.type == ContentBlockType.TEXT


class TestToolUse:
    """Tests for ToolUse dataclass."""

    def test_create_tool_use(self):
        tool_use = ToolUse(
            id="tool-123",
            name="bash",
            input={"command": "ls -la"},
        )
        assert tool_use.id == "tool-123"
        assert tool_use.name == "bash"
        assert tool_use.input == {"command": "ls -la"}
        assert tool_use.type == ContentBlockType.TOOL_USE

    def test_tool_use_empty_input(self):
        tool_use = ToolUse(id="t1", name="test", input={})
        assert tool_use.input == {}


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_create_tool_result_success(self):
        result = ToolResult(
            tool_use_id="tool-123",
            content="Command executed successfully",
        )
        assert result.tool_use_id == "tool-123"
        assert result.content == "Command executed successfully"
        assert result.is_error is False
        assert result.type == ContentBlockType.TOOL_RESULT

    def test_create_tool_result_error(self):
        result = ToolResult(
            tool_use_id="tool-123",
            content="Error: command not found",
            is_error=True,
        )
        assert result.is_error is True


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_simple_message(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_create_message_with_blocks(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent(text="Let me help."),
                ToolUse(id="t1", name="bash", input={"cmd": "ls"}),
            ],
        )
        assert msg.role == Role.ASSISTANT
        assert len(msg.content) == 2

    def test_get_text_from_string_content(self):
        msg = Message(role=Role.USER, content="Hello, world!")
        assert msg.get_text() == "Hello, world!"

    def test_get_text_from_blocks(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent(text="First part."),
                ToolUse(id="t1", name="test", input={}),
                TextContent(text="Second part."),
            ],
        )
        assert msg.get_text() == "First part.\nSecond part."

    def test_get_text_no_text_blocks(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[ToolUse(id="t1", name="test", input={})],
        )
        assert msg.get_text() == ""

    def test_get_tool_uses_from_string_content(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.get_tool_uses() == []

    def test_get_tool_uses_from_blocks(self):
        tool_use = ToolUse(id="t1", name="bash", input={})
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent(text="Running command..."),
                tool_use,
            ],
        )
        tool_uses = msg.get_tool_uses()
        assert len(tool_uses) == 1
        assert tool_uses[0] is tool_use


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_create_tool_definition(self):
        definition = ToolDefinition(
            name="bash",
            description="Execute bash commands",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        )
        assert definition.name == "bash"
        assert definition.description == "Execute bash commands"
        assert "command" in definition.input_schema["properties"]


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_text_delta_chunk(self):
        chunk = StreamChunk(
            type=StreamEventType.TEXT_DELTA,
            content="Hello",
        )
        assert chunk.type == StreamEventType.TEXT_DELTA
        assert chunk.content == "Hello"

    def test_tool_use_start_chunk(self):
        chunk = StreamChunk(
            type=StreamEventType.TOOL_USE_START,
            tool_use_id="tool-123",
            tool_name="bash",
        )
        assert chunk.type == StreamEventType.TOOL_USE_START
        assert chunk.tool_use_id == "tool-123"
        assert chunk.tool_name == "bash"

    def test_message_end_chunk(self):
        chunk = StreamChunk(type=StreamEventType.MESSAGE_END)
        assert chunk.type == StreamEventType.MESSAGE_END
        assert chunk.content is None


class TestUsage:
    """Tests for Usage dataclass."""

    def test_create_usage(self):
        usage = Usage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50


class TestCompletionResponse:
    """Tests for CompletionResponse dataclass."""

    def test_create_completion_response(self):
        message = Message(role=Role.ASSISTANT, content="Hello!")
        response = CompletionResponse(
            message=message,
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
            model="claude-3-sonnet",
        )
        assert response.message is message
        assert response.usage is not None
        assert response.usage.input_tokens == 10
        assert response.stop_reason == "end_turn"
        assert response.model == "claude-3-sonnet"

    def test_completion_response_defaults(self):
        message = Message(role=Role.ASSISTANT, content="Hi")
        response = CompletionResponse(message=message)
        assert response.usage is None
        assert response.stop_reason is None
        assert response.raw == {}
