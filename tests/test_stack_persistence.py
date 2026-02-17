"""Tests for agent stack persistence (save/load/restore)."""

from __future__ import annotations

import json

import pytest

from ash.agents.types import AgentContext, StackFrame
from ash.core.session import SessionState
from ash.sessions.manager import SessionManager
from ash.sessions.types import (
    AgentSessionCompleteEntry,
    AgentSessionEntry,
    StackFrameMeta,
)


def _make_frame(
    *,
    agent_name: str = "skill:test",
    agent_type: str = "skill",
    frame_id: str = "frame-1",
    agent_session_id: str | None = "sess-1",
    model: str | None = "test-model",
    iteration: int = 3,
    max_iterations: int = 25,
    parent_tool_use_id: str | None = "tool-1",
    effective_tools: list[str] | None = None,
    is_skill_agent: bool = True,
    voice: str | None = "casual",
    environment: dict[str, str] | None = None,
) -> StackFrame:
    """Create a StackFrame for testing."""
    session = SessionState(
        session_id="test-session",
        provider="telegram",
        chat_id="123",
        user_id="456",
    )
    context = AgentContext(
        session_id="test-session",
        user_id="456",
        chat_id="123",
        provider="telegram",
    )
    return StackFrame(
        frame_id=frame_id,
        agent_name=agent_name,
        agent_type=agent_type,
        session=session,
        system_prompt="You are a test agent.",
        context=context,
        model=model,
        environment=environment,
        iteration=iteration,
        max_iterations=max_iterations,
        effective_tools=effective_tools or ["bash", "web_search"],
        is_skill_agent=is_skill_agent,
        voice=voice,
        parent_tool_use_id=parent_tool_use_id,
        agent_session_id=agent_session_id,
    )


class TestStackFrameToMeta:
    """Tests for StackFrame.to_meta() conversion."""

    def test_converts_all_fields(self):
        frame = _make_frame()
        meta = frame.to_meta()

        assert isinstance(meta, StackFrameMeta)
        assert meta.frame_id == "frame-1"
        assert meta.agent_session_id == "sess-1"
        assert meta.agent_name == "skill:test"
        assert meta.agent_type == "skill"
        assert meta.model == "test-model"
        assert meta.iteration == 3
        assert meta.max_iterations == 25
        assert meta.parent_tool_use_id == "tool-1"
        assert meta.effective_tools == ["bash", "web_search"]
        assert meta.is_skill_agent is True
        assert meta.voice == "casual"
        assert meta.environment == {}

    def test_main_frame_no_agent_session_id(self):
        """Main frames have agent_session_id=None."""
        frame = _make_frame(
            agent_name="main",
            agent_type="main",
            agent_session_id=None,
            is_skill_agent=False,
            parent_tool_use_id=None,
        )
        meta = frame.to_meta()

        assert meta.agent_session_id is None
        assert meta.agent_type == "main"
        assert meta.parent_tool_use_id is None

    def test_strips_non_serializable_fields(self):
        """to_meta() should not contain session, system_prompt, or context."""
        frame = _make_frame()
        meta = frame.to_meta()

        # StackFrameMeta is a Pydantic model — verify it doesn't have runtime-only fields
        data = meta.model_dump()
        assert "session" not in data
        assert "system_prompt" not in data
        assert "context" not in data

    def test_environment_dict_is_copied(self):
        """Environment should be a copy, not a reference."""
        env = {"API_KEY": "secret"}
        frame = _make_frame(environment=env)
        meta = frame.to_meta()

        assert meta.environment == {"API_KEY": "secret"}
        # Mutating the original shouldn't affect the meta
        env["NEW_KEY"] = "value"
        assert "NEW_KEY" not in meta.environment

    def test_effective_tools_is_copied(self):
        """effective_tools should be a copy, not a reference."""
        tools = ["bash", "web_search"]
        frame = _make_frame(effective_tools=tools)
        meta = frame.to_meta()

        tools.append("new_tool")
        assert "new_tool" not in meta.effective_tools


class TestSaveLoadActiveStack:
    """Tests for SessionManager.save_active_stack() / load_active_stack()."""

    def test_round_trip(self, tmp_path):
        """Saving and loading should produce identical StackFrameMeta objects."""
        sm = SessionManager(
            provider="telegram",
            chat_id="123",
            user_id="456",
            sessions_path=tmp_path,
        )
        sm._ensure_state_file()

        metas = [
            StackFrameMeta(
                frame_id="f1",
                agent_session_id=None,
                agent_name="main",
                agent_type="main",
            ),
            StackFrameMeta(
                frame_id="f2",
                agent_session_id="sess-2",
                agent_name="skill:research",
                agent_type="skill",
                model="haiku",
                iteration=5,
                max_iterations=25,
                parent_tool_use_id="tu-1",
                effective_tools=["bash"],
                is_skill_agent=True,
                environment={"KEY": "val"},
                voice="casual",
            ),
        ]

        sm.save_active_stack(metas)
        loaded = sm.load_active_stack()

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].frame_id == "f1"
        assert loaded[0].agent_session_id is None
        assert loaded[0].agent_type == "main"
        assert loaded[1].frame_id == "f2"
        assert loaded[1].agent_session_id == "sess-2"
        assert loaded[1].model == "haiku"
        assert loaded[1].iteration == 5
        assert loaded[1].effective_tools == ["bash"]
        assert loaded[1].environment == {"KEY": "val"}
        assert loaded[1].voice == "casual"

    def test_save_none_clears_stack(self, tmp_path):
        """Saving None should clear the active_stack field."""
        sm = SessionManager(
            provider="telegram",
            chat_id="123",
            user_id="456",
            sessions_path=tmp_path,
        )
        sm._ensure_state_file()

        # First save a stack
        metas = [
            StackFrameMeta(
                frame_id="f1",
                agent_name="main",
                agent_type="main",
            ),
        ]
        sm.save_active_stack(metas)
        assert sm.load_active_stack() is not None

        # Clear it
        sm.save_active_stack(None)
        assert sm.load_active_stack() is None

    def test_save_empty_list_clears_stack(self, tmp_path):
        """Saving an empty list should also clear the active_stack."""
        sm = SessionManager(
            provider="telegram",
            chat_id="123",
            user_id="456",
            sessions_path=tmp_path,
        )
        sm._ensure_state_file()

        metas = [
            StackFrameMeta(
                frame_id="f1",
                agent_name="main",
                agent_type="main",
            ),
        ]
        sm.save_active_stack(metas)
        assert sm.load_active_stack() is not None

        sm.save_active_stack([])
        assert sm.load_active_stack() is None

    def test_load_no_state_file(self, tmp_path):
        """Loading from non-existent state.json should return None."""
        sm = SessionManager(
            provider="telegram",
            chat_id="123",
            user_id="456",
            sessions_path=tmp_path,
        )
        assert sm.load_active_stack() is None

    def test_preserves_other_state_fields(self, tmp_path):
        """Saving active_stack should not clobber other state.json fields."""
        sm = SessionManager(
            provider="telegram",
            chat_id="123",
            user_id="456",
            sessions_path=tmp_path,
        )
        sm._ensure_state_file()

        # Verify provider is preserved after saving stack
        metas = [
            StackFrameMeta(
                frame_id="f1",
                agent_name="main",
                agent_type="main",
            ),
        ]
        sm.save_active_stack(metas)

        raw = json.loads(sm.state_path.read_text())
        assert raw["provider"] == "telegram"
        assert raw["chat_id"] == "123"
        assert raw["active_stack"] is not None


class TestStaleStackDetection:
    """Tests for stale stack detection via AgentSessionCompleteEntry."""

    @pytest.mark.asyncio
    async def test_completed_session_is_stale(self, tmp_path):
        """A stack whose top frame has a completed session should be detected as stale."""
        sm = SessionManager(
            provider="telegram",
            chat_id="123",
            user_id="456",
            sessions_path=tmp_path,
        )
        await sm.ensure_session()

        # Write an agent_session and its completion to context.jsonl
        agent_session = AgentSessionEntry.create(
            parent_tool_use_id="tu-1",
            agent_type="skill",
            agent_name="skill:test",
        )
        await sm._writer.write_agent_session(agent_session)

        complete_entry = AgentSessionCompleteEntry.create(
            agent_session_id=agent_session.id,
            result="done",
        )
        # Write the completion entry directly via the writer
        await sm._writer._append_context(complete_entry.to_dict())

        # Verify the entries are readable
        entries = await sm._reader.load_entries()
        complete_entries = [
            e
            for e in entries
            if isinstance(e, AgentSessionCompleteEntry)
            and e.agent_session_id == agent_session.id
        ]
        assert len(complete_entries) == 1

    @pytest.mark.asyncio
    async def test_incomplete_session_is_not_stale(self, tmp_path):
        """A stack whose top frame has no completion entry should not be stale."""
        sm = SessionManager(
            provider="telegram",
            chat_id="123",
            user_id="456",
            sessions_path=tmp_path,
        )
        await sm.ensure_session()

        # Write an agent_session without completion
        agent_session = AgentSessionEntry.create(
            parent_tool_use_id="tu-1",
            agent_type="skill",
            agent_name="skill:test",
        )
        await sm._writer.write_agent_session(agent_session)

        entries = await sm._reader.load_entries()
        complete_entries = [
            e
            for e in entries
            if isinstance(e, AgentSessionCompleteEntry)
            and e.agent_session_id == agent_session.id
        ]
        assert len(complete_entries) == 0


class TestStackFrameMetaSerialization:
    """Tests for StackFrameMeta Pydantic serialization."""

    def test_optional_agent_session_id(self):
        """agent_session_id should be optional (None for main frames)."""
        meta = StackFrameMeta(
            frame_id="f1",
            agent_name="main",
            agent_type="main",
        )
        assert meta.agent_session_id is None

        data = meta.model_dump(mode="json")
        restored = StackFrameMeta.model_validate(data)
        assert restored.agent_session_id is None

    def test_voice_field(self):
        """voice field should round-trip correctly."""
        meta = StackFrameMeta(
            frame_id="f1",
            agent_session_id="s1",
            agent_name="skill:test",
            agent_type="skill",
            voice="casual and friendly",
        )
        data = meta.model_dump(mode="json")
        restored = StackFrameMeta.model_validate(data)
        assert restored.voice == "casual and friendly"

    def test_defaults(self):
        """Verify all default values are correct."""
        meta = StackFrameMeta(
            frame_id="f1",
            agent_name="main",
            agent_type="main",
        )
        assert meta.agent_session_id is None
        assert meta.model is None
        assert meta.iteration == 0
        assert meta.max_iterations == 25
        assert meta.parent_tool_use_id is None
        assert meta.effective_tools == []
        assert meta.is_skill_agent is False
        assert meta.environment == {}
        assert meta.voice is None
