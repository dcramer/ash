"""Tests for sandboxed CLI schedule commands."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from ash_sandbox_cli.commands.schedule import app
from ash_sandbox_cli.rpc import RPCError
from typer.testing import CliRunner


@pytest.fixture
def cli_runner():
    """CLI runner with routing context set."""
    return CliRunner(
        env={
            "ASH_SESSION_ID": "test-session",
            "ASH_USER_ID": "user123",
            "ASH_CHAT_ID": "chat456",
            "ASH_PROVIDER": "telegram",
            "ASH_USERNAME": "testuser",
        }
    )


@pytest.fixture
def cli_runner_no_context():
    """CLI runner without routing context."""
    return CliRunner(env={})


@pytest.fixture
def mock_rpc():
    """Mock rpc_call for schedule commands."""
    with patch("ash_sandbox_cli.commands.schedule.rpc_call") as mock:
        yield mock


class TestScheduleCreate:
    """Tests for 'ash schedule create' command."""

    def test_create_one_shot(self, cli_runner, mock_rpc):
        """Test creating a one-shot task."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        future_z = future.replace("+00:00", "Z")
        mock_rpc.return_value = {
            "id": "abc12345",
            "entry": {
                "id": "abc12345",
                "message": "Test reminder",
                "trigger_at": future_z,
                "chat_id": "chat456",
                "provider": "telegram",
            },
        }

        result = cli_runner.invoke(app, ["create", "Test reminder", "--at", future])

        assert result.exit_code == 0
        assert "Scheduled reminder" in result.stdout
        assert "id=abc12345" in result.stdout

        # Verify RPC was called with correct params
        mock_rpc.assert_called_once()
        call_args = mock_rpc.call_args[0]
        assert call_args[0] == "schedule.create"
        params = call_args[1]
        assert params["message"] == "Test reminder"
        assert params["chat_id"] == "chat456"
        assert params["provider"] == "telegram"

    def test_create_periodic(self, cli_runner, mock_rpc):
        """Test creating a periodic task."""
        mock_rpc.return_value = {
            "id": "def67890",
            "entry": {
                "id": "def67890",
                "message": "Daily check",
                "cron": "0 8 * * *",
            },
        }

        result = cli_runner.invoke(
            app, ["create", "Daily check", "--cron", "0 8 * * *"]
        )

        assert result.exit_code == 0
        assert "Scheduled recurring task" in result.stdout
        assert "0 8 * * *" in result.stdout

    def test_create_requires_trigger(self, cli_runner, mock_rpc):
        """Test that create requires --at or --cron."""
        result = cli_runner.invoke(app, ["create", "Missing trigger"])

        assert result.exit_code == 1
        assert "Must specify either --at" in result.output
        mock_rpc.assert_not_called()

    def test_create_rejects_both_triggers(self, cli_runner, mock_rpc):
        """Test that create rejects both --at and --cron."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        result = cli_runner.invoke(
            app, ["create", "Both triggers", "--at", future, "--cron", "0 8 * * *"]
        )

        assert result.exit_code == 1
        assert "Cannot specify both" in result.output
        mock_rpc.assert_not_called()

    def test_create_rejects_past_time(self, cli_runner, mock_rpc):
        """Test that --at rejects past times."""
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        result = cli_runner.invoke(app, ["create", "Past time", "--at", past])

        assert result.exit_code == 1
        assert "in the past" in result.output
        mock_rpc.assert_not_called()

    def test_create_requires_routing_context(self, cli_runner_no_context, mock_rpc):
        """Test that create requires routing context."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        result = cli_runner_no_context.invoke(
            app, ["create", "No context", "--at", future]
        )

        assert result.exit_code == 1
        assert "Scheduling requires a provider context" in result.output
        mock_rpc.assert_not_called()


class TestScheduleList:
    """Tests for 'ash schedule list' command."""

    def test_list_empty(self, cli_runner, mock_rpc):
        """Test listing with no tasks."""
        mock_rpc.return_value = []

        result = cli_runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "No scheduled tasks found" in result.stdout

    def test_list_with_entries(self, cli_runner, mock_rpc):
        """Test listing tasks."""
        mock_rpc.return_value = [
            {
                "id": "abc12345",
                "trigger_at": "2026-01-12T09:00:00Z",
                "message": "Task 1",
                "user_id": "user123",
                "chat_id": "chat456",
            },
            {
                "id": "def67890",
                "cron": "0 8 * * *",
                "message": "Task 2",
                "user_id": "user123",
                "chat_id": "chat456",
            },
        ]

        result = cli_runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "abc12345" in result.stdout
        assert "def67890" in result.stdout
        assert "Task 1" in result.stdout
        assert "Task 2" in result.stdout
        assert "one-shot" in result.stdout
        assert "periodic" in result.stdout
        assert "Total: 2 task(s)" in result.stdout

    def test_list_passes_user_id_and_chat_id(self, cli_runner, mock_rpc):
        """Test that list passes both user_id and chat_id to RPC."""
        mock_rpc.return_value = []

        cli_runner.invoke(app, ["list"])

        mock_rpc.assert_called_once()
        call_args = mock_rpc.call_args[0]
        assert call_args[0] == "schedule.list"
        params = call_args[1]
        assert params["user_id"] == "user123"
        assert params["chat_id"] == "chat456"

    def test_list_filters_by_chat_id_default(self, cli_runner, mock_rpc):
        """Test that default list sends chat_id for room-scoped filtering."""
        mock_rpc.return_value = [
            {
                "id": "room_task",
                "trigger_at": "2026-01-12T09:00:00Z",
                "message": "Room task",
                "chat_id": "chat456",
            },
        ]

        result = cli_runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "room_task" in result.stdout
        # Verify chat_id was passed to RPC
        params = mock_rpc.call_args[0][1]
        assert params["chat_id"] == "chat456"

    def test_list_all_shows_all_rooms(self, cli_runner, mock_rpc):
        """Test that --all shows tasks from all rooms with Room label."""
        mock_rpc.return_value = [
            {
                "id": "task_a",
                "trigger_at": "2026-01-12T09:00:00Z",
                "message": "Task in room A",
                "chat_id": "chatA",
                "chat_title": "Work Chat",
            },
            {
                "id": "task_b",
                "cron": "0 8 * * *",
                "message": "Task in room B",
                "chat_id": "chatB",
                "chat_title": "Personal",
            },
        ]

        result = cli_runner.invoke(app, ["list", "--all"])

        assert result.exit_code == 0
        assert "task_a" in result.stdout
        assert "task_b" in result.stdout
        assert "Room: Work Chat" in result.stdout
        assert "Room: Personal" in result.stdout
        assert "Total: 2 task(s)" in result.stdout

        # Verify chat_id was NOT passed to RPC (all rooms)
        params = mock_rpc.call_args[0][1]
        assert "chat_id" not in params

    def test_list_all_falls_back_to_chat_id(self, cli_runner, mock_rpc):
        """Test that --all uses chat_id when no chat_title available."""
        mock_rpc.return_value = [
            {
                "id": "task_x",
                "trigger_at": "2026-01-12T09:00:00Z",
                "message": "No title task",
                "chat_id": "chatX",
            },
        ]

        result = cli_runner.invoke(app, ["list", "--all"])

        assert result.exit_code == 0
        assert "Room: chatX" in result.stdout

    def test_list_no_room_label_without_all(self, cli_runner, mock_rpc):
        """Test that Room label is not shown without --all."""
        mock_rpc.return_value = [
            {
                "id": "task_a",
                "trigger_at": "2026-01-12T09:00:00Z",
                "message": "Task A",
                "chat_id": "chatA",
                "chat_title": "Work Chat",
            },
        ]

        result = cli_runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "Room:" not in result.stdout

    def test_list_no_chat_id_shows_all(self, mock_rpc):
        """Test that missing ASH_CHAT_ID shows all tasks (graceful fallback)."""
        runner = CliRunner(
            env={
                "ASH_SESSION_ID": "test-session",
                "ASH_USER_ID": "user123",
                "ASH_PROVIDER": "telegram",
                "ASH_USERNAME": "testuser",
            }
        )
        mock_rpc.return_value = []

        runner.invoke(app, ["list"])

        # Without ASH_CHAT_ID, chat_id should not be in params
        params = mock_rpc.call_args[0][1]
        assert "chat_id" not in params


class TestScheduleCancel:
    """Tests for 'ash schedule cancel' command."""

    def test_cancel_success(self, cli_runner, mock_rpc):
        """Test cancelling a task by ID."""
        mock_rpc.return_value = {
            "cancelled": True,
            "entry": {
                "id": "abc12345",
                "message": "To cancel",
            },
        }

        result = cli_runner.invoke(app, ["cancel", "--id", "abc12345"])

        assert result.exit_code == 0
        assert "Cancelled" in result.stdout

    def test_cancel_not_found(self, cli_runner, mock_rpc):
        """Test cancelling non-existent task."""
        mock_rpc.return_value = {"cancelled": False}

        result = cli_runner.invoke(app, ["cancel", "--id", "nonexist"])

        assert result.exit_code == 1
        assert "No task found with ID" in result.output

    def test_cancel_other_user_task(self, cli_runner, mock_rpc):
        """Test that cancel rejects tasks owned by other users."""
        mock_rpc.side_effect = RPCError(
            code=-32000,
            message="Task other123 does not belong to you",
        )

        result = cli_runner.invoke(app, ["cancel", "--id", "other123"])

        assert result.exit_code == 1
        assert "does not belong to you" in result.output

    def test_cancel_requires_id(self, cli_runner, mock_rpc):
        """Test that cancel requires --id."""
        result = cli_runner.invoke(app, ["cancel"])

        assert result.exit_code != 0


class TestNaturalLanguageTime:
    """Tests for natural language time parsing in schedule create."""

    @pytest.fixture
    def cli_runner_with_tz(self, cli_runner):
        """CLI runner with timezone set (extends base cli_runner)."""
        cli_runner.env["ASH_TIMEZONE"] = "America/Los_Angeles"
        return cli_runner

    @pytest.fixture
    def _mock_create_rpc(self, mock_rpc):
        """Mock RPC for create commands."""

        def _create_response(*args, **kwargs):
            params = args[1] if len(args) > 1 else kwargs.get("params", {})
            entry = dict(params)
            entry["id"] = "nl_task1"
            return {"id": "nl_task1", "entry": entry}

        mock_rpc.side_effect = _create_response
        return mock_rpc

    @pytest.mark.parametrize(
        "time_input,message",
        [
            ("3pm", "Afternoon check"),
            ("at 3pm", "Meeting reminder"),
            ("9am", "Morning standup"),
            ("noon", "Lunch break"),
            ("midnight", "End of day"),
        ],
    )
    def test_create_with_clock_time_variants(
        self, cli_runner_with_tz, _mock_create_rpc, time_input, message
    ):
        """Test creating tasks with various clock time formats."""
        result = cli_runner_with_tz.invoke(app, ["create", message, "--at", time_input])

        assert result.exit_code == 0
        assert "Scheduled reminder" in result.stdout

        # Verify RPC was called with parsed time
        params = _mock_create_rpc.call_args[0][1]
        assert params["message"] == message
        assert "trigger_at" in params

    def test_create_with_natural_language_time(
        self, cli_runner_with_tz, _mock_create_rpc
    ):
        """Test creating a task with 'in 2 hours'."""
        result = cli_runner_with_tz.invoke(
            app, ["create", "Test reminder", "--at", "in 2 hours"]
        )

        assert result.exit_code == 0
        assert "Scheduled reminder" in result.stdout
        assert "Time:" in result.stdout
        assert "UTC:" in result.stdout
        assert "Task:" in result.stdout

    def test_create_with_clock_time(self, cli_runner_with_tz, _mock_create_rpc):
        """Test creating a task with 'tomorrow at 9am'."""
        result = cli_runner_with_tz.invoke(
            app, ["create", "Morning meeting", "--at", "tomorrow at 9am"]
        )

        assert result.exit_code == 0
        assert "Scheduled reminder" in result.stdout
        assert "America/Los_Angeles" in result.stdout

    def test_create_with_iso8601_still_works(
        self, cli_runner_with_tz, _mock_create_rpc
    ):
        """Test that ISO 8601 timestamps still work."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        result = cli_runner_with_tz.invoke(
            app, ["create", "ISO reminder", "--at", future]
        )

        assert result.exit_code == 0
        assert "Scheduled reminder" in result.stdout

    def test_create_rejects_invalid_time(self, cli_runner_with_tz, mock_rpc):
        """Test that invalid time strings are rejected."""
        result = cli_runner_with_tz.invoke(
            app, ["create", "Bad time", "--at", "not a valid time string xyz123"]
        )

        assert result.exit_code == 1
        assert "Could not parse time" in result.output
        mock_rpc.assert_not_called()

    def test_output_shows_local_time(self, cli_runner_with_tz, _mock_create_rpc):
        """Test that output shows time in local timezone."""
        result = cli_runner_with_tz.invoke(
            app, ["create", "Local time test", "--at", "in 1 hour"]
        )

        assert result.exit_code == 0
        # Should show timezone in output
        assert "America/Los_Angeles" in result.stdout
        # Should show UTC time too
        assert "UTC:" in result.stdout
