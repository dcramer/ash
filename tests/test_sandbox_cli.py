"""Tests for sandboxed CLI schedule commands."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ash.sandbox.cli.commands.schedule import app


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
def schedule_file(tmp_path: Path, monkeypatch):
    """Create a temporary schedule file."""
    schedule = tmp_path / "schedule.jsonl"
    monkeypatch.setattr("ash.sandbox.cli.commands.schedule.SCHEDULE_FILE", schedule)
    return schedule


class TestScheduleCreate:
    """Tests for 'ash schedule create' command."""

    def test_create_one_shot(self, cli_runner, schedule_file):
        """Test creating a one-shot task."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        result = cli_runner.invoke(app, ["create", "Test reminder", "--at", future])

        assert result.exit_code == 0
        assert "Scheduled one-time task" in result.stdout
        assert "id=" in result.stdout

        # Verify file contents
        entries = [
            json.loads(line) for line in schedule_file.read_text().strip().split("\n")
        ]
        assert len(entries) == 1
        assert entries[0]["message"] == "Test reminder"
        assert entries[0]["trigger_at"] == future
        assert entries[0]["chat_id"] == "chat456"
        assert entries[0]["provider"] == "telegram"
        assert "id" in entries[0]

    def test_create_periodic(self, cli_runner, schedule_file):
        """Test creating a periodic task."""
        result = cli_runner.invoke(
            app, ["create", "Daily check", "--cron", "0 8 * * *"]
        )

        assert result.exit_code == 0
        assert "Scheduled recurring task" in result.stdout
        assert "0 8 * * *" in result.stdout

        entries = [
            json.loads(line) for line in schedule_file.read_text().strip().split("\n")
        ]
        assert len(entries) == 1
        assert entries[0]["message"] == "Daily check"
        assert entries[0]["cron"] == "0 8 * * *"

    def test_create_requires_trigger(self, cli_runner, schedule_file):
        """Test that create requires --at or --cron."""
        result = cli_runner.invoke(app, ["create", "Missing trigger"])

        assert result.exit_code == 1
        assert "Must specify either --at" in result.output

    def test_create_rejects_both_triggers(self, cli_runner, schedule_file):
        """Test that create rejects both --at and --cron."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        result = cli_runner.invoke(
            app, ["create", "Both triggers", "--at", future, "--cron", "0 8 * * *"]
        )

        assert result.exit_code == 1
        assert "Cannot specify both" in result.output

    def test_create_rejects_past_time(self, cli_runner, schedule_file):
        """Test that --at rejects past times."""
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        result = cli_runner.invoke(app, ["create", "Past time", "--at", past])

        assert result.exit_code == 1
        assert "must be in the future" in result.output

    def test_create_requires_routing_context(
        self, cli_runner_no_context, schedule_file
    ):
        """Test that create requires routing context."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        result = cli_runner_no_context.invoke(
            app, ["create", "No context", "--at", future]
        )

        assert result.exit_code == 1
        assert "Scheduling requires a provider context" in result.output


class TestScheduleList:
    """Tests for 'ash schedule list' command."""

    def test_list_empty(self, cli_runner, schedule_file):
        """Test listing with no tasks."""
        result = cli_runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "No scheduled tasks found" in result.stdout

    def test_list_with_entries(self, cli_runner, schedule_file):
        """Test listing tasks."""
        schedule_file.write_text(
            '{"id": "abc12345", "trigger_at": "2026-01-12T09:00:00Z", "message": "Task 1"}\n'
            '{"id": "def67890", "cron": "0 8 * * *", "message": "Task 2"}\n'
        )

        result = cli_runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "abc12345" in result.stdout
        assert "def67890" in result.stdout
        assert "Task 1" in result.stdout
        assert "Task 2" in result.stdout
        assert "one-shot" in result.stdout
        assert "periodic" in result.stdout
        assert "Total: 2 task(s)" in result.stdout


class TestScheduleCancel:
    """Tests for 'ash schedule cancel' command."""

    def test_cancel_success(self, cli_runner, schedule_file):
        """Test cancelling a task by ID."""
        schedule_file.write_text(
            '{"id": "abc12345", "trigger_at": "2026-01-12T09:00:00Z", "message": "To cancel"}\n'
            '{"id": "def67890", "trigger_at": "2026-01-13T09:00:00Z", "message": "To keep"}\n'
        )

        result = cli_runner.invoke(app, ["cancel", "--id", "abc12345"])

        assert result.exit_code == 0
        assert "Cancelled" in result.stdout

        # Verify remaining entries
        content = schedule_file.read_text()
        assert "To cancel" not in content
        assert "To keep" in content

    def test_cancel_not_found(self, cli_runner, schedule_file):
        """Test cancelling non-existent task."""
        schedule_file.write_text('{"id": "abc12345", "message": "Existing task"}\n')

        result = cli_runner.invoke(app, ["cancel", "--id", "nonexist"])

        assert result.exit_code == 1
        assert "No task found with ID" in result.output

    def test_cancel_requires_id(self, cli_runner, schedule_file):
        """Test that cancel requires --id."""
        result = cli_runner.invoke(app, ["cancel"])

        assert result.exit_code != 0


class TestScheduleClear:
    """Tests for 'ash schedule clear' command."""

    def test_clear_empty(self, cli_runner, schedule_file):
        """Test clearing with no tasks."""
        result = cli_runner.invoke(app, ["clear"])

        assert result.exit_code == 0
        assert "No scheduled tasks to clear" in result.stdout

    def test_clear_success(self, cli_runner, schedule_file):
        """Test clearing all tasks."""
        schedule_file.write_text(
            '{"id": "abc12345", "message": "Task 1"}\n'
            '{"id": "def67890", "message": "Task 2"}\n'
        )

        result = cli_runner.invoke(app, ["clear"])

        assert result.exit_code == 0
        assert "Cleared 2 scheduled task(s)" in result.stdout
        assert schedule_file.read_text() == ""
