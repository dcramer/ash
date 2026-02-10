"""Tests for sandboxed CLI schedule commands."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from ash_sandbox_cli.commands.schedule import app
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
def schedule_file(tmp_path: Path, monkeypatch):
    """Create a temporary schedule file."""
    schedule = tmp_path / "schedule.jsonl"
    monkeypatch.setattr("ash_sandbox_cli.commands.schedule.SCHEDULE_FILE", schedule)
    return schedule


class TestScheduleCreate:
    """Tests for 'ash schedule create' command."""

    def test_create_one_shot(self, cli_runner, schedule_file):
        """Test creating a one-shot task."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        result = cli_runner.invoke(app, ["create", "Test reminder", "--at", future])

        assert result.exit_code == 0
        assert "Scheduled reminder" in result.stdout
        assert "id=" in result.stdout

        # Verify file contents
        entries = [
            json.loads(line) for line in schedule_file.read_text().strip().split("\n")
        ]
        assert len(entries) == 1
        assert entries[0]["message"] == "Test reminder"
        # Stored time normalizes to Z suffix
        assert entries[0]["trigger_at"] == future.replace("+00:00", "Z")
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
        assert "in the past" in result.output

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
        """Test listing tasks owned by current user."""
        schedule_file.write_text(
            '{"id": "abc12345", "trigger_at": "2026-01-12T09:00:00Z", "message": "Task 1", "user_id": "user123"}\n'
            '{"id": "def67890", "cron": "0 8 * * *", "message": "Task 2", "user_id": "user123"}\n'
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

    def test_list_filters_by_user(self, cli_runner, schedule_file):
        """Test that list only shows tasks owned by current user."""
        schedule_file.write_text(
            '{"id": "mine", "trigger_at": "2026-01-12T09:00:00Z", "message": "My task", "user_id": "user123"}\n'
            '{"id": "other", "trigger_at": "2026-01-12T09:00:00Z", "message": "Other task", "user_id": "other999"}\n'
        )

        result = cli_runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "mine" in result.stdout
        assert "My task" in result.stdout
        assert "other" not in result.stdout
        assert "Other task" not in result.stdout
        assert "Total: 1 task(s)" in result.stdout


class TestScheduleCancel:
    """Tests for 'ash schedule cancel' command."""

    def test_cancel_success(self, cli_runner, schedule_file):
        """Test cancelling a task by ID."""
        schedule_file.write_text(
            '{"id": "abc12345", "trigger_at": "2026-01-12T09:00:00Z", "message": "To cancel", "user_id": "user123"}\n'
            '{"id": "def67890", "trigger_at": "2026-01-13T09:00:00Z", "message": "To keep", "user_id": "user123"}\n'
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
        schedule_file.write_text(
            '{"id": "abc12345", "message": "Existing task", "user_id": "user123"}\n'
        )

        result = cli_runner.invoke(app, ["cancel", "--id", "nonexist"])

        assert result.exit_code == 1
        assert "No task found with ID" in result.output

    def test_cancel_other_user_task(self, cli_runner, schedule_file):
        """Test that cancel rejects tasks owned by other users."""
        schedule_file.write_text(
            '{"id": "other123", "message": "Other user task", "user_id": "other999"}\n'
        )

        result = cli_runner.invoke(app, ["cancel", "--id", "other123"])

        assert result.exit_code == 1
        assert "does not belong to you" in result.output

    def test_cancel_requires_id(self, cli_runner, schedule_file):
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
        self, cli_runner_with_tz, schedule_file, time_input, message
    ):
        """Test creating tasks with various clock time formats."""
        result = cli_runner_with_tz.invoke(app, ["create", message, "--at", time_input])

        assert result.exit_code == 0
        assert "Scheduled reminder" in result.stdout

        entries = [
            json.loads(line) for line in schedule_file.read_text().strip().split("\n")
        ]
        assert len(entries) == 1
        assert entries[0]["message"] == message
        assert "trigger_at" in entries[0]

    def test_create_with_natural_language_time(self, cli_runner_with_tz, schedule_file):
        """Test creating a task with 'in 2 hours'."""
        result = cli_runner_with_tz.invoke(
            app, ["create", "Test reminder", "--at", "in 2 hours"]
        )

        assert result.exit_code == 0
        assert "Scheduled reminder" in result.stdout
        assert "Time:" in result.stdout
        assert "UTC:" in result.stdout
        assert "Task:" in result.stdout

        # Verify file contents
        entries = [
            json.loads(line) for line in schedule_file.read_text().strip().split("\n")
        ]
        assert len(entries) == 1
        assert entries[0]["message"] == "Test reminder"
        assert "trigger_at" in entries[0]
        # Should be in ISO 8601 format with Z suffix
        assert entries[0]["trigger_at"].endswith("Z")

    def test_create_with_clock_time(self, cli_runner_with_tz, schedule_file):
        """Test creating a task with 'tomorrow at 9am'."""
        result = cli_runner_with_tz.invoke(
            app, ["create", "Morning meeting", "--at", "tomorrow at 9am"]
        )

        assert result.exit_code == 0
        assert "Scheduled reminder" in result.stdout
        assert "America/Los_Angeles" in result.stdout

    def test_create_with_iso8601_still_works(self, cli_runner_with_tz, schedule_file):
        """Test that ISO 8601 timestamps still work."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        result = cli_runner_with_tz.invoke(
            app, ["create", "ISO reminder", "--at", future]
        )

        assert result.exit_code == 0
        assert "Scheduled reminder" in result.stdout

    def test_create_rejects_invalid_time(self, cli_runner_with_tz, schedule_file):
        """Test that invalid time strings are rejected."""
        result = cli_runner_with_tz.invoke(
            app, ["create", "Bad time", "--at", "not a valid time string xyz123"]
        )

        assert result.exit_code == 1
        assert "Could not parse time" in result.output

    def test_output_shows_local_time(self, cli_runner_with_tz, schedule_file):
        """Test that output shows time in local timezone."""
        result = cli_runner_with_tz.invoke(
            app, ["create", "Local time test", "--at", "in 1 hour"]
        )

        assert result.exit_code == 0
        # Should show timezone in output
        assert "America/Los_Angeles" in result.stdout
        # Should show UTC time too
        assert "UTC:" in result.stdout
