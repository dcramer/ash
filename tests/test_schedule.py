"""Tests for file-based schedule watcher."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ash.events.schedule import ScheduleEntry, ScheduleWatcher


class TestScheduleEntry:
    """Tests for ScheduleEntry parsing."""

    def test_from_line_one_shot(self):
        """Test parsing one-shot entry."""
        line = '{"trigger_at": "2026-01-12T09:00:00+00:00", "message": "Test"}'
        entry = ScheduleEntry.from_line(line, 0)

        assert entry is not None
        assert entry.message == "Test"
        assert entry.trigger_at is not None
        assert entry.is_periodic is False

    def test_from_line_periodic(self):
        """Test parsing periodic entry."""
        line = '{"cron": "0 8 * * *", "message": "Daily task"}'
        entry = ScheduleEntry.from_line(line, 0)

        assert entry is not None
        assert entry.message == "Daily task"
        assert entry.cron == "0 8 * * *"
        assert entry.is_periodic is True

    def test_from_line_periodic_with_last_run(self):
        """Test parsing periodic entry with last_run."""
        line = '{"cron": "0 8 * * *", "message": "Daily", "last_run": "2026-01-11T08:00:00+00:00"}'
        entry = ScheduleEntry.from_line(line, 0)

        assert entry is not None
        assert entry.last_run is not None
        assert entry.last_run.day == 11

    def test_from_line_missing_message(self):
        """Test parsing entry without message."""
        line = '{"trigger_at": "2026-01-12T09:00:00+00:00"}'
        assert ScheduleEntry.from_line(line, 0) is None

    def test_from_line_missing_trigger(self):
        """Test parsing entry without trigger_at or cron."""
        line = '{"message": "Test"}'
        assert ScheduleEntry.from_line(line, 0) is None

    def test_from_line_invalid_json(self):
        """Test parsing invalid JSON."""
        assert ScheduleEntry.from_line("not json", 0) is None

    def test_from_line_empty(self):
        """Test parsing empty line."""
        assert ScheduleEntry.from_line("", 0) is None
        assert ScheduleEntry.from_line("# comment", 0) is None

    def test_is_due_one_shot_past(self):
        """Test one-shot entry in the past is due."""
        entry = ScheduleEntry(
            message="Test",
            trigger_at=datetime.now(UTC) - timedelta(hours=1),
        )
        assert entry.is_due() is True

    def test_is_due_one_shot_future(self):
        """Test one-shot entry in the future is not due."""
        entry = ScheduleEntry(
            message="Test",
            trigger_at=datetime.now(UTC) + timedelta(hours=1),
        )
        assert entry.is_due() is False

    def test_is_due_periodic_no_last_run(self):
        """Test periodic entry without last_run waits for next occurrence."""
        entry = ScheduleEntry(
            message="Test",
            cron="0 8 * * *",  # 8 AM daily
        )
        # New cron task should NOT be immediately due - it should wait
        # for the next scheduled occurrence
        next_run = entry._next_run_time()
        assert next_run is not None
        assert next_run > datetime.now(UTC)  # Next run is in the future
        assert entry.is_due() is False  # Therefore not due yet

    def test_to_json_line_one_shot(self):
        """Test serializing one-shot entry."""
        entry = ScheduleEntry(
            message="Test",
            trigger_at=datetime(2026, 1, 12, 9, 0, 0, tzinfo=UTC),
        )
        line = entry.to_json_line()
        assert '"message": "Test"' in line
        assert '"trigger_at"' in line

    def test_to_json_line_periodic(self):
        """Test serializing periodic entry."""
        entry = ScheduleEntry(
            message="Daily",
            cron="0 8 * * *",
            last_run=datetime(2026, 1, 11, 8, 0, 0, tzinfo=UTC),
        )
        line = entry.to_json_line()
        assert '"cron": "0 8 * * *"' in line
        assert '"last_run"' in line


class TestScheduleWatcher:
    """Tests for ScheduleWatcher."""

    def test_init(self, tmp_path: Path):
        """Test watcher initialization."""
        schedule_file = tmp_path / "schedule.jsonl"
        watcher = ScheduleWatcher(schedule_file)

        assert watcher.schedule_file == schedule_file
        assert watcher._running is False

    def test_get_entries_empty(self, tmp_path: Path):
        """Test getting entries from missing file."""
        watcher = ScheduleWatcher(tmp_path / "schedule.jsonl")
        assert watcher.get_entries() == []

    def test_get_entries_parses_file(self, tmp_path: Path):
        """Test getting entries from JSONL file."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"cron": "0 8 * * *", "message": "Task 2"}\n'
        )

        watcher = ScheduleWatcher(schedule_file)
        entries = watcher.get_entries()

        assert len(entries) == 2
        assert not entries[0].is_periodic
        assert entries[1].is_periodic

    def test_get_stats(self, tmp_path: Path):
        """Test getting watcher statistics."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"trigger_at": "2026-01-12T09:00:00+00:00", "message": "One-shot"}\n'
            '{"cron": "0 8 * * *", "message": "Periodic"}\n'
        )

        watcher = ScheduleWatcher(schedule_file)
        stats = watcher.get_stats()

        assert stats["total"] == 2
        assert stats["one_shot"] == 1
        assert stats["periodic"] == 1

    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path: Path):
        """Test starting and stopping watcher."""
        watcher = ScheduleWatcher(tmp_path / "schedule.jsonl", poll_interval=0.1)

        await watcher.start()
        assert watcher._running is True

        await watcher.stop()
        assert watcher._running is False

    @pytest.mark.asyncio
    async def test_triggers_due_one_shot(self, tmp_path: Path):
        """Test that due one-shot entries trigger handlers."""
        schedule_file = tmp_path / "schedule.jsonl"
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        schedule_file.write_text(f'{{"trigger_at": "{past}", "message": "Due"}}\n')

        watcher = ScheduleWatcher(schedule_file)
        triggered: list[ScheduleEntry] = []

        @watcher.on_due
        async def handler(entry: ScheduleEntry):
            triggered.append(entry)

        await watcher._check_schedule()

        assert len(triggered) == 1
        assert triggered[0].message == "Due"

    @pytest.mark.asyncio
    async def test_removes_triggered_one_shot(self, tmp_path: Path):
        """Test that triggered one-shot entries are removed."""
        schedule_file = tmp_path / "schedule.jsonl"
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        schedule_file.write_text(
            f'{{"trigger_at": "{past}", "message": "Due"}}\n'
            f'{{"trigger_at": "{future}", "message": "Not due"}}\n'
        )

        watcher = ScheduleWatcher(schedule_file)

        @watcher.on_due
        async def handler(entry: ScheduleEntry):
            pass

        await watcher._check_schedule()

        remaining = schedule_file.read_text()
        assert "Due" not in remaining
        assert "Not due" in remaining

    @pytest.mark.asyncio
    async def test_updates_periodic_last_run(self, tmp_path: Path):
        """Test that periodic entries get last_run updated."""
        schedule_file = tmp_path / "schedule.jsonl"
        # Create a periodic entry that's due (last_run far in past)
        old_time = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        schedule_file.write_text(
            f'{{"cron": "* * * * *", "message": "Every minute", "last_run": "{old_time}"}}\n'
        )

        watcher = ScheduleWatcher(schedule_file)

        @watcher.on_due
        async def handler(entry: ScheduleEntry):
            pass

        await watcher._check_schedule()

        # File should still have the entry but with updated last_run
        content = schedule_file.read_text()
        assert "Every minute" in content
        assert old_time not in content  # last_run should be updated

    @pytest.mark.asyncio
    async def test_does_not_trigger_future(self, tmp_path: Path):
        """Test that future entries don't trigger."""
        schedule_file = tmp_path / "schedule.jsonl"
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        schedule_file.write_text(f'{{"trigger_at": "{future}", "message": "Future"}}\n')

        watcher = ScheduleWatcher(schedule_file)
        triggered: list[ScheduleEntry] = []

        @watcher.on_due
        async def handler(entry: ScheduleEntry):
            triggered.append(entry)

        await watcher._check_schedule()

        assert triggered == []
        assert "Future" in schedule_file.read_text()

    def test_remove_entry_success(self, tmp_path: Path):
        """Test removing an entry by line number."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"id": "task0002", "trigger_at": "2026-01-13T09:00:00+00:00", "message": "Task 2"}\n'
            '{"id": "task0003", "trigger_at": "2026-01-14T09:00:00+00:00", "message": "Task 3"}\n'
        )

        watcher = ScheduleWatcher(schedule_file)
        result = watcher.remove_entry("task0002")  # Remove middle entry

        assert result is True
        content = schedule_file.read_text()
        assert "Task 1" in content
        assert "Task 2" not in content
        assert "Task 3" in content

    def test_remove_entry_first(self, tmp_path: Path):
        """Test removing the first entry."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"id": "task0002", "trigger_at": "2026-01-13T09:00:00+00:00", "message": "Task 2"}\n'
        )

        watcher = ScheduleWatcher(schedule_file)
        result = watcher.remove_entry("task0001")

        assert result is True
        content = schedule_file.read_text()
        assert "Task 1" not in content
        assert "Task 2" in content

    def test_remove_entry_last(self, tmp_path: Path):
        """Test removing the last entry."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"id": "task0002", "trigger_at": "2026-01-13T09:00:00+00:00", "message": "Task 2"}\n'
        )

        watcher = ScheduleWatcher(schedule_file)
        result = watcher.remove_entry("task0002")

        assert result is True
        content = schedule_file.read_text()
        assert "Task 1" in content
        assert "Task 2" not in content

    def test_remove_entry_invalid_id(self, tmp_path: Path):
        """Test removing with invalid ID."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
        )

        watcher = ScheduleWatcher(schedule_file)

        assert watcher.remove_entry("nonexistent") is False
        assert watcher.remove_entry("") is False
        assert watcher.remove_entry("task9999") is False

    def test_remove_entry_missing_file(self, tmp_path: Path):
        """Test removing from non-existent file."""
        watcher = ScheduleWatcher(tmp_path / "schedule.jsonl")
        assert watcher.remove_entry("nonexistent") is False

    def test_clear_all_success(self, tmp_path: Path):
        """Test clearing all entries."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"trigger_at": "2026-01-13T09:00:00+00:00", "message": "Task 2"}\n'
            '{"cron": "0 8 * * *", "message": "Task 3"}\n'
        )

        watcher = ScheduleWatcher(schedule_file)
        count = watcher.clear_all()

        assert count == 3
        assert schedule_file.read_text() == ""

    def test_clear_all_empty_file(self, tmp_path: Path):
        """Test clearing an empty file."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text("")

        watcher = ScheduleWatcher(schedule_file)
        count = watcher.clear_all()

        assert count == 0

    def test_clear_all_missing_file(self, tmp_path: Path):
        """Test clearing a non-existent file."""
        watcher = ScheduleWatcher(tmp_path / "schedule.jsonl")
        count = watcher.clear_all()

        assert count == 0

    @pytest.mark.asyncio
    async def test_failed_one_shot_removed(self, tmp_path: Path):
        """Test that failed one-shot entries are removed (not retried forever)."""
        schedule_file = tmp_path / "schedule.jsonl"
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        schedule_file.write_text(f'{{"trigger_at": "{past}", "message": "Fail me"}}\n')

        watcher = ScheduleWatcher(schedule_file)

        @watcher.on_due
        async def handler(entry: ScheduleEntry):
            raise RuntimeError("Handler failed intentionally")

        await watcher._check_schedule()

        # Entry should be removed even though handler failed
        assert "Fail me" not in schedule_file.read_text()

    @pytest.mark.asyncio
    async def test_failed_periodic_updates_last_run(self, tmp_path: Path):
        """Test that failed periodic entries update last_run (prevent immediate retry)."""
        schedule_file = tmp_path / "schedule.jsonl"
        old_time = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        schedule_file.write_text(
            f'{{"cron": "* * * * *", "message": "Fail periodic", "last_run": "{old_time}"}}\n'
        )

        watcher = ScheduleWatcher(schedule_file)

        @watcher.on_due
        async def handler(entry: ScheduleEntry):
            raise RuntimeError("Periodic handler failed")

        await watcher._check_schedule()

        # Entry should still exist with updated last_run
        content = schedule_file.read_text()
        assert "Fail periodic" in content
        assert old_time not in content  # last_run should be updated


class TestScheduledTaskHandler:
    """Tests for ScheduledTaskHandler."""

    @pytest.mark.asyncio
    async def test_handle_missing_context(self):
        """Test handler rejects entries without routing context."""
        from unittest.mock import MagicMock

        from ash.events.handler import ScheduledTaskHandler

        mock_agent = MagicMock()
        handler = ScheduledTaskHandler(agent=mock_agent, senders={})

        # Entry without provider/chat_id
        entry = ScheduleEntry(message="Test", trigger_at=datetime.now(UTC))

        with pytest.raises(ValueError, match="Missing required routing context"):
            await handler.handle(entry)

    @pytest.mark.asyncio
    async def test_handle_valid_context(self):
        """Test handler accepts entries with valid routing context."""
        from unittest.mock import AsyncMock, MagicMock

        from ash.events.handler import ScheduledTaskHandler

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_agent.process_message = AsyncMock(return_value=mock_response)

        mock_sender = AsyncMock(return_value="msg_123")
        handler = ScheduledTaskHandler(
            agent=mock_agent, senders={"telegram": mock_sender}
        )

        entry = ScheduleEntry(
            message="Test",
            trigger_at=datetime.now(UTC),
            provider="telegram",
            chat_id="123",
            user_id="456",
        )

        await handler.handle(entry)

        # Verify agent was called
        mock_agent.process_message.assert_called_once()
        # Verify response was sent
        mock_sender.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_calls_registrar(self):
        """Test handler calls registrar after sending message."""
        from unittest.mock import AsyncMock, MagicMock

        from ash.events.handler import ScheduledTaskHandler

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_agent.process_message = AsyncMock(return_value=mock_response)

        mock_sender = AsyncMock(return_value="msg_123")
        mock_registrar = AsyncMock()
        handler = ScheduledTaskHandler(
            agent=mock_agent,
            senders={"telegram": mock_sender},
            registrars={"telegram": mock_registrar},
        )

        entry = ScheduleEntry(
            message="Test",
            trigger_at=datetime.now(UTC),
            provider="telegram",
            chat_id="123",
            user_id="456",
        )

        await handler.handle(entry)

        # Verify registrar was called with chat_id and message_id
        mock_registrar.assert_called_once_with("123", "msg_123")


class TestScheduleEntryTimezone:
    """Tests for ScheduleEntry timezone handling."""

    def test_entry_with_stored_timezone(self):
        """Test entry stores and uses its own timezone."""
        entry = ScheduleEntry(
            message="Test",
            cron="0 8 * * *",
            timezone="America/Los_Angeles",
        )
        assert entry.timezone == "America/Los_Angeles"

    def test_timezone_serialization(self):
        """Test timezone is serialized in to_json_line."""
        entry = ScheduleEntry(
            message="Test",
            cron="0 8 * * *",
            timezone="America/Los_Angeles",
        )
        line = entry.to_json_line()
        assert '"timezone": "America/Los_Angeles"' in line

    def test_timezone_deserialization(self):
        """Test timezone is parsed from JSON line."""
        line = '{"cron": "0 8 * * *", "message": "Test", "timezone": "America/Los_Angeles"}'
        entry = ScheduleEntry.from_line(line)
        assert entry is not None
        assert entry.timezone == "America/Los_Angeles"

    def test_cron_evaluated_in_local_timezone(self):
        """Test cron expressions are evaluated in the stored local timezone."""
        # Same cron, different stored timezone - should give different UTC times
        entry_la = ScheduleEntry(
            message="Test",
            cron="0 8 * * *",
            timezone="America/Los_Angeles",
        )
        entry_utc = ScheduleEntry(
            message="Test",
            cron="0 8 * * *",
            timezone="UTC",
        )
        entry_none = ScheduleEntry(
            message="Test",
            cron="0 8 * * *",
            timezone=None,
        )

        next_la = entry_la.next_fire_time()
        next_utc = entry_utc.next_fire_time()
        next_none = entry_none.next_fire_time()

        assert next_la is not None
        assert next_utc is not None
        assert next_none is not None

        # LA time is 8 hours behind UTC (or 7 during DST)
        # 8 AM LA = 16:00 UTC (or 15:00 during DST)
        # 8 AM UTC = 8:00 UTC
        assert next_la != next_utc  # Different timezones = different UTC times
        assert next_utc.hour == 8  # 8 AM UTC
        # LA is UTC-8 (PST) or UTC-7 (PDT), so 8 AM LA is 15:00 or 16:00 UTC
        assert next_la.hour in (15, 16)  # 8 AM Pacific in UTC

        # None timezone defaults to UTC
        assert next_none == next_utc

    def test_one_shot_timezone_stored(self):
        """Test one-shot entry can store timezone."""
        entry = ScheduleEntry(
            message="Test",
            trigger_at=datetime(2026, 1, 15, 8, 0, 0, tzinfo=UTC),
            timezone="America/Los_Angeles",
        )
        assert entry.timezone == "America/Los_Angeles"
        line = entry.to_json_line()
        assert '"timezone": "America/Los_Angeles"' in line

    def test_cron_next_fire_is_utc(self):
        """Test cron next fire time is returned in UTC."""
        entry = ScheduleEntry(
            message="Test",
            cron="0 15 * * *",  # 3 PM UTC = 7 AM PST
            last_run=datetime(2026, 1, 15, 15, 0, 0, tzinfo=UTC),
        )

        next_fire = entry.next_fire_time()
        assert next_fire is not None
        assert next_fire.tzinfo == UTC
        assert next_fire.hour == 15  # 3 PM UTC
