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
        """Test periodic entry without last_run calculates from now."""
        entry = ScheduleEntry(
            message="Test",
            cron="0 8 * * *",  # 8 AM daily
        )
        # Should calculate next_run from now
        # The entry is due if next_run <= now, which won't be true
        # for a future cron time
        assert entry._next_run_time() is not None

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
