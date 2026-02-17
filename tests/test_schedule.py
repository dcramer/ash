"""Tests for scheduling subsystem."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ash.scheduling import ScheduleEntry, ScheduleStore, ScheduleWatcher


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


class TestScheduleStore:
    """Tests for ScheduleStore CRUD operations."""

    def test_get_entries_empty(self, tmp_path: Path):
        """Test getting entries from missing file."""
        store = ScheduleStore(tmp_path / "schedule.jsonl")
        assert store.get_entries() == []

    def test_get_entries_parses_file(self, tmp_path: Path):
        """Test getting entries from JSONL file."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"cron": "0 8 * * *", "message": "Task 2"}\n'
        )

        store = ScheduleStore(schedule_file)
        entries = store.get_entries()

        assert len(entries) == 2
        assert not entries[0].is_periodic
        assert entries[1].is_periodic

    def test_get_entry_found(self, tmp_path: Path):
        """Test getting a single entry by ID."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"id": "task0002", "trigger_at": "2026-01-13T09:00:00+00:00", "message": "Task 2"}\n'
        )

        store = ScheduleStore(schedule_file)
        entry = store.get_entry("task0002")

        assert entry is not None
        assert entry.message == "Task 2"

    def test_get_entry_not_found(self, tmp_path: Path):
        """Test getting a non-existent entry returns None."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
        )

        store = ScheduleStore(schedule_file)
        assert store.get_entry("nonexistent") is None

    def test_get_stats(self, tmp_path: Path):
        """Test getting store statistics."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"trigger_at": "2026-01-12T09:00:00+00:00", "message": "One-shot"}\n'
            '{"cron": "0 8 * * *", "message": "Periodic"}\n'
        )

        store = ScheduleStore(schedule_file)
        stats = store.get_stats()

        assert stats["total"] == 2
        assert stats["one_shot"] == 1
        assert stats["periodic"] == 1
        assert "running" not in stats  # Store doesn't track running state

    def test_add_entry(self, tmp_path: Path):
        """Test adding an entry to the file."""
        schedule_file = tmp_path / "schedule.jsonl"

        store = ScheduleStore(schedule_file)
        entry = ScheduleEntry(
            id="newtask1",
            message="New task",
            trigger_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
        )
        store.add_entry(entry)

        # Verify file was written
        entries = store.get_entries()
        assert len(entries) == 1
        assert entries[0].id == "newtask1"
        assert entries[0].message == "New task"

    def test_add_entry_creates_parent_dirs(self, tmp_path: Path):
        """Test add_entry creates parent directories if needed."""
        schedule_file = tmp_path / "subdir" / "schedule.jsonl"
        store = ScheduleStore(schedule_file)

        entry = ScheduleEntry(
            id="task1",
            message="Test",
            trigger_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
        )
        store.add_entry(entry)
        assert schedule_file.exists()

    def test_add_entry_appends(self, tmp_path: Path):
        """Test add_entry appends to existing file."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-01-12T09:00:00+00:00", "message": "Existing"}\n'
        )

        store = ScheduleStore(schedule_file)
        entry = ScheduleEntry(
            id="task0002",
            message="New",
            trigger_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
        )
        store.add_entry(entry)

        entries = store.get_entries()
        assert len(entries) == 2

    def test_remove_entry_success(self, tmp_path: Path):
        """Test removing an entry by ID."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"id": "task0002", "trigger_at": "2026-01-13T09:00:00+00:00", "message": "Task 2"}\n'
            '{"id": "task0003", "trigger_at": "2026-01-14T09:00:00+00:00", "message": "Task 3"}\n'
        )

        store = ScheduleStore(schedule_file)
        result = store.remove_entry("task0002")  # Remove middle entry

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

        store = ScheduleStore(schedule_file)
        result = store.remove_entry("task0001")

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

        store = ScheduleStore(schedule_file)
        result = store.remove_entry("task0002")

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

        store = ScheduleStore(schedule_file)

        assert store.remove_entry("nonexistent") is False
        assert store.remove_entry("") is False
        assert store.remove_entry("task9999") is False

    def test_remove_entry_missing_file(self, tmp_path: Path):
        """Test removing from non-existent file."""
        store = ScheduleStore(tmp_path / "schedule.jsonl")
        assert store.remove_entry("nonexistent") is False

    def test_clear_all_success(self, tmp_path: Path):
        """Test clearing all entries."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"trigger_at": "2026-01-13T09:00:00+00:00", "message": "Task 2"}\n'
            '{"cron": "0 8 * * *", "message": "Task 3"}\n'
        )

        store = ScheduleStore(schedule_file)
        count = store.clear_all()

        assert count == 3
        assert schedule_file.read_text() == ""

    def test_clear_all_empty_file(self, tmp_path: Path):
        """Test clearing an empty file."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text("")

        store = ScheduleStore(schedule_file)
        count = store.clear_all()

        assert count == 0

    def test_clear_all_missing_file(self, tmp_path: Path):
        """Test clearing a non-existent file."""
        store = ScheduleStore(tmp_path / "schedule.jsonl")
        count = store.clear_all()

        assert count == 0

    def test_update_entry_message_only(self, tmp_path: Path):
        """Test updating only the message of an entry."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-12-12T09:00:00+00:00", "message": "Original"}\n'
        )

        store = ScheduleStore(schedule_file)
        result = store.update_entry("task0001", message="Updated message")

        assert result is not None
        assert result.message == "Updated message"
        assert result.id == "task0001"

        # Verify file was updated
        content = schedule_file.read_text()
        assert "Updated message" in content
        assert "Original" not in content

    def test_update_entry_trigger_at(self, tmp_path: Path):
        """Test updating trigger_at for a one-shot entry."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-12-12T09:00:00+00:00", "message": "Test"}\n'
        )

        store = ScheduleStore(schedule_file)
        new_time = datetime.now(UTC) + timedelta(hours=2)
        result = store.update_entry("task0001", trigger_at=new_time)

        assert result is not None
        assert result.trigger_at == new_time

    def test_update_entry_cron(self, tmp_path: Path):
        """Test updating cron expression for a periodic entry."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "cron": "0 8 * * *", "message": "Daily"}\n'
        )

        store = ScheduleStore(schedule_file)
        result = store.update_entry("task0001", cron="0 9 * * *")

        assert result is not None
        assert result.cron == "0 9 * * *"

        # Verify file was updated
        content = schedule_file.read_text()
        assert "0 9 * * *" in content

    def test_update_entry_timezone(self, tmp_path: Path):
        """Test updating timezone of an entry."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "cron": "0 8 * * *", "message": "Daily", "timezone": "UTC"}\n'
        )

        store = ScheduleStore(schedule_file)
        result = store.update_entry("task0001", timezone="America/Los_Angeles")

        assert result is not None
        assert result.timezone == "America/Los_Angeles"

    def test_update_entry_not_found(self, tmp_path: Path):
        """Test updating non-existent entry returns None."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-12-12T09:00:00+00:00", "message": "Test"}\n'
        )

        store = ScheduleStore(schedule_file)
        result = store.update_entry("nonexistent", message="Updated")

        assert result is None

    def test_update_entry_switch_oneshot_to_periodic_fails(self, tmp_path: Path):
        """Test that switching from one-shot to periodic fails."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-12-12T09:00:00+00:00", "message": "Test"}\n'
        )

        store = ScheduleStore(schedule_file)

        with pytest.raises(
            ValueError, match="Cannot change one-shot entry to periodic"
        ):
            store.update_entry("task0001", cron="0 8 * * *")

    def test_update_entry_switch_periodic_to_oneshot_fails(self, tmp_path: Path):
        """Test that switching from periodic to one-shot fails."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "cron": "0 8 * * *", "message": "Daily"}\n'
        )

        store = ScheduleStore(schedule_file)
        future_time = datetime.now(UTC) + timedelta(hours=2)

        with pytest.raises(
            ValueError, match="Cannot change periodic entry to one-shot"
        ):
            store.update_entry("task0001", trigger_at=future_time)

    def test_update_entry_trigger_at_in_past_fails(self, tmp_path: Path):
        """Test that setting trigger_at to past time fails."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-12-12T09:00:00+00:00", "message": "Test"}\n'
        )

        store = ScheduleStore(schedule_file)
        past_time = datetime.now(UTC) - timedelta(hours=1)

        with pytest.raises(ValueError, match="trigger_at must be in the future"):
            store.update_entry("task0001", trigger_at=past_time)

    def test_update_entry_invalid_cron_fails(self, tmp_path: Path):
        """Test that invalid cron expression fails."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "cron": "0 8 * * *", "message": "Daily"}\n'
        )

        store = ScheduleStore(schedule_file)

        with pytest.raises(ValueError, match="Invalid cron expression"):
            store.update_entry("task0001", cron="invalid cron")

    def test_update_entry_no_fields_fails(self, tmp_path: Path):
        """Test that updating with no fields fails."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-12-12T09:00:00+00:00", "message": "Test"}\n'
        )

        store = ScheduleStore(schedule_file)

        with pytest.raises(ValueError, match="At least one updatable field"):
            store.update_entry("task0001")

    def test_update_entry_missing_file(self, tmp_path: Path):
        """Test updating from non-existent file returns None."""
        store = ScheduleStore(tmp_path / "schedule.jsonl")
        result = store.update_entry("task0001", message="Updated")

        assert result is None

    def test_update_entry_preserves_other_entries(self, tmp_path: Path):
        """Test that updating one entry doesn't affect others."""
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "task0001", "trigger_at": "2026-12-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"id": "task0002", "trigger_at": "2026-12-13T09:00:00+00:00", "message": "Original message"}\n'
            '{"id": "task0003", "trigger_at": "2026-12-14T09:00:00+00:00", "message": "Task 3"}\n'
        )

        store = ScheduleStore(schedule_file)
        result = store.update_entry("task0002", message="Updated message")

        assert result is not None
        assert result.message == "Updated message"

        # Verify all entries are still present
        content = schedule_file.read_text()
        assert "Task 1" in content
        assert "Updated message" in content
        assert "Task 3" in content
        assert "Original message" not in content  # Old message should be gone

    def test_remove_and_update(self, tmp_path: Path):
        """Test atomic remove + update operation."""
        schedule_file = tmp_path / "schedule.jsonl"
        old_time = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        schedule_file.write_text(
            '{"id": "oneshot1", "trigger_at": "2026-01-12T09:00:00+00:00", "message": "One-shot"}\n'
            f'{{"id": "periodic1", "cron": "* * * * *", "message": "Periodic", "last_run": "{old_time}"}}\n'
            '{"id": "keep1", "trigger_at": "2027-01-12T09:00:00+00:00", "message": "Keep"}\n'
        )

        store = ScheduleStore(schedule_file)

        # Build updated periodic entry
        periodic = store.get_entry("periodic1")
        assert periodic is not None
        periodic.last_run = datetime.now(UTC)

        store.remove_and_update(
            remove_ids={"oneshot1"},
            updates={"periodic1": periodic},
        )

        entries = store.get_entries()
        assert len(entries) == 2
        # One-shot removed
        assert not any(e.id == "oneshot1" for e in entries)
        # Periodic updated
        periodic_entry = next(e for e in entries if e.id == "periodic1")
        assert periodic_entry.last_run is not None
        assert old_time not in schedule_file.read_text()
        # Other entry preserved
        assert any(e.id == "keep1" for e in entries)

    def test_remove_and_update_no_file(self, tmp_path: Path):
        """Test remove_and_update with missing file is a no-op."""
        store = ScheduleStore(tmp_path / "schedule.jsonl")
        store.remove_and_update(remove_ids={"x"}, updates={})  # Should not raise


class TestScheduleWatcher:
    """Tests for ScheduleWatcher (polling/handler behavior)."""

    def _make_watcher(
        self, tmp_path: Path, content: str = ""
    ) -> tuple[ScheduleStore, ScheduleWatcher]:
        schedule_file = tmp_path / "schedule.jsonl"
        if content:
            schedule_file.write_text(content)
        store = ScheduleStore(schedule_file)
        watcher = ScheduleWatcher(store)
        return store, watcher

    def test_init(self, tmp_path: Path):
        """Test watcher initialization."""
        store = ScheduleStore(tmp_path / "schedule.jsonl")
        watcher = ScheduleWatcher(store)

        assert watcher.store is store
        assert watcher._running is False

    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path: Path):
        """Test starting and stopping watcher."""
        store = ScheduleStore(tmp_path / "schedule.jsonl")
        watcher = ScheduleWatcher(store, poll_interval=0.1)

        await watcher.start()
        assert watcher._running is True

        await watcher.stop()
        assert watcher._running is False

    @pytest.mark.asyncio
    async def test_triggers_due_one_shot(self, tmp_path: Path):
        """Test that due one-shot entries trigger handlers."""
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        _, watcher = self._make_watcher(
            tmp_path, f'{{"trigger_at": "{past}", "message": "Due"}}\n'
        )
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
            f'{{"id": "due1", "trigger_at": "{past}", "message": "Due"}}\n'
            f'{{"id": "notdue1", "trigger_at": "{future}", "message": "Not due"}}\n'
        )

        store = ScheduleStore(schedule_file)
        watcher = ScheduleWatcher(store)

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
        old_time = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        schedule_file.write_text(
            f'{{"id": "p1", "cron": "* * * * *", "message": "Every minute", "last_run": "{old_time}"}}\n'
        )

        store = ScheduleStore(schedule_file)
        watcher = ScheduleWatcher(store)

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

        store = ScheduleStore(schedule_file)
        watcher = ScheduleWatcher(store)
        triggered: list[ScheduleEntry] = []

        @watcher.on_due
        async def handler(entry: ScheduleEntry):
            triggered.append(entry)

        await watcher._check_schedule()

        assert triggered == []
        assert "Future" in schedule_file.read_text()

    @pytest.mark.asyncio
    async def test_failed_one_shot_removed(self, tmp_path: Path):
        """Test that failed one-shot entries are removed (not retried forever)."""
        schedule_file = tmp_path / "schedule.jsonl"
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        schedule_file.write_text(
            f'{{"id": "fail1", "trigger_at": "{past}", "message": "Fail me"}}\n'
        )

        store = ScheduleStore(schedule_file)
        watcher = ScheduleWatcher(store)

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
            f'{{"id": "pfail1", "cron": "* * * * *", "message": "Fail periodic", "last_run": "{old_time}"}}\n'
        )

        store = ScheduleStore(schedule_file)
        watcher = ScheduleWatcher(store)

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

        from ash.scheduling import ScheduledTaskHandler

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

        from ash.scheduling import ScheduledTaskHandler

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
        # Verify response was sent with reply_to kwarg
        mock_sender.assert_called_once()
        call_kwargs = mock_sender.call_args
        assert "reply_to" in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_handle_calls_registrar(self):
        """Test handler calls registrar after sending message."""
        from unittest.mock import AsyncMock, MagicMock

        from ash.scheduling import ScheduledTaskHandler

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

    def test_timezone_cron_no_last_run_fires_at_scheduled_time(self):
        """Test cron entry with timezone but no last_run fires correctly.

        This tests the exact scenario that was failing: a cron entry created
        with a timezone like "America/Los_Angeles", with no last_run (first
        execution), should fire when the scheduled time arrives.
        """
        from zoneinfo import ZoneInfo

        # Create entry for 7:45 AM Mon/Tue/Thu in LA timezone (like user's entry)
        entry = ScheduleEntry(
            id="test1234",
            message="Test weekly meeting reminder",
            cron="45 7 * * 1,2,4",  # 7:45 AM Mon/Tue/Thu
            timezone="America/Los_Angeles",
            last_run=None,  # First execution - no last_run
        )

        # Calculate next fire time
        next_fire = entry.next_fire_time()
        assert next_fire is not None
        assert next_fire.tzinfo == UTC

        # Convert to LA time to verify it's 7:45 AM local
        la_tz = ZoneInfo("America/Los_Angeles")
        next_fire_la = next_fire.astimezone(la_tz)
        assert next_fire_la.hour == 7
        assert next_fire_la.minute == 45
        # Should be Mon(0), Tue(1), or Thu(3)
        assert next_fire_la.weekday() in (0, 1, 3)

    def test_timezone_cron_is_due_with_old_last_run(self):
        """Test is_due returns True for cron entries with old last_run.

        A cron entry with a last_run in the past should be due if the
        next scheduled time has passed.
        """
        # Entry for every minute with last_run 2 days ago
        # This ensures next fire time is in the past
        entry = ScheduleEntry(
            id="test5678",
            message="Morning check",
            cron="* * * * *",  # Every minute
            timezone="America/Los_Angeles",
            last_run=datetime.now(UTC) - timedelta(days=2),
        )

        # With last_run 2 days ago and cron "every minute",
        # the next fire time is definitely in the past
        next_fire = entry.next_fire_time()
        assert next_fire is not None
        assert next_fire < datetime.now(UTC)

        # Should be due since next fire time is in the past
        assert entry.is_due() is True

    def test_first_cron_run_waits_for_next_occurrence(self):
        """Test that a new cron entry waits for the next occurrence, not now.

        A cron entry with no last_run should calculate its next fire time
        from now, meaning it should not be immediately due.
        """
        # Create entry for 8 AM daily - this should calculate next 8 AM
        entry = ScheduleEntry(
            message="Daily task",
            cron="0 8 * * *",
            timezone="UTC",
            last_run=None,
        )

        # Next fire should be in the future (next 8 AM)
        next_fire = entry.next_fire_time()
        assert next_fire is not None
        assert next_fire > datetime.now(UTC)

        # Therefore should NOT be due yet
        assert entry.is_due() is False

    def test_weekly_cron_fires_at_scheduled_time(self):
        """Test Monday 8am cron calculates next fire correctly with timezone.

        Verifies that a weekly cron expression like '0 8 * * 1' (Monday 8am)
        correctly calculates the next fire time in the specified timezone.
        """
        from zoneinfo import ZoneInfo

        la_tz = ZoneInfo("America/Los_Angeles")

        # Monday 8am LA time - last ran a week ago
        # Set last_run to a known Monday 8am LA time
        last_monday_8am_la = datetime(2026, 2, 2, 8, 0, 0, tzinfo=la_tz)  # Mon Feb 2
        last_run_utc = last_monday_8am_la.astimezone(UTC)

        entry = ScheduleEntry(
            message="Weekly standup",
            cron="0 8 * * 1",  # Monday 8am
            timezone="America/Los_Angeles",
            last_run=last_run_utc,
        )

        next_fire = entry.next_fire_time()
        assert next_fire is not None

        # Convert to LA time and verify it's Monday 8am
        next_fire_la = next_fire.astimezone(la_tz)
        assert next_fire_la.weekday() == 0  # Monday
        assert next_fire_la.hour == 8
        assert next_fire_la.minute == 0

        # Should be exactly one week after last_run
        expected_next = datetime(2026, 2, 9, 8, 0, 0, tzinfo=la_tz)  # Mon Feb 9
        assert next_fire_la == expected_next

    def test_weekly_cron_is_due_after_scheduled_time(self):
        """Test Monday 8am cron is due when scheduled time has passed.

        Sets last_run to previous week and verifies is_due() returns True
        when the next scheduled Monday 8am has passed.
        """
        from zoneinfo import ZoneInfo

        la_tz = ZoneInfo("America/Los_Angeles")

        # Last ran Monday Feb 2 at 8am LA
        last_monday_8am_la = datetime(2026, 2, 2, 8, 0, 0, tzinfo=la_tz)
        last_run_utc = last_monday_8am_la.astimezone(UTC)

        # Next fire would be Monday Feb 9 at 8am LA
        # If current time is after that, it should be due

        entry = ScheduleEntry(
            message="Weekly standup",
            cron="0 8 * * 1",  # Monday 8am
            timezone="America/Los_Angeles",
            last_run=last_run_utc,
        )

        next_fire = entry.next_fire_time()
        assert next_fire is not None

        # Verify next fire is Feb 9 8am LA
        next_fire_la = next_fire.astimezone(la_tz)
        assert next_fire_la.day == 9
        assert next_fire_la.month == 2

        # If now is after Feb 9 8am LA (Feb 9 4pm UTC for PST), should be due
        # Since today is Feb 9 2026, check if we're past 8am LA
        now = datetime.now(UTC)
        if now > next_fire:
            assert entry.is_due() is True
        else:
            # If we're before the scheduled time, it should not be due
            assert entry.is_due() is False

    def test_cron_with_created_at_fires_on_first_occurrence(self):
        """Test cron entry with created_at (no last_run) fires at first scheduled time.

        This tests the fix for: scheduler tasks not firing at scheduled time.

        Bug scenario:
        - Task created at 6:50 PM Monday for `0 19 * * 1` (7pm Monday)
        - At 7:00:13 PM, without the fix, get_next(7:00:13 PM) → 7pm NEXT Monday
        - With the fix, get_next(created_at at 6:50pm) → 7pm current Monday

        The fix uses created_at as the base time for entries without last_run,
        ensuring the first scheduled occurrence after creation is captured.
        """
        from zoneinfo import ZoneInfo

        la_tz = ZoneInfo("America/Los_Angeles")

        # Simulate: task created at 6:50 PM for 7 PM schedule
        # created_at is 10 minutes before scheduled time
        created_time = datetime(2026, 2, 9, 18, 50, 0, tzinfo=la_tz)  # 6:50 PM LA

        entry = ScheduleEntry(
            id="weekly01",
            message="Weekly reminder",
            cron="0 19 * * 1",  # 7 PM Monday
            timezone="America/Los_Angeles",
            created_at=created_time.astimezone(UTC),
            last_run=None,  # First execution - no last_run
        )

        # With the fix, next_fire should be 7 PM on the same day (Feb 9)
        # because we use created_at (6:50 PM) as base, and get_next gives 7 PM
        next_fire = entry.next_fire_time()
        assert next_fire is not None

        next_fire_la = next_fire.astimezone(la_tz)
        # Should be 7 PM on Feb 9 (Monday), not Feb 16 (next Monday)
        assert next_fire_la.month == 2
        assert next_fire_la.day == 9
        assert next_fire_la.hour == 19
        assert next_fire_la.minute == 0

    def test_cron_without_created_at_or_last_run_uses_now(self):
        """Test cron entry without created_at or last_run uses datetime.now() as fallback.

        This ensures backward compatibility for entries that don't have created_at.
        """
        entry = ScheduleEntry(
            message="Legacy entry",
            cron="0 8 * * *",  # 8 AM daily
            timezone="UTC",
            created_at=None,
            last_run=None,
        )

        # Should calculate from now, so next fire is in the future
        next_fire = entry.next_fire_time()
        assert next_fire is not None
        assert next_fire > datetime.now(UTC)
        assert entry.is_due() is False

    def test_cron_past_scheduled_time_with_created_at_is_due(self):
        """Test cron entry is due when current time is past scheduled time.

        Scenario: created well in the past, scheduled for every minute.
        Should be due because the next scheduled time has passed.
        """
        # Created 2 days ago, so any "every minute" schedule should be past
        created_time = datetime.now(UTC) - timedelta(days=2)

        entry = ScheduleEntry(
            id="weekly02",
            message="Frequent reminder",
            cron="* * * * *",  # Every minute
            timezone="UTC",
            created_at=created_time,
            last_run=None,
        )

        # Next fire from 2 days ago + 1 minute is definitely in the past
        next_fire = entry.next_fire_time()
        assert next_fire is not None
        assert next_fire < datetime.now(UTC)

        # Should be due since next fire time is in the past
        assert entry.is_due() is True


class TestPreviousFireTime:
    """Tests for ScheduleEntry.previous_fire_time()."""

    def test_one_shot_returns_trigger_at(self):
        """One-shot entries return their trigger_at as previous fire time."""
        trigger = datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)
        entry = ScheduleEntry(
            message="Test",
            trigger_at=trigger,
        )
        assert entry.previous_fire_time() == trigger

    def test_cron_returns_most_recent_occurrence(self):
        """Cron entries return the most recent scheduled time before now."""
        entry = ScheduleEntry(
            message="Daily task",
            cron="0 8 * * *",  # 8 AM daily
            timezone="UTC",
        )
        prev = entry.previous_fire_time()
        assert prev is not None
        # Previous fire time should be in the past
        assert prev < datetime.now(UTC)
        # Should be at 8:00 AM UTC
        assert prev.hour == 8
        assert prev.minute == 0

    def test_cron_respects_timezone(self):
        """Previous fire time is calculated in entry's timezone."""
        from zoneinfo import ZoneInfo

        la_tz = ZoneInfo("America/Los_Angeles")

        entry = ScheduleEntry(
            message="LA task",
            cron="0 8 * * *",  # 8 AM LA time
            timezone="America/Los_Angeles",
        )
        prev = entry.previous_fire_time()
        assert prev is not None

        # Convert to LA time - should be 8:00 AM local
        prev_la = prev.astimezone(la_tz)
        assert prev_la.hour == 8
        assert prev_la.minute == 0

    def test_no_trigger_returns_none(self):
        """Entry with neither trigger_at nor cron returns None."""
        entry = ScheduleEntry(message="Test")
        # Manually set both to None to test edge case
        entry.trigger_at = None
        entry.cron = None
        assert entry.previous_fire_time() is None

    def test_fallback_timezone_used(self):
        """Fallback timezone is used when entry has no stored timezone."""
        entry = ScheduleEntry(
            message="Test",
            cron="0 8 * * *",
            timezone=None,  # No stored timezone
        )
        # Pass fallback timezone
        prev_utc = entry.previous_fire_time("UTC")
        prev_la = entry.previous_fire_time("America/Los_Angeles")

        assert prev_utc is not None
        assert prev_la is not None
        # Different fallback timezones should give different UTC times
        # (8 AM UTC vs 8 AM LA = different UTC instants)
        assert prev_utc != prev_la


class TestFormatDelay:
    """Tests for format_delay() helper function."""

    def test_just_now(self):
        """Delays under 1 minute show as 'just now'."""
        from ash.scheduling import format_delay

        assert format_delay(0) == "just now"
        assert format_delay(30) == "just now"
        assert format_delay(59) == "just now"

    def test_minutes(self):
        """Delays between 1-60 minutes show as '~N minutes'."""
        from ash.scheduling import format_delay

        assert format_delay(60) == "~1 minutes"
        assert format_delay(120) == "~2 minutes"
        assert format_delay(5 * 60) == "~5 minutes"
        assert format_delay(59 * 60) == "~59 minutes"

    def test_hours(self):
        """Delays between 1-24 hours show as '~N.N hours'."""
        from ash.scheduling import format_delay

        assert format_delay(60 * 60) == "~1.0 hours"
        assert format_delay(90 * 60) == "~1.5 hours"
        assert format_delay(2 * 60 * 60) == "~2.0 hours"
        assert format_delay(23 * 60 * 60) == "~23.0 hours"

    def test_days(self):
        """Delays over 24 hours show as '~N.N days'."""
        from ash.scheduling import format_delay

        assert format_delay(24 * 60 * 60) == "~1.0 days"
        assert format_delay(36 * 60 * 60) == "~1.5 days"
        assert format_delay(48 * 60 * 60) == "~2.0 days"


class TestScheduledTaskWrapper:
    """Tests for scheduled task wrapper formatting."""

    @pytest.mark.asyncio
    async def test_wrapper_contains_timing_context(self):
        """Handler wraps message with timing context."""
        from unittest.mock import AsyncMock, MagicMock

        from ash.scheduling import ScheduledTaskHandler

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_agent.process_message = AsyncMock(return_value=mock_response)

        mock_sender = AsyncMock(return_value="msg_123")
        handler = ScheduledTaskHandler(
            agent=mock_agent,
            senders={"telegram": mock_sender},
            timezone="UTC",
        )

        entry = ScheduleEntry(
            id="abc12345",
            message="Check the weather",
            trigger_at=datetime.now(UTC) - timedelta(minutes=5),
            provider="telegram",
            chat_id="123",
            username="alice",
        )

        await handler.handle(entry)

        # Get the message passed to agent
        call_args = mock_agent.process_message.call_args
        wrapped_message = call_args[0][0]

        # Verify wrapper structure
        assert "<context>" in wrapped_message
        assert "Entry ID: abc12345" in wrapped_message
        assert "Scheduled by: @alice" in wrapped_message
        assert "</context>" in wrapped_message

        assert "<timing>" in wrapped_message
        assert "Current time:" in wrapped_message
        assert "Scheduled fire time:" in wrapped_message
        assert "Delay:" in wrapped_message
        assert "</timing>" in wrapped_message

        assert "<decision-guidance>" in wrapped_message
        assert "TIME-SENSITIVE" in wrapped_message
        assert "TIME-INDEPENDENT" in wrapped_message
        assert "</decision-guidance>" in wrapped_message

        assert "<task>" in wrapped_message
        assert "Check the weather" in wrapped_message
        assert "</task>" in wrapped_message

    @pytest.mark.asyncio
    async def test_wrapper_shows_cron_schedule(self):
        """Wrapper shows cron expression for periodic entries."""
        from unittest.mock import AsyncMock, MagicMock

        from ash.scheduling import ScheduledTaskHandler

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_agent.process_message = AsyncMock(return_value=mock_response)

        mock_sender = AsyncMock(return_value="msg_123")
        handler = ScheduledTaskHandler(
            agent=mock_agent,
            senders={"telegram": mock_sender},
            timezone="UTC",
        )

        entry = ScheduleEntry(
            id="cron1234",
            message="Daily report",
            cron="0 8 * * *",
            provider="telegram",
            chat_id="123",
        )

        await handler.handle(entry)

        call_args = mock_agent.process_message.call_args
        wrapped_message = call_args[0][0]

        assert "Schedule: 0 8 * * * (recurring)" in wrapped_message

    @pytest.mark.asyncio
    async def test_wrapper_shows_one_shot_trigger(self):
        """Wrapper shows trigger time for one-shot entries."""
        from unittest.mock import AsyncMock, MagicMock

        from ash.scheduling import ScheduledTaskHandler

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_agent.process_message = AsyncMock(return_value=mock_response)

        mock_sender = AsyncMock(return_value="msg_123")
        handler = ScheduledTaskHandler(
            agent=mock_agent,
            senders={"telegram": mock_sender},
            timezone="UTC",
        )

        trigger_time = datetime(2026, 2, 9, 14, 30, 0, tzinfo=UTC)
        entry = ScheduleEntry(
            id="oneshot1",
            message="Reminder",
            trigger_at=trigger_time,
            provider="telegram",
            chat_id="123",
        )

        await handler.handle(entry)

        call_args = mock_agent.process_message.call_args
        wrapped_message = call_args[0][0]

        assert "Trigger:" in wrapped_message
        assert "(one-shot)" in wrapped_message
        assert "2026-02-09 14:30" in wrapped_message

    @pytest.mark.asyncio
    async def test_handler_uses_configured_timezone(self):
        """Handler formats times in its configured timezone."""
        from unittest.mock import AsyncMock, MagicMock

        from ash.scheduling import ScheduledTaskHandler

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_agent.process_message = AsyncMock(return_value=mock_response)

        mock_sender = AsyncMock(return_value="msg_123")
        handler = ScheduledTaskHandler(
            agent=mock_agent,
            senders={"telegram": mock_sender},
            timezone="America/Los_Angeles",  # LA timezone
        )

        # Entry in LA timezone too
        entry = ScheduleEntry(
            id="tz_test",
            message="Test",
            cron="0 8 * * *",
            timezone="America/Los_Angeles",
            provider="telegram",
            chat_id="123",
        )

        await handler.handle(entry)

        call_args = mock_agent.process_message.call_args
        wrapped_message = call_args[0][0]

        # Times should be formatted in LA timezone
        # The previous 8 AM LA should show as 08:00, not UTC equivalent
        assert "08:00" in wrapped_message


class TestStalenessGuard:
    """Tests for the staleness guard in ScheduleWatcher."""

    @pytest.mark.asyncio
    async def test_stale_periodic_task_skipped_silently(self, tmp_path: Path):
        """Periodic task overdue by >2h should be skipped without invoking handler."""
        schedule_file = tmp_path / "schedule.jsonl"
        # Entry for every minute, last ran 3 hours ago — will be very stale
        old_time = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
        schedule_file.write_text(
            f'{{"id": "stale1", "cron": "* * * * *", "message": "Stale task", "last_run": "{old_time}"}}\n'
        )

        store = ScheduleStore(schedule_file)
        watcher = ScheduleWatcher(store)
        triggered: list[ScheduleEntry] = []

        @watcher.on_due
        async def handler(entry: ScheduleEntry):
            triggered.append(entry)

        await watcher._check_schedule()

        # Handler should NOT have been called
        assert len(triggered) == 0
        # Entry should still exist with updated last_run
        content = schedule_file.read_text()
        assert "Stale task" in content
        assert old_time not in content  # last_run advanced

    @pytest.mark.asyncio
    async def test_fresh_periodic_task_fires_normally(self, tmp_path: Path):
        """Periodic task overdue by <2h should fire normally."""
        schedule_file = tmp_path / "schedule.jsonl"
        # Entry for every minute, last ran 30 minutes ago — fresh enough
        old_time = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
        schedule_file.write_text(
            f'{{"id": "fresh1", "cron": "* * * * *", "message": "Fresh task", "last_run": "{old_time}"}}\n'
        )

        store = ScheduleStore(schedule_file)
        watcher = ScheduleWatcher(store)
        triggered: list[ScheduleEntry] = []

        @watcher.on_due
        async def handler(entry: ScheduleEntry):
            triggered.append(entry)

        await watcher._check_schedule()

        # Handler SHOULD have been called
        assert len(triggered) == 1
        assert triggered[0].message == "Fresh task"

    @pytest.mark.asyncio
    async def test_stale_one_shot_still_fires(self, tmp_path: Path):
        """One-shot tasks always fire regardless of staleness."""
        schedule_file = tmp_path / "schedule.jsonl"
        # One-shot task from 5 hours ago
        past = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
        schedule_file.write_text(
            f'{{"id": "oneshot_stale", "trigger_at": "{past}", "message": "Old one-shot"}}\n'
        )

        store = ScheduleStore(schedule_file)
        watcher = ScheduleWatcher(store)
        triggered: list[ScheduleEntry] = []

        @watcher.on_due
        async def handler(entry: ScheduleEntry):
            triggered.append(entry)

        await watcher._check_schedule()

        # One-shot entries always fire
        assert len(triggered) == 1
        assert triggered[0].message == "Old one-shot"
        # And get removed
        assert "Old one-shot" not in schedule_file.read_text()


class TestHandlerNoReply:
    """Tests for [NO_REPLY] suppression in ScheduledTaskHandler."""

    @pytest.mark.asyncio
    async def test_no_reply_suppresses_message(self):
        """Handler should not send message when agent responds with [NO_REPLY]."""
        from unittest.mock import AsyncMock, MagicMock

        from ash.scheduling import ScheduledTaskHandler

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "[NO_REPLY]"
        mock_agent.process_message = AsyncMock(return_value=mock_response)

        mock_sender = AsyncMock(return_value="msg_123")
        handler = ScheduledTaskHandler(
            agent=mock_agent, senders={"telegram": mock_sender}
        )

        entry = ScheduleEntry(
            message="Good morning!",
            trigger_at=datetime.now(UTC) - timedelta(hours=5),
            provider="telegram",
            chat_id="123",
            user_id="456",
        )

        await handler.handle(entry)

        # Agent was called
        mock_agent.process_message.assert_called_once()
        # But sender was NOT called
        mock_sender.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_response_still_sent(self):
        """Handler should still send normal (non-NO_REPLY) responses."""
        from unittest.mock import AsyncMock, MagicMock

        from ash.scheduling import ScheduledTaskHandler

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Good morning! Here's your weather..."
        mock_agent.process_message = AsyncMock(return_value=mock_response)

        mock_sender = AsyncMock(return_value="msg_123")
        handler = ScheduledTaskHandler(
            agent=mock_agent, senders={"telegram": mock_sender}
        )

        entry = ScheduleEntry(
            message="Good morning!",
            trigger_at=datetime.now(UTC) - timedelta(minutes=2),
            provider="telegram",
            chat_id="123",
            user_id="456",
        )

        await handler.handle(entry)

        # Both agent and sender should be called
        mock_agent.process_message.assert_called_once()
        mock_sender.assert_called_once()
