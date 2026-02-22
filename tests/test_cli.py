"""Tests for CLI commands."""

from ash.cli.app import app
from ash.config.paths import ENV_VAR, get_ash_home


class TestConfigCommand:
    """Tests for 'ash config' command."""

    def test_config_show_displays_content(self, cli_runner, config_file):
        result = cli_runner.invoke(app, ["config", "show", "--path", str(config_file)])
        assert result.exit_code == 0
        assert "[models.default]" in result.stdout

    def test_config_show_missing_file(self, cli_runner, tmp_path):
        result = cli_runner.invoke(
            app, ["config", "show", "--path", str(tmp_path / "missing.toml")]
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_config_validate_success(self, cli_runner, config_file):
        result = cli_runner.invoke(
            app, ["config", "validate", "--path", str(config_file)]
        )
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_config_validate_invalid_toml(self, cli_runner, tmp_path):
        invalid_file = tmp_path / "invalid.toml"
        invalid_file.write_text("not valid toml [[[")

        result = cli_runner.invoke(
            app, ["config", "validate", "--path", str(invalid_file)]
        )
        assert result.exit_code == 1

    def test_config_validate_invalid_config(self, cli_runner, tmp_path):
        invalid_config = tmp_path / "bad_config.toml"
        invalid_config.write_text("""
[models.default]
provider = "invalid_provider"
model = "test"
""")
        result = cli_runner.invoke(
            app, ["config", "validate", "--path", str(invalid_config)]
        )
        assert result.exit_code == 1
        assert (
            "validation failed" in result.stdout.lower()
            or "error" in result.stdout.lower()
        )

    def test_config_unknown_action(self, cli_runner):
        result = cli_runner.invoke(app, ["config", "unknown"])
        assert result.exit_code == 1
        assert "Unknown action" in result.stdout


class TestMemoryCommand:
    """Tests for 'ash memory' command."""

    def test_memory_add_requires_query(self, cli_runner, config_file):
        result = cli_runner.invoke(app, ["memory", "add", "--config", str(config_file)])
        assert result.exit_code == 1
        assert "--query" in result.stdout or "required" in result.stdout.lower()

    def test_memory_remove_requires_id(self, cli_runner, config_file):
        result = cli_runner.invoke(
            app, ["memory", "remove", "--config", str(config_file)]
        )
        assert result.exit_code == 1
        assert "--all" in result.stdout or "required" in result.stdout.lower()

    def test_memory_unknown_action(self, cli_runner, config_file):
        result = cli_runner.invoke(
            app, ["memory", "unknown", "--config", str(config_file)]
        )
        assert result.exit_code == 1

    def test_memory_search_requires_query(self, cli_runner, config_file):
        result = cli_runner.invoke(
            app, ["memory", "search", "--config", str(config_file)]
        )
        assert result.exit_code == 1
        assert "search" in result.stdout.lower() or "query" in result.stdout.lower()

    def test_memory_help(self, cli_runner):
        result = cli_runner.invoke(app, ["memory", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "search" in result.stdout
        assert "stats" in result.stdout
        assert "add" in result.stdout
        assert "remove" in result.stdout
        assert "clear" in result.stdout

    def test_memory_doctor_defaults_to_all(self, cli_runner, config_file, monkeypatch):
        calls: list[tuple[str, bool]] = []

        class DummyStore:
            pass

        async def _fake_get_store(_config):
            return DummyStore()

        def _mk(name: str):
            async def _f(*args, **kwargs):
                calls.append((name, kwargs["force"]))

            return _f

        monkeypatch.setattr(
            "ash.cli.commands.memory._helpers.get_store", _fake_get_store
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_prune_missing_provenance",
            _mk("prune-missing-provenance"),
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_self_facts", _mk("self-facts")
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_backfill_subjects",
            _mk("backfill-subjects"),
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_attribution",
            _mk("attribution"),
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_fix_names", _mk("fix-names")
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_normalize_semantics",
            _mk("normalize-semantics"),
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_reclassify",
            _mk("reclassify"),
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_quality", _mk("quality")
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_dedup", _mk("dedup")
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_contradictions",
            _mk("contradictions"),
        )

        result = cli_runner.invoke(
            app, ["memory", "doctor", "--config", str(config_file)]
        )
        assert result.exit_code == 0
        assert [name for name, _force in calls] == [
            "prune-missing-provenance",
            "self-facts",
            "backfill-subjects",
            "attribution",
            "fix-names",
            "normalize-semantics",
            "reclassify",
            "quality",
            "dedup",
            "contradictions",
        ]
        assert all(force is False for _name, force in calls)

    def test_memory_doctor_subcommand_runs_interactive_without_force(
        self, cli_runner, config_file, monkeypatch
    ):
        calls: list[bool] = []

        class DummyStore:
            pass

        async def _fake_get_store(_config):
            return DummyStore()

        async def _fake_embed_missing(*args, **kwargs):
            calls.append(kwargs["force"])

        monkeypatch.setattr(
            "ash.cli.commands.memory._helpers.get_store", _fake_get_store
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_embed_missing",
            _fake_embed_missing,
        )
        result = cli_runner.invoke(
            app, ["memory", "doctor", "embed-missing", "--config", str(config_file)]
        )
        assert result.exit_code == 0
        assert calls == [False]

    def test_memory_doctor_all_runs_force_mode(
        self, cli_runner, config_file, monkeypatch
    ):
        calls: list[tuple[str, bool]] = []

        class DummyStore:
            pass

        async def _fake_get_store(_config):
            return DummyStore()

        def _mk(name: str):
            async def _f(*args, **kwargs):
                calls.append((name, kwargs["force"]))

            return _f

        monkeypatch.setattr(
            "ash.cli.commands.memory._helpers.get_store", _fake_get_store
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_prune_missing_provenance",
            _mk("prune-missing-provenance"),
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_self_facts", _mk("self-facts")
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_backfill_subjects",
            _mk("backfill-subjects"),
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_attribution",
            _mk("attribution"),
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_fix_names", _mk("fix-names")
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_normalize_semantics",
            _mk("normalize-semantics"),
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_reclassify",
            _mk("reclassify"),
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_quality", _mk("quality")
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_dedup", _mk("dedup")
        )
        monkeypatch.setattr(
            "ash.cli.commands.memory.doctor.memory_doctor_contradictions",
            _mk("contradictions"),
        )

        result = cli_runner.invoke(
            app, ["memory", "doctor", "all", "--force", "--config", str(config_file)]
        )
        assert result.exit_code == 0
        assert len(calls) == 10
        assert all(force is True for _name, force in calls)


class TestSessionsCommand:
    """Tests for 'ash sessions' command."""

    def test_sessions_search_requires_query(self, cli_runner):
        # Sessions command reads from JSONL files, no config needed
        result = cli_runner.invoke(app, ["sessions", "search"])
        assert result.exit_code == 1
        assert "query" in result.stdout.lower() or "required" in result.stdout.lower()

    def test_sessions_unknown_session(self, cli_runner):
        # Unknown session key should give "No session found" error
        result = cli_runner.invoke(app, ["sessions", "nonexistent_session_xyz"])
        assert result.exit_code == 0  # exits normally, just prints error
        assert "no session found" in result.stdout.lower()

    def test_sessions_help(self, cli_runner):
        result = cli_runner.invoke(app, ["sessions", "--help"])
        assert result.exit_code == 0
        # Check for key features in the help/examples
        assert "events" in result.stdout
        assert "tools" in result.stdout
        assert "search" in result.stdout
        assert "clear" in result.stdout


class TestServeCommand:
    """Tests for 'ash serve' command."""

    def test_serve_help(self, cli_runner):
        result = cli_runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.stdout or "-c" in result.stdout
        assert "--host" in result.stdout or "-h" in result.stdout
        assert "--port" in result.stdout or "-p" in result.stdout


class TestChatCommand:
    """Tests for 'ash chat' command."""

    def test_chat_help(self, cli_runner):
        result = cli_runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.stdout or "-c" in result.stdout
        assert "--streaming" in result.stdout


class TestUpgradeCommand:
    """Tests for 'ash upgrade' command."""

    def test_upgrade_help(self, cli_runner):
        result = cli_runner.invoke(app, ["upgrade", "--help"])
        assert result.exit_code == 0
        assert (
            "migration" in result.stdout.lower() or "upgrade" in result.stdout.lower()
        )


class TestSandboxCommand:
    """Tests for 'ash sandbox' command."""

    def test_sandbox_help(self, cli_runner):
        result = cli_runner.invoke(app, ["sandbox", "--help"])
        assert result.exit_code == 0
        assert "build" in result.stdout
        assert "status" in result.stdout
        assert "clean" in result.stdout

    def test_sandbox_status(self, cli_runner):
        # Status should always work, even without Docker
        result = cli_runner.invoke(app, ["sandbox", "status"])
        assert result.exit_code == 0
        assert "Docker" in result.stdout
        assert "Sandbox Image" in result.stdout

    def test_sandbox_unknown_action(self, cli_runner):
        result = cli_runner.invoke(app, ["sandbox", "unknown"])
        assert result.exit_code == 1
        assert "Unknown action" in result.stdout


class TestScheduleCommand:
    """Tests for 'ash schedule' command."""

    def test_schedule_help(self, cli_runner):
        result = cli_runner.invoke(app, ["schedule", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "cancel" in result.stdout
        assert "clear" in result.stdout

    def test_schedule_unknown_action(self, cli_runner, monkeypatch, tmp_path):
        # Mock config loading
        from ash.config import models

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setattr(
            "ash.config.load_config",
            lambda: models.AshConfig(
                models={
                    "default": models.ModelConfig(provider="openai", model="gpt-5.2")
                },
                workspace=workspace,
            ),
        )

        result = cli_runner.invoke(app, ["schedule", "unknown"])
        assert result.exit_code == 1
        assert "Unknown action" in result.stdout

    def test_schedule_cancel_requires_id(self, cli_runner, monkeypatch, tmp_path):
        from ash.config import models

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setattr(
            "ash.config.load_config",
            lambda: models.AshConfig(
                models={
                    "default": models.ModelConfig(provider="openai", model="gpt-5.2")
                },
                workspace=workspace,
            ),
        )

        result = cli_runner.invoke(app, ["schedule", "cancel"])
        assert result.exit_code == 1
        assert "--id" in result.stdout or "required" in result.stdout.lower()

    def test_schedule_list_empty(self, cli_runner, monkeypatch, tmp_path):
        from ash.config import models

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setattr(
            "ash.config.load_config",
            lambda: models.AshConfig(
                models={
                    "default": models.ModelConfig(provider="openai", model="gpt-5.2")
                },
                workspace=workspace,
            ),
        )

        result = cli_runner.invoke(app, ["schedule", "list"])
        assert result.exit_code == 0
        assert "No scheduled tasks" in result.stdout

    def test_schedule_list_with_entries(self, cli_runner, monkeypatch, tmp_path):
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"trigger_at": "2026-01-12T09:00:00+00:00", "message": "Test task"}\n'
        )
        monkeypatch.setattr(
            "ash.config.paths.get_schedule_file",
            lambda: schedule_file,
        )

        result = cli_runner.invoke(app, ["schedule", "list"])
        assert result.exit_code == 0
        assert "Test task" in result.stdout

    def test_schedule_cancel_success(self, cli_runner, monkeypatch, tmp_path):
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"id": "abc12345", "trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task to cancel"}\n'
        )
        monkeypatch.setattr(
            "ash.config.paths.get_schedule_file",
            lambda: schedule_file,
        )

        result = cli_runner.invoke(app, ["schedule", "cancel", "--id", "abc12345"])
        assert result.exit_code == 0
        assert "Cancelled" in result.stdout

        # Verify file is empty
        assert schedule_file.read_text().strip() == ""

    def test_schedule_clear_with_force(self, cli_runner, monkeypatch, tmp_path):
        schedule_file = tmp_path / "schedule.jsonl"
        schedule_file.write_text(
            '{"trigger_at": "2026-01-12T09:00:00+00:00", "message": "Task 1"}\n'
            '{"trigger_at": "2026-01-13T09:00:00+00:00", "message": "Task 2"}\n'
        )
        monkeypatch.setattr(
            "ash.config.paths.get_schedule_file",
            lambda: schedule_file,
        )

        result = cli_runner.invoke(app, ["schedule", "clear", "--force"])
        assert result.exit_code == 0
        assert "Cleared 2" in result.stdout
        assert schedule_file.read_text() == ""


class TestPeopleCommand:
    """Tests for `ash people`."""

    def test_people_doctor_defaults_to_all(self, cli_runner, config_file, monkeypatch):
        calls: list[tuple[bool, str]] = []

        async def _fake_people_doctor(config, force: bool, subcommand: str = "all"):
            calls.append((force, subcommand))

        monkeypatch.setattr(
            "ash.cli.commands.people._people_doctor", _fake_people_doctor
        )
        result = cli_runner.invoke(
            app, ["people", "doctor", "--config", str(config_file)]
        )
        assert result.exit_code == 0
        assert calls == [(False, "all")]

    def test_people_doctor_subcommand_runs_interactive_without_force(
        self, cli_runner, config_file, monkeypatch
    ):
        calls: list[tuple[bool, str]] = []

        async def _fake_people_doctor(config, force: bool, subcommand: str = "all"):
            calls.append((force, subcommand))

        monkeypatch.setattr(
            "ash.cli.commands.people._people_doctor", _fake_people_doctor
        )
        result = cli_runner.invoke(
            app,
            ["people", "doctor", "duplicates", "--config", str(config_file)],
        )
        assert result.exit_code == 0
        assert calls == [(False, "duplicates")]

    def test_people_doctor_subcommand_runs_force_mode(
        self, cli_runner, config_file, monkeypatch
    ):
        calls: list[tuple[bool, str]] = []

        async def _fake_people_doctor(config, force: bool, subcommand: str = "all"):
            calls.append((force, subcommand))

        monkeypatch.setattr(
            "ash.cli.commands.people._people_doctor", _fake_people_doctor
        )
        result = cli_runner.invoke(
            app,
            ["people", "doctor", "all", "--force", "--config", str(config_file)],
        )
        assert result.exit_code == 0
        assert calls == [(True, "all")]

    def test_people_doctor_unknown_subcommand(self, cli_runner, config_file):
        result = cli_runner.invoke(
            app,
            ["people", "doctor", "unknown-check", "--config", str(config_file)],
        )
        assert result.exit_code == 1
        assert "Unknown people doctor subcommand" in result.stdout


class TestAppHelp:
    """Tests for main app help."""

    def test_app_no_args_shows_help(self, cli_runner):
        result = cli_runner.invoke(app, [])
        # Exit code 0 or 2 is acceptable (2 is for help display in some Typer versions)
        assert result.exit_code in (0, 2)
        assert "ash" in result.stdout.lower()

    def test_app_help_flag(self, cli_runner):
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "serve" in result.stdout
        assert "doctor" in result.stdout
        assert "chat" in result.stdout
        assert "config" in result.stdout
        assert "memory" in result.stdout
        assert "schedule" in result.stdout
        assert "sessions" in result.stdout
        assert "sandbox" in result.stdout
        assert "stats" in result.stdout
        assert "upgrade" in result.stdout


class TestDoctorCommand:
    """Tests for `ash doctor`."""

    def test_doctor_reports_ok_on_clean_home(self, cli_runner, monkeypatch, tmp_path):
        ash_home = tmp_path / ".ash"
        ash_home.mkdir(parents=True, exist_ok=True)
        for name in ("graph", "sessions", "chats", "logs", "run", "workspace"):
            (ash_home / name).mkdir(exist_ok=True)

        monkeypatch.setenv(ENV_VAR, str(ash_home))
        get_ash_home.cache_clear()
        try:
            result = cli_runner.invoke(app, ["doctor"])
        finally:
            monkeypatch.delenv(ENV_VAR, raising=False)
            get_ash_home.cache_clear()

        assert result.exit_code == 0
        assert "Ash Doctor" in result.stdout
        assert "Summary:" in result.stdout
        assert "Doctor checks passed" in result.stdout
        assert "Doctor Commands" in result.stdout
        assert "ash memory doctor" in result.stdout
        assert "ash people doctor" in result.stdout

    def test_doctor_reports_warnings_for_stale_pid_and_bad_schedule(
        self, cli_runner, monkeypatch, tmp_path
    ):
        ash_home = tmp_path / ".ash"
        run_dir = ash_home / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ash.pid").write_text("99999999\n")
        (ash_home / "schedule.jsonl").write_text("{bad-json}\n")

        monkeypatch.setenv(ENV_VAR, str(ash_home))
        get_ash_home.cache_clear()
        try:
            result = cli_runner.invoke(app, ["doctor"])
        finally:
            monkeypatch.delenv(ENV_VAR, raising=False)
            get_ash_home.cache_clear()

        assert result.exit_code == 0
        assert "stale pid file" in result.stdout
        assert "invalid JSONL lines" in result.stdout
        assert "Doctor found non-blocking issues" in result.stdout

    def test_doctor_warns_when_image_enabled_without_openai_key(
        self, cli_runner, monkeypatch, tmp_path
    ):
        ash_home = tmp_path / ".ash"
        ash_home.mkdir(parents=True, exist_ok=True)
        (ash_home / "config.toml").write_text(
            "\n".join(
                [
                    "[models.default]",
                    "provider='openai'",
                    "model='gpt-5.2'",
                    "",
                    "[image]",
                    "enabled=true",
                    "provider='openai'",
                ]
            )
            + "\n"
        )

        monkeypatch.setenv(ENV_VAR, str(ash_home))
        get_ash_home.cache_clear()
        try:
            result = cli_runner.invoke(app, ["doctor"])
        finally:
            monkeypatch.delenv(ENV_VAR, raising=False)
            get_ash_home.cache_clear()

        assert result.exit_code == 0
        assert "OpenAI API key is" in result.stdout
        assert "OPENAI_API_KEY" in result.stdout


class TestStatsCommand:
    """Tests for `ash stats` and `ash info`."""

    def test_stats_prints_home_and_directory_stats(
        self, cli_runner, monkeypatch, tmp_path
    ):
        ash_home = tmp_path / ".ash"
        (ash_home / "sessions").mkdir(parents=True)
        (ash_home / "sessions" / "history.jsonl").write_text("{}\n")
        (ash_home / "logs").mkdir(parents=True)
        (ash_home / "logs" / "2026-02-21.jsonl").write_text("{}\n")
        (ash_home / "config.toml").write_text("[models.default]\nprovider='openai'\n")

        monkeypatch.setenv(ENV_VAR, str(ash_home))
        get_ash_home.cache_clear()
        try:
            result = cli_runner.invoke(app, ["stats"])
        finally:
            monkeypatch.delenv(ENV_VAR, raising=False)
            get_ash_home.cache_clear()

        assert result.exit_code == 0
        assert str(ash_home) in result.stdout
        assert "Directory Stats" in result.stdout
        assert "sessions" in result.stdout
        assert "Core Files" in result.stdout
        assert "config.toml" in result.stdout

    def test_info_is_alias_for_stats(self, cli_runner, monkeypatch, tmp_path):
        ash_home = tmp_path / ".ash"
        ash_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(ENV_VAR, str(ash_home))
        get_ash_home.cache_clear()
        try:
            result = cli_runner.invoke(app, ["info"])
        finally:
            monkeypatch.delenv(ENV_VAR, raising=False)
            get_ash_home.cache_clear()

        assert result.exit_code == 0
        assert "Ash Home" in result.stdout

    def test_stats_includes_memory_quality_from_logs(
        self, cli_runner, monkeypatch, tmp_path
    ):
        ash_home = tmp_path / ".ash"
        logs = ash_home / "logs"
        logs.mkdir(parents=True)
        (ash_home / "config.toml").write_text("[models.default]\nprovider='openai'\n")
        (logs / "2026-02-22.jsonl").write_text(
            "\n".join(
                [
                    '{"ts":"2026-02-22T10:00:00+00:00","message":"memory_extraction_filter_stats","fact.total_candidates":5,"fact.accepted_count":3,"fact.dropped_low_confidence":1,"fact.dropped_secret":1}',
                    '{"ts":"2026-02-22T10:01:00+00:00","message":"memory_verification_stats","fact.total_candidates":3,"fact.accepted_count":2,"fact.rewritten_count":1,"fact.dropped_ambiguous":1,"fact.dropped_meta_system":0,"fact.dropped_stale_status":0,"fact.dropped_low_utility":0}',
                    '{"ts":"2026-02-22T10:02:00+00:00","message":"image_preprocess_started","image.count":1}',
                    '{"ts":"2026-02-22T10:02:01+00:00","message":"image_preprocess_succeeded","image.count":1,"duration_ms":120}',
                    '{"ts":"2026-02-22T10:03:00+00:00","message":"image_preprocess_skipped","skip_reason":"no_usable_images","image.count":1,"duration_ms":4}',
                    '{"ts":"2026-02-22T10:04:00+00:00","message":"image_preprocess_failed","error.message":"timeout","duration_ms":8000}',
                ]
            )
            + "\n"
        )

        monkeypatch.setenv(ENV_VAR, str(ash_home))
        get_ash_home.cache_clear()
        try:
            result = cli_runner.invoke(app, ["stats"])
        finally:
            monkeypatch.delenv(ENV_VAR, raising=False)
            get_ash_home.cache_clear()

        assert result.exit_code == 0
        assert "Memory Quality (from logs)" in result.stdout
        assert "Extraction runs" in result.stdout
        assert "Verification runs" in result.stdout
        assert "Verification rewritten" in result.stdout
        assert "Vision (from logs)" in result.stdout
        assert "Runs started" in result.stdout
        assert "Runs succeeded" in result.stdout
        assert "Runs failed" in result.stdout
        assert "Runs skipped" in result.stdout
        assert "Avg success latency" in result.stdout
        assert "Skipped (no usable images)" in result.stdout
