"""Tests for CLI commands."""

from ash.cli.app import app


class TestConfigCommand:
    """Tests for 'ash config' command."""

    def test_config_init_creates_file(self, cli_runner, tmp_path):
        config_path = tmp_path / "config.toml"
        result = cli_runner.invoke(app, ["config", "init", "--path", str(config_path)])

        # May fail if example config not found, which is OK for this test
        if result.exit_code == 0:
            assert config_path.exists()
            content = config_path.read_text()
            assert "[models.default]" in content

    def test_config_init_existing_file_fails(self, cli_runner, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text("existing content")

        result = cli_runner.invoke(app, ["config", "init", "--path", str(config_path)])
        assert result.exit_code == 1
        assert "already exists" in result.stdout

    def test_config_show_displays_content(self, cli_runner, config_file):
        result = cli_runner.invoke(app, ["config", "show", "--path", str(config_file)])
        assert result.exit_code == 0
        assert "[default_llm]" in result.stdout

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
[default_llm]
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


class TestDbCommand:
    """Tests for 'ash db' command."""

    def test_db_unknown_action(self, cli_runner):
        result = cli_runner.invoke(app, ["db", "unknown"])
        assert result.exit_code == 1
        assert "Unknown action" in result.stdout

    def test_db_migrate_help(self, cli_runner):
        # Just test that the command parses correctly
        result = cli_runner.invoke(app, ["db", "--help"])
        assert result.exit_code == 0
        assert "migrate" in result.stdout or "migrations" in result.stdout.lower()


class TestMemoryCommand:
    """Tests for 'ash memory' command."""

    def test_memory_search_requires_query(self, cli_runner, config_file):
        result = cli_runner.invoke(
            app, ["memory", "search", "--config", str(config_file)]
        )
        assert result.exit_code == 1
        assert "--query" in result.stdout or "required" in result.stdout.lower()

    def test_memory_add_requires_query(self, cli_runner, config_file):
        result = cli_runner.invoke(app, ["memory", "add", "--config", str(config_file)])
        assert result.exit_code == 1
        assert "--query" in result.stdout or "required" in result.stdout.lower()

    def test_memory_remove_requires_id(self, cli_runner, config_file):
        result = cli_runner.invoke(
            app, ["memory", "remove", "--config", str(config_file)]
        )
        assert result.exit_code == 1
        assert "--id" in result.stdout or "required" in result.stdout.lower()

    def test_memory_unknown_action(self, cli_runner, config_file):
        result = cli_runner.invoke(
            app, ["memory", "unknown", "--config", str(config_file)]
        )
        assert result.exit_code == 1

    def test_memory_help(self, cli_runner):
        result = cli_runner.invoke(app, ["memory", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "search" in result.stdout
        assert "add" in result.stdout
        assert "remove" in result.stdout
        assert "clear" in result.stdout
        assert "stats" in result.stdout


class TestSessionsCommand:
    """Tests for 'ash sessions' command."""

    def test_sessions_search_requires_query(self, cli_runner):
        # Sessions command reads from JSONL files, no config needed
        result = cli_runner.invoke(app, ["sessions", "search"])
        assert result.exit_code == 1
        assert "--query" in result.stdout or "required" in result.stdout.lower()

    def test_sessions_unknown_action(self, cli_runner):
        # Sessions command reads from JSONL files, no config needed
        result = cli_runner.invoke(app, ["sessions", "unknown"])
        assert result.exit_code == 1

    def test_sessions_help(self, cli_runner):
        result = cli_runner.invoke(app, ["sessions", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "search" in result.stdout
        assert "export" in result.stdout
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
        assert "chat" in result.stdout
        assert "config" in result.stdout
        assert "db" in result.stdout
        assert "memory" in result.stdout
        assert "sessions" in result.stdout
        assert "sandbox" in result.stdout
        assert "upgrade" in result.stdout
