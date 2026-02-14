"""Tests for path management."""

from pathlib import Path

from ash.config.paths import (
    ENV_VAR,
    get_all_paths,
    get_ash_home,
    get_config_path,
    get_database_path,
    get_logs_path,
    get_workspace_path,
)


class TestGetAshHome:
    """Tests for get_ash_home()."""

    def test_default_is_home_dot_ash(self, monkeypatch):
        # Clear env var and cache
        monkeypatch.delenv(ENV_VAR, raising=False)
        get_ash_home.cache_clear()

        home = get_ash_home()
        assert home == Path.home() / ".ash"

    def test_respects_env_var(self, monkeypatch, tmp_path):
        custom_path = tmp_path / "custom-ash"
        monkeypatch.setenv(ENV_VAR, str(custom_path))
        get_ash_home.cache_clear()

        home = get_ash_home()
        assert home == custom_path

    def test_expands_tilde_in_env_var(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "~/my-ash")
        get_ash_home.cache_clear()

        home = get_ash_home()
        assert home == Path.home() / "my-ash"


class TestDerivedPaths:
    """Tests for derived path functions."""

    def test_config_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        get_ash_home.cache_clear()

        assert get_config_path() == tmp_path / "config.toml"

    def test_database_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        get_ash_home.cache_clear()

        assert get_database_path() == tmp_path / "index" / "vectors.db"

    def test_workspace_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        get_ash_home.cache_clear()

        assert get_workspace_path() == tmp_path / "workspace"

    def test_logs_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        get_ash_home.cache_clear()

        assert get_logs_path() == tmp_path / "logs"


class TestGetAllPaths:
    """Tests for get_all_paths()."""

    def test_returns_all_standard_paths(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        get_ash_home.cache_clear()

        paths = get_all_paths()

        assert "home" in paths
        assert "config" in paths
        assert "database" in paths
        assert "workspace" in paths
        assert "logs" in paths

        assert paths["home"] == tmp_path
        assert paths["config"] == tmp_path / "config.toml"
