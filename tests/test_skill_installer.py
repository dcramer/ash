"""Tests for skill installer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ash.config.models import SkillSource
from ash.config.writer import ConfigWriter
from ash.skills.installer import (
    GitNotFoundError,
    InstalledSource,
    SkillInstaller,
    SkillInstallerError,
    SourceSyncState,
)


class TestSkillSource:
    """Tests for SkillSource model."""

    def test_valid_repo(self):
        """Test valid repo source."""
        source = SkillSource(repo="owner/repo")
        assert source.repo == "owner/repo"
        assert source.path is None
        assert source.ref is None

    def test_valid_repo_with_ref(self):
        """Test valid repo source with ref."""
        source = SkillSource(repo="owner/repo", ref="v1.0")
        assert source.repo == "owner/repo"
        assert source.ref == "v1.0"

    def test_valid_path(self):
        """Test valid path source."""
        source = SkillSource(path="~/my-skills")
        assert source.path == "~/my-skills"
        assert source.repo is None

    def test_invalid_empty(self):
        """Test that empty source is invalid."""
        with pytest.raises(ValueError, match="Must specify either"):
            SkillSource()

    def test_invalid_both(self):
        """Test that both repo and path is invalid."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            SkillSource(repo="owner/repo", path="~/my-skills")

    def test_invalid_ref_with_path(self):
        """Test that ref with path is invalid."""
        with pytest.raises(ValueError, match="'ref' only applies to repo"):
            SkillSource(path="~/my-skills", ref="v1.0")


class TestInstalledSource:
    """Tests for InstalledSource dataclass."""

    def test_source_key_repo(self):
        """Test source key for repo."""
        source = InstalledSource(repo="owner/repo")
        assert source.source_key == "repo:owner/repo"

    def test_source_key_path(self):
        """Test source key for path."""
        source = InstalledSource(path="~/my-skills")
        assert source.source_key == "path:~/my-skills"

    def test_to_dict(self):
        """Test serialization to dict."""
        source = InstalledSource(
            repo="owner/repo",
            ref="v1.0",
            installed_at="2024-01-01T00:00:00",
            updated_at="2024-01-02T00:00:00",
            install_path="/path/to/install",
            commit_sha="abc123",
            skills=["skill1", "skill2"],
        )
        data = source.to_dict()
        assert data["repo"] == "owner/repo"
        assert data["ref"] == "v1.0"
        assert data["skills"] == ["skill1", "skill2"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "repo": "owner/repo",
            "ref": "v1.0",
            "installed_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
            "install_path": "/path/to/install",
            "commit_sha": "abc123",
            "skills": ["skill1", "skill2"],
        }
        source = InstalledSource.from_dict(data)
        assert source.repo == "owner/repo"
        assert source.ref == "v1.0"
        assert source.skills == ["skill1", "skill2"]


class TestSkillInstaller:
    """Tests for SkillInstaller."""

    @pytest.fixture
    def installer(self, tmp_path):
        """Create an installer with a temp install path."""
        return SkillInstaller(install_path=tmp_path / "installed")

    @pytest.fixture
    def skill_dir(self, tmp_path):
        """Create a skill directory with a valid SKILL.md."""
        skill_path = tmp_path / "my-skill"
        skill_path.mkdir()
        skill_file = skill_path / "SKILL.md"
        skill_file.write_text(
            """---
description: A test skill
---

# Test Skill

Instructions here.
"""
        )
        return skill_path

    def test_check_git_missing(self, installer):
        """Test git check when git is not available."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            with pytest.raises(GitNotFoundError, match="git is required"):
                installer._check_git()

    def test_check_git_available(self, installer):
        """Test git check when git is available."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Should not raise
            installer._check_git()

    def test_repo_to_path(self, installer):
        """Test repo name to path conversion."""
        path = installer._repo_to_path("owner/repo")
        assert path.name == "owner__repo"
        assert path.parent.name == "github"

    def test_local_to_symlink_path(self, installer):
        """Test local path to symlink path conversion."""
        path = installer._local_to_symlink_path(Path("/path/to/my-skills"))
        assert path.name == "my-skills"
        assert path.parent.name == "local"

    def test_discover_skills_in_path(self, installer, skill_dir):
        """Test skill discovery in a directory."""
        parent = skill_dir.parent
        skills = installer._discover_skills_in_path(parent)
        assert "my-skill" in skills

    def test_install_path_source(self, installer, skill_dir):
        """Test installing a local path source."""
        result = installer.install_path_source(skill_dir)

        assert result.path == str(skill_dir)
        assert result.install_path
        assert Path(result.install_path).is_symlink()

        # Check metadata was saved
        metadata_path = installer._metadata_path()
        assert metadata_path.exists()

    def test_install_path_source_not_found(self, installer, tmp_path):
        """Test installing a non-existent path."""
        with pytest.raises(SkillInstallerError, match="does not exist"):
            installer.install_path_source(tmp_path / "nonexistent")

    def test_install_path_source_not_dir(self, installer, tmp_path):
        """Test installing a file instead of directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("not a directory")
        with pytest.raises(SkillInstallerError, match="not a directory"):
            installer.install_path_source(file_path)

    def test_uninstall_path(self, installer, skill_dir):
        """Test uninstalling a path source."""
        installer.install_path_source(skill_dir)

        result = installer.uninstall(path=str(skill_dir))
        assert result is True

        # Check symlink was removed
        symlink_path = installer._local_to_symlink_path(skill_dir)
        assert not symlink_path.exists()

    def test_uninstall_not_found(self, installer):
        """Test uninstalling a source that doesn't exist."""
        result = installer.uninstall(path="/nonexistent")
        assert result is False

    def test_list_installed(self, installer, skill_dir):
        """Test listing installed sources."""
        installer.install_path_source(skill_dir)

        installed = installer.list_installed()
        assert len(installed) == 1
        assert installed[0].path == str(skill_dir)

    def test_get_installed_skills_dirs(self, installer, skill_dir):
        """Test getting directories containing installed skills."""
        installer.install_path_source(skill_dir)

        dirs = installer.get_installed_skills_dirs()
        assert len(dirs) >= 1
        # The symlink should resolve to the original path
        assert any(d.resolve() == skill_dir.resolve() for d in dirs)

    def test_metadata_persistence(self, installer, skill_dir):
        """Test that metadata persists across installer instances."""
        installer.install_path_source(skill_dir)

        # Create new installer instance
        new_installer = SkillInstaller(install_path=installer.install_path)
        installed = new_installer.list_installed()

        assert len(installed) == 1
        assert installed[0].path == str(skill_dir)

    def test_install_source_from_config(self, installer, skill_dir):
        """Test installing from SkillSource config object."""
        source = SkillSource(path=str(skill_dir))
        result = installer.install_source(source)

        assert result.path == str(skill_dir)

    def test_sync_all_updates_existing_repos(self, installer):
        """sync_all should install and then update repo sources."""
        source = SkillSource(repo="owner/repo")
        installed = InstalledSource(
            repo="owner/repo",
            install_path="installed/owner__repo",
        )
        updated = InstalledSource(
            repo="owner/repo",
            install_path="installed/owner__repo",
            commit_sha="abc123",
        )

        with (
            patch.object(
                installer, "install_source", return_value=installed
            ) as mock_install,
            patch.object(installer, "update", return_value=updated) as mock_update,
        ):
            results = installer.sync_all([source])

        mock_install.assert_called_once_with(source)
        mock_update.assert_called_once_with(repo="owner/repo")
        assert len(results) == 1
        assert results[0].commit_sha == "abc123"

    def test_sync_all_refreshes_path_sources(self, installer):
        """sync_all should force-refresh local path sources."""
        source = SkillSource(path="~/skills")
        refreshed = InstalledSource(
            path="~/skills",
            install_path="installed/local/skills",
            skills=["a", "b"],
        )

        with (
            patch.object(
                installer,
                "install_path_source",
                return_value=refreshed,
            ) as mock_install_path,
            patch.object(installer, "install_source") as mock_install_source,
            patch.object(installer, "update") as mock_update,
        ):
            results = installer.sync_all([source])

        mock_install_path.assert_called_once_with("~/skills", force=True)
        mock_install_source.assert_not_called()
        mock_update.assert_not_called()
        assert len(results) == 1
        assert results[0].skills == ["a", "b"]

    def test_sync_all_report_records_success_state(self, installer):
        """sync_all_report should persist success health metadata."""
        source = SkillSource(repo="owner/repo")
        installed = InstalledSource(
            repo="owner/repo",
            install_path="installed/owner__repo",
            commit_sha="oldsha",
        )
        updated = InstalledSource(
            repo="owner/repo",
            install_path="installed/owner__repo",
            commit_sha="newsha",
        )
        installer._sources = {"repo:owner/repo": installed}

        with (
            patch.object(installer, "install_source", return_value=installed),
            patch.object(installer, "update", return_value=updated),
        ):
            report = installer.sync_all_report([source])

        assert len(report.synced) == 1
        assert len(report.failed) == 0
        state = installer.list_sync_state()["repo:owner/repo"]
        assert state.last_status == "ok"
        assert state.last_action == "updated"
        assert state.previous_commit_sha == "oldsha"
        assert state.current_commit_sha == "newsha"
        assert state.commit_changed is True

    def test_sync_all_report_records_failure_state(self, installer):
        """sync_all_report should persist failure health metadata."""
        source = SkillSource(path="~/skills")
        with patch.object(
            installer,
            "install_path_source",
            side_effect=SkillInstallerError("boom"),
        ):
            report = installer.sync_all_report([source])

        assert len(report.synced) == 0
        assert len(report.failed) == 1
        state = installer.list_sync_state()["path:~/skills"]
        assert state.last_status == "error"
        assert state.last_action == "sync_failed"
        assert state.last_error == "boom"

    def test_remove_sync_state_on_uninstall(self, installer):
        """Uninstall should delete sync state entry for removed source."""
        fake_install_path = installer.install_path / "local" / "skills"
        installer._sync_state = {
            "path:~/skills": SourceSyncState(last_status="ok", last_action="installed")
        }
        installer._sources = {
            "path:~/skills": InstalledSource(
                path="~/skills",
                install_path=str(fake_install_path),
            )
        }

        result = installer.uninstall(path="~/skills")
        assert result is True
        assert "path:~/skills" not in installer.list_sync_state()


class TestConfigWriter:
    """Tests for ConfigWriter."""

    @pytest.fixture
    def config_path(self, tmp_path):
        """Create a config file path."""
        return tmp_path / "config.toml"

    @pytest.fixture
    def writer(self, config_path):
        """Create a ConfigWriter instance."""
        return ConfigWriter(config_path)

    def test_add_skill_source_repo(self, writer, config_path):
        """Test adding a repo source."""
        result = writer.add_skill_source(repo="owner/repo")
        assert result is True

        # Check file contents
        content = config_path.read_text()
        assert "owner/repo" in content
        assert "[[skills.sources]]" in content

    def test_add_skill_source_repo_with_ref(self, writer, config_path):
        """Test adding a repo source with ref."""
        writer.add_skill_source(repo="owner/repo", ref="v1.0")

        content = config_path.read_text()
        assert "owner/repo" in content
        assert 'ref = "v1.0"' in content

    def test_add_skill_source_path(self, writer, config_path):
        """Test adding a path source."""
        writer.add_skill_source(path="~/my-skills")

        content = config_path.read_text()
        assert "~/my-skills" in content

    def test_add_skill_source_duplicate(self, writer):
        """Test adding a duplicate source."""
        writer.add_skill_source(repo="owner/repo")
        result = writer.add_skill_source(repo="owner/repo")
        assert result is False

    def test_remove_skill_source(self, writer, config_path):
        """Test removing a source."""
        writer.add_skill_source(repo="owner/repo")
        result = writer.remove_skill_source(repo="owner/repo")
        assert result is True

        content = config_path.read_text()
        assert "owner/repo" not in content

    def test_remove_skill_source_not_found(self, writer):
        """Test removing a source that doesn't exist."""
        result = writer.remove_skill_source(repo="owner/repo")
        assert result is False

    def test_list_skill_sources(self, writer):
        """Test listing sources."""
        writer.add_skill_source(repo="owner/repo", ref="v1.0")
        writer.add_skill_source(path="~/my-skills")

        sources = writer.list_skill_sources()
        assert len(sources) == 2

        repos = [s for s in sources if s["repo"]]
        paths = [s for s in sources if s["path"]]

        assert len(repos) == 1
        assert repos[0]["repo"] == "owner/repo"
        assert repos[0]["ref"] == "v1.0"

        assert len(paths) == 1
        assert paths[0]["path"] == "~/my-skills"

    def test_update_skill_source_ref(self, writer, config_path):
        """Test updating a source ref."""
        writer.add_skill_source(repo="owner/repo", ref="v1.0")
        result = writer.update_skill_source_ref("owner/repo", "v2.0")
        assert result is True

        content = config_path.read_text()
        assert 'ref = "v2.0"' in content

    def test_preserves_existing_config(self, writer, config_path):
        """Test that existing config is preserved."""
        # Write initial config
        config_path.write_text(
            """# My config file
[models.default]
provider = "openai"
model = "gpt-5.2"

[skills.research]
enabled = true
"""
        )

        writer.add_skill_source(repo="owner/repo")

        content = config_path.read_text()
        # Check existing content is preserved
        assert "[models.default]" in content
        assert "[skills.research]" in content
        # Check new content is added
        assert "owner/repo" in content


class TestSourceParsing:
    """Tests for source parsing in CLI."""

    def test_parse_github_repo(self):
        """Test parsing GitHub repo."""
        from ash.cli.commands.skill import _parse_source

        repo, path, ref = _parse_source("owner/repo")
        assert repo == "owner/repo"
        assert path is None
        assert ref is None

    def test_parse_github_repo_with_ref(self):
        """Test parsing GitHub repo with ref."""
        from ash.cli.commands.skill import _parse_source

        repo, path, ref = _parse_source("owner/repo@v1.0")
        assert repo == "owner/repo"
        assert path is None
        assert ref == "v1.0"

    def test_parse_local_path_home(self):
        """Test parsing local path with ~."""
        from ash.cli.commands.skill import _parse_source

        repo, path, ref = _parse_source("~/my-skills")
        assert repo is None
        assert path == "~/my-skills"
        assert ref is None

    def test_parse_local_path_absolute(self):
        """Test parsing absolute local path."""
        from ash.cli.commands.skill import _parse_source

        repo, path, ref = _parse_source("/path/to/skills")
        assert repo is None
        assert path == "/path/to/skills"
        assert ref is None

    def test_parse_relative_path(self):
        """Test parsing relative path (fallback)."""
        from ash.cli.commands.skill import _parse_source

        repo, path, ref = _parse_source("./local-skills")
        assert repo is None
        assert path == "./local-skills"
        assert ref is None
