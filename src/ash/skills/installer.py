"""Skill installer for external skill sources.

Handles cloning GitHub repos and symlinking local paths to the installed skills directory.
"""

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ash.config.models import SkillSource
from ash.config.paths import get_installed_skills_path

logger = logging.getLogger(__name__)

SOURCES_METADATA_FILE = ".sources.json"
GITHUB_DIR = "github"
LOCAL_DIR = "local"


class SkillInstallerError(Exception):
    """Error during skill installation."""


class GitNotFoundError(SkillInstallerError):
    """Git is not installed or not available."""


@dataclass
class InstalledSource:
    """Metadata for an installed skill source."""

    # Source identification
    repo: str | None = None  # owner/repo
    path: str | None = None  # Original local path
    ref: str | None = None  # Git ref (branch/tag/commit)

    # Installation metadata
    installed_at: str = ""  # ISO timestamp
    updated_at: str = ""  # ISO timestamp
    install_path: str = ""  # Absolute path to installed location
    commit_sha: str | None = None  # Current commit SHA (for repos)
    skills: list[str] = field(default_factory=list)  # Discovered skill names

    @property
    def source_key(self) -> str:
        """Unique key for this source."""
        if self.repo:
            return f"repo:{self.repo}"
        return f"path:{self.path}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "repo": self.repo,
            "path": self.path,
            "ref": self.ref,
            "installed_at": self.installed_at,
            "updated_at": self.updated_at,
            "install_path": self.install_path,
            "commit_sha": self.commit_sha,
            "skills": self.skills,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InstalledSource":
        """Create from dictionary."""
        return cls(
            repo=data.get("repo"),
            path=data.get("path"),
            ref=data.get("ref"),
            installed_at=data.get("installed_at", ""),
            updated_at=data.get("updated_at", ""),
            install_path=data.get("install_path", ""),
            commit_sha=data.get("commit_sha"),
            skills=data.get("skills", []),
        )


class SkillInstaller:
    """Installer for external skill sources."""

    def __init__(self, install_path: Path | None = None):
        self.install_path = install_path or get_installed_skills_path()
        self._sources: dict[str, InstalledSource] | None = None

    def _check_git(self) -> None:
        """Ensure git is available on host."""
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise GitNotFoundError(
                "git is required for installing skill repos. "
                "Install with: apt install git / brew install git"
            )

    def _metadata_path(self) -> Path:
        """Get path to sources metadata file."""
        return self.install_path / SOURCES_METADATA_FILE

    def _load_metadata(self) -> dict[str, InstalledSource]:
        """Load installed sources metadata."""
        if self._sources is not None:
            return self._sources

        metadata_path = self._metadata_path()
        if not metadata_path.exists():
            self._sources = {}
            return self._sources

        try:
            data = json.loads(metadata_path.read_text())
            self._sources = {
                key: InstalledSource.from_dict(value)
                for key, value in data.get("sources", {}).items()
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "sources_metadata_load_failed", extra={"error.message": str(e)}
            )
            self._sources = {}

        return self._sources

    def _save_metadata(self) -> None:
        """Save installed sources metadata."""
        if self._sources is None:
            return

        self.install_path.mkdir(parents=True, exist_ok=True)
        metadata_path = self._metadata_path()

        data = {
            "version": 1,
            "sources": {key: source.to_dict() for key, source in self._sources.items()},
        }
        metadata_path.write_text(json.dumps(data, indent=2))

    def _repo_to_path(self, repo: str) -> Path:
        """Convert repo name to installation path.

        owner/repo -> github/owner__repo
        """
        safe_name = repo.replace("/", "__")
        return self.install_path / GITHUB_DIR / safe_name

    def _local_to_symlink_path(self, original_path: Path) -> Path:
        """Convert local path to symlink location.

        ~/my-skills -> local/my-skills
        /path/to/skills -> local/skills
        """
        return self.install_path / LOCAL_DIR / original_path.name

    def _discover_skills_in_path(self, path: Path) -> list[str]:
        """Discover skill names in a directory."""
        if not path.exists():
            return []

        skills = []

        # Check for SKILL.md in subdirectories
        for subdir in path.iterdir():
            if subdir.is_dir() and (subdir / "SKILL.md").exists():
                skills.append(subdir.name)

        return skills

    def _discover_all_skills(self, base_path: Path) -> list[str]:
        """Discover skills in root and skills/ subdirectory."""
        return self._discover_skills_in_path(base_path) + self._discover_skills_in_path(
            base_path / "skills"
        )

    def _get_commit_sha(self, repo_path: Path) -> str | None:
        """Get current commit SHA for a repo."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None

    def _get_default_branch(self, repo_path: Path) -> str:
        """Get the default branch ref for a repo."""
        result = subprocess.run(
            ["git", "remote", "show", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.split("\n"):
            if "HEAD branch:" in line:
                branch = line.split(":")[1].strip()
                return f"origin/{branch}"
        return "origin/main"

    def install_repo(
        self,
        repo: str,
        ref: str | None = None,
        *,
        force: bool = False,
    ) -> InstalledSource:
        """Install a skill source from a GitHub repo.

        Args:
            repo: GitHub repo in owner/repo format
            ref: Git ref (branch/tag/commit) to checkout
            force: Force reinstall even if already installed

        Returns:
            InstalledSource metadata

        Raises:
            GitNotFoundError: If git is not available
            SkillInstallerError: If installation fails
        """
        self._check_git()

        sources = self._load_metadata()
        source_key = f"repo:{repo}"
        install_path = self._repo_to_path(repo)

        # Check if already installed
        if source_key in sources and not force:
            existing = sources[source_key]
            if Path(existing.install_path).exists():
                logger.info(
                    "skill_source_already_installed", extra={"skill.source": repo}
                )
                return existing

        # Clone the repo
        github_url = f"https://github.com/{repo}.git"
        install_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing if force reinstall
        if install_path.exists():
            shutil.rmtree(install_path)

        logger.info("skill_source_cloning", extra={"skill.source": repo})
        clone_cmd = ["git", "clone", "--depth", "1"]
        if ref:
            clone_cmd.extend(["--branch", ref])
        clone_cmd.extend([github_url, str(install_path)])

        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SkillInstallerError(
                f"Failed to clone {repo}: {result.stderr.strip()}"
            )

        # Discover skills (root level or skills/ subdirectory)
        discovered_skills = self._discover_all_skills(install_path)

        # Create metadata
        now = datetime.now(UTC).isoformat()
        source = InstalledSource(
            repo=repo,
            ref=ref,
            installed_at=now,
            updated_at=now,
            install_path=str(install_path),
            commit_sha=self._get_commit_sha(install_path),
            skills=discovered_skills,
        )

        sources[source_key] = source
        self._save_metadata()

        logger.info(
            "skill_source_installed",
            extra={
                "skill.source": repo,
                "count": len(discovered_skills),
                "skills": discovered_skills,
            },
        )
        return source

    def install_path_source(
        self,
        path: str | Path,
        *,
        force: bool = False,
    ) -> InstalledSource:
        """Install a skill source from a local path via symlink.

        Args:
            path: Local path to skill directory
            force: Force reinstall even if already installed

        Returns:
            InstalledSource metadata

        Raises:
            SkillInstallerError: If installation fails
        """
        # Resolve the path
        resolved_path = Path(path).expanduser().resolve()
        original_path_str = str(path)

        if not resolved_path.exists():
            raise SkillInstallerError(f"Path does not exist: {resolved_path}")
        if not resolved_path.is_dir():
            raise SkillInstallerError(f"Path is not a directory: {resolved_path}")

        sources = self._load_metadata()
        source_key = f"path:{original_path_str}"
        symlink_path = self._local_to_symlink_path(resolved_path)

        # Check if already installed
        if source_key in sources and not force:
            existing = sources[source_key]
            if Path(existing.install_path).exists():
                logger.info(
                    "skill_source_already_installed", extra={"skill.source": str(path)}
                )
                return existing

        # Create symlink
        symlink_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing if force reinstall
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        symlink_path.symlink_to(resolved_path)
        logger.info(
            "skill_source_symlinked",
            extra={
                "symlink_path": str(symlink_path),
                "target_path": str(resolved_path),
            },
        )

        # Discover skills (root level or skills/ subdirectory)
        discovered_skills = self._discover_all_skills(resolved_path)

        # Create metadata
        now = datetime.now(UTC).isoformat()
        source = InstalledSource(
            path=original_path_str,
            installed_at=now,
            updated_at=now,
            install_path=str(symlink_path),
            skills=discovered_skills,
        )

        sources[source_key] = source
        self._save_metadata()

        logger.info(
            "skill_source_installed",
            extra={
                "skill.source": str(path),
                "count": len(discovered_skills),
                "skills": discovered_skills,
            },
        )
        return source

    def install_source(
        self,
        source: SkillSource,
        *,
        force: bool = False,
    ) -> InstalledSource:
        """Install a skill source from a SkillSource config.

        Args:
            source: SkillSource configuration
            force: Force reinstall even if already installed

        Returns:
            InstalledSource metadata
        """
        if source.repo:
            return self.install_repo(source.repo, source.ref, force=force)
        elif source.path:
            return self.install_path_source(source.path, force=force)
        else:
            raise SkillInstallerError("Invalid source: no repo or path specified")

    def uninstall(
        self,
        *,
        repo: str | None = None,
        path: str | None = None,
    ) -> bool:
        """Uninstall a skill source.

        Args:
            repo: GitHub repo to uninstall
            path: Local path to uninstall

        Returns:
            True if uninstalled, False if not found
        """
        if not repo and not path:
            raise ValueError("Must specify either 'repo' or 'path'")

        sources = self._load_metadata()

        if repo:
            source_key = f"repo:{repo}"
        else:
            source_key = f"path:{path}"

        if source_key not in sources:
            logger.debug(f"Source not found: {source_key}")
            return False

        source = sources[source_key]
        install_path = Path(source.install_path)

        # Remove files
        if install_path.is_symlink():
            install_path.unlink()
            logger.info(
                "skill_source_symlink_removed", extra={"file.path": str(install_path)}
            )
        elif install_path.exists():
            shutil.rmtree(install_path)
            logger.info(
                "skill_source_directory_removed", extra={"file.path": str(install_path)}
            )

        # Remove from metadata
        del sources[source_key]
        self._save_metadata()

        logger.info(
            "skill_source_uninstalled", extra={"skill.source": repo or str(path)}
        )
        return True

    def update(
        self,
        *,
        repo: str | None = None,
    ) -> InstalledSource | None:
        """Update an installed repo to latest.

        Args:
            repo: Specific repo to update, or None for all repos

        Returns:
            Updated source metadata, or None if not found
        """
        self._check_git()

        sources = self._load_metadata()

        if repo:
            source_key = f"repo:{repo}"
            if source_key not in sources:
                logger.warning("skill_source_not_found", extra={"skill.source": repo})
                return None
            sources_to_update = [sources[source_key]]
        else:
            sources_to_update = [s for s in sources.values() if s.repo]

        updated = None
        for source in sources_to_update:
            install_path = Path(source.install_path)
            if not install_path.exists():
                logger.warning(
                    "skill_source_install_path_missing",
                    extra={"file.path": str(install_path)},
                )
                continue

            logger.info("skill_source_updating", extra={"skill.source": source.repo})

            # Fetch and reset to origin
            fetch_cmd = ["git", "fetch", "origin"]
            result = subprocess.run(
                fetch_cmd, cwd=install_path, capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.warning(
                    "skill_source_fetch_failed",
                    extra={
                        "skill.source": source.repo,
                        "error.message": result.stderr.strip(),
                    },
                )
                continue

            # Determine the ref to reset to
            reset_ref = (
                f"origin/{source.ref}"
                if source.ref
                else self._get_default_branch(install_path)
            )

            reset_cmd = ["git", "reset", "--hard", reset_ref]
            result = subprocess.run(
                reset_cmd, cwd=install_path, capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.warning(
                    "skill_source_reset_failed",
                    extra={
                        "skill.source": source.repo,
                        "error.message": result.stderr.strip(),
                    },
                )
                continue

            # Update metadata
            source.updated_at = datetime.now(UTC).isoformat()
            source.commit_sha = self._get_commit_sha(install_path)

            # Re-discover skills
            source.skills = self._discover_all_skills(install_path)

            self._save_metadata()
            logger.info(
                "skill_source_updated",
                extra={"skill.source": source.repo, "commit_sha": source.commit_sha},
            )
            updated = source

        return updated

    def sync_all(self, sources: list[SkillSource]) -> list[InstalledSource]:
        """Sync all sources from config.

        Installs new sources and updates existing ones.

        Args:
            sources: List of SkillSource configs

        Returns:
            List of installed/updated sources
        """
        results = []

        for source in sources:
            try:
                # Always refresh path sources so changed SKILL.md trees are picked up.
                if source.path:
                    installed = self.install_path_source(source.path, force=True)
                else:
                    installed = self.install_source(source)
                if source.repo:
                    updated = self.update(repo=source.repo)
                    if updated is not None:
                        installed = updated
                results.append(installed)
            except SkillInstallerError as e:
                logger.error(
                    "skill_install_failed",
                    extra={
                        "skill.source": source.repo or str(source.path),
                        "error.message": str(e),
                    },
                )

        return results

    def list_installed(self) -> list[InstalledSource]:
        """List all installed sources.

        Returns:
            List of InstalledSource metadata
        """
        sources = self._load_metadata()
        return list(sources.values())

    def get_installed_skills_dirs(self) -> list[Path]:
        """Get all directories containing installed skills.

        Returns paths where skills can be discovered:
        - github/owner__repo/
        - github/owner__repo/skills/
        - local/name/
        - local/name/skills/

        Returns:
            List of paths containing skills
        """
        dirs = []
        sources = self._load_metadata()

        for source in sources.values():
            install_path = Path(source.install_path)
            if not install_path.exists():
                continue

            # Resolve symlinks for local sources
            if install_path.is_symlink():
                install_path = install_path.resolve()

            # Add root and skills/ subdirectory if they exist
            if install_path.exists():
                dirs.append(install_path)
                skills_subdir = install_path / "skills"
                if skills_subdir.exists():
                    dirs.append(skills_subdir)

        return dirs
