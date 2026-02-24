"""Skill installer for external skill sources.

Handles cloning GitHub repos and symlinking local paths to the installed skills directory.
"""

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ash.config.models import SkillSource
from ash.config.paths import get_installed_skills_path

logger = logging.getLogger(__name__)

SOURCES_METADATA_FILE = ".sources.json"
SYNC_STATE_FILE = ".sync_state.json"
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


@dataclass
class SourceSyncState:
    """Per-source sync health and recency."""

    last_attempt_at: str = ""
    last_success_at: str = ""
    last_status: str = "unknown"  # unknown|ok|error
    last_action: str = (
        ""  # installed|updated|checked_no_change|refreshed_path|sync_failed
    )
    last_error: str | None = None
    previous_commit_sha: str | None = None
    current_commit_sha: str | None = None
    commit_changed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_attempt_at": self.last_attempt_at,
            "last_success_at": self.last_success_at,
            "last_status": self.last_status,
            "last_action": self.last_action,
            "last_error": self.last_error,
            "previous_commit_sha": self.previous_commit_sha,
            "current_commit_sha": self.current_commit_sha,
            "commit_changed": self.commit_changed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceSyncState":
        return cls(
            last_attempt_at=str(data.get("last_attempt_at") or ""),
            last_success_at=str(data.get("last_success_at") or ""),
            last_status=str(data.get("last_status") or "unknown"),
            last_action=str(data.get("last_action") or ""),
            last_error=(
                str(data["last_error"])
                if isinstance(data.get("last_error"), str)
                else None
            ),
            previous_commit_sha=(
                str(data["previous_commit_sha"])
                if isinstance(data.get("previous_commit_sha"), str)
                else None
            ),
            current_commit_sha=(
                str(data["current_commit_sha"])
                if isinstance(data.get("current_commit_sha"), str)
                else None
            ),
            commit_changed=(
                bool(data["commit_changed"])
                if isinstance(data.get("commit_changed"), bool)
                else None
            ),
        )


@dataclass
class SyncReport:
    """Outcome summary for a sync pass."""

    synced: list[InstalledSource] = field(default_factory=list)
    failed: list[tuple[SkillSource, str]] = field(default_factory=list)


class SkillInstaller:
    """Installer for external skill sources."""

    def __init__(self, install_path: Path | None = None):
        self.install_path = install_path or get_installed_skills_path()
        self._sources: dict[str, InstalledSource] | None = None
        self._sync_state: dict[str, SourceSyncState] | None = None

    def _check_git(self) -> None:
        """Ensure git is available on host."""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as e:
            raise GitNotFoundError(
                "git is required for installing skill repos. "
                "Install with: apt install git / brew install git"
            ) from e
        if result.returncode != 0:
            raise GitNotFoundError(
                "git is required for installing skill repos. "
                "Install with: apt install git / brew install git"
            )

    def _metadata_path(self) -> Path:
        """Get path to sources metadata file."""
        return self.install_path / SOURCES_METADATA_FILE

    def _sync_state_path(self) -> Path:
        """Get path to per-source sync health state file."""
        return self.install_path / SYNC_STATE_FILE

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

    def _load_sync_state(self) -> dict[str, SourceSyncState]:
        """Load per-source sync health state."""
        if self._sync_state is not None:
            return self._sync_state

        state_path = self._sync_state_path()
        if not state_path.exists():
            self._sync_state = {}
            return self._sync_state

        try:
            data = json.loads(state_path.read_text())
            self._sync_state = {
                key: SourceSyncState.from_dict(value)
                for key, value in data.get("sources", {}).items()
                if isinstance(value, dict)
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("sync_state_load_failed", extra={"error.message": str(e)})
            self._sync_state = {}

        return self._sync_state

    def _save_sync_state(self) -> None:
        """Persist per-source sync health state."""
        if self._sync_state is None:
            return
        self.install_path.mkdir(parents=True, exist_ok=True)
        state_path = self._sync_state_path()
        payload = {
            "version": 1,
            "sources": {
                key: value.to_dict() for key, value in self._sync_state.items()
            },
        }
        state_path.write_text(json.dumps(payload, indent=2))

    def _source_key(self, *, repo: str | None = None, path: str | None = None) -> str:
        if repo:
            return f"repo:{repo}"
        return f"path:{self._normalize_path(path)}"

    @staticmethod
    def _normalize_path(path: str | None) -> str | None:
        if path is None:
            return None
        try:
            return str(Path(path).expanduser().resolve(strict=False))
        except Exception:
            return str(Path(path).expanduser())

    def _record_sync_state(
        self,
        *,
        source_key: str,
        ok: bool,
        action: str,
        error: str | None = None,
        previous_commit_sha: str | None = None,
        current_commit_sha: str | None = None,
    ) -> None:
        states = self._load_sync_state()
        state = states.get(source_key, SourceSyncState())
        now = datetime.now(UTC).isoformat()
        state.last_attempt_at = now
        state.last_status = "ok" if ok else "error"
        state.last_action = action
        if ok:
            state.last_success_at = now
            state.last_error = None
        else:
            state.last_error = error or "unknown sync failure"
        state.previous_commit_sha = previous_commit_sha
        state.current_commit_sha = current_commit_sha
        if previous_commit_sha and current_commit_sha:
            state.commit_changed = previous_commit_sha != current_commit_sha
        else:
            state.commit_changed = None
        states[source_key] = state
        self._save_sync_state()

    def _remove_sync_state(self, source_key: str) -> None:
        states = self._load_sync_state()
        if source_key in states:
            del states[source_key]
            self._save_sync_state()

    def list_sync_state(self) -> dict[str, SourceSyncState]:
        """List per-source sync health state."""
        return dict(self._load_sync_state())

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
        digest = hashlib.sha256(str(original_path).encode("utf-8")).hexdigest()[:8]
        safe_name = f"{original_path.name}-{digest}"
        return self.install_path / LOCAL_DIR / safe_name

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

    def _git_ref_exists(self, repo_path: Path, ref: str) -> bool:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _resolve_reset_ref(self, repo_path: Path, ref: str | None) -> str:
        """Resolve a ref for reset/checkouts (branch, tag, or commit)."""
        if not ref:
            return self._get_default_branch(repo_path)

        branch_ref = f"refs/remotes/origin/{ref}"
        if self._git_ref_exists(repo_path, branch_ref):
            return f"origin/{ref}"

        tag_ref = f"refs/tags/{ref}"
        if self._git_ref_exists(repo_path, tag_ref):
            return tag_ref

        commit_ref = f"{ref}^{{commit}}"
        if self._git_ref_exists(repo_path, commit_ref):
            return ref

        raise SkillInstallerError(f"Ref not found in repository: {ref}")

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
        source_key = self._source_key(repo=repo)
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
        clone_cmd = ["git", "clone", github_url, str(install_path)]

        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SkillInstallerError(
                f"Failed to clone {repo}: {result.stderr.strip()}"
            )

        if ref:
            checkout_cmd = ["git", "checkout", "--detach", ref]
            result = subprocess.run(
                checkout_cmd,
                cwd=install_path,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise SkillInstallerError(
                    f"Failed to checkout ref '{ref}' for {repo}: {result.stderr.strip()}"
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
        normalized_path_str = str(resolved_path)

        if not resolved_path.exists():
            raise SkillInstallerError(f"Path does not exist: {resolved_path}")
        if not resolved_path.is_dir():
            raise SkillInstallerError(f"Path is not a directory: {resolved_path}")

        sources = self._load_metadata()
        source_key = self._source_key(path=normalized_path_str)
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
            path=normalized_path_str,
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
        source_key = self._source_key(repo=repo, path=path)

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
        self._remove_sync_state(source_key)

        logger.info(
            "skill_source_uninstalled", extra={"skill.source": repo or str(path)}
        )
        return True

    def update(
        self,
        *,
        repo: str | None = None,
        strict: bool = False,
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
                if strict:
                    raise SkillInstallerError(f"Repo not installed: {repo}")
                return None
            sources_to_update = [sources[source_key]]
        else:
            sources_to_update = [s for s in sources.values() if s.repo]

        updated = None
        for source in sources_to_update:
            install_path = Path(source.install_path)
            if not install_path.exists():
                message = f"Install path missing: {install_path}"
                logger.warning(
                    "skill_source_install_path_missing",
                    extra={"file.path": str(install_path)},
                )
                if strict:
                    raise SkillInstallerError(message)
                continue

            logger.info("skill_source_updating", extra={"skill.source": source.repo})

            # Fetch and reset to origin
            fetch_cmd = ["git", "fetch", "--tags", "origin"]
            result = subprocess.run(
                fetch_cmd, cwd=install_path, capture_output=True, text=True
            )
            if result.returncode != 0:
                message = f"Failed to fetch {source.repo}: {result.stderr.strip() or 'unknown error'}"
                logger.warning(
                    "skill_source_fetch_failed",
                    extra={
                        "skill.source": source.repo,
                        "error.message": result.stderr.strip(),
                    },
                )
                if strict:
                    raise SkillInstallerError(message)
                continue

            # Determine the ref to reset to
            reset_ref = self._resolve_reset_ref(install_path, source.ref)

            reset_cmd = ["git", "reset", "--hard", reset_ref]
            result = subprocess.run(
                reset_cmd, cwd=install_path, capture_output=True, text=True
            )
            if result.returncode != 0:
                message = (
                    f"Failed to reset {source.repo} to {reset_ref}: "
                    f"{result.stderr.strip() or 'unknown error'}"
                )
                logger.warning(
                    "skill_source_reset_failed",
                    extra={
                        "skill.source": source.repo,
                        "error.message": result.stderr.strip(),
                    },
                )
                if strict:
                    raise SkillInstallerError(message)
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

    def sync_all_report(self, sources: list[SkillSource]) -> SyncReport:
        """Sync all sources and return a success/failure report."""
        report = SyncReport()
        installed = self._load_metadata()

        for source in sources:
            source_key = self._source_key(repo=source.repo, path=source.path)
            existing_source = installed.get(source_key)
            previous_commit_sha = (
                existing_source.commit_sha if source.repo and existing_source else None
            )
            try:
                if source.path:
                    synced = self.install_path_source(source.path, force=True)
                    action = "refreshed_path"
                else:
                    synced = self.install_source(source)
                    action = "installed"
                    if source.repo:
                        updated = self.update(repo=source.repo, strict=True)
                        if updated is None:
                            raise SkillInstallerError(
                                f"Failed to update repo: {source.repo}"
                            )
                        synced = updated
                        action = "updated"
                if (
                    source.repo
                    and previous_commit_sha
                    and synced.commit_sha
                    and previous_commit_sha == synced.commit_sha
                ):
                    action = "checked_no_change"
                self._record_sync_state(
                    source_key=source_key,
                    ok=True,
                    action=action,
                    previous_commit_sha=previous_commit_sha,
                    current_commit_sha=synced.commit_sha if source.repo else None,
                )
                report.synced.append(synced)
            except SkillInstallerError as e:
                self._record_sync_state(
                    source_key=source_key,
                    ok=False,
                    action="sync_failed",
                    error=str(e),
                    previous_commit_sha=previous_commit_sha,
                )
                logger.error(
                    "skill_install_failed",
                    extra={
                        "skill.source": source.repo or str(source.path),
                        "error.message": str(e),
                    },
                )
                report.failed.append((source, str(e)))

        return report

    def sync_all(self, sources: list[SkillSource]) -> list[InstalledSource]:
        """Sync all sources from config.

        Installs new sources and updates existing ones.

        Args:
            sources: List of SkillSource configs

        Returns:
            List of installed/updated sources
        """
        return self.sync_all_report(sources).synced

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
