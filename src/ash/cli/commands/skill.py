"""Skill management commands."""

import re
from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, error, info, success, warning
from ash.cli.context import get_config

# Pattern to detect source type from input
GITHUB_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


def _parse_source(source: str) -> tuple[str | None, str | None, str | None]:
    """Parse a source string into repo, path, and ref.

    Args:
        source: Source string (owner/repo, owner/repo@ref, or path)

    Returns:
        Tuple of (repo, path, ref)

    Examples:
        owner/repo -> ("owner/repo", None, None)
        owner/repo@v1.0 -> ("owner/repo", None, "v1.0")
        ~/my-skills -> (None, "~/my-skills", None)
        /path/to/skills -> (None, "/path/to/skills", None)
    """
    # Check for local path (starts with /, ~, or .)
    if source.startswith("/") or source.startswith("~") or source.startswith("."):
        return None, source, None

    # Check for repo with ref (owner/repo@ref)
    if "@" in source:
        repo, ref = source.rsplit("@", 1)
        if GITHUB_REPO_PATTERN.match(repo):
            return repo, None, ref

    # Check for plain repo (owner/repo)
    if GITHUB_REPO_PATTERN.match(source):
        return source, None, None

    # Fallback: treat as local path
    return None, source, None


def _format_skills_list(skills: list[str]) -> str:
    """Format a list of skills for display."""
    return ", ".join(skills) if skills else "none"


def _format_sha(sha: str | None) -> str:
    """Format a commit SHA for display."""
    return sha[:8] if sha else "unknown"


def _source_key(repo: str | None, path: str | None) -> str:
    """Generate a source key from repo or path."""
    return f"repo:{repo}" if repo else f"path:{path}"


def register(app: typer.Typer) -> None:
    """Register the skill command group."""
    skill_app = typer.Typer(name="skill", help="Manage skills", no_args_is_help=True)

    @skill_app.command("validate")
    def validate(
        path: Annotated[
            Path,
            typer.Argument(help="Path to skill file or directory"),
        ],
    ) -> None:
        """Validate a skill file format.

        Examples:
            ash skill validate workspace/skills/my-skill
            ash skill validate workspace/skills/my-skill/SKILL.md
        """
        from ash.skills.registry import SkillRegistry

        # Resolve path to SKILL.md
        if path.is_dir():
            skill_file = path / "SKILL.md"
        else:
            skill_file = path

        if not skill_file.exists():
            error(f"File not found: {skill_file}")
            raise typer.Exit(1)

        registry = SkillRegistry()
        is_valid, err = registry.validate_skill_file(skill_file)

        if is_valid:
            success(f"Valid skill: {skill_file}")
        else:
            error(f"Invalid skill: {err}")
            raise typer.Exit(1)

    @skill_app.command("list")
    def list_skills(
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
        show_source: Annotated[
            bool,
            typer.Option(
                "--source",
                "-s",
                help="Show source type for each skill",
            ),
        ] = False,
    ) -> None:
        """List registered skills.

        Examples:
            ash skill list
            ash skill list --source
        """
        from rich.table import Table

        from ash.skills.registry import SkillRegistry

        config = get_config(config_path)
        registry = SkillRegistry()
        registry.discover(config.workspace)

        skills = registry.list_available()

        if not skills:
            warning("No skills found")
            return

        table = Table(title="Skills")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        if show_source:
            table.add_column("Source", style="yellow")
            table.add_column("Repo/Path", style="dim")
        else:
            table.add_column("Path", style="dim")

        for skill in sorted(skills, key=lambda s: s.name):
            if show_source:
                source_info = skill.source_repo or (
                    str(skill.skill_path) if skill.skill_path else "-"
                )
                table.add_row(
                    skill.name,
                    skill.description,
                    skill.source_type.value,
                    source_info,
                )
            else:
                path = str(skill.skill_path) if skill.skill_path else "-"
                table.add_row(skill.name, skill.description, path)

        console.print(table)

    @skill_app.command("add")
    def add_source(
        source: Annotated[
            str,
            typer.Argument(
                help="Skill source: owner/repo, owner/repo@ref, or local path"
            ),
        ],
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
    ) -> None:
        """Add and install a skill source.

        Adds the source to config.toml and installs it immediately.

        Examples:
            ash skill add anthropic/skills
            ash skill add anthropic/skills@v1.0
            ash skill add ~/my-local-skills
        """
        from ash.config.paths import get_config_path
        from ash.config.writer import ConfigWriter
        from ash.skills.installer import (
            GitNotFoundError,
            SkillInstaller,
            SkillInstallerError,
        )

        repo, path, ref = _parse_source(source)

        # Add to config
        config_file = config_path or get_config_path()
        writer = ConfigWriter(config_file)

        added = writer.add_skill_source(repo=repo, path=path, ref=ref)
        if not added:
            info(f"Source already in config: {source}")

        # Install the source
        installer = SkillInstaller()
        try:
            if repo:
                result = installer.install_repo(repo, ref)
            elif path:
                result = installer.install_path_source(path)
            else:
                return

            success(
                f"Installed {repo or path}: {len(result.skills)} skill(s) "
                f"({_format_skills_list(result.skills)})"
            )
        except GitNotFoundError as e:
            error(str(e))
            raise typer.Exit(1) from None
        except SkillInstallerError as e:
            error(f"Installation failed: {e}")
            raise typer.Exit(1) from None

    @skill_app.command("remove")
    def remove_source(
        source: Annotated[
            str,
            typer.Argument(help="Skill source to remove: owner/repo or local path"),
        ],
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
    ) -> None:
        """Remove a skill source.

        Removes the source from config.toml and deletes installed files.

        Examples:
            ash skill remove anthropic/skills
            ash skill remove ~/my-local-skills
        """
        from ash.config.paths import get_config_path
        from ash.config.writer import ConfigWriter
        from ash.skills.installer import SkillInstaller

        repo, path, _ = _parse_source(source)

        # Remove from config
        config_file = config_path or get_config_path()
        writer = ConfigWriter(config_file)

        removed_from_config = writer.remove_skill_source(repo=repo, path=path)
        if not removed_from_config:
            warning(f"Source not found in config: {source}")

        # Uninstall
        installer = SkillInstaller()
        removed_files = installer.uninstall(repo=repo, path=path)

        if removed_files:
            success(f"Removed: {source}")
        elif not removed_from_config:
            error(f"Source not found: {source}")
            raise typer.Exit(1)
        else:
            info(f"Removed from config (no installed files): {source}")

    @skill_app.command("sync")
    def sync_sources(
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
    ) -> None:
        """Sync all skill sources from config.

        Installs any sources in [[skills.sources]] that aren't installed yet.

        Examples:
            ash skill sync
        """
        from ash.skills.installer import SkillInstaller, SkillInstallerError

        config = get_config(config_path)

        if not config.skill_sources:
            warning("No skill sources configured in config.toml")
            return

        installer = SkillInstaller()
        results = []
        errors = []

        for source in config.skill_sources:
            try:
                result = installer.install_source(source)
                results.append(result)
                info(
                    f"Synced {source.repo or source.path}: "
                    f"{len(result.skills)} skill(s)"
                )
            except SkillInstallerError as e:
                errors.append((source, e))
                error(f"Failed to sync {source.repo or source.path}: {e}")

        if results:
            total_skills = sum(len(r.skills) for r in results)
            success(f"Synced {len(results)} source(s), {total_skills} skill(s) total")

        if errors:
            raise typer.Exit(1)

    @skill_app.command("update")
    def update_source(
        source: Annotated[
            str | None,
            typer.Argument(
                help="Specific repo to update (owner/repo), or all if omitted"
            ),
        ] = None,
    ) -> None:
        """Update installed skill repos.

        Updates to the latest commit (or pinned ref).

        Examples:
            ash skill update                  # Update all repos
            ash skill update anthropic/skills # Update specific repo
        """
        from ash.skills.installer import (
            GitNotFoundError,
            SkillInstaller,
            SkillInstallerError,
        )

        installer = SkillInstaller()

        try:
            if source:
                repo, _, _ = _parse_source(source)
                if not repo:
                    error("update only works with repos, not local paths")
                    raise typer.Exit(1)

                result = installer.update(repo=repo)
                if result:
                    success(f"Updated {repo} to {_format_sha(result.commit_sha)}")
                else:
                    warning(f"Repo not installed: {repo}")
            else:
                # Update all
                repos = [s for s in installer.list_installed() if s.repo]

                if not repos:
                    warning("No installed repos to update")
                    return

                for src in repos:
                    try:
                        result = installer.update(repo=src.repo)
                        if result:
                            info(
                                f"Updated {src.repo} to {_format_sha(result.commit_sha)}"
                            )
                    except SkillInstallerError as e:
                        error(f"Failed to update {src.repo}: {e}")

                success(f"Updated {len(repos)} repo(s)")

        except GitNotFoundError as e:
            error(str(e))
            raise typer.Exit(1) from None

    @skill_app.command("sources")
    def list_sources(
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
    ) -> None:
        """List configured skill sources.

        Shows [[skills.sources]] entries from config.toml and their install status.

        Examples:
            ash skill sources
        """
        from rich.table import Table

        from ash.skills.installer import SkillInstaller

        config = get_config(config_path)
        installer = SkillInstaller()
        installed = {s.source_key: s for s in installer.list_installed()}

        if not config.skill_sources and not installed:
            warning("No skill sources configured or installed")
            return

        table = Table(title="Skill Sources")
        table.add_column("Type", style="cyan")
        table.add_column("Source", style="white")
        table.add_column("Ref", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Skills", style="dim")

        # Build set of config keys for orphan detection
        config_keys = {_source_key(s.repo, s.path) for s in config.skill_sources}

        # Show configured sources
        for source in config.skill_sources:
            key = _source_key(source.repo, source.path)
            source_type = "repo" if source.repo else "path"

            if key in installed:
                status = "installed"
                skills = _format_skills_list(installed[key].skills) or "-"
            else:
                status = "not installed"
                skills = "-"

            table.add_row(
                source_type,
                source.repo or source.path or "",
                source.ref or "-",
                status,
                skills,
            )

        # Show installed sources not in config (orphaned)
        for key, source in installed.items():
            if key in config_keys:
                continue
            source_type = "repo" if source.repo else "path"
            table.add_row(
                source_type,
                source.repo or source.path or "",
                source.ref or "-",
                "[yellow]orphaned[/yellow]",
                _format_skills_list(source.skills) or "-",
            )

        console.print(table)

    app.add_typer(skill_app)
