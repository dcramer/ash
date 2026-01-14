"""Skill management commands."""

from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, error, success, warning
from ash.cli.context import get_config


def register(app: typer.Typer) -> None:
    """Register the skill command group."""
    skill_app = typer.Typer(name="skill", help="Manage skills")

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
    ) -> None:
        """List registered skills.

        Examples:
            ash skill list
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
        table.add_column("Path", style="dim")

        for skill in sorted(skills, key=lambda s: s.name):
            path = str(skill.skill_path) if skill.skill_path else "-"
            table.add_row(skill.name, skill.description, path)

        console.print(table)

    app.add_typer(skill_app)
