"""Skill management commands."""

from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, dim, error, success, warning
from ash.cli.context import get_config

# Template for new skills
SKILL_TEMPLATE = """---
description: {description}
---

{instructions}
""".strip()


def register(app: typer.Typer) -> None:
    """Register the skill command group."""
    skill_app = typer.Typer(name="skill", help="Manage skills")

    @skill_app.command("init")
    def init(
        name: Annotated[
            str,
            typer.Argument(help="Skill name (lowercase, hyphens)"),
        ],
        description: Annotated[
            str,
            typer.Option(
                "--description",
                "-d",
                help="One-line description",
            ),
        ] = "A new skill",
        resources: Annotated[
            str,
            typer.Option(
                "--resources",
                "-r",
                help="Comma-separated: scripts,references,assets",
            ),
        ] = "",
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
    ) -> None:
        """Initialize a new skill directory with template.

        Examples:
            ash skill init my-skill
            ash skill init my-skill -d "Check weather forecast"
            ash skill init my-skill --resources scripts,references
        """
        config = get_config(config_path)
        skills_dir = config.workspace / "skills"
        skill_dir = skills_dir / name

        # Validate name
        if not name.replace("-", "").replace("_", "").isalnum():
            error("Skill name must be alphanumeric with hyphens/underscores")
            raise typer.Exit(1)

        # Check if skill already exists
        if skill_dir.exists():
            error(f"Skill '{name}' already exists at {skill_dir}")
            raise typer.Exit(1)

        # Create directory structure
        skill_dir.mkdir(parents=True)

        # Create optional resource directories
        if resources:
            for resource in resources.split(","):
                resource = resource.strip()
                if resource in ("scripts", "references", "assets"):
                    (skill_dir / resource).mkdir()
                    dim(f"  Created {resource}/")

        # Create SKILL.md with template
        skill_file = skill_dir / "SKILL.md"
        content = SKILL_TEMPLATE.format(
            description=description,
            instructions="Instructions for the skill go here.\n\n"
            "## Implementation\n\n"
            "Describe what the agent should do.",
        )
        skill_file.write_text(content)

        success(f"Created skill: {skill_dir}")
        dim(f"  Edit: {skill_file}")
        dim("  Then run: ash skill validate <path>")

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
        show_all: Annotated[
            bool,
            typer.Option(
                "--all",
                "-a",
                help="Include unavailable skills",
            ),
        ] = False,
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
    ) -> None:
        """List available skills.

        Examples:
            ash skill list
            ash skill list --all
        """
        from rich.table import Table

        from ash.skills.registry import SkillRegistry

        config = get_config(config_path)
        registry = SkillRegistry()
        registry.discover(config.workspace)

        if show_all:
            skills = list(registry._skills.values())
        else:
            skills = registry.list_available()

        if not skills:
            warning("No skills found")
            return

        table = Table(title="Skills")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        table.add_column("Path", style="dim")

        for skill in sorted(skills, key=lambda s: s.name):
            is_available, reason = skill.is_available()
            if is_available:
                status = "[green]available[/green]"
            else:
                status = f"[red]{reason}[/red]"

            path = str(skill.skill_path) if skill.skill_path else "-"
            table.add_row(skill.name, skill.description, status, path)

        console.print(table)

    @skill_app.command("reload")
    def reload(
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
    ) -> None:
        """Reload skills from workspace.

        This is useful after manually creating skill files.

        Examples:
            ash skill reload
        """
        from ash.skills.registry import SkillRegistry

        config = get_config(config_path)
        registry = SkillRegistry()
        registry.discover(config.workspace)

        available = registry.list_available()
        success(f"Loaded {len(available)} available skills")
        for skill in sorted(available, key=lambda s: s.name):
            dim(f"  - {skill.name}: {skill.description}")

    app.add_typer(skill_app)
