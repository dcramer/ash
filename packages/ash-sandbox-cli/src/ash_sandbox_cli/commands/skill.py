"""Skill management commands for sandbox CLI."""

from pathlib import Path

import typer
import yaml

app = typer.Typer(
    name="skill",
    help="Manage skills in the workspace.",
    no_args_is_help=True,
)

WORKSPACE_SKILLS = Path("/workspace/skills")


def _scan_skills_dir(skills_dir: Path) -> list[tuple[str, str]]:
    """Scan a single directory for skill definitions.

    Returns list of (name, description) tuples.
    """
    results: list[tuple[str, str]] = []
    if not skills_dir.exists():
        return results

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue

        try:
            content = skill_file.read_text()
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 2:
                    frontmatter = yaml.safe_load(parts[1])
                    if isinstance(frontmatter, dict):
                        desc = frontmatter.get("description", "(no description)")
                        results.append((skill_dir.name, desc))
                        continue
        except Exception:  # noqa: BLE001
            results.append((skill_dir.name, "(unable to read)"))
            continue

        results.append((skill_dir.name, "(unable to read)"))

    return results


@app.command()
def validate(path: Path) -> None:
    """Validate a SKILL.md file format.

    Checks that the file has valid YAML frontmatter and required fields.
    """
    if not path.exists():
        typer.echo(f"Error: {path} does not exist", err=True)
        raise typer.Exit(1)

    content = path.read_text()

    # Check for frontmatter
    if not content.startswith("---"):
        typer.echo("Error: SKILL.md must start with YAML frontmatter (---)", err=True)
        raise typer.Exit(1)

    # Extract frontmatter
    parts = content.split("---", 2)
    if len(parts) < 3:
        typer.echo("Error: Invalid frontmatter format (missing closing ---)", err=True)
        raise typer.Exit(1)

    frontmatter_str = parts[1].strip()
    body = parts[2].strip()

    # Parse YAML
    try:
        frontmatter = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        typer.echo(f"Error: Invalid YAML in frontmatter: {e}", err=True)
        raise typer.Exit(1) from None

    if not isinstance(frontmatter, dict):
        typer.echo("Error: Frontmatter must be a YAML mapping", err=True)
        raise typer.Exit(1)

    # Check required fields
    if "description" not in frontmatter:
        typer.echo(
            "Error: Missing required field 'description' in frontmatter", err=True
        )
        raise typer.Exit(1)

    # Check optional fields have valid types
    # Accept allowed_tools, allowed-tools, or tools (legacy alias)
    tools_key = next(
        (k for k in ("allowed_tools", "allowed-tools", "tools") if k in frontmatter),
        None,
    )
    if tools_key is not None:
        if not isinstance(frontmatter[tools_key], list):
            typer.echo(f"Error: '{tools_key}' must be a list", err=True)
            raise typer.Exit(1)

    if "requires" in frontmatter:
        req = frontmatter["requires"]
        if not isinstance(req, dict):
            typer.echo("Error: 'requires' must be a mapping", err=True)
            raise typer.Exit(1)

        for key in ("bins", "env", "os"):
            if key in req and not isinstance(req[key], list):
                typer.echo(f"Error: 'requires.{key}' must be a list", err=True)
                raise typer.Exit(1)

    # Check body has content
    if not body:
        typer.echo("Warning: Skill has no instructions (body is empty)", err=True)

    typer.echo(f"Valid: {path}")
    typer.echo(f"  Description: {frontmatter['description']}")

    if tools_key is not None:
        typer.echo(f"  Tools: {', '.join(frontmatter[tools_key])}")

    if "requires" in frontmatter:
        req = frontmatter["requires"]
        if "bins" in req:
            typer.echo(f"  Binaries: {', '.join(req['bins'])}")
        if "env" in req:
            typer.echo(f"  Env vars: {', '.join(req['env'])}")
        if "os" in req:
            typer.echo(f"  OS: {', '.join(req['os'])}")


@app.command("list")
def list_skills() -> None:
    """List skills from all mounted skill directories."""
    import os

    raw = os.environ.get("ASH_SKILL_DIRS", "")
    scan_dirs = [Path(p) for p in raw.split(":") if p] if raw else [WORKSPACE_SKILLS]

    skills: list[tuple[str, str]] = []
    for d in scan_dirs:
        skills.extend(_scan_skills_dir(d))

    if not skills:
        typer.echo("No skills found")
        return

    typer.echo("Available skills:")
    for name, desc in skills:
        typer.echo(f"  {name}: {desc}")
