"""Sandbox management commands."""

import subprocess
from pathlib import Path
from typing import Annotated

import click
import typer

from ash.cli.console import console, dim, error, success, warning


def _get_dockerfile_path() -> Path | None:
    """Find the Dockerfile.sandbox path.

    Returns:
        Path to Dockerfile.sandbox, or None if not found.
    """
    # Try relative to this file: cli/commands/sandbox.py -> project root
    project_root = Path(__file__).parents[4]
    dockerfile_path = project_root / "docker" / "Dockerfile.sandbox"
    if dockerfile_path.exists():
        return dockerfile_path

    # Try relative to ash package
    import ash

    if ash.__file__:
        package_root = Path(ash.__file__).parents[2]
        dockerfile_path = package_root / "docker" / "Dockerfile.sandbox"
        if dockerfile_path.exists():
            return dockerfile_path

    return None


def register(app: typer.Typer) -> None:
    """Register the sandbox command."""

    @app.command()
    def sandbox(
        action: Annotated[
            str | None,
            typer.Argument(help="Action: build, status, clean"),
        ] = None,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                "-f",
                help="For clean: also remove the sandbox image",
            ),
        ] = False,
        config: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Config file for build-time packages",
            ),
        ] = None,
    ) -> None:
        """Manage the Docker sandbox environment."""
        if action is None:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            raise typer.Exit(0)

        if action == "build":
            dockerfile_path = _get_dockerfile_path()
            if not dockerfile_path:
                error("Dockerfile.sandbox not found")
                raise typer.Exit(1)
            if not _sandbox_build(dockerfile_path, config):
                raise typer.Exit(1)

        elif action == "status":
            _sandbox_status()

        elif action == "clean":
            _sandbox_clean(force)

        else:
            error(f"Unknown action: {action}")
            console.print("Valid actions: build, status, clean")
            raise typer.Exit(1)


def _sandbox_build(dockerfile_path: Path, config_path: Path | None = None) -> bool:
    """Build the sandbox Docker image.

    Args:
        dockerfile_path: Path to Dockerfile.sandbox.
        config_path: Optional config file path for build-time packages.

    Returns:
        True if build succeeded, False otherwise.
    """
    from ash.cli.console import DockerStatus, check_docker, warn_docker_unavailable

    # Check if Docker is available
    docker_status = check_docker()
    if docker_status != DockerStatus.AVAILABLE:
        warn_docker_unavailable(docker_status)
        return False

    if not dockerfile_path.exists():
        error(f"Dockerfile not found: {dockerfile_path}")
        return False

    # Load config for build-time packages
    build_args: list[str] = []
    from ash.config import load_config
    from ash.sandbox.packages import _validate_package_names, collect_skill_packages
    from ash.skills import SkillRegistry

    try:
        cfg = load_config(config_path)  # Uses default path if None

        # Discover skill packages from workspace
        skill_apt: list[str] = []
        if cfg.workspace and cfg.workspace.exists():
            skill_registry = SkillRegistry(skill_config=cfg.skills)
            skill_registry.discover(cfg.workspace, include_bundled=False)
            skill_apt, _, _ = collect_skill_packages(skill_registry)
            if skill_apt:
                dim(f"from skills: {', '.join(skill_apt)}")

        # Merge config + skill packages
        all_apt = sorted(set(cfg.sandbox.apt_packages + skill_apt))
        valid_apt = _validate_package_names(all_apt)
        if valid_apt:
            apt_str = " ".join(valid_apt)
            build_args.extend(["--build-arg", f"EXTRA_APT_PACKAGES={apt_str}"])
            dim(f"apt packages: {apt_str}")

        valid_python = _validate_package_names(cfg.sandbox.python_packages)
        if valid_python:
            python_str = " ".join(valid_python)
            build_args.extend(["--build-arg", f"EXTRA_PYTHON_PACKAGES={python_str}"])
            dim(f"python packages: {python_str}")
    except Exception as e:
        warning(f"Could not load config: {e}")

    console.print("[bold]Building sandbox image...[/bold]")
    dim(f"Using {dockerfile_path}")
    console.print()

    # Build context is the project root (parent of docker/)
    build_context = dockerfile_path.parent.parent
    result = subprocess.run(
        [
            "docker",
            "build",
            "-t",
            "ash-sandbox:latest",
            "-f",
            str(dockerfile_path),
            *build_args,
            str(build_context),
        ],
    )

    if result.returncode == 0:
        console.print()
        success("Sandbox image built successfully!")
        console.print("You can now use the sandbox with [cyan]ash chat[/cyan]")
        return True
    else:
        console.print()
        error("Failed to build sandbox image")
        return False


def _sandbox_status() -> None:
    """Show sandbox status."""
    from ash.cli.console import DockerStatus, check_docker, create_table

    docker_running = check_docker() == DockerStatus.AVAILABLE

    # Check image
    image_exists = False
    image_info = None
    if docker_running:
        result = subprocess.run(
            [
                "docker",
                "images",
                "ash-sandbox:latest",
                "--format",
                "{{.ID}}\t{{.CreatedAt}}\t{{.Size}}",
            ],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            image_exists = True
            parts = result.stdout.strip().split("\t")
            if len(parts) >= 3:
                image_info = {"id": parts[0], "created": parts[1], "size": parts[2]}

    # Check running containers
    running_containers = 0
    if docker_running:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "ancestor=ash-sandbox:latest"],
            capture_output=True,
            text=True,
        )
        running_containers = (
            len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
        )

    table = create_table(
        "Sandbox Status",
        [
            ("Component", "cyan"),
            ("Status", "green"),
        ],
    )

    table.add_row(
        "Docker",
        "[green]Running[/green]" if docker_running else "[red]Not available[/red]",
    )
    table.add_row(
        "Sandbox Image",
        "[green]Built[/green]" if image_exists else "[yellow]Not built[/yellow]",
    )
    if image_info:
        table.add_row("  Image ID", image_info["id"][:12])
        table.add_row("  Created", image_info["created"])
        table.add_row("  Size", image_info["size"])
    table.add_row(
        "Running Containers",
        str(running_containers),
    )

    console.print(table)

    if not docker_running:
        console.print("\n[yellow]Start Docker to use the sandbox[/yellow]")
    elif not image_exists:
        console.print(
            "\n[yellow]Run 'ash sandbox build' to create the sandbox image[/yellow]"
        )


def _sandbox_clean(force: bool) -> None:
    """Clean sandbox resources."""
    console.print("[bold]Cleaning sandbox resources...[/bold]")

    # Stop and remove containers
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "ancestor=ash-sandbox:latest"],
        capture_output=True,
        text=True,
    )
    container_ids = result.stdout.strip().split("\n") if result.stdout.strip() else []

    if container_ids and container_ids[0]:
        console.print(f"Removing {len(container_ids)} container(s)...")
        subprocess.run(["docker", "rm", "-f"] + container_ids, capture_output=True)

    if force:
        # Remove image
        result = subprocess.run(
            ["docker", "rmi", "ash-sandbox:latest"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            success("Removed sandbox image")
        else:
            dim("No image to remove")
    else:
        dim("Use --force to also remove the sandbox image")

    success("Cleanup complete")
