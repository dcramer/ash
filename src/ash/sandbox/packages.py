"""Aggregate package requirements from skills for sandbox setup."""

import logging
import re

from ash.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# Package name pattern: alphanumeric, dash, underscore, dot, brackets for extras
# Rejects shell metacharacters to prevent accidental command injection
_SAFE_PACKAGE_NAME = re.compile(r"^[a-zA-Z0-9._\-\[\],>=<! ]+$")


def _validate_package_names(packages: list[str]) -> list[str]:
    """Filter out package names with potentially dangerous characters.

    Args:
        packages: List of package names to validate.

    Returns:
        List of valid package names (invalid ones are logged and skipped).
    """
    valid = []
    for pkg in packages:
        if _SAFE_PACKAGE_NAME.match(pkg):
            valid.append(pkg)
        else:
            logger.warning(f"Skipping invalid package name: {pkg!r}")
    return valid


def collect_skill_packages(
    registry: SkillRegistry,
) -> tuple[list[str], list[str], list[str]]:
    """Collect system package requirements from available skills.

    Skills declare system packages via the `packages:` field in SKILL.md.
    Python dependencies should use PEP 723 inline script metadata instead.

    Args:
        registry: Skill registry to scan.

    Returns:
        Tuple of (apt_packages, python_packages, python_tools).
        Only apt_packages is populated from skill packages fields.
    """
    apt_packages: set[str] = set()

    for skill in registry.list_available():
        apt_packages.update(skill.packages)

    valid_apt = _validate_package_names(sorted(apt_packages))
    return valid_apt, [], []


def build_setup_command(
    python_packages: list[str],
    python_tools: list[str],
    base_setup_command: str | None = None,
) -> str | None:
    """Build a setup command that installs required packages.

    Combines skill package requirements with any user-configured setup command.

    Note: apt_packages cannot be installed at runtime (sandbox runs as non-root).
    They should be added to config and baked into the image at build time.

    Args:
        python_packages: Python packages to install via uv.
        python_tools: Python CLI tools (logged but run via uvx at invocation time).
        base_setup_command: User-configured setup command from config.

    Returns:
        Combined setup command, or None if nothing to do.
    """
    commands: list[str] = []

    if base_setup_command:
        commands.append(base_setup_command)

    # Validate package names to prevent shell injection
    valid_packages = _validate_package_names(python_packages)
    valid_tools = _validate_package_names(python_tools)

    if valid_packages:
        pkg_str = " ".join(valid_packages)
        commands.append(f"uv pip install --user --quiet {pkg_str}")
        logger.debug(f"Skills require python packages: {pkg_str}")

    if valid_tools:
        # python_tools run via uvx at invocation time (no install needed)
        # but we can pre-cache them
        for tool in valid_tools:
            commands.append(f"uvx --quiet {tool} --version 2>/dev/null || true")
        logger.debug(f"Skills require python tools: {', '.join(valid_tools)}")

    return " && ".join(commands) if commands else None


def warn_missing_apt_packages(apt_packages: list[str]) -> None:
    """Warn about apt packages that must be added to config.

    Apt packages cannot be installed at runtime because the sandbox
    runs as a non-root user.

    Args:
        apt_packages: List of required apt packages from skills.
    """
    if apt_packages:
        logger.warning(
            f"Skills require apt packages not installable at runtime: {', '.join(apt_packages)}. "
            f"Add to [sandbox].apt_packages in config and run 'ash sandbox build --force'."
        )
