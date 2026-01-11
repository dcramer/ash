"""Workspace and personality file loading."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Regex to match YAML frontmatter: starts with ---, ends with ---
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# Built-in personalities that can be extended
PERSONALITIES: dict[str, str] = {
    "ash": """# Ash

You are Ash, a personal assistant inspired by Ash Ketchum from Pokemon.

## Personality

- Enthusiastic and determined - you never give up on helping
- Friendly and encouraging - you believe in the user's potential
- Action-oriented - you prefer doing over just talking
- Loyal and supportive - you're always on the user's side
- Curious and eager to learn - you love discovering new things

## Communication Style

- Energetic and positive tone
- Use encouraging phrases like "Let's do this!" or "We've got this!"
- Be direct and action-focused
- Ask clarifying questions when the path forward isn't clear
- Celebrate successes, no matter how small

## Catchphrases (use sparingly)

- "I choose you!" (when selecting a tool or approach)
- "Gotta catch 'em all!" (when gathering information)
- "Time to battle!" (when tackling a challenge)

## Principles

- Never give up - there's always a way
- Trust your instincts but verify with data
- Learn from every experience, success or failure
- Teamwork makes the dream work
- Respect boundaries and privacy
""",
}


@dataclass
class SoulConfig:
    """Configuration parsed from SOUL.md frontmatter."""

    extends: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workspace:
    """Loaded workspace configuration.

    Contains the SOUL (personality) that defines how the assistant
    behaves and interacts.
    """

    path: Path
    soul: str = ""
    soul_config: SoulConfig = field(default_factory=SoulConfig)
    custom_files: dict[str, str] = field(default_factory=dict)


class WorkspaceLoader:
    """Load workspace configuration from directory."""

    SOUL_FILENAME = "SOUL.md"

    def __init__(self, workspace_path: Path):
        """Initialize loader.

        Args:
            workspace_path: Path to workspace directory.
        """
        self._path = workspace_path.expanduser().resolve()

    @property
    def path(self) -> Path:
        """Get workspace path."""
        return self._path

    def load(self) -> Workspace:
        """Load workspace from directory.

        Returns:
            Loaded workspace.

        Raises:
            FileNotFoundError: If workspace directory doesn't exist.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Workspace directory not found: {self._path}")

        workspace = Workspace(path=self._path)

        # Load SOUL.md (personality)
        soul_path = self._path / self.SOUL_FILENAME
        if soul_path.exists():
            raw_content = self._read_file(soul_path)
            workspace.soul, workspace.soul_config = self._parse_soul(raw_content)
            logger.debug(f"Loaded SOUL.md ({len(workspace.soul)} chars)")
        else:
            # Use default personality
            workspace.soul = PERSONALITIES["ash"]
            workspace.soul_config = SoulConfig(extends="ash")
            logger.info("No SOUL.md found, using default Ash personality")

        return workspace

    def _parse_soul(self, content: str) -> tuple[str, SoulConfig]:
        """Parse SOUL.md with optional frontmatter and inheritance.

        Frontmatter format:
            ---
            extends: ash  # Inherit from built-in personality
            ---

            # Custom additions here...

        Args:
            content: Raw file content.

        Returns:
            Tuple of (final soul content, parsed config).
        """
        config = SoulConfig()
        body = content

        # Check for frontmatter
        match = FRONTMATTER_PATTERN.match(content)
        if match:
            frontmatter_yaml = match.group(1)
            body = content[match.end() :].strip()

            try:
                data = yaml.safe_load(frontmatter_yaml)
                if isinstance(data, dict):
                    config.extends = data.get("extends")
                    # Store any extra frontmatter fields
                    config.extra = {k: v for k, v in data.items() if k != "extends"}
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse SOUL.md frontmatter: {e}")

        # Build final soul content
        if config.extends:
            base_name = config.extends.lower().replace("-", "_").replace(" ", "_")
            if base_name in PERSONALITIES:
                base = PERSONALITIES[base_name]
                if body:
                    # Append custom content after base personality
                    return f"{base}\n\n{body}", config
                return base, config
            else:
                logger.warning(
                    f"Unknown personality '{config.extends}', "
                    f"available: {', '.join(PERSONALITIES.keys())}"
                )

        # No inheritance or unknown base - use content as-is
        return body or PERSONALITIES["ash"], config

    def load_custom_file(self, filename: str, workspace: Workspace) -> str | None:
        """Load a custom file from workspace.

        Args:
            filename: Name of file to load.
            workspace: Workspace to add file to.

        Returns:
            File content or None if not found.
        """
        file_path = self._path / filename
        if file_path.exists():
            content = self._read_file(file_path)
            workspace.custom_files[filename] = content
            return content
        return None

    def _read_file(self, path: Path) -> str:
        """Read file content.

        Args:
            path: File path.

        Returns:
            File content.
        """
        return path.read_text(encoding="utf-8").strip()

    def ensure_workspace(self) -> None:
        """Ensure workspace directory exists with default files."""
        self._path.mkdir(parents=True, exist_ok=True)

        # Create default SOUL.md if not exists
        soul_path = self._path / self.SOUL_FILENAME
        if not soul_path.exists():
            soul_path.write_text(self._default_soul(), encoding="utf-8")
            logger.info(f"Created default {self.SOUL_FILENAME}")

    @staticmethod
    def _default_soul() -> str:
        """Generate default SOUL.md content."""
        return """---
extends: ash
---

# Customizations

Add your personality customizations here. They will be appended
to the base Ash personality.
"""
