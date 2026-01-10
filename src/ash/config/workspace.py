"""Workspace and personality file loading."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Workspace:
    """Loaded workspace configuration.

    Contains the SOUL (personality) and USER (user profile) documents
    that define how the assistant behaves and interacts.
    """

    path: Path
    soul: str = ""
    user: str = ""
    tools: str = ""
    custom_files: dict[str, str] = field(default_factory=dict)

    @property
    def system_prompt(self) -> str:
        """Generate system prompt from workspace files.

        Returns:
            Combined system prompt.
        """
        parts = []

        if self.soul:
            parts.append(self.soul)

        if self.user:
            parts.append(f"\n\n## User Profile\n\n{self.user}")

        if self.tools:
            parts.append(f"\n\n## Available Tools\n\n{self.tools}")

        return "\n".join(parts)


class WorkspaceLoader:
    """Load workspace configuration from directory."""

    SOUL_FILENAME = "SOUL.md"
    USER_FILENAME = "USER.md"
    TOOLS_FILENAME = "TOOLS.md"

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
            workspace.soul = self._read_file(soul_path)
            logger.debug(f"Loaded SOUL.md ({len(workspace.soul)} chars)")
        else:
            logger.warning(f"No SOUL.md found in {self._path}")

        # Load USER.md (user profile)
        user_path = self._path / self.USER_FILENAME
        if user_path.exists():
            workspace.user = self._read_file(user_path)
            logger.debug(f"Loaded USER.md ({len(workspace.user)} chars)")

        # Load TOOLS.md (tool documentation)
        tools_path = self._path / self.TOOLS_FILENAME
        if tools_path.exists():
            workspace.tools = self._read_file(tools_path)
            logger.debug(f"Loaded TOOLS.md ({len(workspace.tools)} chars)")

        return workspace

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

        # Create default USER.md if not exists
        user_path = self._path / self.USER_FILENAME
        if not user_path.exists():
            user_path.write_text(self._default_user(), encoding="utf-8")
            logger.info(f"Created default {self.USER_FILENAME}")

    @staticmethod
    def _default_soul() -> str:
        """Generate default SOUL.md content."""
        return """# Ash

You are Ash, a helpful personal assistant.

## Personality

- Friendly and approachable
- Clear and concise in communication
- Proactive in offering helpful suggestions
- Honest about limitations

## Communication Style

- Use natural, conversational language
- Be direct but polite
- Ask clarifying questions when needed
- Provide explanations when helpful

## Principles

- Respect user privacy
- Be transparent about capabilities
- Prioritize accuracy over speed
- Learn from interactions
"""

    @staticmethod
    def _default_user() -> str:
        """Generate default USER.md content."""
        return """# User Profile

## Preferences

- Language: English
- Communication style: Balanced (not too formal, not too casual)

## Notes

Add notes about the user here as you learn their preferences.
"""
