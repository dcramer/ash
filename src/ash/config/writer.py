"""Configuration writer for modifying config.toml while preserving formatting.

Uses tomlkit to preserve comments, formatting, and ordering in TOML files.
"""

import logging
from pathlib import Path

import tomlkit
from tomlkit import TOMLDocument, aot, table

from ash.config.paths import get_config_path

logger = logging.getLogger(__name__)


class ConfigWriter:
    """Writer for modifying config.toml while preserving formatting."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or get_config_path()
        self._doc: TOMLDocument | None = None

    def _load(self) -> TOMLDocument:
        """Load the config file, creating it if it doesn't exist."""
        if self._doc is not None:
            return self._doc

        if self.config_path.exists():
            content = self.config_path.read_text()
            self._doc = tomlkit.parse(content)
        else:
            self._doc = tomlkit.document()

        return self._doc

    def _save(self) -> None:
        """Save the config file."""
        if self._doc is None:
            return

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(tomlkit.dumps(self._doc))
        logger.debug(f"Saved config to {self.config_path}")

    def _ensure_skills_table(self, doc: TOMLDocument):
        """Ensure [skills] table exists and return it."""
        if "skills" not in doc:
            doc["skills"] = table()
        return doc["skills"]

    def add_skill_source(
        self,
        *,
        repo: str | None = None,
        path: str | None = None,
        ref: str | None = None,
    ) -> bool:
        """Add a skill source to [[skills.sources]].

        Args:
            repo: GitHub repo in owner/repo format
            path: Local path (~/... or /...)
            ref: Git ref (branch/tag/commit) - only for repos

        Returns:
            True if added, False if already exists.
        """
        if not repo and not path:
            raise ValueError("Must specify either 'repo' or 'path'")
        if repo and path:
            raise ValueError("Cannot specify both 'repo' and 'path'")

        doc = self._load()
        skills = self._ensure_skills_table(doc)

        # Get or create sources array
        if "sources" not in skills:
            skills["sources"] = aot()
        sources = skills["sources"]

        # Check if source already exists
        for source in sources:
            if repo and source.get("repo") == repo:
                logger.debug(f"Source already exists: repo={repo}")
                return False
            if path and source.get("path") == path:
                logger.debug(f"Source already exists: path={path}")
                return False

        # Add new source
        new_source = table()
        if repo:
            new_source["repo"] = repo
            if ref:
                new_source["ref"] = ref
        else:
            new_source["path"] = path

        sources.append(new_source)
        self._save()
        logger.info("skill_source_added", extra={"skill.source": repo or str(path)})
        return True

    def remove_skill_source(
        self,
        *,
        repo: str | None = None,
        path: str | None = None,
    ) -> bool:
        """Remove a skill source from [[skills.sources]].

        Args:
            repo: GitHub repo to remove
            path: Local path to remove

        Returns:
            True if removed, False if not found.
        """
        if not repo and not path:
            raise ValueError("Must specify either 'repo' or 'path'")

        doc = self._load()
        skills = doc.get("skills")
        if not skills:
            return False

        sources = skills.get("sources")
        if not sources:
            return False

        # Find and remove the source
        for i, source in enumerate(sources):
            if repo and source.get("repo") == repo:
                del sources[i]
                self._save()
                logger.info("skill_source_removed", extra={"skill.source": repo})
                return True
            if path and source.get("path") == path:
                del sources[i]
                self._save()
                logger.info("skill_source_removed", extra={"skill.source": str(path)})
                return True

        return False

    def list_skill_sources(self) -> list[dict[str, str | None]]:
        """List all skill sources from [[skills.sources]].

        Returns:
            List of source dicts with repo/path/ref keys.
        """
        doc = self._load()
        sources = doc.get("skills", {}).get("sources", [])
        return [
            {
                key: str(s[key]) if s.get(key) else None
                for key in ("repo", "path", "ref")
            }
            for s in sources
        ]

    def update_skill_source_ref(self, repo: str, ref: str | None) -> bool:
        """Update the ref for a repo source.

        Args:
            repo: GitHub repo to update
            ref: New ref (branch/tag/commit) or None to remove

        Returns:
            True if updated, False if not found.
        """
        doc = self._load()
        skills = doc.get("skills")
        if not skills:
            return False

        sources = skills.get("sources")
        if not sources:
            return False

        for source in sources:
            if source.get("repo") == repo:
                if ref:
                    source["ref"] = ref
                elif "ref" in source:
                    del source["ref"]
                self._save()
                logger.info(
                    "skill_source_ref_updated", extra={"skill.source": repo, "ref": ref}
                )
                return True

        return False
