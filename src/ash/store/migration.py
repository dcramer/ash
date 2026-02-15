"""Filesystem auto-migration.

Handles restructuring files from old layout to new layout.
"""

from __future__ import annotations

import logging
import shutil

from ash.config.paths import get_ash_home, get_index_dir

logger = logging.getLogger(__name__)


def migrate_filesystem() -> bool:
    """Migrate filesystem from old layout to new layout.

    Moves:
    - graph/embeddings.jsonl -> index/embeddings.jsonl
    - data/memory.db -> index/vectors.db
    - data/skills/ -> skills/state/
    - Removes empty data/ directory

    Returns:
        True if any migration was performed.
    """
    ash_home = get_ash_home()
    migrated = False

    # Move embeddings from graph/ to index/
    old_embeddings = ash_home / "graph" / "embeddings.jsonl"
    new_embeddings = get_index_dir() / "embeddings.jsonl"
    if old_embeddings.exists() and not new_embeddings.exists():
        new_embeddings.parent.mkdir(parents=True, exist_ok=True)
        old_embeddings.rename(new_embeddings)
        logger.info("Migrated embeddings.jsonl from graph/ to index/")
        migrated = True

    # Move vector DB from data/ to index/
    old_db = ash_home / "data" / "memory.db"
    new_db = get_index_dir() / "vectors.db"
    if old_db.exists() and not new_db.exists():
        new_db.parent.mkdir(parents=True, exist_ok=True)
        old_db.rename(new_db)
        logger.info("Migrated memory.db from data/ to index/vectors.db")
        migrated = True

    # Move skill state from data/skills/ to skills/state/
    old_skills = ash_home / "data" / "skills"
    new_skills = ash_home / "skills" / "state"
    if old_skills.exists() and old_skills.is_dir() and not new_skills.exists():
        new_skills.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_skills), str(new_skills))
        logger.info("Migrated skill state from data/skills/ to skills/state/")
        migrated = True

    # Remove empty data/ directory
    data_dir = ash_home / "data"
    if data_dir.exists() and data_dir.is_dir():
        try:
            data_dir.rmdir()  # Only removes if empty
            logger.info("Removed empty data/ directory")
            migrated = True
        except OSError:
            # Not empty — other files remain, leave it
            pass

    return migrated
