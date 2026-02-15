"""Add covering indexes for common query patterns.

Revision ID: 008
Revises: 007
Create Date: 2026-02-14

Adds indexes to optimize frequently-used queries:
- ix_memory_subjects_person: Lookup memories by person_id
- ix_people_name_lower: Case-insensitive person name search
- ix_person_aliases_value_lower: Case-insensitive alias matching
- ix_memories_active: Filter for active (not archived/superseded) memories
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Covering index on memory_subjects for efficient person -> memory lookup
    # Includes memory_id in the index for index-only scans
    op.create_index(
        "ix_memory_subjects_person",
        "memory_subjects",
        ["person_id", "memory_id"],
    )

    # Case-insensitive name lookup on people table
    # SQLite doesn't support expression indexes natively, but this helps
    # the optimizer when doing LOWER(name) queries on non-merged people
    op.create_index(
        "ix_people_name_lower",
        "people",
        [sa.text("LOWER(name)")],
        sqlite_where=sa.text("merged_into IS NULL"),
    )

    # Case-insensitive alias value lookup
    op.create_index(
        "ix_person_aliases_value_lower",
        "person_aliases",
        [sa.text("LOWER(value)")],
    )

    # Composite index for active memories (common filter pattern)
    # Covers queries filtering by archived_at IS NULL AND superseded_at IS NULL
    op.create_index(
        "ix_memories_active",
        "memories",
        ["archived_at", "superseded_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_memories_active", table_name="memories")
    op.drop_index("ix_person_aliases_value_lower", table_name="person_aliases")
    op.drop_index("ix_people_name_lower", table_name="people")
    op.drop_index("ix_memory_subjects_person", table_name="memory_subjects")
