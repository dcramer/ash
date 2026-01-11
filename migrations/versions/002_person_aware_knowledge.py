"""Add person-aware knowledge.

Revision ID: 002
Revises: 001
Create Date: 2026-01-10

Adds Person model to track people mentioned by users, and links
knowledge entries to specific people they are about.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "002"
down_revision: str = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_exists(table_name: str) -> bool:
    """Check if a table exists."""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def _column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    # Create people table if it doesn't exist
    if not _table_exists("people"):
        op.create_table(
            "people",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("owner_user_id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("relation", sa.String(), nullable=True),
            sa.Column("aliases", sa.JSON(), nullable=True),
            sa.Column("metadata", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_people_owner_user_id", "people", ["owner_user_id"])
        op.create_index("ix_people_name", "people", ["name"])

    # Add columns to knowledge table if they don't exist
    # Using batch mode for SQLite compatibility with foreign key
    if not _column_exists("knowledge", "owner_user_id"):
        with op.batch_alter_table("knowledge") as batch_op:
            batch_op.add_column(sa.Column("owner_user_id", sa.String(), nullable=True))
            batch_op.add_column(sa.Column("subject_person_id", sa.String(), nullable=True))
            batch_op.create_index("ix_knowledge_owner_user_id", ["owner_user_id"])
            batch_op.create_index("ix_knowledge_subject_person_id", ["subject_person_id"])
            batch_op.create_foreign_key(
                "fk_knowledge_subject_person",
                "people",
                ["subject_person_id"],
                ["id"],
                ondelete="SET NULL",
            )
    else:
        # Columns exist, just ensure foreign key is set up
        # This handles partial migration states
        with op.batch_alter_table("knowledge") as batch_op:
            batch_op.create_foreign_key(
                "fk_knowledge_subject_person",
                "people",
                ["subject_person_id"],
                ["id"],
                ondelete="SET NULL",
            )


def downgrade() -> None:
    # Remove columns from knowledge table using batch mode for SQLite compatibility
    with op.batch_alter_table("knowledge") as batch_op:
        batch_op.drop_constraint("fk_knowledge_subject_person", type_="foreignkey")
        batch_op.drop_index("ix_knowledge_subject_person_id")
        batch_op.drop_index("ix_knowledge_owner_user_id")
        batch_op.drop_column("subject_person_id")
        batch_op.drop_column("owner_user_id")

    op.drop_index("ix_people_name", "people")
    op.drop_index("ix_people_owner_user_id", "people")
    op.drop_table("people")
