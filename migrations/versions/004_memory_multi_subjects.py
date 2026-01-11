"""Convert subject_person_id to subject_person_ids JSON array.

Revision ID: 004
Revises: 003
Create Date: 2026-01-10

Allows memories to be about multiple people (e.g., "Sarah and John are getting married").
The subject_person_ids column stores a JSON array of person UUIDs.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.sqlite import JSON

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Convert subject_person_id to subject_person_ids JSON array."""
    with op.batch_alter_table("memories") as batch_op:
        # Add new JSON array column
        batch_op.add_column(sa.Column("subject_person_ids", JSON(), nullable=True))

    # Migrate existing data: wrap single ID in array
    op.execute("""
        UPDATE memories
        SET subject_person_ids = json_array(subject_person_id)
        WHERE subject_person_id IS NOT NULL
    """)

    with op.batch_alter_table("memories") as batch_op:
        # Drop index and old column
        batch_op.drop_index("ix_memories_subject_person_id")
        batch_op.drop_column("subject_person_id")


def downgrade() -> None:
    """Convert subject_person_ids back to single subject_person_id."""
    with op.batch_alter_table("memories") as batch_op:
        # Add back old column
        batch_op.add_column(sa.Column("subject_person_id", sa.String(), nullable=True))

    # Take first element from array
    op.execute("""
        UPDATE memories
        SET subject_person_id = json_extract(subject_person_ids, '$[0]')
        WHERE subject_person_ids IS NOT NULL
    """)

    with op.batch_alter_table("memories") as batch_op:
        batch_op.create_index("ix_memories_subject_person_id", ["subject_person_id"])
        batch_op.drop_column("subject_person_ids")
