"""Add supersession tracking to memories table.

Revision ID: 003
Revises: 002
Create Date: 2026-01-10

When a new memory conflicts with an old one (e.g., "favorite color is blue"
supersedes "favorite color is red"), the old memory is soft-deleted by setting
superseded_at and superseded_by_id. This preserves history while keeping
retrieval clean.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add supersession columns to memories table."""
    with op.batch_alter_table("memories") as batch_op:
        batch_op.add_column(sa.Column("superseded_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("superseded_by_id", sa.String(), nullable=True))
        batch_op.create_index("ix_memories_superseded_at", ["superseded_at"])


def downgrade() -> None:
    """Remove supersession columns from memories table."""
    with op.batch_alter_table("memories") as batch_op:
        batch_op.drop_index("ix_memories_superseded_at")
        batch_op.drop_column("superseded_by_id")
        batch_op.drop_column("superseded_at")
