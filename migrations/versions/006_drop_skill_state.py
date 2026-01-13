"""Drop skill_state table - skill state moved to file-based storage.

Revision ID: 006
Revises: 005
Create Date: 2026-01-12

Skill state has been moved from SQLite to file-based storage at
~/.ash/data/skills/<skill-name>.json for simpler inspection and maintenance.
The table was never used in practice.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Drop skill_state table."""
    op.drop_table("skill_state")


def downgrade() -> None:
    """Recreate skill_state table."""
    op.create_table(
        "skill_state",
        sa.Column("skill_name", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False, default=""),
        sa.Column("value", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("skill_name", "key", "user_id"),
    )
