"""Add chat_id column to memories table for group memory scoping.

Revision ID: 002
Revises: 001
Create Date: 2025-01-10

Memory scoping:
- Personal: owner_user_id set, chat_id NULL - only visible to that user
- Group: owner_user_id NULL, chat_id set - visible to everyone in that chat
- Global: both NULL - visible everywhere (rare)
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add chat_id column to memories table."""
    with op.batch_alter_table("memories") as batch_op:
        batch_op.add_column(sa.Column("chat_id", sa.String(), nullable=True))
        batch_op.create_index("ix_memories_chat_id", ["chat_id"])


def downgrade() -> None:
    """Remove chat_id column from memories table."""
    with op.batch_alter_table("memories") as batch_op:
        batch_op.drop_index("ix_memories_chat_id")
        batch_op.drop_column("chat_id")
