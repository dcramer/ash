"""Drop unused sessions, messages, and tool_executions tables.

Revision ID: 005
Revises: 004
Create Date: 2026-01-11

These tables were created in 001_initial_schema.py but never used.
Sessions and messages are now stored in JSONL files (see ash.sessions module).
Tool executions are recorded in the session JSONL as ToolUseEntry/ToolResultEntry.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Drop unused tables."""
    # Drop tool_executions first (has FK to sessions)
    op.drop_index("ix_tool_executions_created_at", table_name="tool_executions")
    op.drop_index("ix_tool_executions_session_id", table_name="tool_executions")
    op.drop_table("tool_executions")

    # Drop messages (has FK to sessions)
    op.drop_index("ix_messages_created_at", table_name="messages")
    op.drop_index("ix_messages_session_id", table_name="messages")
    op.drop_table("messages")

    # Drop sessions
    op.drop_index("ix_sessions_provider_chat", table_name="sessions")
    op.drop_table("sessions")


def downgrade() -> None:
    """Recreate tables (schema from 001_initial_schema.py)."""
    # Sessions table
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("chat_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_sessions_provider_chat",
        "sessions",
        ["provider", "chat_id"],
        unique=True,
    )

    # Messages table
    op.create_table(
        "messages",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])

    # Tool executions table
    op.create_table(
        "tool_executions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("tool_name", sa.String(), nullable=False),
        sa.Column("input", sa.JSON(), nullable=False),
        sa.Column("output", sa.Text(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tool_executions_session_id", "tool_executions", ["session_id"])
    op.create_index("ix_tool_executions_created_at", "tool_executions", ["created_at"])
