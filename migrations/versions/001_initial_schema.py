"""Initial schema.

Revision ID: 001
Revises:
Create Date: 2026-01-10

Complete database schema with sessions, messages, memories, people,
user profiles, tool executions, and skill state.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
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

    # People table (for person-aware memory)
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

    # Memories table (facts and preferences)
    op.create_table(
        "memories",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("owner_user_id", sa.String(), nullable=True),
        sa.Column("subject_person_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["subject_person_id"],
            ["people.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memories_owner_user_id", "memories", ["owner_user_id"])
    op.create_index("ix_memories_subject_person_id", "memories", ["subject_person_id"])

    # User profiles table
    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("username", sa.String(), nullable=True),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("profile_data", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )

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

    # Skill state table
    op.create_table(
        "skill_state",
        sa.Column("skill_name", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False, default=""),
        sa.Column("value", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("skill_name", "key", "user_id"),
    )


def downgrade() -> None:
    op.drop_table("skill_state")
    op.drop_table("tool_executions")
    op.drop_table("user_profiles")
    op.drop_table("memories")
    op.drop_table("people")
    op.drop_table("messages")
    op.drop_table("sessions")
