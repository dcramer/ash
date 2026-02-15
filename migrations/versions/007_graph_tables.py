"""Create graph tables for SQLite-backed graph store.

Revision ID: 007
Revises: 006
Create Date: 2026-02-14

Replaces JSONL-based storage for memories, people, users, and chats
with typed SQLite tables. Also drops the legacy user_profiles table
(consolidated into the new users table).
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop legacy tables that are being replaced
    # (memories and people were from the original schema, user_profiles added later)
    op.drop_table("memories")
    op.drop_table("people")
    op.drop_table("user_profiles")

    # -- Node tables --

    op.create_table(
        "memories",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("memory_type", sa.Text(), nullable=False, server_default="knowledge"),
        sa.Column("source", sa.Text(), nullable=False, server_default="user"),
        sa.Column("owner_user_id", sa.Text(), nullable=True),
        sa.Column("chat_id", sa.Text(), nullable=True),
        sa.Column("source_username", sa.Text(), nullable=True),
        sa.Column("source_display_name", sa.Text(), nullable=True),
        sa.Column("source_session_id", sa.Text(), nullable=True),
        sa.Column("source_message_id", sa.Text(), nullable=True),
        sa.Column("extraction_confidence", sa.Float(), nullable=True),
        sa.Column("sensitivity", sa.Text(), nullable=True),
        sa.Column("portable", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("observed_at", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.Text(), nullable=True),
        sa.Column("superseded_at", sa.Text(), nullable=True),
        sa.Column(
            "superseded_by_id",
            sa.Text(),
            sa.ForeignKey("memories.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("archived_at", sa.Text(), nullable=True),
        sa.Column("archive_reason", sa.Text(), nullable=True),
        sa.Column("metadata", sa.Text(), nullable=True),  # JSON
    )
    op.create_index(
        "ix_mem_owner",
        "memories",
        ["owner_user_id"],
        sqlite_where=sa.text("archived_at IS NULL"),
    )
    op.create_index(
        "ix_mem_chat",
        "memories",
        ["chat_id"],
        sqlite_where=sa.text("archived_at IS NULL"),
    )
    op.create_index(
        "ix_mem_created",
        "memories",
        ["created_at"],
        sqlite_where=sa.text("archived_at IS NULL"),
    )
    op.create_index(
        "ix_mem_active",
        "memories",
        ["id"],
        sqlite_where=sa.text("archived_at IS NULL AND superseded_at IS NULL"),
    )

    op.create_table(
        "people",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_by", sa.Text(), nullable=False, server_default=""),
        sa.Column("name", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "merged_into",
            sa.Text(),
            sa.ForeignKey("people.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("metadata", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_people_name",
        "people",
        ["name"],
        sqlite_where=sa.text("merged_into IS NULL"),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("provider", sa.Text(), nullable=False, server_default=""),
        sa.Column("provider_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("username", sa.Text(), nullable=True),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column(
            "person_id",
            sa.Text(),
            sa.ForeignKey("people.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("metadata", sa.Text(), nullable=True),
    )
    op.create_index("ix_users_prov", "users", ["provider", "provider_id"], unique=True)
    op.create_index(
        "ix_users_uname",
        "users",
        ["username"],
        sqlite_where=sa.text("username IS NOT NULL"),
    )

    op.create_table(
        "chats",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("provider", sa.Text(), nullable=False, server_default=""),
        sa.Column("provider_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("chat_type", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("metadata", sa.Text(), nullable=True),
    )
    op.create_index("ix_chats_prov", "chats", ["provider", "provider_id"], unique=True)

    # -- Relationship/edge tables --

    op.create_table(
        "memory_subjects",
        sa.Column(
            "memory_id",
            sa.Text(),
            sa.ForeignKey("memories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "person_id",
            sa.Text(),
            sa.ForeignKey("people.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("memory_id", "person_id"),
    )
    op.create_index("ix_memsub_person", "memory_subjects", ["person_id"])

    op.create_table(
        "person_relationships",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "person_id",
            sa.Text(),
            sa.ForeignKey("people.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relationship", sa.Text(), nullable=False),
        sa.Column("stated_by", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=True),
    )
    op.create_index("ix_prel_person", "person_relationships", ["person_id"])
    op.create_index("ix_prel_rel", "person_relationships", ["relationship"])
    op.create_index("ix_prel_stated", "person_relationships", ["stated_by"])

    op.create_table(
        "person_aliases",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "person_id",
            sa.Text(),
            sa.ForeignKey("people.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("added_by", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=True),
    )
    op.create_index("ix_palias_person", "person_aliases", ["person_id"])
    op.create_index("ix_palias_value", "person_aliases", ["value"])


def downgrade() -> None:
    # Drop new tables
    op.drop_table("person_aliases")
    op.drop_table("person_relationships")
    op.drop_table("memory_subjects")
    op.drop_table("chats")
    op.drop_table("users")
    op.drop_table("people")
    op.drop_table("memories")

    # Recreate legacy tables
    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.String(), primary_key=True),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("username", sa.String(), nullable=True),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("profile_data", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "people",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("owner_user_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("relation", sa.String(), nullable=True),
        sa.Column("aliases", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "memories",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("owner_user_id", sa.String(), nullable=True),
        sa.Column("chat_id", sa.String(), nullable=True),
        sa.Column("subject_person_ids", sa.JSON(), nullable=True),
        sa.Column("superseded_at", sa.DateTime(), nullable=True),
        sa.Column(
            "superseded_by_id",
            sa.String(),
            sa.ForeignKey("memories.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
