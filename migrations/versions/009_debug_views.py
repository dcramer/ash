"""Add debug views for introspection.

Revision ID: 009
Revises: 008
Create Date: 2026-02-14

Adds SQL views for debugging and introspection:
- v_active_memories: Active memories with aggregated subject names
- v_person_detail: Person records with alias/relationship counts
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # View for active memories with subject person names
    op.execute("""
        CREATE VIEW IF NOT EXISTS v_active_memories AS
        SELECT
            m.id,
            m.content,
            m.memory_type,
            m.source,
            m.owner_user_id,
            m.chat_id,
            m.created_at,
            m.observed_at,
            m.expires_at,
            m.sensitivity,
            m.portable,
            (
                SELECT GROUP_CONCAT(p.name, ', ')
                FROM memory_subjects ms
                JOIN people p ON p.id = ms.person_id
                WHERE ms.memory_id = m.id
            ) AS subject_names,
            (
                SELECT COUNT(*)
                FROM memory_subjects ms
                WHERE ms.memory_id = m.id
            ) AS subject_count
        FROM memories m
        WHERE m.archived_at IS NULL
          AND m.superseded_at IS NULL
          AND (m.expires_at IS NULL OR m.expires_at > datetime('now'))
        ORDER BY m.created_at DESC
    """)

    # View for person details with alias and relationship counts
    op.execute("""
        CREATE VIEW IF NOT EXISTS v_person_detail AS
        SELECT
            p.id,
            p.name,
            p.created_by,
            p.merged_into,
            p.created_at,
            p.updated_at,
            (
                SELECT GROUP_CONCAT(pa.value, ', ')
                FROM person_aliases pa
                WHERE pa.person_id = p.id
            ) AS aliases,
            (
                SELECT COUNT(*)
                FROM person_aliases pa
                WHERE pa.person_id = p.id
            ) AS alias_count,
            (
                SELECT GROUP_CONCAT(pr.relationship, ', ')
                FROM person_relationships pr
                WHERE pr.person_id = p.id
            ) AS relationships,
            (
                SELECT COUNT(*)
                FROM person_relationships pr
                WHERE pr.person_id = p.id
            ) AS relationship_count,
            (
                SELECT COUNT(*)
                FROM memory_subjects ms
                JOIN memories m ON m.id = ms.memory_id
                WHERE ms.person_id = p.id AND m.archived_at IS NULL
            ) AS active_memory_count
        FROM people p
        WHERE p.merged_into IS NULL
        ORDER BY p.name
    """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS v_person_detail")
    op.execute("DROP VIEW IF EXISTS v_active_memories")
