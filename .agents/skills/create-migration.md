# /create-migration

Create a new database migration using Alembic.

## Usage

```
/create-migration <description>
```

Example: `/create-migration add status column to memories`

## Process

1. **Check current state**:
   ```bash
   ls migrations/versions/*.py | sort | tail -1
   ```

2. **Update the model** in `src/ash/db/models.py` if not already done

3. **Create migration file** in `migrations/versions/<NNN>_<snake_case_desc>.py`

## Migration Template

```python
"""<Description of what this migration does>.

Revision ID: <NNN>
Revises: <NNN-1>
Create Date: YYYY-MM-DD

<Brief context about why this migration is needed>
"""

from alembic import op
import sqlalchemy as sa

revision = "<NNN>"
down_revision = "<NNN-1>"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """<Describe upgrade>."""
    with op.batch_alter_table("<table>") as batch_op:
        batch_op.add_column(
            sa.Column("<column>", sa.<Type>(), nullable=True)
        )
        batch_op.create_index("ix_<table>_<column>", ["<column>"])


def downgrade() -> None:
    """<Describe downgrade>."""
    with op.batch_alter_table("<table>") as batch_op:
        batch_op.drop_index("ix_<table>_<column>")
        batch_op.drop_column("<column>")
```

## Rules

- **ALWAYS use batch mode** for SQLite: `with op.batch_alter_table()`
- **Keep migrations simple**: one logical change per migration
- **Include indexes** when adding columns used in WHERE/ORDER BY
- **Match model exactly**: migration column types must match `models.py`
- **Number sequentially**: 001, 002, 003...
- **Remind user to run**: `uv run alembic upgrade head`

## Common Operations

### Add nullable column
```python
with op.batch_alter_table("memories") as batch_op:
    batch_op.add_column(sa.Column("new_col", sa.String(), nullable=True))
```

### Add column with default
```python
with op.batch_alter_table("memories") as batch_op:
    batch_op.add_column(
        sa.Column("status", sa.String(), nullable=False, server_default="active")
    )
```

### Create new table
```python
op.create_table(
    "new_table",
    sa.Column("id", sa.String(), primary_key=True),
    sa.Column("name", sa.String(), nullable=False),
    sa.Column("created_at", sa.DateTime(), nullable=False),
)
op.create_index("ix_new_table_name", "new_table", ["name"])
```

### Add foreign key
```python
with op.batch_alter_table("child_table") as batch_op:
    batch_op.add_column(sa.Column("parent_id", sa.String(), nullable=True))
    batch_op.create_foreign_key(
        "fk_child_parent", "parent_table", ["parent_id"], ["id"]
    )
```

## Notes

- Database path: `~/.ash/data/memory.db`
- Virtual tables (sqlite-vec) are NOT managed by migrations - they're created at runtime
