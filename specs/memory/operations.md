# Memory Operations

## Forget Person

When a user requests to be forgotten, or a person record needs to be purged:

1. Find all active memories with ABOUT edges to this person (`subject_person_ids` contains person_id)
2. Archive in-place: set `archived_at` + `archive_reason="forgotten"`
3. Remove from vector index
4. If `delete_person_record=True`, delete the person node

Available via:
- `memory.forget_person` RPC method
- `ash memory forget --person <id>` CLI command


## RPC Interface (Sandbox Access)

Tools running in the sandbox can access memory via RPC:

| Method | Purpose | Parameters |
|--------|---------|------------|
| `memory.search` | Semantic search | `query` (required), `limit`, `user_id`, `chat_id` |
| `memory.add` | Add a memory | `content` (required), `source`, `expires_days`, `user_id`, `chat_id`, `subjects` |
| `memory.list` | List recent memories | `limit`, `include_expired`, `user_id`, `chat_id` |
| `memory.delete` | Delete a memory | `memory_id` (required), `user_id`, `chat_id` |
| `memory.forget_person` | Archive all memories about a person | `person_id` (required), `delete_person_record` (default false) |

The sandbox CLI (`ash-sb memory`) wraps these RPC calls with a host-signed
`ASH_CONTEXT_TOKEN`. The RPC server verifies the token and injects trusted
identity/routing fields (`user_id`, `chat_id`, `chat_type`, etc.) before
authorization checks run.

## Rebuild & Repair

If the vector index is missing embeddings (e.g., after a crash or manual DB edit), it can be rebuilt:

```bash
uv run ash memory rebuild-index
```

**How rebuild works:**
1. Query active memories from the `memories` table
2. Check which memory IDs already have embeddings in `memory_embeddings`
3. Generate embeddings via API only for missing entries
4. Insert new embeddings into the vector index

### Supersession History

View the chain of memories that led to a current memory:

```bash
# Show supersession history for a memory
uv run ash memory history <memory-id>
```

## Export, Import & Backup

The database can be exported, imported, and backed up using CLI commands:

### Export

Export all data (memories, people, users, chats) to a single JSONL file:

```bash
# Export to stdout
uv run ash db export > backup.jsonl

# Export to file
uv run ash db export --output backup.jsonl
```

Each line contains a record with a `_type` field indicating the entity type.

### Import

Import data from a previously exported JSONL file:

```bash
# Replace existing data
uv run ash db import backup.jsonl

# Merge with existing data
uv run ash db import backup.jsonl --merge
```

After import, run `ash memory rebuild-index` to rebuild the vector index.

### Backup

Create an atomic backup of the SQLite vector index using `VACUUM INTO`:

```bash
# Backup to timestamped file in ~/.ash/backups/
uv run ash db backup

# Backup to specific path
uv run ash db backup --output /tmp/ash-backup.db
```

The backup is a standalone database file that can be copied or restored directly.
