# Memory Verification

## Verification

```bash
# Unit tests
uv run pytest tests/test_memory.py tests/test_memory_extractor.py tests/test_people.py -v

# Integration test
uv run ash chat "Remember my favorite color is blue"
uv run ash chat "What's my favorite color?"
# Should answer "blue"

uv run ash chat "Remember my wife Sarah likes Italian food"
uv run ash chat "What does my wife like?"
# Should mention Italian food and attribute to Sarah

# CLI inspection
uv run ash memory list
uv run ash memory search "favorite"
uv run ash memory gc

# Verify storage via CLI
uv run ash memory list
uv run ash db export | head -5

# Check supersession removes old memory
uv run ash chat "Remember my favorite color is red"
uv run ash memory search "favorite color"  # should show red
uv run ash chat "Remember my favorite color is blue"
uv run ash memory gc
uv run ash memory search "favorite color"  # should show only blue

# Rebuild index for missing embeddings
uv run ash memory rebuild-index

# Secrets filtering tests:
# 1. Via CLI - should reject
uv run ash memory add -q "my password is hunter2"
# Should error: "Memory content contains potential secrets"

# 2. Via CLI with API key - should reject
uv run ash memory add -q "API key: sk-abc123def456789"
# Should error: "Memory content contains potential secrets"

# 3. Normal content should work
uv run ash memory add -q "I prefer dark mode"
uv run ash memory search "dark mode"  # should find it

# Export/Import/Backup tests:
# 1. Export data
uv run ash db export --output /tmp/ash-export.jsonl
head -5 /tmp/ash-export.jsonl

# 2. Import data (with merge)
uv run ash db import /tmp/ash-export.jsonl --merge

# 3. Create backup
uv run ash db backup --output /tmp/ash-backup.db
ls -la /tmp/ash-backup.db
```

### Checklist

- [ ] Agent answers questions using previously stored facts
- [ ] `remember` tool stores facts when user explicitly requests
- [ ] Automatic context injection retrieves relevant memories before each response
- [ ] Person entities created for "my wife", "Sarah", etc.
- [ ] Known people appear in system prompt
- [ ] Memory results show subject attribution ("about Sarah")
- [ ] Conflicting facts supersede old versions (with LLM verification)
- [ ] Superseded facts archived but excluded from retrieval
- [ ] `max_entries` evicts oldest when exceeded
- [ ] `ash memory gc` removes expired/superseded/decayed entries
- [ ] Background extraction captures facts from conversations
- [ ] Extraction assigns memory types (preference, identity, etc.)
- [ ] Extraction skips low-confidence and duplicate facts
- [ ] Ephemeral memory types decay based on their TTL (from `observed_at`)
- [ ] SQLite database is the source of truth
- [ ] `ash memory rebuild-index` generates embeddings for missing entries
- [ ] `ash memory history <id>` shows supersession chain
- [ ] Extracted memories have `source_username` populated
- [ ] `ash memory list` shows About, Source, and Trust columns
- [ ] `ash memory show <id>` displays full attribution details
- [ ] Extraction classifies sensitivity (public/personal/sensitive)
- [ ] Sensitive memories not disclosed in group chats
- [ ] Personal memories only shown to subject or owner
- [ ] Cross-context retrieval finds facts from other owners
- [ ] Privacy filter applied to cross-context memories
- [ ] Secrets rejected via CLI: `ash memory add -q "password is hunter2"` → error
- [ ] Secrets rejected via RPC: agent storing API key → error
- [ ] Secrets filtered during extraction (LLM output post-filtered)
- [ ] Relative times converted to absolute dates during extraction
- [ ] `observed_at` field populated on extracted memories
- [ ] `ash memory search <query>` performs semantic search via MemoryManager
- [ ] `ash memory add` generates embeddings and indexes via MemoryManager
- [ ] `ash db export` exports all data to JSONL format
- [ ] `ash db import` imports data from exported JSONL
- [ ] `ash db backup` creates atomic SQLite backup with VACUUM INTO
