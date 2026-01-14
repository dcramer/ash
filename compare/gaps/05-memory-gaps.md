# Memory System Gap Analysis

This document analyzes potential improvements to Ash's memory system, comparing it with the simpler MEMORY.md approach used in Archer and pi-mono/mom.

**Note:** Ash is **ahead** on memory capabilities with SQLite + sqlite-vec for semantic search, supersession tracking, and person entity resolution. The reference implementations use simple markdown files. This analysis focuses on features Ash could add to enhance its existing system.

**Files Analyzed:**
- Ash: `/home/dcramer/src/ash/src/ash/memory/manager.py`
- Ash: `/home/dcramer/src/ash/src/ash/memory/store.py`
- Ash: `/home/dcramer/src/ash/src/ash/memory/retrieval.py`
- Ash: `/home/dcramer/src/ash/src/ash/memory/extractor.py`
- Ash: `/home/dcramer/src/ash/src/ash/memory/types.py`
- Ash: `/home/dcramer/src/ash/src/ash/cli/commands/memory.py`
- Ash: `/home/dcramer/src/ash/src/ash/db/models.py`
- Archer: `/home/dcramer/src/archer/src/agent.ts` (MEMORY.md approach)
- Pi-mono: `/home/dcramer/src/pi-mono/packages/mom/src/agent.ts` (MEMORY.md approach)

---

## Gap 1: MEMORY.md Fallback Mode

### What Ash is Missing

Ash requires SQLite + sqlite-vec for memory storage. While more powerful, this creates deployment complexity for simple use cases. Archer and pi-mono use plain MEMORY.md files that the agent reads/writes directly:

```typescript
// archer/src/agent.ts lines 67-101
function getMemory(channelDir: string): string {
    const parts: string[] = [];

    // Read workspace-level memory (shared across all channels)
    const workspaceMemoryPath = join(channelDir, "..", "MEMORY.md");
    if (existsSync(workspaceMemoryPath)) {
        try {
            const content = readFileSync(workspaceMemoryPath, "utf-8").trim();
            if (content) {
                parts.push(`### Global Workspace Memory\n${content}`);
            }
        } catch (error) {
            log.logWarning("Failed to read workspace memory", `${workspaceMemoryPath}: ${error}`);
        }
    }

    // Read channel-specific memory
    const channelMemoryPath = join(channelDir, "MEMORY.md");
    if (existsSync(channelMemoryPath)) {
        try {
            const content = readFileSync(channelMemoryPath, "utf-8").trim();
            if (content) {
                parts.push(`### Channel-Specific Memory\n${content}`);
            }
        } catch (error) {
            log.logWarning("Failed to read channel memory", `${channelMemoryPath}: ${error}`);
        }
    }

    return parts.length === 0 ? "(no working memory yet)" : parts.join("\n\n");
}
```

### Why It Matters

- **Simpler deployments**: No database setup required for basic use cases
- **Human-editable**: Users can directly edit MEMORY.md files
- **Debuggable**: Memory state is visible in plain text
- **Portable**: Easy to backup, version control, or move between systems
- **Offline-capable**: Works without embedding API access

### Files to Modify

- `/home/dcramer/src/ash/src/ash/memory/__init__.py` - Add fallback store
- `/home/dcramer/src/ash/src/ash/memory/file_store.py` - New file-based store
- `/home/dcramer/src/ash/src/ash/config/models.py` - Add storage mode config

### Concrete Python Code Changes

```python
# New file: src/ash/memory/file_store.py
"""File-based memory store using MEMORY.md files.

A simpler alternative to SQLite storage for deployments that don't
need semantic search or complex querying.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileMemoryStore:
    """Store memories in markdown files.

    Structure:
    - ~/.ash/MEMORY.md - Global memories
    - ~/.ash/users/{user_id}/MEMORY.md - User-specific memories
    - ~/.ash/chats/{chat_id}/MEMORY.md - Chat-specific memories
    """

    def __init__(self, base_dir: Path):
        """Initialize file-based memory store.

        Args:
            base_dir: Base directory for memory files (e.g., ~/.ash)
        """
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _get_memory_path(
        self,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> Path:
        """Get path to memory file based on scope."""
        if owner_user_id:
            return self._base_dir / "users" / owner_user_id / "MEMORY.md"
        elif chat_id:
            return self._base_dir / "chats" / chat_id / "MEMORY.md"
        else:
            return self._base_dir / "MEMORY.md"

    def _read_memory_file(self, path: Path) -> str:
        """Read memory file content."""
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            logger.warning("Failed to read memory file: %s", path, exc_info=True)
            return ""

    def _write_memory_file(self, path: Path, content: str) -> None:
        """Write content to memory file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _parse_memories(self, content: str) -> list[dict[str, Any]]:
        """Parse memory entries from markdown content.

        Format:
        ## Memory Entry
        - **Added**: 2025-01-13
        - **Source**: user

        Content of the memory here.

        ---
        """
        memories = []
        if not content.strip():
            return memories

        # Split by separator
        entries = content.split("\n---\n")
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            memory = {"content": "", "metadata": {}}
            lines = entry.split("\n")
            content_lines = []
            in_content = False

            for line in lines:
                if line.startswith("## "):
                    continue
                elif line.startswith("- **Added**:"):
                    memory["created_at"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Source**:"):
                    memory["source"] = line.split(":", 1)[1].strip()
                elif line.startswith("- **Expires**:"):
                    memory["expires_at"] = line.split(":", 1)[1].strip()
                elif line.strip():
                    content_lines.append(line)

            memory["content"] = "\n".join(content_lines).strip()
            if memory["content"]:
                memories.append(memory)

        return memories

    def _format_memory(
        self,
        content: str,
        source: str | None = None,
        expires_at: datetime | None = None,
    ) -> str:
        """Format a memory entry as markdown."""
        lines = ["## Memory Entry"]
        lines.append(f"- **Added**: {datetime.now(UTC).strftime('%Y-%m-%d')}")
        if source:
            lines.append(f"- **Source**: {source}")
        if expires_at:
            lines.append(f"- **Expires**: {expires_at.strftime('%Y-%m-%d')}")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        return "\n".join(lines)

    async def add_memory(
        self,
        content: str,
        source: str | None = None,
        expires_at: datetime | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Add a memory entry."""
        path = self._get_memory_path(owner_user_id, chat_id)
        existing = self._read_memory_file(path)

        new_entry = self._format_memory(content, source, expires_at)

        if existing.strip():
            updated = f"{existing.strip()}\n\n{new_entry}"
        else:
            updated = new_entry

        self._write_memory_file(path, updated)

        return {
            "id": f"file-{datetime.now(UTC).timestamp()}",
            "content": content,
            "source": source,
            "expires_at": expires_at,
        }

    async def get_memories(
        self,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Get all memories for the given scope."""
        path = self._get_memory_path(owner_user_id, chat_id)
        content = self._read_memory_file(path)
        return self._parse_memories(content)

    def get_memory_text(
        self,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> str:
        """Get raw memory text for prompt injection."""
        parts = []

        # Always include global memory
        global_path = self._get_memory_path()
        global_content = self._read_memory_file(global_path)
        if global_content.strip():
            parts.append(f"### Global Memory\n{global_content}")

        # Include user-specific memory if applicable
        if owner_user_id:
            user_path = self._get_memory_path(owner_user_id=owner_user_id)
            user_content = self._read_memory_file(user_path)
            if user_content.strip():
                parts.append(f"### Personal Memory\n{user_content}")

        # Include chat-specific memory if applicable
        if chat_id:
            chat_path = self._get_memory_path(chat_id=chat_id)
            chat_content = self._read_memory_file(chat_path)
            if chat_content.strip():
                parts.append(f"### Chat Memory\n{chat_content}")

        return "\n\n".join(parts) if parts else "(no memories yet)"
```

```python
# Add to src/ash/config/models.py

class MemoryConfig(BaseModel):
    """Memory subsystem configuration."""

    storage_mode: Literal["sqlite", "file"] = "sqlite"
    """Storage backend: 'sqlite' for full features, 'file' for simple MEMORY.md"""

    file_storage_path: Path | None = None
    """Path for file-based storage (defaults to ~/.ash)"""

    # ... existing fields ...
```

### Effort Estimate

**Size: Medium**
- New file store implementation: ~200 lines
- Config changes: ~20 lines
- Integration with existing MemoryManager: ~50 lines
- Tests: ~100 lines

**Priority: Low**
- Nice-to-have for simple deployments
- SQLite mode is already well-implemented
- Could be useful for CLI-only usage without server

---

## Gap 2: Memory Source Attribution

### What Ash is Missing

Ash tracks `source` as a simple string (e.g., "user", "extraction") but doesn't track the specific conversation or session that produced each memory. This makes it hard to audit where memories came from.

Current Memory model (db/models.py):
```python
class Memory(Base):
    # ...
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    # No session_id, conversation_id, or message_id tracking
```

### Why It Matters

- **Auditability**: Users want to know "when did I tell you that?"
- **Trust**: Ability to trace memories back to source conversation
- **Debugging**: Helps diagnose incorrect memories
- **Cleanup**: Delete all memories from a specific session

### Files to Modify

- `/home/dcramer/src/ash/src/ash/db/models.py` - Add source fields
- `/home/dcramer/src/ash/src/ash/memory/store.py` - Pass through source info
- `/home/dcramer/src/ash/src/ash/memory/manager.py` - Accept source context
- `/home/dcramer/src/ash/alembic/versions/` - New migration

### Concrete Python Code Changes

```python
# Update db/models.py Memory class

class Memory(Base):
    """Memory entry - a stored fact or piece of information."""

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True
    )

    # Owner and scope tracking (existing)
    owner_user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    chat_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    subject_person_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # NEW: Source attribution tracking
    source_session_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    """Session ID where this memory was created"""

    source_message_index: Mapped[int | None] = mapped_column(
        nullable=True
    )
    """Index of the message in the session that triggered extraction"""

    # Supersession tracking (existing)
    superseded_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, index=True
    )
    superseded_by_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("memories.id", ondelete="SET NULL"), nullable=True
    )
```

```python
# Update memory/store.py add_memory method

async def add_memory(
    self,
    content: str,
    source: str | None = None,
    expires_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    owner_user_id: str | None = None,
    chat_id: str | None = None,
    subject_person_ids: list[str] | None = None,
    # NEW: Source attribution
    source_session_id: str | None = None,
    source_message_index: int | None = None,
) -> Memory:
    """Add a memory entry.

    Args:
        content: Memory content.
        source: Source type (user, extraction, tool, etc.).
        expires_at: When this memory expires.
        metadata: Optional metadata.
        owner_user_id: User who added this memory.
        chat_id: Chat this memory belongs to.
        subject_person_ids: List of person IDs this memory is about.
        source_session_id: Session ID where this memory was created.
        source_message_index: Message index in session that triggered creation.

    Returns:
        Created memory entry.
    """
    # Validate subject_person_ids exist
    if subject_person_ids:
        for person_id in subject_person_ids:
            person = await self.get_person(person_id)
            if not person:
                raise ValueError(f"Invalid subject person ID: {person_id}")

    memory = Memory(
        id=str(uuid.uuid4()),
        content=content,
        source=source,
        expires_at=expires_at,
        metadata_=metadata,
        owner_user_id=owner_user_id,
        chat_id=chat_id,
        subject_person_ids=subject_person_ids,
        source_session_id=source_session_id,
        source_message_index=source_message_index,
    )
    self._session.add(memory)
    await self._session.flush()
    return memory
```

```python
# Update memory/manager.py add_memory method signature

async def add_memory(
    self,
    content: str,
    source: str = "user",
    expires_at: datetime | None = None,
    expires_in_days: int | None = None,
    owner_user_id: str | None = None,
    chat_id: str | None = None,
    subject_person_ids: list[str] | None = None,
    # NEW: Source attribution
    source_session_id: str | None = None,
    source_message_index: int | None = None,
) -> Memory:
    """Add memory entry (used by remember tool).

    Args:
        content: Memory content.
        source: Source of memory (default: "user").
        expires_at: Explicit expiration datetime.
        expires_in_days: Days until expiration.
        owner_user_id: User who added this memory.
        chat_id: Chat this memory belongs to.
        subject_person_ids: List of person IDs this memory is about.
        source_session_id: Session ID where memory was created.
        source_message_index: Message index that triggered creation.

    Returns:
        Created memory entry.
    """
    # Calculate expiration if days provided
    if expires_in_days is not None and expires_at is None:
        expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

    # Store memory
    memory = await self._store.add_memory(
        content=content,
        source=source,
        expires_at=expires_at,
        owner_user_id=owner_user_id,
        chat_id=chat_id,
        subject_person_ids=subject_person_ids,
        source_session_id=source_session_id,
        source_message_index=source_message_index,
    )
    # ... rest of method unchanged
```

```python
# New migration: alembic/versions/xxxx_add_memory_source_attribution.py

"""Add memory source attribution fields.

Revision ID: xxxx
Revises: previous_revision
Create Date: 2025-01-13
"""

from alembic import op
import sqlalchemy as sa

revision = "xxxx"
down_revision = "previous_revision"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column("source_session_id", sa.String(), nullable=True),
    )
    op.add_column(
        "memories",
        sa.Column("source_message_index", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_memories_source_session_id",
        "memories",
        ["source_session_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_memories_source_session_id", table_name="memories")
    op.drop_column("memories", "source_message_index")
    op.drop_column("memories", "source_session_id")
```

### Effort Estimate

**Size: Small**
- Schema changes: ~20 lines
- Store/manager updates: ~30 lines
- Migration: ~20 lines
- Pass-through in callers: ~20 lines

**Priority: Medium**
- Improves auditability significantly
- Low risk change (additive only)
- Useful for debugging memory issues

---

## Gap 3: Memory Confidence Scores

### What Ash is Missing

The extractor generates confidence scores for extracted facts, but they're not persisted:

```python
# extractor.py lines 35-42
@dataclass
class ExtractedFact:
    """A fact extracted from conversation."""

    content: str
    subjects: list[str]
    shared: bool
    confidence: float  # Generated but not stored!
```

The confidence is used for filtering during extraction (threshold >= 0.7) but then discarded. It could be valuable for:
- Prioritizing which memories to show
- Allowing users to filter low-confidence memories
- Letting the agent express uncertainty

### Why It Matters

- **Quality filtering**: Show high-confidence memories first
- **Transparency**: Users can see why agent remembers something
- **Cleanup**: Easy to prune low-confidence extractions
- **Decay**: Could reduce confidence over time without reinforcement

### Files to Modify

- `/home/dcramer/src/ash/src/ash/db/models.py` - Add confidence field
- `/home/dcramer/src/ash/src/ash/memory/store.py` - Accept confidence
- `/home/dcramer/src/ash/src/ash/memory/retrieval.py` - Return confidence in results
- `/home/dcramer/src/ash/alembic/versions/` - New migration

### Concrete Python Code Changes

```python
# Update db/models.py Memory class

class Memory(Base):
    """Memory entry - a stored fact or piece of information."""

    __tablename__ = "memories"

    # ... existing fields ...

    # NEW: Confidence tracking
    confidence: Mapped[float | None] = mapped_column(
        nullable=True, default=1.0
    )
    """Confidence score from extraction (0.0-1.0). NULL for user-provided."""

    reinforcement_count: Mapped[int] = mapped_column(
        nullable=False, default=0
    )
    """Number of times this memory was reinforced/mentioned."""
```

```python
# Update memory/store.py add_memory

async def add_memory(
    self,
    content: str,
    source: str | None = None,
    expires_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    owner_user_id: str | None = None,
    chat_id: str | None = None,
    subject_person_ids: list[str] | None = None,
    source_session_id: str | None = None,
    source_message_index: int | None = None,
    confidence: float | None = None,  # NEW
) -> Memory:
    """Add a memory entry."""
    memory = Memory(
        id=str(uuid.uuid4()),
        content=content,
        source=source,
        expires_at=expires_at,
        metadata_=metadata,
        owner_user_id=owner_user_id,
        chat_id=chat_id,
        subject_person_ids=subject_person_ids,
        source_session_id=source_session_id,
        source_message_index=source_message_index,
        confidence=confidence,
        reinforcement_count=0,
    )
    self._session.add(memory)
    await self._session.flush()
    return memory


async def reinforce_memory(self, memory_id: str) -> bool:
    """Increment reinforcement count for a memory.

    Called when the same fact is mentioned again, increasing confidence
    that this memory is important.

    Args:
        memory_id: Memory to reinforce.

    Returns:
        True if memory was found and updated.
    """
    stmt = select(Memory).where(Memory.id == memory_id)
    result = await self._session.execute(stmt)
    memory = result.scalar_one_or_none()

    if not memory:
        return False

    memory.reinforcement_count += 1
    # Optionally boost confidence based on reinforcement
    if memory.confidence is not None and memory.confidence < 1.0:
        # Asymptotic approach to 1.0
        memory.confidence = min(1.0, memory.confidence + (1.0 - memory.confidence) * 0.1)

    await self._session.flush()
    return True
```

```python
# Update memory/types.py SearchResult

@dataclass
class SearchResult:
    """Search result with similarity score."""

    id: str
    content: str
    similarity: float
    metadata: dict[str, Any] | None = None
    source_type: str = "memory"
    confidence: float | None = None  # NEW: Include extraction confidence
    reinforcement_count: int = 0  # NEW: How many times reinforced
```

```python
# Update memory/retrieval.py to include confidence

# In search_memories method, update the SQL query:
sql = text(f"""
    SELECT
        me.memory_id,
        m.content,
        m.metadata,
        m.subject_person_ids,
        m.confidence,
        m.reinforcement_count,
        vec_distance_cosine(me.embedding, :query_embedding) as distance
    FROM memory_embeddings me
    JOIN memories m ON me.memory_id = m.id
    {where_clause}
    ORDER BY distance ASC
    LIMIT :limit
""")

# And in building results:
results.append(
    SearchResult(
        id=row[0],
        content=row[1],
        metadata={
            **base_metadata,
            "subject_person_ids": subject_ids,
            "subject_name": subject_name,
        },
        similarity=1.0 - row[6],  # Adjust index for new columns
        source_type="memory",
        confidence=row[4],
        reinforcement_count=row[5],
    )
)
```

```python
# New migration

"""Add memory confidence and reinforcement fields.

Revision ID: yyyy
"""

from alembic import op
import sqlalchemy as sa

revision = "yyyy"
down_revision = "xxxx"


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column("confidence", sa.Float(), nullable=True, server_default="1.0"),
    )
    op.add_column(
        "memories",
        sa.Column("reinforcement_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("memories", "reinforcement_count")
    op.drop_column("memories", "confidence")
```

### Effort Estimate

**Size: Small**
- Schema changes: ~10 lines
- Store updates: ~30 lines
- Retrieval updates: ~20 lines
- Migration: ~15 lines

**Priority: Medium**
- Already extracting confidence, just need to store it
- Enables future quality-based filtering
- Low implementation risk

---

## Gap 4: Memory Categories/Tags

### What Ash is Missing

Memories are stored as flat text without categorization. Users can't easily filter by type (preferences, facts about people, work items, etc.).

### Why It Matters

- **Organization**: "Show me all my preferences" or "What do you know about my family?"
- **Privacy**: Could mark certain categories as more sensitive
- **Retention**: Different categories might have different expiration policies
- **Context**: Include only relevant categories in prompts

### Files to Modify

- `/home/dcramer/src/ash/src/ash/db/models.py` - Add category field
- `/home/dcramer/src/ash/src/ash/memory/store.py` - Filter by category
- `/home/dcramer/src/ash/src/ash/memory/extractor.py` - Extract category
- `/home/dcramer/src/ash/src/ash/cli/commands/memory.py` - Filter CLI

### Concrete Python Code Changes

```python
# Add to memory/types.py

from enum import Enum

class MemoryCategory(str, Enum):
    """Categories for organizing memories."""

    PREFERENCE = "preference"
    """User preferences (likes, dislikes, habits)"""

    PERSON = "person"
    """Facts about people (relationships, details)"""

    FACT = "fact"
    """General facts the user shared"""

    EVENT = "event"
    """Dates, appointments, occasions"""

    WORK = "work"
    """Work-related information"""

    LOCATION = "location"
    """Places, addresses, locations"""

    CONTACT = "contact"
    """Contact information, communication details"""

    OTHER = "other"
    """Uncategorized memories"""
```

```python
# Update db/models.py Memory class

class Memory(Base):
    """Memory entry - a stored fact or piece of information."""

    __tablename__ = "memories"

    # ... existing fields ...

    # NEW: Categorization
    category: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    """Memory category (preference, person, fact, event, etc.)"""

    tags: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True
    )
    """Free-form tags for additional organization"""
```

```python
# Update memory/extractor.py EXTRACTION_PROMPT

EXTRACTION_PROMPT = """You are a memory extraction system. Analyze this conversation and identify facts worth remembering about the user(s).

## What to extract:
- User preferences (likes, dislikes, habits) -> category: preference
- Facts about people in their life (names, relationships, details) -> category: person
- Important dates or events -> category: event
- Work-related information (projects, colleagues, deadlines) -> category: work
- Location information (addresses, places) -> category: location
- General facts -> category: fact
- Explicit requests to remember something -> infer category from content
- Corrections to previously known information

## What NOT to extract:
- Actions the assistant took
- Temporary task context ("working on X project")
- Generic conversation flow
- Credentials or sensitive data
- Things already in memory (avoid duplicates)

## CRITICAL: Resolve references
Convert pronouns and references to concrete facts:
- "I liked that restaurant" -> Find which restaurant from context, store "User liked [restaurant name]"
- "She's visiting next week" -> Find who "she" is, store "[Person name] is visiting [date]"
- "Yes, that one" -> Don't extract - too ambiguous

{existing_memories_section}

## Conversation to analyze:
{conversation}

## Output format:
Return a JSON array of facts. Each fact has:
- content: The fact (MUST be standalone, no unresolved pronouns)
- subjects: Names of people this is about (empty array if about user themselves)
- shared: true if this is group/team knowledge, false if personal
- confidence: 0.0-1.0 how confident this should be stored
- category: one of [preference, person, fact, event, work, location, contact, other]
- tags: optional array of additional tags

Only include facts with confidence >= 0.7. If you cannot resolve a reference, do not extract it.

Return ONLY valid JSON, no other text. Example:
[
  {{"content": "User prefers dark mode", "subjects": [], "shared": false, "confidence": 0.9, "category": "preference", "tags": ["ui", "settings"]}},
  {{"content": "Sarah's birthday is March 15", "subjects": ["Sarah"], "shared": false, "confidence": 0.85, "category": "person", "tags": ["birthday"]}}
]

If there are no facts worth extracting, return an empty array: []"""
```

```python
# Update memory/types.py ExtractedFact

@dataclass
class ExtractedFact:
    """A fact extracted from conversation."""

    content: str
    subjects: list[str]
    shared: bool
    confidence: float
    category: str = "other"  # NEW
    tags: list[str] | None = None  # NEW
```

```python
# Update memory/extractor.py _parse_extraction_response

def _parse_extraction_response(self, response_text: str) -> list[ExtractedFact]:
    """Parse the LLM's JSON response into ExtractedFact objects."""
    # ... existing parsing ...

    facts = []
    for item in data:
        if not isinstance(item, dict):
            continue

        content = item.get("content", "").strip()
        if not content:
            continue

        confidence = float(item.get("confidence", 0.0))
        if confidence < self._confidence_threshold:
            continue

        subjects = item.get("subjects", [])
        if not isinstance(subjects, list):
            subjects = []
        subjects = [str(s) for s in subjects if s]

        shared = bool(item.get("shared", False))
        category = item.get("category", "other")
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        facts.append(
            ExtractedFact(
                content=content,
                subjects=subjects,
                shared=shared,
                confidence=confidence,
                category=category,
                tags=tags,
            )
        )

    return facts
```

```python
# Update cli/commands/memory.py to add category filter

@app.command()
def memory(
    action: Annotated[str, typer.Argument(help="Action: list, add, remove, clear")],
    query: Annotated[str | None, typer.Option("--query", "-q")] = None,
    # ... existing options ...
    category: Annotated[
        str | None,
        typer.Option(
            "--category",
            "-C",
            help="Filter by category (preference, person, fact, event, work, location)",
        ),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option(
            "--tag",
            "-t",
            help="Filter by tag",
        ),
    ] = None,
) -> None:
    """Manage memory entries.

    Examples:
        ash memory list --category preference   # List all preferences
        ash memory list --category person       # List facts about people
        ash memory list --tag birthday          # List memories tagged 'birthday'
    """
    # ... pass through to async handler


async def _memory_list(
    session,
    query: str | None,
    limit: int,
    include_expired: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
    category: str | None,  # NEW
    tag: str | None,  # NEW
) -> None:
    """List memory entries."""
    # ... existing query building ...

    if category:
        stmt = stmt.where(MemoryModel.category == category)

    if tag:
        # SQLite JSON contains check
        stmt = stmt.where(
            text("EXISTS (SELECT 1 FROM json_each(memories.tags) WHERE json_each.value = :tag)")
            .bindparams(tag=tag)
        )

    # ... rest of method
```

### Effort Estimate

**Size: Medium**
- Schema changes: ~10 lines
- Extractor updates: ~50 lines (prompt and parsing)
- CLI updates: ~30 lines
- Store/retrieval filtering: ~20 lines
- Migration: ~15 lines

**Priority: Medium**
- Good organization benefit
- Requires updating extraction prompt
- Could break existing memory extraction if not careful

---

## Gap 5: Memory Export/Import

### What Ash is Missing

No way to export memories to a portable format or import from external sources. Users are locked into the SQLite database.

### Why It Matters

- **Backup**: Export memories for safekeeping
- **Portability**: Move memories between Ash instances
- **Migration**: Import from other systems (MEMORY.md files, notes apps)
- **Sharing**: Export subset of memories to share with others
- **Inspection**: Human-readable export for review

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/commands/memory.py` - Add export/import actions
- `/home/dcramer/src/ash/src/ash/memory/io.py` - New file for import/export logic

### Concrete Python Code Changes

```python
# New file: src/ash/memory/io.py
"""Memory import/export utilities."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ash.db.models import Memory

logger = logging.getLogger(__name__)


class MemoryExporter:
    """Export memories to various formats."""

    @staticmethod
    def to_json(memories: list[Memory]) -> str:
        """Export memories to JSON format."""
        data = []
        for m in memories:
            data.append({
                "id": m.id,
                "content": m.content,
                "source": m.source,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "expires_at": m.expires_at.isoformat() if m.expires_at else None,
                "owner_user_id": m.owner_user_id,
                "chat_id": m.chat_id,
                "category": getattr(m, "category", None),
                "tags": getattr(m, "tags", None),
                "confidence": getattr(m, "confidence", None),
                "subject_person_ids": m.subject_person_ids,
            })
        return json.dumps(data, indent=2)

    @staticmethod
    def to_markdown(memories: list[Memory]) -> str:
        """Export memories to markdown format.

        Compatible with MEMORY.md files used by Archer/mom.
        """
        lines = ["# Exported Memories", ""]

        for m in memories:
            lines.append(f"## {m.content[:50]}...")
            lines.append(f"- **Added**: {m.created_at.strftime('%Y-%m-%d') if m.created_at else 'unknown'}")
            if m.source:
                lines.append(f"- **Source**: {m.source}")
            if m.expires_at:
                lines.append(f"- **Expires**: {m.expires_at.strftime('%Y-%m-%d')}")
            if getattr(m, "category", None):
                lines.append(f"- **Category**: {m.category}")
            if getattr(m, "tags", None):
                lines.append(f"- **Tags**: {', '.join(m.tags)}")
            lines.append("")
            lines.append(m.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_jsonl(memories: list[Memory]) -> str:
        """Export memories to JSONL format (one JSON object per line)."""
        lines = []
        for m in memories:
            data = {
                "content": m.content,
                "source": m.source,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "category": getattr(m, "category", None),
                "tags": getattr(m, "tags", None),
            }
            lines.append(json.dumps(data))
        return "\n".join(lines)


class MemoryImporter:
    """Import memories from various formats."""

    @staticmethod
    def from_json(content: str) -> list[dict[str, Any]]:
        """Import memories from JSON format."""
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON must contain an array of memories")

        memories = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "content" not in item:
                continue
            memories.append({
                "content": item["content"],
                "source": item.get("source", "import"),
                "category": item.get("category"),
                "tags": item.get("tags"),
            })
        return memories

    @staticmethod
    def from_markdown(content: str) -> list[dict[str, Any]]:
        """Import memories from MEMORY.md format.

        Parses markdown with entries separated by ---.
        """
        memories = []
        entries = content.split("\n---\n")

        for entry in entries:
            entry = entry.strip()
            if not entry or entry.startswith("# "):
                continue

            lines = entry.split("\n")
            memory_content = []
            category = None
            tags = None
            source = "import"

            for line in lines:
                if line.startswith("## "):
                    continue
                elif line.startswith("- **Category**:"):
                    category = line.split(":", 1)[1].strip()
                elif line.startswith("- **Tags**:"):
                    tags = [t.strip() for t in line.split(":", 1)[1].split(",")]
                elif line.startswith("- **Source**:"):
                    source = line.split(":", 1)[1].strip()
                elif line.startswith("- **"):
                    continue  # Skip other metadata
                elif line.strip():
                    memory_content.append(line)

            if memory_content:
                memories.append({
                    "content": "\n".join(memory_content).strip(),
                    "source": source,
                    "category": category,
                    "tags": tags,
                })

        return memories

    @staticmethod
    def from_jsonl(content: str) -> list[dict[str, Any]]:
        """Import memories from JSONL format."""
        memories = []
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "content" in data:
                    memories.append({
                        "content": data["content"],
                        "source": data.get("source", "import"),
                        "category": data.get("category"),
                        "tags": data.get("tags"),
                    })
            except json.JSONDecodeError:
                continue
        return memories

    @staticmethod
    def from_file(path: Path) -> list[dict[str, Any]]:
        """Import memories from file, detecting format from extension."""
        content = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()

        if suffix == ".json":
            return MemoryImporter.from_json(content)
        elif suffix == ".md":
            return MemoryImporter.from_markdown(content)
        elif suffix == ".jsonl":
            return MemoryImporter.from_jsonl(content)
        else:
            # Try to detect format
            content = content.strip()
            if content.startswith("["):
                return MemoryImporter.from_json(content)
            elif content.startswith("{"):
                return MemoryImporter.from_jsonl(content)
            else:
                return MemoryImporter.from_markdown(content)
```

```python
# Update cli/commands/memory.py to add export/import

@app.command()
def memory(
    action: Annotated[
        str,
        typer.Argument(help="Action: list, add, remove, clear, export, import"),
    ],
    # ... existing options ...
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for export",
        ),
    ] = None,
    input_file: Annotated[
        Path | None,
        typer.Option(
            "--input",
            "-i",
            help="Input file for import",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Export format: json, markdown, jsonl",
        ),
    ] = "json",
) -> None:
    """Manage memory entries.

    Examples:
        ash memory export -o memories.json           # Export all to JSON
        ash memory export -o memories.md --format markdown
        ash memory import -i memories.json           # Import from file
        ash memory import -i MEMORY.md               # Import from markdown
    """
    # ... implementation


async def _memory_export(
    session,
    output_file: Path | None,
    format: str,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
    category: str | None,
) -> None:
    """Export memories to file."""
    from ash.memory.io import MemoryExporter
    from sqlalchemy import select
    from ash.db.models import Memory as MemoryModel

    # Build query with filters
    stmt = select(MemoryModel).order_by(MemoryModel.created_at.desc())
    # ... apply filters similar to list ...

    result = await session.execute(stmt)
    memories = list(result.scalars().all())

    if not memories:
        warning("No memories to export")
        return

    # Export to chosen format
    if format == "json":
        content = MemoryExporter.to_json(memories)
    elif format == "markdown":
        content = MemoryExporter.to_markdown(memories)
    elif format == "jsonl":
        content = MemoryExporter.to_jsonl(memories)
    else:
        error(f"Unknown format: {format}")
        raise typer.Exit(1)

    if output_file:
        output_file.write_text(content, encoding="utf-8")
        success(f"Exported {len(memories)} memories to {output_file}")
    else:
        console.print(content)


async def _memory_import(
    session,
    input_file: Path,
    owner_user_id: str | None,
    force: bool,
) -> None:
    """Import memories from file."""
    from ash.memory.io import MemoryImporter
    from ash.memory import MemoryStore

    if not input_file.exists():
        error(f"File not found: {input_file}")
        raise typer.Exit(1)

    memories = MemoryImporter.from_file(input_file)

    if not memories:
        warning("No memories found in file")
        return

    if not force:
        console.print(f"Found {len(memories)} memories to import:")
        for m in memories[:5]:
            console.print(f"  - {m['content'][:60]}...")
        if len(memories) > 5:
            console.print(f"  ... and {len(memories) - 5} more")

        confirm = typer.confirm("Import these memories?")
        if not confirm:
            dim("Cancelled")
            return

    store = MemoryStore(session)
    imported = 0

    for m in memories:
        try:
            await store.add_memory(
                content=m["content"],
                source=m.get("source", "import"),
                owner_user_id=owner_user_id,
            )
            imported += 1
        except Exception as e:
            warning(f"Failed to import: {m['content'][:40]}... ({e})")

    await session.commit()
    success(f"Imported {imported} memories")
```

### Effort Estimate

**Size: Medium**
- New io.py module: ~150 lines
- CLI updates: ~80 lines
- Tests: ~100 lines

**Priority: Medium**
- Useful for backup/migration
- Enables interop with MEMORY.md tools
- Low risk (new functionality, no changes to existing)

---

## Gap 6: Memory Deduplication

### What Ash is Missing

Ash has supersession (newer memory replaces older conflicting one) but not deduplication. If the same fact is extracted multiple times with slightly different wording, they all get stored.

Current conflict detection (manager.py):
```python
async def find_conflicting_memories(
    self,
    new_content: str,
    owner_user_id: str | None = None,
    chat_id: str | None = None,
    subject_person_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Find existing memories that may conflict with new content.

    Looks for memories with high semantic similarity in the same scope,
    which likely represent updated information about the same topic.
    """
    # Uses CONFLICT_SIMILARITY_THRESHOLD = 0.75
    # This is for supersession, not exact deduplication
```

### Why It Matters

- **Storage efficiency**: Avoid duplicate memories
- **Cleaner context**: Don't show same fact multiple times
- **Reinforcement**: Merge duplicates into one with higher confidence

### Files to Modify

- `/home/dcramer/src/ash/src/ash/memory/manager.py` - Add deduplication logic
- `/home/dcramer/src/ash/src/ash/memory/dedup.py` - New deduplication utilities

### Concrete Python Code Changes

```python
# New file: src/ash/memory/dedup.py
"""Memory deduplication utilities."""

import logging
from dataclasses import dataclass

from ash.memory.types import SearchResult

logger = logging.getLogger(__name__)

# Threshold for considering memories as duplicates (very high similarity)
DUPLICATE_SIMILARITY_THRESHOLD = 0.92


@dataclass
class DuplicateGroup:
    """Group of duplicate memories."""

    canonical_id: str
    """ID of the memory to keep (most reinforced or newest)."""

    duplicate_ids: list[str]
    """IDs of duplicate memories to merge/remove."""

    similarity_scores: list[float]
    """Similarity scores between canonical and each duplicate."""


async def find_duplicate_groups(
    retriever,
    store,
    owner_user_id: str | None = None,
    chat_id: str | None = None,
    batch_size: int = 100,
) -> list[DuplicateGroup]:
    """Find groups of duplicate memories.

    Scans memories and groups those with very high semantic similarity
    (above DUPLICATE_SIMILARITY_THRESHOLD).

    Args:
        retriever: SemanticRetriever for similarity search.
        store: MemoryStore for getting memories.
        owner_user_id: Scope to user's memories.
        chat_id: Scope to chat's memories.
        batch_size: How many memories to check at once.

    Returns:
        List of DuplicateGroup objects.
    """
    # Get all active memories in scope
    memories = await store.get_memories(
        limit=1000,
        include_expired=False,
        include_superseded=False,
        owner_user_id=owner_user_id,
        chat_id=chat_id,
    )

    if len(memories) < 2:
        return []

    seen_ids = set()
    groups: list[DuplicateGroup] = []

    for memory in memories:
        if memory.id in seen_ids:
            continue

        # Find similar memories
        similar = await retriever.search_memories(
            query=memory.content,
            limit=10,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            include_expired=False,
            include_superseded=False,
        )

        # Filter to duplicates (high similarity, different ID)
        duplicates = [
            (s.id, s.similarity)
            for s in similar
            if s.id != memory.id
            and s.similarity >= DUPLICATE_SIMILARITY_THRESHOLD
            and s.id not in seen_ids
        ]

        if duplicates:
            # Memory with highest reinforcement (or newest) becomes canonical
            group = DuplicateGroup(
                canonical_id=memory.id,
                duplicate_ids=[d[0] for d in duplicates],
                similarity_scores=[d[1] for d in duplicates],
            )
            groups.append(group)

            # Mark these as seen
            seen_ids.add(memory.id)
            seen_ids.update(d[0] for d in duplicates)

    return groups


async def merge_duplicate_group(
    group: DuplicateGroup,
    store,
    retriever,
) -> int:
    """Merge a group of duplicates into the canonical memory.

    Transfers reinforcement counts and supersedes duplicates.

    Args:
        group: The duplicate group to merge.
        store: MemoryStore for updates.
        retriever: SemanticRetriever for embedding cleanup.

    Returns:
        Number of duplicates merged.
    """
    from datetime import UTC, datetime

    merged_count = 0

    # Get canonical memory for updating reinforcement count
    canonical = await store.get_memory(group.canonical_id)
    if not canonical:
        logger.warning("Canonical memory not found: %s", group.canonical_id)
        return 0

    total_reinforcement = getattr(canonical, "reinforcement_count", 0)

    for dup_id in group.duplicate_ids:
        dup = await store.get_memory(dup_id)
        if not dup:
            continue

        # Aggregate reinforcement counts
        total_reinforcement += getattr(dup, "reinforcement_count", 0) + 1

        # Mark as superseded
        success = await store.mark_memory_superseded(
            memory_id=dup_id,
            superseded_by_id=group.canonical_id,
        )

        if success:
            # Clean up embedding
            try:
                await retriever.delete_memory_embedding(dup_id)
            except Exception:
                pass
            merged_count += 1

    # Update canonical with aggregated reinforcement
    if hasattr(canonical, "reinforcement_count"):
        canonical.reinforcement_count = total_reinforcement

    return merged_count
```

```python
# Add to memory/manager.py

async def deduplicate_memories(
    self,
    owner_user_id: str | None = None,
    chat_id: str | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Find and merge duplicate memories.

    Scans for memories with very high semantic similarity and merges
    them, keeping the most reinforced or newest version.

    Args:
        owner_user_id: Scope to user's memories.
        chat_id: Scope to chat's memories.
        dry_run: If True, only report what would be merged.

    Returns:
        Tuple of (groups_found, memories_merged).
    """
    from ash.memory.dedup import find_duplicate_groups, merge_duplicate_group

    groups = await find_duplicate_groups(
        retriever=self._retriever,
        store=self._store,
        owner_user_id=owner_user_id,
        chat_id=chat_id,
    )

    if not groups:
        return (0, 0)

    if dry_run:
        total_dups = sum(len(g.duplicate_ids) for g in groups)
        return (len(groups), total_dups)

    total_merged = 0
    for group in groups:
        merged = await merge_duplicate_group(
            group=group,
            store=self._store,
            retriever=self._retriever,
        )
        total_merged += merged

    await self._session.commit()

    logger.info(
        "Deduplication complete",
        extra={"groups": len(groups), "merged": total_merged},
    )

    return (len(groups), total_merged)
```

```python
# Add CLI command for deduplication

@app.command()
def memory_dedupe(
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be merged without making changes",
        ),
    ] = False,
    user_id: Annotated[str | None, typer.Option("--user", "-u")] = None,
    chat_id: Annotated[str | None, typer.Option("--chat")] = None,
) -> None:
    """Find and merge duplicate memories.

    Examples:
        ash memory dedupe --dry-run    # Preview deduplication
        ash memory dedupe              # Actually merge duplicates
    """
    asyncio.run(_run_dedupe(dry_run, user_id, chat_id))


async def _run_dedupe(dry_run: bool, user_id: str | None, chat_id: str | None) -> None:
    config = get_config(None)
    database = await get_database(config)

    async with database.session() as session:
        from ash.memory import create_memory_manager
        from ash.llm import LLMRegistry

        # ... setup memory manager ...

        groups, merged = await manager.deduplicate_memories(
            owner_user_id=user_id,
            chat_id=chat_id,
            dry_run=dry_run,
        )

        if dry_run:
            console.print(f"Found {groups} duplicate groups ({merged} total duplicates)")
            console.print("Run without --dry-run to merge")
        else:
            success(f"Merged {merged} duplicate memories from {groups} groups")
```

### Effort Estimate

**Size: Medium**
- New dedup.py module: ~100 lines
- Manager integration: ~40 lines
- CLI command: ~40 lines
- Tests: ~80 lines

**Priority: Low**
- Nice to have for cleanup
- Supersession already handles most cases
- Could run as periodic maintenance task

---

## Gap 7: Memory Search CLI Improvements

### What Ash is Missing

The current CLI (`ash memory list -q "query"`) does basic substring matching. It doesn't use the semantic search capabilities that exist in the retrieval layer.

Current list implementation (cli/commands/memory.py):
```python
# Filter by content if query provided
if query:
    stmt = stmt.where(MemoryModel.content.ilike(f"%{query}%"))
```

### Why It Matters

- **Better search**: "food preferences" should find "User likes Italian cuisine"
- **Similarity scores**: Show how relevant each result is
- **Parity**: CLI should have same capabilities as agent

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/commands/memory.py` - Add semantic search

### Concrete Python Code Changes

```python
# Update cli/commands/memory.py

@app.command()
def memory(
    action: Annotated[
        str,
        typer.Argument(help="Action: list, search, add, remove, clear, export, import"),
    ],
    query: Annotated[str | None, typer.Option("--query", "-q")] = None,
    # ... existing options ...
    semantic: Annotated[
        bool,
        typer.Option(
            "--semantic",
            help="Use semantic search instead of substring matching",
        ),
    ] = False,
    min_similarity: Annotated[
        float,
        typer.Option(
            "--min-similarity",
            help="Minimum similarity score for semantic search (0.0-1.0)",
        ),
    ] = 0.5,
) -> None:
    """Manage memory entries.

    Examples:
        ash memory list -q "food"                    # Substring search
        ash memory search -q "food preferences"     # Semantic search (alias)
        ash memory list -q "food" --semantic         # Semantic search
        ash memory list --semantic --min-similarity 0.7
    """
    # Map 'search' action to 'list' with semantic=True
    if action == "search":
        action = "list"
        semantic = True
    # ... rest of handler


async def _memory_list(
    session,
    query: str | None,
    limit: int,
    include_expired: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
    category: str | None,
    tag: str | None,
    semantic: bool,  # NEW
    min_similarity: float,  # NEW
    config,  # Need config for LLM registry
) -> None:
    """List memory entries."""
    from rich.table import Table

    if semantic and query:
        await _memory_semantic_search(
            session=session,
            query=query,
            limit=limit,
            user_id=user_id,
            chat_id=chat_id,
            min_similarity=min_similarity,
            config=config,
        )
        return

    # ... existing substring-based list logic ...


async def _memory_semantic_search(
    session,
    query: str,
    limit: int,
    user_id: str | None,
    chat_id: str | None,
    min_similarity: float,
    config,
) -> None:
    """Perform semantic search over memories."""
    from rich.table import Table
    from ash.memory import create_memory_manager
    from ash.llm import LLMRegistry

    # Set up LLM registry for embeddings
    llm_registry = LLMRegistry()
    llm_registry.configure(config.llm)

    manager = await create_memory_manager(
        db_session=session,
        llm_registry=llm_registry,
        embedding_model=config.memory.embedding_model,
    )

    results = await manager.search(
        query=query,
        limit=limit,
        owner_user_id=user_id,
        chat_id=chat_id,
    )

    # Filter by similarity threshold
    results = [r for r in results if r.similarity >= min_similarity]

    if not results:
        warning(f"No memories found matching '{query}' (similarity >= {min_similarity})")
        return

    table = Table(title=f"Semantic Search: '{query}'")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Similarity", style="green", justify="right")
    table.add_column("Confidence", style="yellow", justify="right")
    table.add_column("Content", style="white", max_width=50)

    for result in results:
        content = result.content[:70] + "..." if len(result.content) > 70 else result.content
        content = content.replace("\n", " ")

        sim_pct = f"{result.similarity * 100:.0f}%"
        conf = f"{result.confidence * 100:.0f}%" if result.confidence else "-"

        # Color similarity based on strength
        if result.similarity >= 0.8:
            sim_style = "[bold green]"
        elif result.similarity >= 0.6:
            sim_style = "[green]"
        else:
            sim_style = "[dim green]"

        table.add_row(
            result.id[:8],
            f"{sim_style}{sim_pct}[/]",
            conf,
            content,
        )

    console.print(table)
    dim(f"\nFound {len(results)} results (threshold: {min_similarity * 100:.0f}%)")
```

```python
# Add interactive memory exploration command

@app.command()
def memory_explore(
    user_id: Annotated[str | None, typer.Option("--user", "-u")] = None,
    chat_id: Annotated[str | None, typer.Option("--chat")] = None,
) -> None:
    """Interactively explore memories with semantic search.

    Enter queries and see semantically similar memories.
    Type 'quit' to exit.
    """
    from rich.prompt import Prompt

    console.print("[bold]Memory Explorer[/bold]")
    console.print("Enter search queries to find related memories.")
    console.print("Type 'quit' to exit.\n")

    while True:
        query = Prompt.ask("[cyan]Search[/cyan]")
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query.strip():
            continue

        asyncio.run(_memory_semantic_search_interactive(query, user_id, chat_id))


async def _memory_semantic_search_interactive(
    query: str,
    user_id: str | None,
    chat_id: str | None,
) -> None:
    """Run a single semantic search interactively."""
    config = get_config(None)
    database = await get_database(config)

    try:
        async with database.session() as session:
            await _memory_semantic_search(
                session=session,
                query=query,
                limit=10,
                user_id=user_id,
                chat_id=chat_id,
                min_similarity=0.4,  # Lower threshold for exploration
                config=config,
            )
    finally:
        await database.disconnect()
```

### Effort Estimate

**Size: Small**
- Semantic search integration: ~60 lines
- CLI option changes: ~20 lines
- Interactive explorer: ~40 lines

**Priority: High**
- Low effort, high value
- Uses existing retrieval capabilities
- Improves debugging and exploration

---

## Summary

| Gap | Description | Effort | Priority | Notes |
|-----|-------------|--------|----------|-------|
| 1 | MEMORY.md Fallback Mode | Medium | Low | Nice for simple deployments, but SQLite works well |
| 2 | Memory Source Attribution | Small | Medium | Easy win for auditability |
| 3 | Memory Confidence Scores | Small | Medium | Already extracting, just need to store |
| 4 | Memory Categories/Tags | Medium | Medium | Good organization, requires prompt changes |
| 5 | Memory Export/Import | Medium | Medium | Enables backup and migration |
| 6 | Memory Deduplication | Medium | Low | Supersession handles most cases |
| 7 | Memory Search CLI | Small | High | Low effort, high value |

### Recommended Priority Order

1. **Gap 7: Memory Search CLI** - Quick win, uses existing capabilities
2. **Gap 2: Source Attribution** - Small change, improves auditability
3. **Gap 3: Confidence Scores** - Already extracting, just store it
4. **Gap 4: Categories/Tags** - Improves organization
5. **Gap 5: Export/Import** - Enables portability
6. **Gap 6: Deduplication** - Nice to have cleanup
7. **Gap 1: File Fallback** - Only if deployment simplicity becomes priority
