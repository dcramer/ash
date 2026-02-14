# People

> Person recognition and identity management so the agent maps "my wife" / "Sarah" / "@sksembhi" to the same entity

## Intent

The people subsystem solves person recognition: turning natural-language references into stable identities that persist across conversations. It provides:

1. **Auto-resolution** - References like "my wife", "Sarah", "@sksembhi" resolve to the same person entity
2. **Alias management** - Multiple names and handles tracked per person with provenance
3. **Global identity** - People are shared across all users, not isolated per-owner
4. **Memory attribution** - Facts linked to person entities via `subject_person_ids`
5. **Self-identity** - Each user gets a "self" person record linking their username to display name
6. **Person merging** - When two records turn out to be the same person, merge them

People is NOT for:
- Conversation history or session state
- User authentication or authorization
- Contact management (no emails, phone numbers, addresses)

## Graph Model

The people subsystem is best understood as a graph. Person entities are nodes; relationships, aliases, and identity links are edges.

### Nodes

| Node Type | Backed By | Key Fields |
|-----------|-----------|------------|
| **Person** | `PersonEntry` in `people.jsonl` | id, name, created_by |

### Edges (owned by people subsystem)

| Edge | From → To | Stored As | Metadata |
|------|-----------|-----------|----------|
| **SELF** | user_id → Person | `RelationshipClaim(relationship="self")` | stated_by |
| **KNOWS** | user_id → Person | `RelationshipClaim(relationship="wife"...)` | stated_by, created_at |
| **ALIAS** | Person → label | `AliasEntry(value="sksembhi")` | added_by, created_at |
| **MERGED_INTO** | Person → Person | `merged_into` field | Preserves full merge history; secondary stays in graph as redirect |

### Key Traversals

| Query | Traversal |
|-------|-----------|
| "my wife" (from Alice) | Alice →SELF→ PersonAlice →KNOWS(wife)→ **result** |
| "Sarah" (from Alice) | Alice →SELF→ PersonAlice →KNOWS→ persons →filter(name="Sarah")→ **result**; fallback: global name search |
| "who said this?" | memory →STATED_BY→ user_id →SELF→ person |
| merge follow | PersonB →MERGED_INTO→ PersonA (secondary stays as redirect; traversal resolves transparently) |
| merge history | PersonA ←MERGED_INTO← [PersonB, PersonC, ...] (queryable audit trail) |

See [specs/memory.md](memory.md) for Memory-owned edges (ABOUT, STATED_BY, OWNED_BY, IN_CHAT, SUPERSEDES).

## User Identity Model

A user arrives from a provider (currently Telegram) carrying several identity fields. These have different stability and uniqueness guarantees, which matter for how we anchor person records.

### Identity Fields from Telegram

| Field | Example | Unique? | Immutable? | Always present? |
|-------|---------|---------|------------|-----------------|
| `user.id` | `123456789` | Yes (globally) | Yes | Yes |
| `user.username` | `notzeeg` | Yes (at a point in time) | Practically yes | No (optional) |
| `user.full_name` | `David Cramer` | No | No (user can change) | Yes |

**Telegram user ID** (`user.id`): Integer, assigned at account creation, never changes, never reused. This is the **only guaranteed-stable anchor** for a user's identity.

**Username** (`user.username`): String handle like `notzeeg`. Unique at any point in time but technically mutable (users can change it, though in practice this is rare). Optional — not all Telegram users set one.

**Full name** (`user.full_name`): Display name. Not unique, not stable. Users change this freely.

### How Fields Flow Through the System

```
Telegram message.from_user
  ├─ .id (int: 123456789)
  │    → str(user.id) → IncomingMessage.user_id → session.user_id
  │    → MemoryEntry.owner_user_id (scoping)
  │    → PersonEntry.created_by (provenance)
  │    → AliasEntry.added_by / RelationshipClaim.stated_by
  │
  ├─ .username (str: "notzeeg")
  │    → session.metadata["username"]
  │    → MemoryEntry.source_user_id ← MISNAMED: this is a username, not an ID
  │    → PersonEntry alias (via _ensure_self_person)
  │    → find_person_ids_for_username() lookup key
  │
  └─ .full_name (str: "David Cramer")
       → session.metadata["display_name"]
       → MemoryEntry.source_user_name
       → PersonEntry.name (via _ensure_self_person)
```

### Field Naming Problems (Current State)

Several fields are named `*_id` but contain usernames or display names, not unique IDs:

| Field | Named as | Actually contains | Trust level |
|-------|----------|-------------------|-------------|
| `session.user_id` | ID | Telegram numeric ID (string) | **Stable anchor** |
| `MemoryEntry.owner_user_id` | ID | Telegram numeric ID (string) | **Stable anchor** |
| `MemoryEntry.source_user_id` | ID | Username like `notzeeg` | **Mostly stable** (mutable in theory) |
| `PersonEntry.created_by` | — | Telegram numeric ID (string) | **Stable anchor** |
| `AliasEntry.added_by` | — | Telegram numeric ID (string) | **Stable anchor** |
| `RelationshipClaim.stated_by` | — | Telegram numeric ID (string) | **Stable anchor** |
| `PersonEntry.name` | — | Display name like `David Cramer` | **Unstable** (user can change) |

The critical mismatch: `source_user_id` on memories holds a *username*, while `owner_user_id` holds the actual *numeric ID*. These are used together for scoping and trust determination but represent different kinds of identifiers.

### The Self-Person Bridge

The self-person record is the link between all three identity tiers:

```
PersonEntry:
  id: "person-abc123"            ← Internal UUID (system-generated)
  name: "David Cramer"           ← Display name (unstable, from full_name)
  created_by: "123456789"        ← Numeric ID (stable anchor)
  aliases: ["notzeeg"]           ← Username (mostly stable)
  relationships: [{self}]        ← Marks this as the user's own record
```

`find_person_ids_for_username("notzeeg")` → matches alias → returns `"person-abc123"` → enables trust determination, cross-context retrieval, and hearsay supersession.

### User Stories

**User sends their first message:**
1. Telegram delivers `user_id=123456789`, `username=notzeeg`, `full_name=David Cramer`
2. `_ensure_self_person` creates PersonEntry with `name="David Cramer"`, alias `"notzeeg"`, `relationship="self"`, `created_by="123456789"`
3. Memory extraction sets `source_user_id="notzeeg"` on extracted facts
4. Personal memories scoped with `owner_user_id="123456789"`

**User changes their Telegram display name:**
- Next message arrives with `full_name=Dave Cramer` but same `username=notzeeg`
- `_ensure_self_person` finds existing person by display name `"David Cramer"` — **this lookup fails** because the name changed
- Falls through to username lookup — finds the person via alias `"notzeeg"`
- Self-person still works, but `PersonEntry.name` remains stale as `"David Cramer"`
- **Gap:** No mechanism to update `PersonEntry.name` when display name changes

**User changes their Telegram username (rare):**
- Next message arrives with `username=dcramer` instead of `notzeeg`
- `_ensure_self_person` tries display name first — finds existing if name unchanged
- If found, adds `"dcramer"` as new alias — old `"notzeeg"` alias remains, so old memories still resolve
- **Gap:** If display name also changed, creates a duplicate self-person

**Second user mentions the first user by relationship:**
- Bob says "my coworker David is great with databases"
- Extraction resolves "David" → finds existing PersonEntry by name match
- Memory stored with `subject_person_ids=["person-abc123"]`
- Later, David's own facts can be distinguished from Bob's hearsay about David via `source_user_id` → person ID resolution

**User has no username (Telegram allows this):**
- Message arrives with `user_id=123456789`, `username=None`, `full_name=David Cramer`
- `_ensure_self_person` creates a self-person with the numeric `user_id` as an alias
- This reconnects the graph: `find_person_ids_for_username("123456789")` matches the alias
- Memory `source_user_id` falls back to numeric `user_id` string, which now resolves correctly

**Duplicate names across users:**
- Alice's coworker is named "Sarah". Bob's wife is also named "Sarah".
- When `relationship_stated_by` is provided (e.g., during extraction), `find_for_speaker` walks the speaker's KNOWS edges first
- Alice resolving "Sarah" finds her Sarah (connected via Alice's KNOWS edges); Bob resolving "Sarah" finds his Sarah
- Without speaker context, `find()` still uses global first-match — LLM fuzzy matching and dedup handle remaining edge cases

## Storage Architecture

People uses the same **filesystem-primary** architecture as the memory subsystem:

```
~/.ash/
└── people.jsonl          # Person entities (global, source of truth)
```

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| JSONL is source of truth | Human-readable, version-controllable, survives corruption |
| In-memory cache with mtime invalidation | Fast reads without hitting disk on every call |
| Atomic file writes | Write to temp file, then rename; prevents corruption on crash |
| Global visibility | All users see all people; `created_by` tracks provenance |

## Person Attributes

Each person entity is stored as a `PersonEntry`:

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | UUID primary key (system-generated) |
| `version` | Yes | Schema version (currently 1) |
| `created_by` | Yes | Provider user ID of who created this record (stable anchor) |
| `name` | Yes | Primary display name (from provider, **not stable**) |
| `relationships` | No | List of `RelationshipClaim` assertions with provenance |
| `aliases` | No | List of `AliasEntry` alternate names/handles (non-unique across people) |
| `merged_into` | No | If merged, UUID of primary person |
| `created_at` | Yes | When the record was created |
| `updated_at` | Yes | When the record was last modified |
| `metadata` | No | Optional extra data |

### Backward Compatibility

The `relationship` property returns the first relationship value from the `relationships` list, preserving the old single-relationship API. During deserialization:

- Old `owner_user_id` key maps to `created_by`
- Old `aliases` as `list[str]` converts to `list[AliasEntry]`
- Old `relationship` or `relation` string converts to `list[RelationshipClaim]`

## Alias & Relationship Provenance

### AliasEntry

Each alias tracks who added it:

| Field | Description |
|-------|-------------|
| `value` | The alias string (e.g., "my wife", "sksembhi") — **not unique** across people |
| `added_by` | Provider user ID of who added this alias (stable anchor) |
| `created_at` | When the alias was added |

### RelationshipClaim

Each relationship tracks who stated it:

| Field | Description |
|-------|-------------|
| `relationship` | The relationship term (e.g., "wife", "boss") |
| `stated_by` | Provider user ID of who stated this relationship (stable anchor) |
| `created_at` | When the claim was made |

### Protection Rules

- **Self-added alias**: An alias where `added_by` matches this person (via `find_person_ids_for_username`) cannot be removed by a third party
- **Self-stated relationship**: A relationship where `stated_by` matches this person cannot be overwritten by a third party, unless the third party is the other person in the relationship
- **"Self" determination**: Checked by resolving `added_by`/`stated_by` through `find_person_ids_for_username` to see if it maps to this person's ID

## Relationship Terms

The `RELATIONSHIP_TERMS` set defines recognized relationship words:

```
wife, husband, partner, spouse, mom, mother, dad, father, parent,
son, daughter, child, kid, brother, sister, sibling, boss, manager,
coworker, colleague, friend, best friend, roommate, doctor, therapist, dentist
```

When a user says "my wife", the system:
1. Strips the "my " prefix
2. Matches "wife" against `RELATIONSHIP_TERMS`
3. Creates a `RelationshipClaim(relationship="wife", stated_by=user_id)`
4. Attempts to extract a proper name from surrounding content (supports multi-word names like "Sarah Jane")

## Outcomes

### The agent resolves person references

References like "my wife", "Sarah", "@sksembhi" resolve to the same person entity. Resolution is case-insensitive and strips common prefixes:

| Reference | Normalized | Matches |
|-----------|-----------|---------|
| "My wife" | "wife" | Relationship match |
| "Sarah" | "sarah" | Name match |
| "@sksembhi" | "sksembhi" | Alias match |
| "The boss" | "boss" | Relationship match |
| "SARAH" | "sarah" | Case-insensitive name match |

### New people are auto-created

When a reference cannot be resolved, `resolve_or_create()` creates a new person:
- Relationship terms extracted from "my wife"-style references
- Content hints used to extract proper names (e.g., "My wife Sarah loves hiking" yields name "Sarah", relationship "wife")
- If no name can be extracted, the relationship term is title-cased as name (e.g., "Wife")
- The original reference is stored as an alias when it differs from the extracted name

### Aliases track provenance

Each alias records who added it via `AliasEntry`:
- `value`: The alias string
- `added_by`: Provider user ID of who added it
- `created_at`: Timestamp

Self-added aliases (where the person named by `added_by` is this person) are protected from third-party removal.

**Backward compatibility:** Old plain string aliases loaded during deserialization are converted to `AliasEntry(value=x, added_by=None)`.

### Relationships track provenance

Each relationship records who stated it via `RelationshipClaim`:
- `relationship`: The relationship term
- `stated_by`: Provider user ID of who made the claim
- `created_at`: Timestamp

Self-stated relationships are protected from third-party overwriting.

**Backward compatibility:** Old single `relationship` string loaded during deserialization is converted to `[RelationshipClaim(relationship=x, stated_by=None)]`.

### Self-records enable trust determination

`_ensure_self_person` creates a person with `relationship="self"` for each user:
- Links the user's handle (e.g., "notzeeg") as an alias to their display name (e.g., "David Cramer")
- If the self-person already exists, ensures the username is in the alias list
- Called automatically during conversation setup when speaker identity is known

This enables the memory trust model to determine whether a `source_user_id` is speaking about themselves (FACT) or about someone else (HEARSAY), by resolving the username to a person ID and checking if it overlaps with `subject_person_ids`.

### People are globally visible

People are not scoped per-owner. All users see all people:
- `created_by` tracks who created the record (replaces old `owner_user_id`)
- `find()` searches all records (not filtered by owner)
- `list_all()` returns all non-merged records (replaces `list_for_owner`)
- `get_all()` returns every record across all creators
- `find_person_ids_for_username()` searches across all owners for cross-context retrieval

### Merged persons are consolidated

`merge(primary_id, secondary_id)` combines two person records:

1. Merge aliases from secondary into primary (deduped, case-insensitive)
2. Add secondary's name as an alias on the primary
3. Copy relationships from secondary that the primary lacks
4. Set `secondary.merged_into = primary_id`
5. Merged records (where `merged_into` is set) are excluded from `find()` and `list_all()`
6. `resolve_or_create()` follows the merge chain to the primary record
7. `find_person_ids_for_username()` remaps merged IDs to primary IDs

### Subject authority protects self-confirmed facts

The memory system has three natural protections against accidental supersession:

1. **Self-facts have no subjects** - When a user speaks about themselves, `subject_person_ids` is empty, so their facts never match hearsay about them by subject overlap
2. **Owner-scoped isolation** - Personal memories are scoped to `owner_user_id`, so Alice's hearsay about Bob won't supersede Bob's own memories
3. **One-directional hearsay supersession** - `supersede_confirmed_hearsay` only fires when a user confirms a fact about themselves, superseding others' hearsay

For the RPC/shared-scope edge case where both memories have overlapping `subject_person_ids`, a source-authority check verifies that the new memory's `source_user_id` resolves to a person in `subject_person_ids` before allowing supersession. This prevents third-party hearsay from overwriting first-person facts.

## Resolution Algorithm

Resolution uses graph traversal: starting from the speaker's node and walking their KNOWS/ALIAS edges before falling back to global search.

### Speaker-Scoped Resolution (`find_for_speaker`)

When a speaker context is available (e.g., during extraction with `relationship_stated_by`), resolution starts at the speaker's graph neighborhood:

1. **Load all people** — get the full set of non-merged persons
2. **Filter to speaker's connections** — find persons where the speaker has a KNOWS edge (`RelationshipClaim.stated_by` matching speaker) or ALIAS edge (`AliasEntry.added_by` matching speaker)
3. **Match reference** — among connected persons, match by name, relationship term, or alias value
4. **If found** — return the connected person (speaker-scoped match)
5. **If not found** — fall through to global `find()`

This prevents Alice's coworker Sarah and Bob's wife Sarah from silently merging: when Alice says "Sarah", resolution walks Alice's KNOWS edges first.

### Full `resolve_or_create` Flow

1. **Speaker-scoped search** — If `relationship_stated_by` is provided, call `find_for_speaker(reference, speaker_user_id)` first
2. **Global search** — `find()`: normalize, iterate all records, match against name, relationship, aliases
3. **If found and merged** — Follow `merged_into` chain to reach the primary record
4. **If found and not merged** — Return existing person
5. **Fuzzy match** — LLM-assisted matching if exact match fails
6. **If not found, parse reference** — Extract name and relationship from the reference string
   - If reference starts with "my " and remainder is in `RELATIONSHIP_TERMS`, treat as relationship
   - Use `content_hint` to extract proper name via regex patterns
   - Fall back to title-casing the reference as the name
7. **Create** — New `PersonEntry` with `created_by`, extracted name, relationship claim, and original reference as alias

## Integration with Memory

### Subject Attribution

`subject_person_ids` on `MemoryEntry` links memories to person entities. When a memory is about "Sarah", the system resolves "Sarah" to a person ID and stores it.

### Cross-Context Retrieval

`find_memories_by_subject(person_ids)` searches across all owners:
- Enables facts learned about a person in one context (e.g., group chat) to surface in another (e.g., private chat)
- Privacy filtering still applies (sensitive facts only in private context with the subject)

### System Prompt

`_build_people_section` renders known people into the system prompt, with two adjustments for accuracy:

1. **Self-person filtering** — The current sender's self-person is excluded (avoids "You told me about David Cramer" when David IS the speaker)
2. **Sender-aware relationships** — If the sender has a relationship claim on a person, that relationship is shown. Otherwise all distinct relationships are listed.

```
## Known People

These are people you know about:

- **Sarah** (wife)
- **Bob** (coworker, friend)

Use these when interpreting references like 'my wife' or 'Sarah'.
```

### Supersession Protection

Source-authority check in `supersede_confirmed_hearsay` prevents third-party supersession of self-confirmed facts. The method resolves the new memory's `source_user_id` to person IDs and verifies overlap with `subject_person_ids` before allowing the supersession to proceed.

### Hearsay Directionality

Self-facts (a user speaking about themselves) have empty `subject_person_ids`, so the ABOUT edge is absent. This means:
- Supersession search cannot find self-facts via subject overlap — they are naturally protected
- Hearsay (someone else speaking about a person) has `subject_person_ids` set, creating ABOUT edges
- `supersede_confirmed_hearsay` only fires when a user confirms a fact about themselves, using the ABOUT edges on hearsay to find candidates

### Person Deletion

When a person is deleted via `forget_person`:
1. All memories with ABOUT edges to this person are archived (with `archive_reason="forgotten"`)
2. Archived memories are removed from active store and vector index
3. The person record itself is optionally deleted

## JSONL Schema

### Person Entry

```json
{
  "id": "person-456",
  "version": 1,
  "created_by": "123456789",
  "name": "Sarah",
  "relationships": [
    {"relationship": "wife", "stated_by": "123456789", "created_at": "2026-01-15T10:00:00+00:00"}
  ],
  "aliases": [
    {"value": "my wife", "added_by": "123456789", "created_at": "2026-01-15T10:00:00+00:00"},
    {"value": "sksembhi", "added_by": "987654321", "created_at": "2026-01-20T10:00:00+00:00"}
  ],
  "merged_into": null,
  "created_at": "2026-01-15T10:00:00+00:00",
  "updated_at": "2026-01-20T10:00:00+00:00",
  "metadata": null
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | UUID primary key (system-generated) |
| `version` | Yes | Schema version (currently 1) |
| `created_by` | Yes | Provider user ID of record creator (e.g., Telegram numeric ID as string) |
| `name` | Yes | Primary display name (from provider `full_name`, **not stable**) |
| `relationships` | No | List of `RelationshipClaim` objects with provenance |
| `aliases` | No | List of `AliasEntry` objects with provenance (may be non-unique across people) |
| `merged_into` | No | UUID of primary person if this record was merged |
| `created_at` | Yes | When the record was created |
| `updated_at` | Yes | When the record was last modified |
| `metadata` | No | Extensibility field for additional data |

### AliasEntry

```json
{"value": "sksembhi", "added_by": "987654321", "created_at": "2026-01-20T10:00:00+00:00"}
```

| Field | Required | Description |
|-------|----------|-------------|
| `value` | Yes | The alias string (username, nickname, relationship term — **not guaranteed unique** across people) |
| `added_by` | No | Provider user ID of who added this alias |
| `created_at` | No | When the alias was added |

### RelationshipClaim

```json
{"relationship": "wife", "stated_by": "123456789", "created_at": "2026-01-15T10:00:00+00:00"}
```

| Field | Required | Description |
|-------|----------|-------------|
| `relationship` | Yes | Relationship term from `RELATIONSHIP_TERMS` |
| `stated_by` | No | Provider user ID of who stated this relationship |
| `created_at` | No | When the claim was made |

## Verification

```bash
# Unit tests
uv run pytest tests/test_people.py -v

# Test person resolution
uv run ash chat "My wife Sarah likes hiking"
cat ~/.ash/people.jsonl | grep "Sarah"
# Should show person with name "Sarah", relationship "wife"

# Test alias resolution
uv run ash chat "Tell me about my wife"
# Should resolve "my wife" to Sarah

# Test self-person creation
uv run ash chat "Hello"
cat ~/.ash/people.jsonl | grep '"self"'
# Should show self-person for the current user

# Test global visibility
cat ~/.ash/people.jsonl | python3 -c "import sys,json; [print(json.loads(l).get('name','')) for l in sys.stdin]"
# Should list all people regardless of creator

# Test merging (when implemented via CLI)
# 1. Create two records that are the same person
# 2. Merge them
# 3. Verify secondary has merged_into set
# 4. Verify find() returns primary, not secondary

# Test backward compatibility
# Old format person record should load correctly
echo '{"id":"old-1","version":1,"owner_user_id":"user-1","name":"Bob","relationship":"friend","aliases":["bobby"]}' >> ~/.ash/people.jsonl
# Should load with created_by="user-1", relationships=[RelationshipClaim("friend")], aliases=[AliasEntry("bobby")]

# Verify provenance on aliases
cat ~/.ash/people.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    p = json.loads(line)
    for a in p.get('aliases', []):
        if isinstance(a, dict):
            print(f\"{p['name']}: alias '{a['value']}' added by {a.get('added_by', 'unknown')}\")
"

# Verify provenance on relationships
cat ~/.ash/people.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    p = json.loads(line)
    for r in p.get('relationships', []):
        if isinstance(r, dict):
            print(f\"{p['name']}: {r['relationship']} stated by {r.get('stated_by', 'unknown')}\")
"
```

### Checklist

- [ ] "my wife", "Sarah", "@sksembhi" all resolve to the same person entity
- [ ] Resolution is case-insensitive
- [ ] Prefixes "my ", "the ", "@" stripped during normalization
- [ ] `resolve_or_create` creates new person when reference is unknown
- [ ] Relationship terms extracted from "my X" references
- [ ] Content hints used to extract proper names from surrounding text
- [ ] Aliases stored as `AliasEntry` with `added_by` provenance
- [ ] Relationships stored as `RelationshipClaim` with `stated_by` provenance
- [ ] Old plain string aliases loaded as `AliasEntry(value=x, added_by=None)`
- [ ] Old `relationship` string loaded as `RelationshipClaim(relationship=x, stated_by=None)`
- [ ] Old `owner_user_id` key maps to `created_by` on load
- [ ] Self-person created automatically with `relationship="self"`
- [ ] Self-person links username to display name via alias
- [ ] All users see all people (global visibility)
- [ ] `created_by` tracks record provenance (not access control)
- [ ] `find()` searches all records, not filtered by owner
- [ ] `find_person_ids_for_username()` searches across all owners
- [ ] Merged persons have `merged_into` pointing to primary ID
- [ ] Merged records excluded from `find()` and `list_all()`
- [ ] `resolve_or_create()` follows merge chain to primary
- [ ] Known people appear in system prompt via `_build_people_section`
- [ ] `subject_person_ids` links memories to person entities
- [ ] Cross-context retrieval finds facts about people from other owners
- [ ] Source-authority check prevents third-party supersession of self-confirmed facts
- [ ] `people.jsonl` is the source of truth
- [ ] In-memory cache invalidated after writes
- [ ] Cache refreshed when file mtime changes
