# Memory Foundation

## Intent

The memory subsystem solves a fundamental problem: conversations are ephemeral, but users expect the agent to remember important facts about them. Memory provides:

1. **Automatic context** - Relevant facts surface before each response without user intervention
2. **Explicit storage** - Users can tell the agent to remember specific things
3. **Person awareness** - Facts can be attributed to people in the user's life
4. **Knowledge evolution** - New information supersedes outdated facts
5. **Smart decay** - Ephemeral facts expire naturally based on their type

**Scale expectations:** Tens of users in a chat, each with thousands of memories. Operations like doctor, search, and GC must handle this without degrading.

Memory is NOT for:
- Conversation history (that's sessions)
- Temporary task state (that's session context)
- Credentials or secrets (security risk)


## Memory Types

Memories are classified into types that determine their lifecycle:

### Long-Lived Types (no automatic expiration)

| Type | Description | Examples |
|------|-------------|----------|
| `preference` | User likes, dislikes, preferences | "prefers dark mode", "favorite color is blue" |
| `identity` | Facts about the user themselves | "works as a software engineer", "lives in SF" |
| `relationship` | People in user's life | "Sarah is my wife", "has a dog named Max" |
| `knowledge` | Factual information | "project X uses Python", "company uses Slack" |

### Ephemeral Types (decay over time)

| Type | Default TTL | Description | Examples |
|------|-------------|-------------|----------|
| `context` | 7 days | Current situation/state | "working on project X" |
| `event` | 30 days | Past occurrences | "had dinner with Sarah Tuesday" |
| `task` | 14 days | Things to do/remember | "needs to call dentist" |
| `observation` | 3 days | Fleeting observations | "mentioned being tired" |

### Type Assignment

Types are assigned during:
1. **Explicit remember**: User says "remember X" → LLM infers type from content
2. **Background extraction**: LLM extracts and classifies during conversation
3. **CLI/RPC add**: Optional `type` parameter, defaults to `knowledge`


## Scoping

Memories have visibility scope controlled by two fields:

| Scope | `owner_user_id` | `chat_id` | Visible to |
|-------|-----------------|-----------|------------|
| Personal | Set | NULL | Only the user who created it |
| Group | NULL | Set | Everyone in the chat |

**Scoping rules:**
- Personal memories: `owner_user_id` is set, `chat_id` is NULL
- Group memories: `owner_user_id` is NULL, `chat_id` is set
- When retrieving, a user sees their personal memories + any group memories for the current chat

| Scope | Use case |
|-------|----------|
| Personal | Default - "I like coffee" |
| Group | Team facts - "Our standup is at 9am" |

## Authorization

Best-effort authorization prevents accidental cross-user data access. When context is provided:

| Operation | Rule |
|-----------|------|
| Delete memory | Must be owner (personal) or member of chat (group) |
| Update person | Must be the person's creator (`created_by`) or the person themselves |
| Add person alias | Must be the person's creator (`created_by`) or the person themselves |
| Reference person in memory | Any user (people are globally visible) |
| Get memories about person | Filtered to accessible memories (privacy rules apply) |

**Design notes:**
- Authorization is best-effort, not security boundary (sandbox can't be fully trusted)
- Operations without context (e.g., internal GC) skip checks
- Goal: prevent accidental mistakes, not malicious actors

## Privacy & Sensitivity

Memories can be classified by sensitivity level to control sharing:

### Sensitivity Levels

| Level | Description | Sharing Rules |
|-------|-------------|---------------|
| `public` | Can be shared anywhere (default) | No restrictions |
| `personal` | Personal details not for group disclosure | Only with subject or memory owner |
| `sensitive` | High privacy (medical, financial, mental health) | Only in private chat with subject |

### Examples

| Content | Sensitivity | Why |
|---------|-------------|-----|
| "Prefers dark mode" | public | General preference |
| "Looking for a new job" | personal | Sensitive career info |
| "Has anxiety" | sensitive | Mental health |
| "Salary is $150k" | sensitive | Financial |
| "Is pregnant, due August" | sensitive | Medical/health |
| "Has a dog named Max" | public | General fact |

### Privacy Filtering

Privacy rules depend on the chat context:

**Group chats:**

| Sensitivity | Subject is participant | Subject is NOT participant |
|-------------|----------------------|--------------------------|
| PUBLIC | Shown | Shown |
| PERSONAL | Shown | Excluded |
| SENSITIVE | Shown | Excluded |

**Private chats (DMs) — contextual disclosure:**

A memory is disclosable if any of:
- Memory is ABOUT the DM partner
- Memory was STATED_BY the DM partner
- DM partner has a PARTICIPATES_IN edge to the memory's LEARNED_IN chat (they were present when it was said)
- Memory is a self-memory (no subjects)
- Memory has no LEARNED_IN edge (legacy data — fail-open)

Otherwise, memories about third parties are excluded.

**Cross-context retrieval:**

1. **PUBLIC**: Always shown
2. **PERSONAL**: Only shown to the subject person or memory owner
3. **SENSITIVE**: Only shown in private chat with the subject

### Backward Compatibility

- `sensitivity` field defaults to `null`, treated as `public`
- Existing memories work unchanged


## Cross-Context Retrieval

Facts learned about a person in one context (e.g., group chat) can be recalled in another context (e.g., private chat with that person), subject to privacy rules.

### How It Works

1. When Alice mentions "@bob loves pizza" in a group chat:
   - Memory stored with `subject_person_ids=[bob_person_id]`
   - Person record links username "bob" to the person entity

2. When Bob starts a private chat:
   - System finds memories where Bob is a subject (across all owners)
   - Privacy filter applies: sensitive facts only shown to Bob himself
   - Facts from other users become available context

### Cross-Context Query

`find_memories_by_subject(person_ids)` searches across ALL owners:
- Caller resolves person IDs first via `PersonManager`
- Returns memories where that person is a subject
- Only includes **portable** memories (enduring facts about a person, not chat-operational)
- Optional `exclude_owner_user_id` to avoid double-counting

### Privacy in Groups

When someone asks about a person in a group chat:
- **PUBLIC** facts: shown
- **PERSONAL** facts: shown only if the subject is a chat participant
- **SENSITIVE** facts: shown only if the subject is a chat participant

### Self-Query Example

When Bob asks "what do you know about me?" in different contexts:

| Context | PUBLIC | PERSONAL | SENSITIVE |
|---------|--------|----------|-----------|
| Private chat with Bob | ✓ | ✓ | ✓ |
| Group chat (Bob asking) | ✓ | ✓ | ✗ |
| Group chat (others asking) | ✓ | ✗ | ✗ |


## Portable Memories

Not all group-scoped memories about a person should cross contexts. The `portable` field controls whether a memory's ABOUT edge is traversable from outside the originating chat.

| Content | Portable | Why |
|---------|----------|-----|
| "Bob loves pizza" | true | Enduring trait about the person |
| "Bob is presenting next" | false | Chat-operational, only relevant here |
| "Sarah's birthday is March 15" | true | Enduring fact |
| "Sarah will send the report by EOD" | false | Ephemeral task context |

**Default:** `true` for backward compatibility. Extraction prompt guides the LLM to set `portable=false` for chat-operational facts.

## Known Limitations

- **Contradictory group facts**: no authority model — last-write-wins. Two users stating conflicting facts about the same subject both persist.
- **Hearsay directionality**: Self-facts have the speaker's person_id injected into `subject_person_ids` for graph traversal (so they're discoverable when asking "what do you know about Bob?"). Hearsay supersession uses the original extraction state (no subjects = self-fact) to trigger cross-scope lookup, not the post-injection `subject_person_ids`.

## Configuration

```toml
[memory]
auto_gc = true              # Clean up on startup
max_entries = 1000          # Cap on active memories (optional)
extraction_enabled = true   # Background extraction
extraction_confidence_threshold = 0.7
```
