# Memory Extraction

## Background Extraction

Optionally, facts are extracted automatically from conversations:
- Runs after each exchange (debounced)
- Extracts preferences, facts about people, important dates
- Skips assistant actions, temporary context, credentials
- Confidence threshold filters low-quality extractions

## Secrets Filtering

Memory will NEVER store credentials or secrets. Three-layer defense:

1. **Extraction prompt** - LLM instructed to reject secrets during extraction
2. **Post-extraction filter** - Regex patterns catch secrets before storage
3. **Centralized filter in MemoryManager** - Final check catches all entry points (CLI, RPC, direct)

### Automatically Rejected

| Type | Examples |
|------|----------|
| Passwords | "my password is X", "passwd: hunter2" |
| API Keys | sk-..., ghp_..., gho_..., AKIA... |
| SSN | 123-45-6789 |
| Credit Cards | 16-digit numbers with optional separators |
| Private Keys | -----BEGIN PRIVATE KEY----- |
| Slack Tokens | xoxb-..., xoxp-..., xoxs-... |

Even if the user explicitly asks to remember these, they will be rejected with an error.

## Temporal Context

Facts with relative time references are automatically converted to absolute dates during extraction.

### How It Works

1. Current datetime is passed to the extraction prompt
2. LLM instructed to rewrite relative references → absolute dates
3. `observed_at` field records when the fact was stated

### Examples

| User says | Stored as |
|-----------|-----------|
| "this weekend" | "the weekend of Feb 15-16, 2026" |
| "next Tuesday" | "Tuesday, Feb 18, 2026" |
| "tomorrow" | "Feb 12, 2026" |
| "in 2 days" | "Feb 14, 2026" |

This ensures memories remain meaningful when recalled weeks or months later.

## Multi-User Attribution

Memory supports tracking WHO provided each fact, enabling trust-based reasoning:

### Source Attribution Fields

| Field | Description |
|-------|-------------|
| `source_username` | Username/handle of who stated this fact |
| `source_display_name` | Display name of the source user |
| `subject_person_ids` | Who the memory is ABOUT (third parties) |

### Trust Model

| Source == Subject? | Type | Trustworthiness |
|-------------------|------|-----------------|
| Yes (speaking about self) | **FACT** | High - first-person claim |
| No (speaking about others) | **HEARSAY** | Lower - second-hand claim |

### Examples

| Who said it | Content | Source User | Subjects | Trust |
|-------------|---------|-------------|----------|-------|
| David | "I like pizza" | david | [] | FACT |
| David | "Bob likes pasta" | david | [bob] | HEARSAY |
| Bob | "I like pasta" | bob | [] | FACT |

### CLI Display

The `ash memory list` command shows:
- **About**: Subject person(s), or source user if subjects is empty (speaking about self)
- **Source**: Who provided the information (`@username`)
- **Trust**: "fact" or "hearsay"

The `ash memory show` command displays full attribution details.

### Extraction with Speaker Identity

During background extraction, messages are formatted with speaker identity:
```
@david (David Cramer): I like pizza
@bob: Bob prefers pasta
Assistant: Great choices!
```

The LLM then attributes each extracted fact to the appropriate speaker.

## Multi-Subject Facts

Facts can be about multiple people simultaneously. The `subject_person_ids` list supports multiple entries, creating ABOUT edges to each person.

### Joint Facts

When a fact inherently involves multiple people as participants (not just one person reporting about another), all participants should be subjects:

| Statement | Subjects | Why |
|-----------|----------|-----|
| "Alice and Bob are starting a company" | [Alice, Bob] | Joint venture, both are participants |
| "The team is relocating to Austin" | [Alice, Bob, Carol] | Affects all team members |
| "My coworker and I are working on project X" | [coworker] + speaker | Joint effort |

### Speaker as Subject

When the speaker is one of multiple participants in a joint fact, they should be included as a subject. The processing pipeline only treats the speaker as a self-fact (subjects=[]) when they are the SOLE subject — not when they are one of several.

| Statement | subjects (extraction) | subject_person_ids (after processing) |
|-----------|----------------------|--------------------------------------|
| "I like pizza" | [] | [speaker_pid] (self-fact injection) |
| "Alice and I started a company" | ["Alice", speaker_name] | [alice_pid, speaker_pid] |
| "Alice started a company" | ["Alice"] | [alice_pid] |
