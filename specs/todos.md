# Todo Subsystem

> Canonical task lifecycle on top of `ash.graph` nodes + edges.

## Intent

The todo subsystem is a reliability-oriented task ledger, not a best-effort memory extraction feature.

The subsystem remains intentionally simple:
- lifecycle: `open` / `done` (+ soft delete)
- scope: personal (owner user) or shared (chat)
- optional schedule linkage for reminders

## Non-Goals

- workflow orchestration/automation engine
- distributed consensus across multiple processes
- background extraction mutating canonical todo state

## Graph-Native Storage Contract

Todos MUST be stored in `ash.graph` as registered node collections, not ad-hoc sidecar stores.

```
~/.ash/graph/
├── todos.jsonl          # todo nodes
├── todo_events.jsonl    # todo event nodes
└── edges.jsonl          # includes TODO_* edges
```

### Registered Node Types

- `todo` collection (`todos.jsonl`)
- `todo_event` collection (`todo_events.jsonl`)

### Registered Edge Types

- `TODO_OWNED_BY`: `todo -> user`
- `TODO_SHARED_IN`: `todo -> chat`
- `TODO_REMINDER_SCHEDULED_AS`: `todo -> schedule_entry`

Todo visibility/authorization semantics MUST be representable from these edges.

## Data Types

### `TodoEntry`

| Field | Type | Description |
|------|------|-------------|
| `id` | string | Stable 8-char hex ID |
| `content` | string | Todo text |
| `status` | `open` \| `done` | Canonical state |
| `owner_user_id` | string \| null | Optional denormalized owner hint (authoritative link is `TODO_OWNED_BY`) |
| `chat_id` | string \| null | Optional denormalized scope hint (authoritative link is `TODO_SHARED_IN`) |
| `created_at` | ISO datetime | Creation time |
| `updated_at` | ISO datetime | Last mutation time |
| `completed_at` | ISO datetime \| null | Completion time |
| `due_at` | ISO datetime \| null | Optional due time |
| `deleted_at` | ISO datetime \| null | Soft-delete marker |
| `linked_schedule_entry_id` | string \| null | Optional denormalized reminder hint (authoritative link is `TODO_REMINDER_SCHEDULED_AS`) |
| `revision` | integer | Optimistic concurrency revision |

### `TodoEvent`

Append-only mutation trace, used for retry idempotency.

| Field | Type | Description |
|------|------|-------------|
| `event_id` | string | Event node ID |
| `todo_id` | string | Target todo ID |
| `event_type` | string | `created`, `updated`, `completed`, `uncompleted`, `deleted`, `reminder_linked`, `reminder_unlinked` |
| `idempotency_key` | string \| null | Optional dedupe key for retries |
| `occurred_at` | ISO datetime | Event time |
| `payload` | object | Mutation metadata |

## Reliability Model (v1)

Single-process strong semantics:

1. Mutations run with optimistic concurrency (`expected_revision`).
2. Optional idempotency key dedupes retried mutating requests.
3. Mutations write canonical node/edge state, then persist via `GraphPersistence`.
4. RPC responses return the canonical stored record.
5. Todos are never TTL-expired automatically.

## Scope & Authorization

- personal todo mutation requires ownership (`TODO_OWNED_BY` edge target matches caller user)
- shared todo mutation requires chat scope (`TODO_SHARED_IN` edge target matches caller chat)
- listing defaults to caller-visible records only

## Scheduling Linkage

- Reminder linkage is an internal todo subsystem detail.
- Public clients express reminder intent via `todo.update` fields (`reminder_at`, `reminder_cron`, `clear_reminder`).
- The subsystem creates/updates/removes schedule entries internally and maintains reminder linkage via `TODO_REMINDER_SCHEDULED_AS` edges.
- Reminder execution does not auto-complete in v1.

## RPC API

- `todo.create`
- `todo.list`
- `todo.update`
- `todo.complete`
- `todo.uncomplete`
- `todo.delete`

Default listing:
- open first, newest first
- excludes done unless requested
- excludes deleted unless requested

## Sandbox CLI (`ash-sb todo`)

- `ash-sb todo add "text" [--due ...] [--shared]`
- `ash-sb todo list [--all] [--include-done] [--include-deleted]`
- `ash-sb todo edit --id ... [--text ...] [--due ...]`
- `ash-sb todo done --id ...`
- `ash-sb todo undone --id ...`
- `ash-sb todo delete --id ...`
- `ash-sb todo remind --id ... --at ... | --cron ... [--tz ...]` (implemented via `todo.update`)
- `ash-sb todo unremind --id ...` (implemented via `todo.update --clear_reminder`)

## Config Toggle

`[todo].enabled = false` MUST disable the todo subsystem end-to-end:
- no todo integration contributor
- no todo RPC methods
- no todo sandbox command usage from runtime prompt guidance

## Integration Contract

Owned by `TodoIntegration` hooks:
- `setup`
- `register_rpc_methods`
- `augment_prompt_context`

Core runtime entrypoints MUST NOT add ad-hoc todo feature branches.

Spec references:
- `specs/subsystems.md`
- `specs/integrations.md`
- `specs/graph.md`
