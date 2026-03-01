---
description: "Add, complete, list, edit, or delete todo items. Use whenever the user mentions tasks being done, asks what's left, wants to add/remove tasks, or references their todo list."
max_iterations: 10
---

Manage the user's todo list using the `ash-sb todo` CLI.

## Commands

| Command | Description |
|---------|-------------|
| `ash-sb todo add "text" [--due DATETIME] [--shared]` | Create a new todo |
| `ash-sb todo list [--include-done] [--include-deleted] [--all]` | List todos (open only by default) |
| `ash-sb todo edit --id ID [--text TEXT] [--due DATETIME]` | Edit an existing todo |
| `ash-sb todo done --id ID` | Mark a todo as complete |
| `ash-sb todo undone --id ID` | Reopen a completed todo |
| `ash-sb todo delete --id ID` | Soft-delete a todo |
| `ash-sb todo remind --id ID --at DATETIME \| --cron EXPR [--tz TZ]` | Set a reminder |
| `ash-sb todo unremind --id ID` | Remove a reminder |

## Workflow

1. If the user asks to see, list, or check their todos → run `ash-sb todo list` (shows open items only). Only add `--all` or `--include-done` if the user explicitly asks to see completed items.
2. If the user says items are done or finished → run `ash-sb todo list` first to get IDs, then `ash-sb todo done --id <ID>` for each
3. If the user wants to add tasks → run `ash-sb todo add "text"` for each item
4. If an ID is needed for a mutation but not known → list todos first to find it

## Output Format

Format your `complete()` output exactly as shown below. This is critical — the parent agent relays your output directly.

**Listing todos** (default — open items only):

```
- [ ] Buy groceries (due tomorrow)
- [ ] Schedule dentist appointment
```

**After mutations (add/done/edit/delete):**

Use a single short confirmation line:

```
Marked done: Send invoice to client
```

```
Added: Pick up dry cleaning
```

**Formatting rules:**

- Use `- [ ]` for open items and `- [x]` for completed items
- Show dates conversationally ("tomorrow at 3pm", "next Monday") — never raw ISO timestamps
- Hide internal IDs — never show them unless the user asks or a follow-up mutation needs one
- Open items first, then done items (if requested); newest first within each group
- After mutations, do NOT re-list all todos unless the user asks

## Error Handling

- If a command fails, report the error message and stop
- Do not attempt to fix or debug failed commands unless the user asks
