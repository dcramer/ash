# Gmail Workflows

Use this reference when the user asks for inbox triage, email summaries, or a day-at-a-glance view.

## Query Recipes

Use `search_messages` with focused Gmail query syntax.

| Goal | Query |
|------|-------|
| Unread in last day | `is:unread newer_than:1d` |
| Unread in last week | `is:unread newer_than:7d` |
| Important/unread | `is:important is:unread` |
| From a person/domain | `from:alice@example.com` or `from:@example.com` |
| Time-sensitive subjects | `subject:(urgent OR asap OR today)` |
| Action requests | `("can you" OR "please review" OR "action required") newer_than:7d` |

Prefer a narrow query first, then broaden if needed.

## Summarize Emails Workflow

1. Run `search_messages` (or `list_messages` when folder-based scope is requested).
2. Select top candidates by recency + urgency language + sender relevance.
3. Run `get_message` for each selected message before summarizing.
4. Summarize with actionable bullets tied to sender and subject.

Do not rely on snippets when full message content is available.

## Day-At-A-Glance Workflow (Email Portion)

1. Start with `search_messages` default query: `is:unread newer_than:1d`.
2. If results are sparse, broaden to `newer_than:2d` or remove `is:unread`.
3. Pull full content for top 3-8 messages with `get_message`.
4. Prioritize messages that imply deadlines, asks, or pending replies.

## Thread Usage

Use `get_thread` when:

- The user asks for context on a conversation.
- A message implies prior back-and-forth and reply context matters.
- You need to summarize the evolution of a decision.

Use `get_message` when:

- The user asks about one specific email.
- You only need the latest message content.
