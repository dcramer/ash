# Telegram Provider

> Telegram bot integration using aiogram 3.x

Files: src/ash/providers/telegram/provider.py, src/ash/providers/telegram/handlers.py, src/ash/providers/base.py

## Requirements

### MUST

- Support polling mode (default, no external server needed)
- Support webhook mode (for production with server)
- Authenticate users via allowed_users list
- Silently ignore unauthorized users
- Convert Telegram messages to internal IncomingMessage format
- Send messages via OutgoingMessage format
- Support message reply threading
- Restore conversation history from database on session start
- Send typing indicator before processing messages
- Handle /start and /help commands

### SHOULD

- Support streaming responses via message editing
- Rate limit message edits (Telegram limit: ~1/second)
- Support Markdown parsing in messages
- Support message editing
- Support message deletion
- Accept photo messages and pass to handler
- Set reaction on message when processing starts (👀)
- Clear reaction when processing completes or errors
- Show "Thinking..." message during tool execution

### MAY

- Support inline keyboards
- Support document/file attachments
- Image understanding through integration preprocess hooks

## Group Chat

### MUST

- Support group and supergroup chat types
- Share session context across all users in a group
- Default to mention-only mode (@botname) in groups
- Strip bot mention from message text before processing
- Cache bot username on startup for mention detection

### SHOULD

- Support configurable group_mode ("mention" | "always")
- Support allowed_groups configuration (empty = all groups with authorized users)
- Apply same user authorization in groups as DMs

## Interface

```python
class TelegramProvider(Provider):
    def __init__(
        self,
        bot_token: str,
        allowed_users: list[str] | None = None,  # usernames or IDs
        webhook_url: str | None = None,
        webhook_path: str = "/telegram/webhook",
        allowed_groups: list[str] | None = None,  # group IDs (empty = all)
        group_mode: str = "mention",  # "mention" or "always"
    ): ...

    @property
    def name(self) -> str  # "telegram"
    @property
    def bot(self) -> Bot
    @property
    def dispatcher(self) -> Dispatcher

    async def start(handler: MessageHandler) -> None
    async def stop() -> None

    async def send(message: OutgoingMessage) -> str  # returns message_id
    async def send_streaming(
        chat_id: str,
        stream: AsyncIterator[str],
        reply_to: str | None = None,
    ) -> str

    async def edit(
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
    ) -> None

    async def delete(chat_id: str, message_id: str) -> None

    async def process_webhook_update(update_data: dict) -> None
```

### Message Types

```python
@dataclass
class ImageAttachment:
    file_id: str
    width: int | None
    height: int | None
    file_size: int | None
    mime_type: str | None
    data: bytes | None  # Populated after download

@dataclass
class IncomingMessage:
    id: str
    chat_id: str
    user_id: str
    text: str
    username: str | None
    display_name: str | None
    reply_to_message_id: str | None
    images: list[ImageAttachment]  # Photo attachments
    metadata: dict[str, Any]  # chat_type, chat_title

    @property
    def has_images(self) -> bool

@dataclass
class OutgoingMessage:
    chat_id: str
    text: str
    reply_to_message_id: str | None = None
    parse_mode: str | None = None  # "markdown", "html"

MessageHandler = Callable[[IncomingMessage], Awaitable[None]]
```

## Configuration

```toml
[telegram]
bot_token = "..."  # or TELEGRAM_BOT_TOKEN env
allowed_users = ["@username", "123456789"]
webhook_url = "https://example.com"  # optional, uses polling if absent
webhook_path = "/telegram/webhook"
# Group chat settings
allowed_groups = []  # Group IDs (empty = allow all with authorized users)
group_mode = "mention"  # "mention" (default) or "always"
```

## Message UX

### Reactions

- Set 👀 reaction on user message immediately when processing starts
- Clear reaction when response is sent (success or error)
- Reactions are best-effort (may not work in all chats)

### Thinking Message

When the agent uses tools, show progress via a single "Thinking..." message:

1. First tool call → Send "_Thinking..._" as reply to user message
2. Progress messages (via `send_message` tool) → Edited into the same message ABOVE "Thinking..."
3. "Thinking..." always stays at the bottom of the message
4. Response ready → Remove "Thinking...", keep progress messages, append response

Layout during processing:
```
[progress message 1]
[progress message 2]

_Thinking..._
```

Layout in final response:
```
[progress message 1]
[progress message 2]

[response content]
```

No tool call counts or timing stats are shown.

**No tools used**: Skip thinking message, send response directly.

### Hybrid Streaming

For streaming mode, balance responsiveness with API efficiency:

1. Accumulate response for first 5 seconds (STREAM_DELAY)
2. If still generating after 5 seconds, start showing partial content
3. Rate-limit edits to 1 second minimum (MIN_EDIT_INTERVAL)
4. Final edit/send always includes complete response

**Fast responses (<5s)**: Sent as single message, no streaming edits.
**Slow responses (>5s)**: Start streaming after delay, rate-limited edits.

## Behaviors

| Scenario | Behavior |
|----------|----------|
| Polling mode (no webhook_url) | Deletes webhook, starts long polling |
| Webhook mode | Sets webhook URL, waits for updates |
| Unauthorized user message | Log warning, ignore (no response) |
| Authorized user message | Convert to IncomingMessage, call handler |
| /start command | Send welcome message with bot introduction |
| /help command | Send help message listing capabilities |
| Photo message | Download photo, create IncomingMessage with image attachment |
| Photo with caption | Analyze image and inject structured image context before agent processing |
| Photo without caption | Analyze image and respond with concise description plus one clarifying follow-up question |
| New session | Restore up to 50 messages from database |
| Before processing | Set 👀 reaction, send typing indicator |
| Tool execution | Show "Thinking..." message, update per tool |
| Streaming response | Accumulate 5s, then edit with content, rate limited to 1/sec |
| Final streaming edit | Always edit with complete content |
| Response complete | Clear 👀 reaction |
| Parse mode specified | Use Telegram's markdown/HTML parsing |
| Group message (mention mode) | Only respond when @botname mentioned |
| Group message (always mode) | Respond to all messages from authorized users |
| Group message with mention | Strip @botname from text before processing |
| Group not in allowed_groups | Ignore message silently |

## Errors

| Condition | Response |
|-----------|----------|
| Invalid bot token | aiogram raises on start |
| User not in allowed_users | Silent ignore, log warning |
| Edit rate limit exceeded | Logged, skip edit (final edit still attempted) |
| Message edit failed | Log warning, continue |
| Webhook processing error | Log exception, return 200 (prevent retry) |

## Verification

```bash
uv run pytest tests/test_providers.py -v
# Manual: Start bot, send message as allowed user
```

- [ ] Polling mode starts without webhook
- [ ] Unauthorized users ignored
- [ ] Messages converted to IncomingMessage
- [ ] Streaming edits respect rate limit
- [ ] Webhook updates processed correctly
- [ ] /start command returns welcome message
- [ ] /help command returns capabilities
- [ ] Session messages restored from database
- [ ] Typing indicator sent before processing
- [ ] Photo messages acknowledged
- [ ] Group messages ignored without mention (mention mode)
- [ ] Group messages responded to with mention
- [ ] Bot mention stripped from message text
- [ ] 👀 reaction set on message receipt, cleared after response
- [ ] "Thinking..." message shown only during tool use
- [ ] "Thinking..." message updates with each tool status
- [ ] Fast responses (<5s) sent as single message
- [ ] Slow responses (>5s) stream with rate-limited edits
