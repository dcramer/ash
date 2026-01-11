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

### MAY

- Support inline keyboards
- Support document/file attachments
- Full vision model integration for image analysis

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
| Photo with caption | Process caption with image context |
| Photo without caption | Acknowledge receipt, suggest adding caption |
| New session | Restore up to 50 messages from database |
| Before processing | Send typing indicator |
| Streaming response | Send "...", edit with content, rate limited to 1/sec |
| Final streaming edit | Always edit with complete content |
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
