# Provider Integrations Comparison

This document compares how four codebases handle provider integrations for messaging platforms like Telegram, Slack, Discord, and others.

## Overview

Provider integrations are the bridge between external messaging platforms and the agent's core functionality. They handle:

- **Authentication and authorization** - Validating tokens, checking user permissions
- **Message reception** - Polling, webhooks, or socket connections for incoming messages
- **Message sending** - Text formatting, chunking, media uploads, threading
- **Typing indicators** - Showing activity while processing
- **Rate limiting** - Respecting platform API constraints
- **Error handling** - Graceful degradation on failures

## Comparison Table

| Feature | ash (Python) | archer (TypeScript) | clawdbot (TypeScript) | pi-mono (TypeScript) |
|---------|-------------|---------------------|----------------------|---------------------|
| **Providers** | Telegram, CLI | Telegram | Telegram, Slack, Discord, Signal, WhatsApp, iMessage, Teams | Slack |
| **Library** | aiogram 3.x | grammY | grammY, @slack/bolt, discord.js, signal-cli | @slack/socket-mode, @slack/web-api |
| **Abstraction** | Provider ABC with IncomingMessage/OutgoingMessage | TelegramBotWrapper class | Per-provider modules (accounts, monitor, send, format) | SlackBot class |
| **Auth Model** | allowed_users + allowed_groups lists | permittedUsers Set | Per-channel config with allow lists | Channel membership |
| **Message Handling** | MessageHandler callback | ArcherHandler interface | dispatchReplyFromConfig routing | MomHandler interface |
| **Typing Indicators** | send_typing() + typing loop | sendChatAction() | Built into reply dispatcher | Not explicit |
| **Threading** | thread_id in metadata | threadId parameter | resolveThreadTargets | thread_ts |
| **Streaming** | send_streaming() with edits | Message editing | Draft streams with chunking | Not implemented |

## Detailed Analysis

### 1. ash (Python)

ash uses a clean abstract base class pattern with Python's ABC module to define a provider interface.

**Provider Base Class** (`/home/dcramer/src/ash/src/ash/providers/base.py`):

```python
@dataclass
class IncomingMessage:
    """Message received from a provider."""
    id: str
    chat_id: str
    user_id: str
    text: str
    username: str | None = None
    display_name: str | None = None
    reply_to_message_id: str | None = None
    images: list[ImageAttachment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None

class Provider(ABC):
    """Abstract interface for communication providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'telegram', 'discord')."""
        ...

    @abstractmethod
    async def start(self, handler: MessageHandler) -> None:
        """Start the provider and begin receiving messages."""
        ...

    @abstractmethod
    async def send(self, message: OutgoingMessage) -> str:
        """Send a message."""
        ...

    @abstractmethod
    async def send_streaming(
        self,
        chat_id: str,
        stream: AsyncIterator[str],
        *,
        reply_to: str | None = None,
    ) -> str:
        """Send a message with streaming updates."""
        ...
```

**Authorization Model** (`/home/dcramer/src/ash/src/ash/providers/telegram/provider.py`):

```python
def _is_user_allowed(self, user_id: int, username: str | None) -> bool:
    """Check if a user is allowed to interact with the bot."""
    if not self._allowed_users:
        return True  # Empty list = open access

    return str(user_id) in self._allowed_users or (
        username is not None and f"@{username}" in self._allowed_users
    )

def _is_group_allowed(self, chat_id: int) -> bool:
    """Check if a group is allowed."""
    if not self._allowed_groups:
        return True
    return str(chat_id) in self._allowed_groups
```

**Typing Indicator Loop** (`/home/dcramer/src/ash/src/ash/providers/telegram/handlers.py`):

```python
async def _typing_loop(self, chat_id: str) -> None:
    """Send typing indicators in a loop.

    Telegram typing indicators only last 5 seconds, so we need to
    keep sending them for long operations.
    """
    while True:
        try:
            await self._provider.send_typing(chat_id)
            await asyncio.sleep(4)  # Refresh before 5 second timeout
        except asyncio.CancelledError:
            break
```

**Key Design Choices**:
- Clean separation between provider interface (base.py) and implementation (telegram/)
- Dataclasses for message types ensure immutability and type safety
- Per-chat locks serialize message handling to prevent race conditions
- Streaming support with automatic message editing and rate limiting

---

### 2. archer (TypeScript)

archer uses a wrapper class pattern around the grammY library with per-chat message queuing.

**Bot Wrapper** (`/home/dcramer/src/archer/src/telegram.ts`):

```typescript
export class TelegramBotWrapper {
    private bot: Bot;
    private handler: ArcherHandler;
    private workingDir: string;
    private permittedUsers: Set<string> | null = null;
    private queues = new Map<string, ChatQueue>();

    isUserPermitted(userId: string): boolean {
        if (!this.permittedUsers) {
            return true; // No allow list = open access
        }
        return this.permittedUsers.has(userId);
    }
}
```

**Per-Chat Queue for Sequential Processing**:

```typescript
class ChatQueue {
    private queue: QueuedWork[] = [];
    private processing = false;

    enqueue(work: QueuedWork): void {
        this.queue.push(work);
        this.processNext();
    }

    private async processNext(): Promise<void> {
        if (this.processing || this.queue.length === 0) return;
        this.processing = true;
        const work = this.queue.shift()!;
        try {
            await work();
        } catch (err) {
            log.logWarning("Queue error", err instanceof Error ? err.message : String(err));
        }
        this.processing = false;
        this.processNext();
    }
}
```

**Markdown to HTML Conversion**:

```typescript
export function markdownToTelegramHtml(text: string): string {
    // Converts Markdown to Telegram-compatible HTML
    // Supports: **bold**, *italic*, `code`, ```code blocks```, [text](url)
    // Handles edge cases like nested formatting and HTML tag passthrough
}
```

**Media Type Detection for File Uploads**:

```typescript
async uploadFile(chatId: string, filePath: string, title?: string, threadId?: number): Promise<void> {
    const ext = filePath.toLowerCase().split(".").pop() || "";

    // Images: sendPhoto for inline preview
    if (["jpg", "jpeg", "png", "webp"].includes(ext)) {
        await this.bot.api.sendPhoto(chatId, inputFile, opts);
        return;
    }

    // GIFs: sendAnimation for inline animated preview
    if (ext === "gif") {
        await this.bot.api.sendAnimation(chatId, inputFile, opts);
        return;
    }

    // Videos, Audio, Voice, etc. with appropriate API calls
    // ...

    // Everything else: sendDocument
    await this.bot.api.sendDocument(chatId, inputFile, opts);
}
```

**Key Design Choices**:
- Single wrapper class encapsulates all Telegram functionality
- Message queue ensures sequential processing per chat
- Comprehensive media type detection for optimal file uploads
- JSONL logging to working directory for conversation history

---

### 3. clawdbot (TypeScript)

clawdbot has the most comprehensive provider support with a modular architecture per provider.

**Provider Module Pattern** - Each provider follows a consistent structure:

```
src/telegram/
  accounts.ts    - Account resolution and configuration
  monitor.ts     - Message polling/webhook handling
  send.ts        - Outbound message sending
  format.ts      - Markdown/text formatting
  probe.ts       - Health checks and diagnostics
  bot.ts         - Main bot logic (Telegram-specific)
```

**Account Resolution** (`/home/dcramer/src/clawdbot/src/telegram/accounts.ts`):

```typescript
export type ResolvedTelegramAccount = {
    accountId: string;
    enabled: boolean;
    name?: string;
    token: string;
    tokenSource: "env" | "tokenFile" | "config" | "none";
    config: TelegramAccountConfig;
};

export function resolveTelegramAccount(params: {
    cfg: ClawdbotConfig;
    accountId?: string | null;
}): ResolvedTelegramAccount {
    // Handles account resolution with fallback logic
    // Supports multiple token sources: env vars, files, config
}
```

**Channel-Based Authorization** (`/home/dcramer/src/clawdbot/src/slack/monitor.ts`):

```typescript
function resolveSlackChannelConfig(params: {
    channelId: string;
    channelName?: string;
    channels?: Record<string, {
        enabled?: boolean;
        allow?: boolean;
        requireMention?: boolean;
        allowBots?: boolean;
        users?: Array<string | number>;
        skills?: string[];
        systemPrompt?: string;
    }>;
}): SlackChannelConfigResolved | null {
    // Per-channel configuration with granular controls
}
```

**Signal Provider Send** (`/home/dcramer/src/clawdbot/src/signal/send.ts`):

```typescript
type SignalTarget =
    | { type: "recipient"; recipient: string }
    | { type: "group"; groupId: string }
    | { type: "username"; username: string };

export async function sendMessageSignal(
    to: string,
    text: string,
    opts: SignalSendOpts = {},
): Promise<SignalSendResult> {
    // Uses signal-cli RPC for sending
    const result = await signalRpcRequest<{ timestamp?: number }>(
        "send",
        params,
        { baseUrl, timeoutMs: opts.timeoutMs },
    );
}
```

**Slack Format Conversion** (`/home/dcramer/src/clawdbot/src/slack/format.ts`):

```typescript
// Slack uses "mrkdwn" format which differs from standard Markdown
export function markdownToSlackMrkdwn(text: string): string {
    // Converts standard Markdown to Slack's mrkdwn format
    // Handles: *bold* -> *bold*, _italic_ -> _italic_, etc.
}
```

**Key Design Choices**:
- Modular architecture allows adding new providers easily
- Multi-account support per provider
- Unified configuration schema across all providers
- Rich per-channel and per-user authorization controls
- Consistent send interface across providers with result types

---

### 4. pi-mono (TypeScript)

pi-mono's Slack integration focuses on reliability with channel backfill for offline message catch-up.

**Socket Mode Client** (`/home/dcramer/src/pi-mono/packages/mom/src/slack.ts`):

```typescript
export class SlackBot {
    private socketClient: SocketModeClient;
    private webClient: WebClient;
    private startupTs: string | null = null;

    async start(): Promise<void> {
        const auth = await this.webClient.auth.test();
        this.botUserId = auth.user_id as string;

        await Promise.all([this.fetchUsers(), this.fetchChannels()]);
        await this.backfillAllChannels();

        this.setupEventHandlers();
        await this.socketClient.start();

        // Record startup time - messages older than this are just logged
        this.startupTs = (Date.now() / 1000).toFixed(6);
    }
}
```

**Channel Backfill for Offline Messages**:

```typescript
private async backfillChannel(channelId: string): Promise<number> {
    const existingTs = this.getExistingTimestamps(channelId);

    // Find the biggest ts in log.jsonl
    let latestTs: string | undefined;
    for (const ts of existingTs) {
        if (!latestTs || parseFloat(ts) > parseFloat(latestTs)) latestTs = ts;
    }

    // Fetch messages newer than what we have
    const result = await this.webClient.conversations.history({
        channel: channelId,
        oldest: latestTs,
        inclusive: false,
        limit: 1000,
    });

    // Log each message to log.jsonl
    for (const msg of relevantMessages) {
        this.logToFile(channelId, { /* message data */ });
    }
}
```

**Per-Channel Queue**:

```typescript
class ChannelQueue {
    private queue: QueuedWork[] = [];
    private processing = false;

    enqueue(work: QueuedWork): void {
        this.queue.push(work);
        this.processNext();
    }
}
```

**Event Handling with Startup Time Check**:

```typescript
// Only trigger processing for messages AFTER startup (not replayed old messages)
if (this.startupTs && e.ts < this.startupTs) {
    log.logInfo(`[${e.channel}] Skipping old message (pre-startup)`);
    ack();
    return;
}
```

**Key Design Choices**:
- Socket mode for real-time events (more reliable than HTTP)
- Channel backfill ensures no missed messages during downtime
- Startup timestamp prevents reprocessing old messages
- JSONL logging for conversation history
- Separate handling for DMs vs channel mentions

---

## Key Differences

### Abstraction Level

| Codebase | Abstraction | Trade-off |
|----------|-------------|-----------|
| **ash** | Abstract Provider base class | Clean interface but requires implementing all methods per provider |
| **archer** | Concrete wrapper class | Simple but tightly coupled to Telegram |
| **clawdbot** | Module-per-provider pattern | Very flexible but more code duplication |
| **pi-mono** | Concrete SlackBot class | Simple but single-provider focused |

### Authorization Models

| Codebase | Model | Granularity |
|----------|-------|-------------|
| **ash** | User IDs + Group IDs | Per-user and per-group allow lists |
| **archer** | User IDs only | Single allow list |
| **clawdbot** | Per-channel config | Channel-level with user overrides, skill restrictions |
| **pi-mono** | Channel membership | Implicit via bot membership |

### Message Handling Patterns

| Codebase | Pattern | Concurrent Handling |
|----------|---------|---------------------|
| **ash** | Per-chat async lock | Serialized per chat |
| **archer** | Per-chat queue | Serialized per chat |
| **clawdbot** | grammY runner sequentialization | Configurable concurrency |
| **pi-mono** | Per-channel queue | Serialized per channel |

### Streaming Support

| Codebase | Streaming | Implementation |
|----------|-----------|----------------|
| **ash** | Yes | send_streaming() with rate-limited edits |
| **archer** | Yes | editMessage() calls |
| **clawdbot** | Yes | Draft streams with block chunking |
| **pi-mono** | No | Single message response |

---

## Recommendations

### For ash

1. **Multi-Provider Support**: Consider the clawdbot module pattern if adding more providers. Each provider gets its own directory with accounts, monitor, send, and format modules.

2. **Backfill Capability**: pi-mono's backfill pattern is valuable for reliability. When the bot restarts, it can catch up on missed messages.

3. **Per-Channel Configuration**: clawdbot's per-channel config with skill restrictions could be useful for group chats with different purposes.

4. **Rate Limiting**: archer's message queue pattern is simple and effective. ash's per-chat lock achieves similar serialization but queuing could be more explicit.

### Provider Integration Checklist

When adding a new provider, ensure:

- [ ] Account/token resolution with multiple sources (env, file, config)
- [ ] Authorization model (user allow list, channel allow list)
- [ ] Message reception (polling, webhooks, or socket)
- [ ] Message sending with format conversion
- [ ] Typing indicators during processing
- [ ] Threading support (reply_to, thread_id)
- [ ] Media handling (upload, download, type detection)
- [ ] Error handling with retries
- [ ] Rate limiting compliance
- [ ] Logging for debugging

### Format Conversion Matrix

| Platform | Text Format | Code Blocks | Links | Bold | Italic |
|----------|-------------|-------------|-------|------|--------|
| Telegram | HTML or MarkdownV2 | `<pre>` or ``` | `<a href="">` | `<b>` or `**` | `<i>` or `_` |
| Slack | mrkdwn | ``` | `<url\|text>` | `*text*` | `_text_` |
| Discord | Markdown | ``` | `[text](url)` | `**text**` | `*text*` |
| Signal | Plain text | N/A | Raw URLs | N/A | N/A |

Each provider needs format conversion from the agent's output (typically Markdown) to the platform's native format.
