# Provider Integration Gap Analysis

This document analyzes gaps between Ash's provider implementation and the reference implementations in Archer, Clawdbot, and Pi-mono.

**Files Analyzed:**
- Ash: `/home/dcramer/src/ash/src/ash/providers/base.py`, `/home/dcramer/src/ash/src/ash/providers/telegram/`
- Archer: `/home/dcramer/src/archer/src/telegram.ts`
- Clawdbot: `/home/dcramer/src/clawdbot/src/telegram/`, `/home/dcramer/src/clawdbot/src/slack/`
- Pi-mono: `/home/dcramer/src/pi-mono/packages/mom/src/slack.ts`

---

## Gap 1: Message Backfill

### What Ash is Missing

Ash does not catch up on messages received while offline. When the bot starts, it only processes new messages from that point forward. Old messages (received when the bot was offline) are skipped based on a 5-minute age check (lines 244-253 in `handlers.py`):

```python
# handlers.py lines 244-253
if message.timestamp:
    age = datetime.now(UTC) - message.timestamp.replace(tzinfo=UTC)
    if age > timedelta(minutes=5):
        logger.debug(
            "Skipping old message %s (age=%ds)",
            message.id,
            age.total_seconds(),
        )
        return
```

### Reference Implementation (Pi-mono)

Pi-mono's Slack implementation has comprehensive backfill functionality (lines 436-556 in `slack.ts`):

```typescript
// slack.ts lines 436-556
private getExistingTimestamps(channelId: string): Set<string> {
    const logPath = join(this.workingDir, channelId, "log.jsonl");
    const timestamps = new Set<string>();
    if (!existsSync(logPath)) return timestamps;

    const content = readFileSync(logPath, "utf-8");
    const lines = content.trim().split("\n").filter(Boolean);
    for (const line of lines) {
        try {
            const entry = JSON.parse(line);
            if (entry.ts) timestamps.add(entry.ts);
        } catch {}
    }
    return timestamps;
}

private async backfillChannel(channelId: string): Promise<number> {
    const existingTs = this.getExistingTimestamps(channelId);

    // Find the biggest ts in log.jsonl
    let latestTs: string | undefined;
    for (const ts of existingTs) {
        if (!latestTs || parseFloat(ts) > parseFloat(latestTs)) latestTs = ts;
    }

    const allMessages: Message[] = [];
    let cursor: string | undefined;
    let pageCount = 0;
    const maxPages = 3;

    do {
        const result = await this.webClient.conversations.history({
            channel: channelId,
            oldest: latestTs, // Only fetch messages newer than what we have
            inclusive: false,
            limit: 1000,
            cursor,
        });
        if (result.messages) {
            allMessages.push(...(result.messages as Message[]));
        }
        cursor = result.response_metadata?.next_cursor;
        pageCount++;
    } while (cursor && pageCount < maxPages);

    // Log each message to log.jsonl
    for (const msg of relevantMessages) {
        this.logToFile(channelId, { /* message entry */ });
    }

    return relevantMessages.length;
}

private async backfillAllChannels(): Promise<void> {
    // Only backfill channels that already have a log.jsonl
    for (const [channelId, channel] of this.channels) {
        const logPath = join(this.workingDir, channelId, "log.jsonl");
        if (existsSync(logPath)) {
            channelsToBackfill.push([channelId, channel]);
        }
    }
    // ...
}
```

Key features:
- Tracks `startupTs` to distinguish between backfilled messages (just logged) and new messages (processed)
- Only backfills channels that have existing log files (prior interaction)
- Uses pagination to fetch up to 3000 messages per channel
- Deduplicates by timestamp before logging

### Files to Modify

- `/home/dcramer/src/ash/src/ash/providers/telegram/provider.py`
- `/home/dcramer/src/ash/src/ash/providers/telegram/handlers.py`
- Create new file: `/home/dcramer/src/ash/src/ash/providers/telegram/backfill.py`

### Concrete Python Code Changes

```python
# New file: /home/dcramer/src/ash/src/ash/providers/telegram/backfill.py
"""Message backfill for Telegram provider."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from aiogram import Bot

logger = logging.getLogger("telegram.backfill")


class TelegramBackfill:
    """Handles catching up on messages missed while offline."""

    def __init__(self, bot: Bot, sessions_dir: Path):
        """Initialize backfill handler.

        Args:
            bot: Telegram bot instance.
            sessions_dir: Directory containing session JSONL files.
        """
        self._bot = bot
        self._sessions_dir = sessions_dir
        self._startup_ts: datetime | None = None

    def get_existing_message_ids(self, chat_id: str) -> set[str]:
        """Get message IDs already in the session file.

        Args:
            chat_id: Chat ID to check.

        Returns:
            Set of external message IDs.
        """
        # Session files are at ~/.ash/sessions/{provider}_{chat_id}_{user_id}/context.jsonl
        # We need to scan for matching directories
        existing_ids: set[str] = set()

        for session_dir in self._sessions_dir.glob(f"telegram_{chat_id}_*"):
            context_file = session_dir / "context.jsonl"
            if not context_file.exists():
                continue

            try:
                for line in context_file.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        metadata = entry.get("metadata", {})
                        if external_id := metadata.get("external_id"):
                            existing_ids.add(str(external_id))
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logger.warning(f"Failed to read session file: {e}")

        return existing_ids

    async def get_unprocessed_updates(
        self,
        chat_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get updates that weren't processed while offline.

        Note: Telegram's getUpdates only works in polling mode and only
        returns pending updates. For proper backfill, we'd need to use
        the chat history API (requires special permissions).

        For now, this returns an empty list - full backfill would require
        bot admin access to chat history.

        Args:
            chat_id: Chat to backfill.
            limit: Maximum updates to fetch.

        Returns:
            List of update dictionaries.
        """
        # Telegram doesn't have a direct "get chat history" API for bots
        # without admin permissions. The getUpdates API only returns
        # pending updates, not historical messages.
        #
        # For real backfill, you'd need:
        # 1. MTProto client (Pyrogram/Telethon) instead of Bot API
        # 2. Or bot admin rights + getChatHistory (not in Bot API)
        #
        # For now, we track startup time and skip old messages gracefully.
        return []

    def set_startup_time(self) -> None:
        """Record the startup time for filtering old messages."""
        self._startup_ts = datetime.now(UTC)
        logger.info(f"Backfill startup time set: {self._startup_ts.isoformat()}")

    def is_message_before_startup(self, message_ts: datetime) -> bool:
        """Check if a message predates bot startup.

        Args:
            message_ts: Message timestamp.

        Returns:
            True if message is from before startup.
        """
        if not self._startup_ts:
            return False
        return message_ts < self._startup_ts

    @property
    def startup_time(self) -> datetime | None:
        """Get the startup timestamp."""
        return self._startup_ts
```

```python
# In handlers.py, modify handle_message to log old messages instead of skipping:

# Add import
from ash.providers.telegram.backfill import TelegramBackfill

# In TelegramMessageHandler.__init__, add:
self._backfill = TelegramBackfill(
    bot=provider.bot,
    sessions_dir=Path(os.environ.get("ASH_DATA_DIR", "~/.ash")).expanduser() / "sessions",
)

# Replace the skip logic (lines 244-253) with:
if message.timestamp:
    # Check if this is an old message (before startup)
    if self._backfill.is_message_before_startup(
        message.timestamp.replace(tzinfo=UTC)
    ):
        # Log to session file for history, but don't process
        thread_id = message.metadata.get("thread_id")
        session_manager = self._get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        await session_manager.add_user_message(
            content=message.text,
            token_count=estimate_tokens(message.text),
            metadata={
                "external_id": message.id,
                "backfilled": True,
            },
            user_id=message.user_id,
            username=message.username,
            display_name=message.display_name,
        )
        logger.debug(
            "Backfilled old message %s (age=%ds)",
            message.id,
            (datetime.now(UTC) - message.timestamp.replace(tzinfo=UTC)).total_seconds(),
        )
        return

# In start() method of TelegramProvider (or wherever the bot starts):
# After bot.get_me() succeeds:
if hasattr(self, '_message_handler') and self._message_handler:
    self._message_handler._backfill.set_startup_time()
```

### Effort: Medium
### Priority: Medium

Backfill ensures conversation continuity after bot restarts. Without it, users may feel ignored if they sent messages while the bot was offline. However, Telegram's Bot API limitations make full backfill challenging.

---

## Gap 2: Multi-Provider Architecture

### What Ash is Missing

Ash only has a Telegram provider. There's no Slack, Discord, or other provider implementations.

The base provider interface (`base.py`) is clean and extensible:

```python
# base.py lines 58-145
class Provider(ABC):
    """Abstract interface for communication providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'telegram', 'discord')."""
        ...

    @abstractmethod
    async def start(self, handler: MessageHandler) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def send(self, message: OutgoingMessage) -> str:
        ...

    @abstractmethod
    async def send_streaming(self, chat_id: str, stream: AsyncIterator[str], ...) -> str:
        ...
```

### Reference Implementation (Clawdbot)

Clawdbot has 7 providers: Telegram, Slack, WhatsApp, Facebook Messenger, Discord, SMS, and Voice. Each has dedicated directories with modular components.

The Slack implementation (`/home/dcramer/src/clawdbot/src/slack/`) includes:
- `monitor.ts` - Event handling and message routing
- `format.ts` - Markdown to Slack mrkdwn conversion
- `actions.ts` - API actions (send, edit, delete, react)
- `threading.ts` - Thread management
- `types.ts` - Type definitions
- `accounts.ts` - Multi-account support

### Files to Create

- `/home/dcramer/src/ash/src/ash/providers/slack/__init__.py`
- `/home/dcramer/src/ash/src/ash/providers/slack/provider.py`
- `/home/dcramer/src/ash/src/ash/providers/slack/format.py`
- `/home/dcramer/src/ash/src/ash/providers/slack/handlers.py`

### Concrete Python Code Changes

```python
# New file: /home/dcramer/src/ash/src/ash/providers/slack/__init__.py
"""Slack provider."""

from ash.providers.slack.handlers import SlackMessageHandler
from ash.providers.slack.provider import SlackProvider

__all__ = [
    "SlackMessageHandler",
    "SlackProvider",
]
```

```python
# New file: /home/dcramer/src/ash/src/ash/providers/slack/provider.py
"""Slack provider using slack-sdk."""

import asyncio
import logging
import re
from collections.abc import AsyncIterator

from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.web.async_client import AsyncWebClient

from ash.providers.base import (
    IncomingMessage,
    MessageHandler,
    OutgoingMessage,
    Provider,
)

logger = logging.getLogger("slack")


class SlackProvider(Provider):
    """Slack provider using Socket Mode for real-time events."""

    def __init__(
        self,
        app_token: str,
        bot_token: str,
        allowed_users: list[str] | None = None,
        allowed_channels: list[str] | None = None,
    ):
        """Initialize Slack provider.

        Args:
            app_token: Slack app-level token (xapp-...).
            bot_token: Slack bot token (xoxb-...).
            allowed_users: List of allowed user IDs.
            allowed_channels: List of allowed channel IDs.
        """
        self._app_token = app_token
        self._bot_token = bot_token
        self._allowed_users = set(allowed_users or [])
        self._allowed_channels = set(allowed_channels or [])

        self._web_client = AsyncWebClient(token=bot_token)
        self._socket_client: SocketModeClient | None = None
        self._handler: MessageHandler | None = None
        self._running = False
        self._bot_user_id: str | None = None

    @property
    def name(self) -> str:
        return "slack"

    @property
    def web_client(self) -> AsyncWebClient:
        """Get the Slack Web API client."""
        return self._web_client

    async def start(self, handler: MessageHandler) -> None:
        """Start the Slack bot.

        Args:
            handler: Callback to handle incoming messages.
        """
        self._handler = handler

        # Get bot user ID
        auth_response = await self._web_client.auth_test()
        self._bot_user_id = auth_response.get("user_id")
        logger.info(f"Bot user ID: {self._bot_user_id}")

        # Initialize socket mode client
        self._socket_client = SocketModeClient(
            app_token=self._app_token,
            web_client=self._web_client,
        )

        # Register event handlers
        self._socket_client.socket_mode_request_listeners.append(
            self._handle_socket_event
        )

        self._running = True
        await self._socket_client.connect()
        logger.info("Slack bot started in Socket Mode")

    async def stop(self) -> None:
        """Stop the Slack bot."""
        self._running = False
        if self._socket_client:
            await self._socket_client.close()
        logger.info("Slack bot stopped")

    async def _handle_socket_event(self, client: SocketModeClient, req) -> None:
        """Handle incoming socket mode events."""
        # Acknowledge the event
        await client.send_socket_mode_response(
            {"envelope_id": req.envelope_id}
        )

        if req.type == "events_api":
            event = req.payload.get("event", {})
            await self._handle_event(event)

    async def _handle_event(self, event: dict) -> None:
        """Handle a Slack event."""
        event_type = event.get("type")

        if event_type == "app_mention":
            await self._handle_mention(event)
        elif event_type == "message":
            # Handle DMs
            channel_type = event.get("channel_type")
            if channel_type == "im":
                await self._handle_dm(event)

    async def _handle_mention(self, event: dict) -> None:
        """Handle @mention in a channel."""
        user_id = event.get("user")
        channel_id = event.get("channel")
        text = event.get("text", "")
        ts = event.get("ts")

        # Check permissions
        if self._allowed_users and user_id not in self._allowed_users:
            logger.debug(f"Ignoring mention from unauthorized user: {user_id}")
            return

        if self._allowed_channels and channel_id not in self._allowed_channels:
            logger.debug(f"Ignoring mention in unauthorized channel: {channel_id}")
            return

        # Strip the bot mention from text
        text = re.sub(rf"<@{self._bot_user_id}>", "", text).strip()

        # Get user info
        user_info = await self._web_client.users_info(user=user_id)
        user_data = user_info.get("user", {})

        incoming = IncomingMessage(
            id=ts or "",
            chat_id=channel_id,
            user_id=user_id,
            text=text,
            username=user_data.get("name"),
            display_name=user_data.get("real_name"),
            metadata={"chat_type": "channel"},
        )

        if self._handler:
            try:
                await self._handler(incoming)
            except Exception:
                logger.exception("Error handling Slack mention")

    async def _handle_dm(self, event: dict) -> None:
        """Handle direct message."""
        # Skip bot messages
        if event.get("bot_id") or event.get("user") == self._bot_user_id:
            return

        user_id = event.get("user")
        channel_id = event.get("channel")
        text = event.get("text", "")
        ts = event.get("ts")

        # Check permissions
        if self._allowed_users and user_id not in self._allowed_users:
            logger.debug(f"Ignoring DM from unauthorized user: {user_id}")
            return

        # Get user info
        user_info = await self._web_client.users_info(user=user_id)
        user_data = user_info.get("user", {})

        incoming = IncomingMessage(
            id=ts or "",
            chat_id=channel_id,
            user_id=user_id,
            text=text,
            username=user_data.get("name"),
            display_name=user_data.get("real_name"),
            metadata={"chat_type": "dm"},
        )

        if self._handler:
            try:
                await self._handler(incoming)
            except Exception:
                logger.exception("Error handling Slack DM")

    async def send(self, message: OutgoingMessage) -> str:
        """Send a message via Slack.

        Args:
            message: Message to send.

        Returns:
            Sent message timestamp (Slack's message ID).
        """
        response = await self._web_client.chat_postMessage(
            channel=message.chat_id,
            text=message.text,
        )
        return response.get("ts", "")

    async def send_streaming(
        self,
        chat_id: str,
        stream: AsyncIterator[str],
        *,
        reply_to: str | None = None,
    ) -> str:
        """Send a message with streaming updates.

        Args:
            chat_id: Chat to send to.
            stream: Async iterator of text chunks.
            reply_to: Thread timestamp to reply in.

        Returns:
            Final message timestamp.
        """
        content = ""
        message_ts: str | None = None
        last_update = 0.0

        async for chunk in stream:
            content += chunk
            now = asyncio.get_event_loop().time()

            if message_ts is None and content.strip():
                # Send initial message
                response = await self._web_client.chat_postMessage(
                    channel=chat_id,
                    text=content,
                    thread_ts=reply_to,
                )
                message_ts = response.get("ts", "")
                last_update = now
            elif message_ts and now - last_update >= 1.0:
                # Update message (rate limited to 1 update/second)
                try:
                    await self._web_client.chat_update(
                        channel=chat_id,
                        ts=message_ts,
                        text=content,
                    )
                    last_update = now
                except Exception:
                    pass  # Ignore update errors

        # Final update
        if message_ts:
            try:
                await self._web_client.chat_update(
                    channel=chat_id,
                    ts=message_ts,
                    text=content,
                )
            except Exception:
                pass

        return message_ts or ""

    async def edit(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        """Edit an existing message."""
        await self._web_client.chat_update(
            channel=chat_id,
            ts=message_id,
            text=text,
        )

    async def delete(self, chat_id: str, message_id: str) -> None:
        """Delete a message."""
        await self._web_client.chat_delete(
            channel=chat_id,
            ts=message_id,
        )
```

```python
# New file: /home/dcramer/src/ash/src/ash/providers/slack/format.py
"""Markdown to Slack mrkdwn conversion."""

import re


def markdown_to_slack_mrkdwn(text: str) -> str:
    """Convert standard Markdown to Slack mrkdwn format.

    Slack mrkdwn differences from standard Markdown:
    - Bold: *text* (single asterisk, not double)
    - Italic: _text_ (same)
    - Strikethrough: ~text~ (single tilde)
    - Code: `code` (same)
    - Code blocks: ```code``` (same, but no language hint)
    - Links: <url|text> format

    Args:
        text: Markdown text to convert.

    Returns:
        Slack mrkdwn formatted text.
    """
    if not text:
        return ""

    result = text

    # Convert bold: **text** -> *text*
    result = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", result)

    # Convert strikethrough: ~~text~~ -> ~text~
    result = re.sub(r"~~([^~]+)~~", r"~\1~", result)

    # Convert links: [text](url) -> <url|text>
    result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", result)

    # Escape special characters (but preserve allowed angle-bracket tokens)
    def escape_segment(segment: str) -> str:
        return segment.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Protect Slack-style tokens
    protected = []
    def protect_token(match):
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    result = re.sub(r"<[@#!][^>]+>|<https?://[^>]+>", protect_token, result)

    # Escape remaining angle brackets
    parts = result.split("\x00")
    for i in range(0, len(parts), 2):
        if i < len(parts):
            parts[i] = escape_segment(parts[i])

    # Restore protected tokens
    result = "\x00".join(parts)
    for i, token in enumerate(protected):
        result = result.replace(f"\x00{i}\x00", token)

    return result
```

### Effort: Large
### Priority: Medium

Adding Slack support doubles the bot's reach. The base provider abstraction is already clean; this is mostly implementation work.

---

## Gap 3: Rich Media Type Detection

### What Ash is Missing

Ash only handles photos explicitly (lines 482-536 in `provider.py`). Other media types (videos, audio, documents, stickers, voice, GIFs) are not processed:

```python
# provider.py lines 482-514
@self._dp.message(F.photo)
async def handle_photo(message: TelegramMessage) -> None:
    """Handle photo messages."""
    # ... only photo handling
```

### Reference Implementation (Archer)

Archer has comprehensive media handling with proper MIME type detection and appropriate upload methods (lines 438-502 in `telegram.ts`):

```typescript
// telegram.ts lines 438-502
async uploadFile(chatId: string, filePath: string, title?: string, threadId?: number): Promise<void> {
    const fileName = title || basename(filePath);
    const ext = filePath.toLowerCase().split(".").pop() || "";
    const inputFile = new InputFile(filePath, fileName);

    const opts: { message_thread_id?: number } = {};
    if (threadId) opts.message_thread_id = threadId;

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

    // Videos: sendVideo for inline playback
    if (["mp4", "mov", "avi", "mkv", "webm"].includes(ext)) {
        await this.bot.api.sendVideo(chatId, inputFile, opts);
        return;
    }

    // Audio: sendAudio for inline player
    if (["mp3", "m4a", "wav", "ogg", "flac"].includes(ext)) {
        await this.bot.api.sendAudio(chatId, inputFile, opts);
        return;
    }

    // Voice notes: sendVoice
    if (ext === "oga") {
        await this.bot.api.sendVoice(chatId, inputFile, opts);
        return;
    }

    // Contacts: parse VCF and send as contact
    if (ext === "vcf") {
        const contact = await this.parseVcfFile(filePath);
        if (contact) {
            await this.bot.api.sendContact(chatId, contact.phone, contact.firstName, {
                ...opts,
                last_name: contact.lastName,
                vcard: contact.vcard,
            });
            return;
        }
    }

    // Locations: parse and send as location
    if (["gpx", "kml", "geojson"].includes(ext)) {
        const location = await this.parseLocationFile(filePath, ext);
        if (location) {
            await this.bot.api.sendLocation(chatId, location.latitude, location.longitude, opts);
            return;
        }
    }

    // Everything else: sendDocument
    await this.bot.api.sendDocument(chatId, inputFile, opts);
}
```

Archer also extracts all media types from incoming messages (lines 800-853 in `telegram.ts`):

```typescript
// telegram.ts lines 800-853
private extractFiles(msg: GrammyContext["message"]): Array<{ name: string; fileId: string }> {
    const files: Array<{ name: string; fileId: string }> = [];

    // Photo (get largest size)
    if (msg.photo && msg.photo.length > 0) {
        const largest = msg.photo[msg.photo.length - 1];
        files.push({ name: `photo_${largest.file_id.substring(0, 8)}.jpg`, fileId: largest.file_id });
    }

    // Document
    if (msg.document) {
        files.push({
            name: msg.document.file_name || `document_${msg.document.file_id.substring(0, 8)}`,
            fileId: msg.document.file_id,
        });
    }

    // Audio
    if (msg.audio) {
        files.push({
            name: msg.audio.file_name || `audio_${msg.audio.file_id.substring(0, 8)}.mp3`,
            fileId: msg.audio.file_id,
        });
    }

    // Video
    if (msg.video) {
        files.push({
            name: msg.video.file_name || `video_${msg.video.file_id.substring(0, 8)}.mp4`,
            fileId: msg.video.file_id,
        });
    }

    // Voice
    if (msg.voice) {
        files.push({
            name: `voice_${msg.voice.file_id.substring(0, 8)}.ogg`,
            fileId: msg.voice.file_id,
        });
    }

    // Sticker
    if (msg.sticker) {
        const ext = msg.sticker.is_animated ? "tgs" : msg.sticker.is_video ? "webm" : "webp";
        files.push({
            name: `sticker_${msg.sticker.file_id.substring(0, 8)}.${ext}`,
            fileId: msg.sticker.file_id,
        });
    }

    return files;
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/providers/base.py`
- `/home/dcramer/src/ash/src/ash/providers/telegram/provider.py`

### Concrete Python Code Changes

```python
# In base.py, add more attachment types (after ImageAttachment):

@dataclass
class VideoAttachment:
    """Video attached to a message."""
    file_id: str
    width: int | None = None
    height: int | None = None
    duration: int | None = None
    file_size: int | None = None
    mime_type: str | None = None
    data: bytes | None = None


@dataclass
class AudioAttachment:
    """Audio attached to a message."""
    file_id: str
    duration: int | None = None
    title: str | None = None
    performer: str | None = None
    file_size: int | None = None
    mime_type: str | None = None
    data: bytes | None = None


@dataclass
class VoiceAttachment:
    """Voice message attached."""
    file_id: str
    duration: int | None = None
    file_size: int | None = None
    mime_type: str | None = None
    data: bytes | None = None


@dataclass
class DocumentAttachment:
    """Document attached to a message."""
    file_id: str
    file_name: str | None = None
    file_size: int | None = None
    mime_type: str | None = None
    data: bytes | None = None


@dataclass
class StickerAttachment:
    """Sticker attached to a message."""
    file_id: str
    width: int | None = None
    height: int | None = None
    emoji: str | None = None
    is_animated: bool = False
    is_video: bool = False
    data: bytes | None = None


# Update IncomingMessage to include all attachment types:
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
    videos: list[VideoAttachment] = field(default_factory=list)
    audio: list[AudioAttachment] = field(default_factory=list)
    voice: list[VoiceAttachment] = field(default_factory=list)
    documents: list[DocumentAttachment] = field(default_factory=list)
    stickers: list[StickerAttachment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    @property
    def has_media(self) -> bool:
        """Check if message has any media attachments."""
        return bool(
            self.images or self.videos or self.audio or
            self.voice or self.documents or self.stickers
        )

    @property
    def media_description(self) -> str:
        """Get a text description of attached media."""
        parts = []
        if self.images:
            parts.append(f"{len(self.images)} image(s)")
        if self.videos:
            parts.append(f"{len(self.videos)} video(s)")
        if self.audio:
            parts.append(f"{len(self.audio)} audio file(s)")
        if self.voice:
            parts.append(f"{len(self.voice)} voice message(s)")
        if self.documents:
            parts.append(f"{len(self.documents)} document(s)")
        if self.stickers:
            parts.append(f"{len(self.stickers)} sticker(s)")
        return ", ".join(parts) if parts else ""
```

```python
# In provider.py, add handlers for all media types:

# Add imports
from aiogram.types import (
    Audio,
    Document,
    PhotoSize,
    Sticker,
    Video,
    Voice,
)
from ash.providers.base import (
    AudioAttachment,
    DocumentAttachment,
    ImageAttachment,
    IncomingMessage,
    MessageHandler,
    OutgoingMessage,
    Provider,
    StickerAttachment,
    VideoAttachment,
    VoiceAttachment,
)


# In _setup_handlers, add:
@self._dp.message(F.video)
async def handle_video(message: TelegramMessage) -> None:
    """Handle video messages."""
    access = self._should_process_message(message)
    if not access:
        return
    user_id, username = access

    video = message.video
    if not video:
        return

    try:
        file = await self._bot.get_file(video.file_id)
        file_data = await self._bot.download_file(file.file_path) if file.file_path else None
        video_bytes = file_data.read() if file_data else None
    except Exception as e:
        logger.warning(f"Failed to download video: {e}")
        video_bytes = None

    attachment = VideoAttachment(
        file_id=video.file_id,
        width=video.width,
        height=video.height,
        duration=video.duration,
        file_size=video.file_size,
        mime_type=video.mime_type,
        data=video_bytes,
    )

    # ... create IncomingMessage with videos=[attachment] ...


@self._dp.message(F.audio)
async def handle_audio(message: TelegramMessage) -> None:
    """Handle audio messages."""
    # Similar pattern...


@self._dp.message(F.voice)
async def handle_voice(message: TelegramMessage) -> None:
    """Handle voice messages."""
    # Similar pattern...


@self._dp.message(F.document)
async def handle_document(message: TelegramMessage) -> None:
    """Handle document messages."""
    # Similar pattern...


@self._dp.message(F.sticker)
async def handle_sticker(message: TelegramMessage) -> None:
    """Handle sticker messages."""
    # Similar pattern...


# Add upload_file method:
async def upload_file(
    self,
    chat_id: str,
    file_path: str,
    *,
    title: str | None = None,
    thread_id: int | None = None,
) -> str:
    """Upload a file with appropriate method based on type.

    Args:
        chat_id: Chat to send to.
        file_path: Path to the file.
        title: Optional title/caption.
        thread_id: Optional thread ID for topics.

    Returns:
        Sent message ID.
    """
    from pathlib import Path
    from aiogram.types import FSInputFile

    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")
    input_file = FSInputFile(file_path, filename=title or path.name)

    kwargs = {"message_thread_id": thread_id} if thread_id else {}

    # Images
    if ext in ("jpg", "jpeg", "png", "webp"):
        result = await self._bot.send_photo(int(chat_id), input_file, **kwargs)
        return str(result.message_id)

    # GIFs
    if ext == "gif":
        result = await self._bot.send_animation(int(chat_id), input_file, **kwargs)
        return str(result.message_id)

    # Videos
    if ext in ("mp4", "mov", "avi", "mkv", "webm"):
        result = await self._bot.send_video(int(chat_id), input_file, **kwargs)
        return str(result.message_id)

    # Audio
    if ext in ("mp3", "m4a", "wav", "ogg", "flac"):
        result = await self._bot.send_audio(int(chat_id), input_file, **kwargs)
        return str(result.message_id)

    # Voice notes
    if ext == "oga":
        result = await self._bot.send_voice(int(chat_id), input_file, **kwargs)
        return str(result.message_id)

    # Default: document
    result = await self._bot.send_document(int(chat_id), input_file, **kwargs)
    return str(result.message_id)
```

### Effort: Medium
### Priority: High

Rich media handling is essential for a capable assistant. Users expect to share files, videos, voice messages, and get appropriate responses.

---

## Gap 4: Markdown-to-Telegram-HTML Conversion

### What Ash is Missing

Ash uses Telegram's native Markdown parsing (ParseMode.MARKDOWN) which is limited and error-prone. When parsing fails, it falls back to plain text:

```python
# provider.py lines 209-251
async def _send_with_fallback(
    self,
    chat_id: int,
    text: str,
    reply_to: int | None = None,
    parse_mode: ParseMode | None = ParseMode.MARKDOWN,
) -> TelegramMessage:
    try:
        return await self._bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_to_message_id=reply_to,
            parse_mode=parse_mode,
        )
    except TelegramBadRequest as e:
        error_msg = str(e).lower()
        if "can't parse" in error_msg and parse_mode is not None:
            logger.debug(f"Markdown parsing failed, sending as plain text: {e}")
            return await self._bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to,
                parse_mode=None,  # Fallback to plain text
            )
```

This loses all formatting when Markdown parsing fails.

### Reference Implementation (Archer & Clawdbot)

Archer converts Markdown to Telegram HTML before sending (lines 14-142 in `telegram.ts`):

```typescript
// telegram.ts lines 14-128
export function escapeHtml(text: string): string {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

export function markdownToTelegramHtml(text: string): string {
    // First, normalize any HTML tags the LLM might have used to markdown
    let normalized = text
        .replace(/<b>([^<]*)<\/b>/g, "**$1**")
        .replace(/<strong>([^<]*)<\/strong>/g, "**$1**")
        .replace(/<i>([^<]*)<\/i>/g, "*$1*")
        .replace(/<em>([^<]*)<\/em>/g, "*$1*")
        .replace(/<code>([^<]*)<\/code>/g, "`$1`")
        .replace(/<pre>([^<]*)<\/pre>/g, "```\n$1\n```");

    const result: string[] = [];
    let i = 0;

    while (i < normalized.length) {
        // Code blocks: ```...```
        if (normalized.slice(i, i + 3) === "```") {
            // ... handle code blocks
            result.push(`<pre>${escapeHtml(code)}</pre>`);
        }

        // Inline code: `...`
        if (normalized[i] === "`") {
            result.push(`<code>${escapeHtml(code)}</code>`);
        }

        // Bold: **...**
        if (normalized.slice(i, i + 2) === "**") {
            result.push(`<b>${markdownToTelegramHtml(content)}</b>`);
        }

        // Italic: *...* or _..._
        // Links: [text](url)
        // Plain character - escape it
    }

    return result.join("");
}
```

Clawdbot uses a more robust approach with markdown-it parser (lines 1-141 in `format.ts`):

```typescript
// clawdbot/src/telegram/format.ts lines 1-141
import MarkdownIt from "markdown-it";

const md = new MarkdownIt({
    html: false,
    linkify: true,
    breaks: false,
    typographer: false,
});

md.enable("strikethrough");

md.renderer.rules.em_open = () => "<i>";
md.renderer.rules.em_close = () => "</i>";
md.renderer.rules.strong_open = () => "<b>";
md.renderer.rules.strong_close = () => "</b>";
md.renderer.rules.s_open = () => "<s>";
md.renderer.rules.s_close = () => "</s>";

md.renderer.rules.code_inline = (tokens, idx) =>
    `<code>${escapeHtml(tokens[idx]?.content ?? "")}</code>`;
md.renderer.rules.code_block = (tokens, idx) =>
    `<pre><code>${escapeHtml(tokens[idx]?.content ?? "")}</code></pre>\n`;

md.renderer.rules.link_open = (tokens, idx, _opts, env) => {
    const href = tokens[idx]?.attrGet("href") ?? "";
    return `<a href="${escapeHtml(href)}">`;
};
md.renderer.rules.link_close = () => "</a>";

export function markdownToTelegramHtml(markdown: string): string {
    const rendered = md.render(markdown ?? "");
    return rendered
        .replace(/\n{3,}/g, "\n\n")
        .trimEnd();
}
```

### Files to Modify

- Create new file: `/home/dcramer/src/ash/src/ash/providers/telegram/format.py`
- `/home/dcramer/src/ash/src/ash/providers/telegram/provider.py`

### Concrete Python Code Changes

```python
# New file: /home/dcramer/src/ash/src/ash/providers/telegram/format.py
"""Markdown to Telegram HTML conversion."""

import re
from html import escape


def markdown_to_telegram_html(text: str) -> str:
    """Convert Markdown to Telegram-compatible HTML.

    Supports: **bold**, *italic*, _italic_, `code`, ```code blocks```, [text](url)
    Also handles <b>, <i>, <code>, <pre>, <a>, <s> tags.

    Args:
        text: Markdown text to convert.

    Returns:
        Telegram HTML formatted text.
    """
    if not text:
        return ""

    # Normalize HTML tags that LLM might output to markdown
    normalized = text
    normalized = re.sub(r"<b>([^<]*)</b>", r"**\1**", normalized)
    normalized = re.sub(r"<strong>([^<]*)</strong>", r"**\1**", normalized)
    normalized = re.sub(r"<i>([^<]*)</i>", r"*\1*", normalized)
    normalized = re.sub(r"<em>([^<]*)</em>", r"*\1*", normalized)
    normalized = re.sub(r"<code>([^<]*)</code>", r"`\1`", normalized)
    normalized = re.sub(r"<pre>([^<]*)</pre>", r"```\n\1\n```", normalized)

    result: list[str] = []
    i = 0

    while i < len(normalized):
        # Code blocks: ```...```
        if normalized[i:i+3] == "```":
            start = i + 3
            # Skip optional language hint on first line
            content_start = start
            newline_pos = normalized.find("\n", start)
            if newline_pos != -1 and newline_pos < start + 20:
                possible_lang = normalized[start:newline_pos].strip()
                if possible_lang and " " not in possible_lang and len(possible_lang) < 15:
                    content_start = newline_pos + 1

            end = normalized.find("```", content_start)
            if end != -1:
                code = normalized[content_start:end]
                result.append(f"<pre>{escape(code)}</pre>")
                i = end + 3
                continue

        # Inline code: `...`
        if normalized[i] == "`":
            end = normalized.find("`", i + 1)
            if end != -1 and "\n" not in normalized[i+1:end]:
                code = normalized[i+1:end]
                result.append(f"<code>{escape(code)}</code>")
                i = end + 1
                continue

        # Bold: **...**
        if normalized[i:i+2] == "**":
            end = normalized.find("**", i + 2)
            if end != -1 and "\n" not in normalized[i+2:end]:
                content = normalized[i+2:end]
                result.append(f"<b>{markdown_to_telegram_html(content)}</b>")
                i = end + 2
                continue

        # Italic: *...* (but not **)
        if normalized[i] == "*" and (i + 1 >= len(normalized) or normalized[i+1] != "*"):
            end = _find_closing_marker(normalized, i + 1, "*")
            if end != -1:
                content = normalized[i+1:end]
                result.append(f"<i>{markdown_to_telegram_html(content)}</i>")
                i = end + 1
                continue

        # Italic: _..._ (but not __)
        if normalized[i] == "_" and (i + 1 >= len(normalized) or normalized[i+1] != "_"):
            end = _find_closing_marker(normalized, i + 1, "_")
            if end != -1:
                content = normalized[i+1:end]
                result.append(f"<i>{markdown_to_telegram_html(content)}</i>")
                i = end + 1
                continue

        # Strikethrough: ~~...~~
        if normalized[i:i+2] == "~~":
            end = normalized.find("~~", i + 2)
            if end != -1 and "\n" not in normalized[i+2:end]:
                content = normalized[i+2:end]
                result.append(f"<s>{markdown_to_telegram_html(content)}</s>")
                i = end + 2
                continue

        # Links: [text](url)
        if normalized[i] == "[":
            close_bracket = normalized.find("]", i + 1)
            if close_bracket != -1 and close_bracket + 1 < len(normalized) and normalized[close_bracket + 1] == "(":
                close_paren = normalized.find(")", close_bracket + 2)
                if close_paren != -1 and "\n" not in normalized[i:close_paren]:
                    link_text = normalized[i+1:close_bracket]
                    url = normalized[close_bracket+2:close_paren]
                    result.append(f'<a href="{escape(url)}">{escape(link_text)}</a>')
                    i = close_paren + 1
                    continue

        # Plain character - escape it
        result.append(escape(normalized[i]))
        i += 1

    return "".join(result)


def _find_closing_marker(text: str, start: int, marker: str) -> int:
    """Find closing marker for italic (* or _), avoiding newlines."""
    for j in range(start, len(text)):
        if text[j] == "\n":
            return -1
        if text[j] == marker:
            # Check it's not preceded by space and not followed by same marker
            if j > 0 and text[j-1] != " ":
                if j + 1 >= len(text) or text[j+1] != marker:
                    return j
    return -1


def truncate_for_telegram(text: str, max_length: int = 4096) -> str:
    """Truncate text to fit Telegram's message limit.

    Args:
        text: Text to truncate.
        max_length: Maximum length (default: Telegram's 4096 limit).

    Returns:
        Truncated text with ellipsis if needed.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
```

```python
# In provider.py, update _send_with_fallback to use HTML:

from ash.providers.telegram.format import markdown_to_telegram_html, truncate_for_telegram


async def _send_with_fallback(
    self,
    chat_id: int,
    text: str,
    reply_to: int | None = None,
    parse_mode: ParseMode | None = ParseMode.HTML,  # Changed default
) -> TelegramMessage:
    """Send a message with automatic plain-text fallback on parse errors."""
    # Convert Markdown to HTML first
    if parse_mode == ParseMode.HTML or parse_mode == ParseMode.MARKDOWN:
        html_text = markdown_to_telegram_html(text)
        html_text = truncate_for_telegram(html_text)
        effective_mode = ParseMode.HTML
    else:
        html_text = truncate_for_telegram(text)
        effective_mode = parse_mode

    try:
        return await self._bot.send_message(
            chat_id=chat_id,
            text=html_text,
            reply_to_message_id=reply_to,
            parse_mode=effective_mode,
        )
    except TelegramBadRequest as e:
        error_msg = str(e).lower()
        if "can't parse" in error_msg and effective_mode is not None:
            logger.debug(f"HTML parsing failed, sending as plain text: {e}")
            return await self._bot.send_message(
                chat_id=chat_id,
                text=text,  # Original text, not HTML
                reply_to_message_id=reply_to,
                parse_mode=None,
            )
        raise
```

### Effort: Medium
### Priority: High

Proper Markdown conversion ensures consistent formatting across LLM responses. Without it, users see broken formatting or plain text fallbacks.

---

## Gap 5: Typing Indicator Improvements

### What Ash is Missing

Ash has a basic typing loop that refreshes every 4 seconds (lines 1034-1051 in `handlers.py`):

```python
# handlers.py lines 1034-1051
async def _typing_loop(self, chat_id: str) -> None:
    """Send typing indicators in a loop."""
    while True:
        try:
            await self._provider.send_typing(chat_id)
            await asyncio.sleep(4)  # Refresh before 5 second timeout
        except asyncio.CancelledError:
            break
        except Exception:
            break
```

Missing improvements:
1. No intelligent action selection (upload_photo, upload_document, etc.)
2. No thread/topic support for typing indicators
3. No debouncing for rapid message handling

### Reference Implementation (Archer & Clawdbot)

Archer supports multiple chat actions (lines 417-436 in `telegram.ts`):

```typescript
// telegram.ts lines 417-436
async sendChatAction(
    chatId: string,
    action:
        | "typing"
        | "upload_photo"
        | "upload_video"
        | "upload_voice"
        | "upload_document"
        | "record_video"
        | "record_voice"
        | "choose_sticker"
        | "find_location"
        | "record_video_note"
        | "upload_video_note",
    threadId?: number,
): Promise<void> {
    const opts: { message_thread_id?: number } = {};
    if (threadId) opts.message_thread_id = threadId;
    await this.bot.api.sendChatAction(chatId, action, opts);
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/providers/telegram/provider.py`
- `/home/dcramer/src/ash/src/ash/providers/telegram/handlers.py`

### Concrete Python Code Changes

```python
# In provider.py, enhance send_typing to support different actions:

from enum import Enum


class ChatAction(str, Enum):
    """Telegram chat actions."""
    TYPING = "typing"
    UPLOAD_PHOTO = "upload_photo"
    UPLOAD_VIDEO = "upload_video"
    UPLOAD_VOICE = "upload_voice"
    UPLOAD_DOCUMENT = "upload_document"
    RECORD_VIDEO = "record_video"
    RECORD_VOICE = "record_voice"
    CHOOSE_STICKER = "choose_sticker"
    FIND_LOCATION = "find_location"
    RECORD_VIDEO_NOTE = "record_video_note"
    UPLOAD_VIDEO_NOTE = "upload_video_note"


async def send_chat_action(
    self,
    chat_id: str,
    action: ChatAction = ChatAction.TYPING,
    *,
    thread_id: int | None = None,
) -> None:
    """Send a chat action indicator.

    Args:
        chat_id: Chat to show action in.
        action: Type of action to show.
        thread_id: Thread ID for forum topics.
    """
    kwargs = {}
    if thread_id is not None:
        kwargs["message_thread_id"] = thread_id

    await self._bot.send_chat_action(
        chat_id=int(chat_id),
        action=action.value,
        **kwargs,
    )


# Keep send_typing for backward compatibility:
async def send_typing(
    self,
    chat_id: str,
    *,
    thread_id: int | None = None,
) -> None:
    """Send typing indicator to a chat.

    Args:
        chat_id: Chat to show typing indicator in.
        thread_id: Thread ID for forum topics.
    """
    await self.send_chat_action(chat_id, ChatAction.TYPING, thread_id=thread_id)
```

```python
# In handlers.py, enhance typing loop with context awareness:

async def _typing_loop(
    self,
    chat_id: str,
    *,
    thread_id: int | None = None,
    action: ChatAction = ChatAction.TYPING,
) -> None:
    """Send chat action indicators in a loop.

    Telegram indicators only last 5 seconds, so we need to
    keep sending them for long operations.

    Args:
        chat_id: Chat to show action in.
        thread_id: Thread ID for forum topics.
        action: Type of action to show.
    """
    while True:
        try:
            await self._provider.send_chat_action(
                chat_id,
                action,
                thread_id=thread_id,
            )
            await asyncio.sleep(4)  # Refresh before 5 second timeout
        except asyncio.CancelledError:
            break
        except Exception:
            # Ignore errors - typing is best effort
            break


# Update _handle_sync to pass thread_id:
async def _handle_sync(
    self,
    message: IncomingMessage,
    session: SessionState,
) -> None:
    # ...
    thread_id = message.metadata.get("thread_id")
    thread_id_int = int(thread_id) if thread_id else None

    # Determine action based on context
    action = ChatAction.TYPING  # Default
    # Could be enhanced to use UPLOAD_PHOTO if we know response will have media

    typing_task = asyncio.create_task(
        self._typing_loop(
            message.chat_id,
            thread_id=thread_id_int,
            action=action,
        )
    )
    # ...
```

### Effort: Small
### Priority: Low

Typing improvements are nice-to-have. The current implementation works; these enhancements add polish for power users in forum topics.

---

## Gap 6: Provider-Specific Formatting

### What Ash is Missing

Ash has no provider-specific message formatting. The same text goes to all providers. Different platforms have different:
- Message length limits (Telegram: 4096, Slack: 40000, Discord: 2000)
- Mention formats (Telegram: @username, Slack: <@U123>)
- Formatting syntax (Telegram: HTML, Slack: mrkdwn)

### Reference Implementation (Clawdbot)

Clawdbot has dedicated format modules for each provider:

**Telegram (`format.ts`):**
```typescript
// Uses HTML tags: <b>, <i>, <code>, <pre>, <a>, <s>
export function markdownToTelegramHtml(markdown: string): string {
    // ...
    md.renderer.rules.strong_open = () => "<b>";
    md.renderer.rules.strong_close = () => "</b>";
    // ...
}
```

**Slack (`format.ts`):**
```typescript
// Uses mrkdwn: *bold*, _italic_, ~strikethrough~, <url|text>
export function markdownToSlackMrkdwn(markdown: string): string {
    // ...
    md.renderer.rules.strong_open = () => "*";  // Single asterisk
    md.renderer.rules.strong_close = () => "*";
    // ...
}
```

### Files to Modify

- Create new file: `/home/dcramer/src/ash/src/ash/providers/format.py`
- Update provider implementations to use formatters

### Concrete Python Code Changes

```python
# New file: /home/dcramer/src/ash/src/ash/providers/format.py
"""Provider-specific message formatting."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderLimits:
    """Message limits for a provider."""
    max_message_length: int
    max_caption_length: int = 1024
    supports_formatting: bool = True
    supports_editing: bool = True


class MessageFormatter(ABC):
    """Base class for provider-specific message formatting."""

    @property
    @abstractmethod
    def limits(self) -> ProviderLimits:
        """Get provider limits."""
        ...

    @abstractmethod
    def format_markdown(self, text: str) -> str:
        """Convert Markdown to provider-specific format.

        Args:
            text: Markdown text.

        Returns:
            Formatted text for the provider.
        """
        ...

    @abstractmethod
    def format_mention(self, user_id: str, display_name: str | None = None) -> str:
        """Format a user mention.

        Args:
            user_id: User ID to mention.
            display_name: Optional display name.

        Returns:
            Provider-specific mention format.
        """
        ...

    def truncate(self, text: str, max_length: int | None = None) -> str:
        """Truncate text to fit provider limits.

        Args:
            text: Text to truncate.
            max_length: Override max length.

        Returns:
            Truncated text.
        """
        limit = max_length or self.limits.max_message_length
        if len(text) <= limit:
            return text
        return text[:limit - 3] + "..."

    def chunk(self, text: str, max_length: int | None = None) -> list[str]:
        """Split text into chunks that fit provider limits.

        Args:
            text: Text to split.
            max_length: Override max length.

        Returns:
            List of text chunks.
        """
        limit = max_length or self.limits.max_message_length
        if len(text) <= limit:
            return [text]

        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break

            # Find a good break point (newline or space)
            break_point = limit
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = remaining.rfind(sep, 0, limit)
                if idx > limit // 2:  # Only use if not too early
                    break_point = idx + len(sep)
                    break

            chunks.append(remaining[:break_point].rstrip())
            remaining = remaining[break_point:].lstrip()

        return chunks


class TelegramFormatter(MessageFormatter):
    """Formatter for Telegram messages."""

    @property
    def limits(self) -> ProviderLimits:
        return ProviderLimits(
            max_message_length=4096,
            max_caption_length=1024,
            supports_formatting=True,
            supports_editing=True,
        )

    def format_markdown(self, text: str) -> str:
        # Import here to avoid circular imports
        from ash.providers.telegram.format import markdown_to_telegram_html
        return markdown_to_telegram_html(text)

    def format_mention(self, user_id: str, display_name: str | None = None) -> str:
        # Telegram uses HTML mention format
        if display_name:
            return f'<a href="tg://user?id={user_id}">{display_name}</a>'
        return f'<a href="tg://user?id={user_id}">User</a>'


class SlackFormatter(MessageFormatter):
    """Formatter for Slack messages."""

    @property
    def limits(self) -> ProviderLimits:
        return ProviderLimits(
            max_message_length=40000,
            max_caption_length=40000,
            supports_formatting=True,
            supports_editing=True,
        )

    def format_markdown(self, text: str) -> str:
        from ash.providers.slack.format import markdown_to_slack_mrkdwn
        return markdown_to_slack_mrkdwn(text)

    def format_mention(self, user_id: str, display_name: str | None = None) -> str:
        # Slack uses angle-bracket mentions
        return f"<@{user_id}>"


class PlainTextFormatter(MessageFormatter):
    """Formatter that strips all formatting."""

    @property
    def limits(self) -> ProviderLimits:
        return ProviderLimits(
            max_message_length=10000,
            supports_formatting=False,
            supports_editing=False,
        )

    def format_markdown(self, text: str) -> str:
        # Strip Markdown formatting
        import re
        result = text
        # Remove bold/italic markers
        result = re.sub(r"\*\*([^*]+)\*\*", r"\1", result)
        result = re.sub(r"\*([^*]+)\*", r"\1", result)
        result = re.sub(r"_([^_]+)_", r"\1", result)
        # Remove code markers
        result = re.sub(r"`([^`]+)`", r"\1", result)
        result = re.sub(r"```[^`]*```", r"\1", result, flags=re.DOTALL)
        # Convert links
        result = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", result)
        return result

    def format_mention(self, user_id: str, display_name: str | None = None) -> str:
        return display_name or user_id


def get_formatter(provider_name: str) -> MessageFormatter:
    """Get the appropriate formatter for a provider.

    Args:
        provider_name: Provider identifier.

    Returns:
        MessageFormatter instance.
    """
    formatters = {
        "telegram": TelegramFormatter(),
        "slack": SlackFormatter(),
    }
    return formatters.get(provider_name, PlainTextFormatter())
```

```python
# In providers, use the formatter:

# In telegram/provider.py
from ash.providers.format import get_formatter

class TelegramProvider(Provider):
    def __init__(self, ...):
        # ...
        self._formatter = get_formatter("telegram")

    async def send(self, message: OutgoingMessage) -> str:
        # Format text before sending
        formatted_text = self._formatter.format_markdown(message.text)

        # Chunk if needed
        chunks = self._formatter.chunk(formatted_text)

        for chunk in chunks:
            sent = await self._send_with_fallback(
                chat_id=int(message.chat_id),
                text=chunk,
                # ...
            )

        return str(sent.message_id)
```

### Effort: Medium
### Priority: Medium

Provider-specific formatting ensures messages look good on each platform. Without it, formatting may break or look inconsistent across providers.

---

## Summary Table

| Gap | Description | Effort | Priority |
|-----|-------------|--------|----------|
| 1 | Message backfill | Medium | Medium |
| 2 | Multi-provider architecture (Slack) | Large | Medium |
| 3 | Rich media type detection | Medium | High |
| 4 | Markdown-to-Telegram-HTML conversion | Medium | High |
| 5 | Typing indicator improvements | Small | Low |
| 6 | Provider-specific formatting | Medium | Medium |

## Recommended Implementation Order

1. **Gap 4: Markdown-to-Telegram-HTML** - High impact, fixes formatting issues immediately
2. **Gap 3: Rich media type detection** - High impact, enables full media support
3. **Gap 6: Provider-specific formatting** - Foundation for multi-provider
4. **Gap 1: Message backfill** - Improves reliability after restarts
5. **Gap 2: Multi-provider (Slack)** - Expands reach significantly
6. **Gap 5: Typing indicator improvements** - Polish for advanced use cases
