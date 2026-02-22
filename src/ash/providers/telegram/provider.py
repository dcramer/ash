"""Telegram provider using aiogram."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from aiogram import Bot, Dispatcher, F

if TYPE_CHECKING:
    from ash.chats import ChatStateManager

from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import CallbackQuery, ReactionTypeEmoji
from aiogram.types import Message as TelegramMessage

from ash.config.models import PassiveListeningConfig
from ash.providers.base import (
    ImageAttachment,
    IncomingMessage,
    MessageHandler,
    OutgoingMessage,
    Provider,
)

# Type alias for message processing result
ProcessingResult = tuple[Literal["active", "passive"], int, str | None] | None

CallbackHandler = Callable[[CallbackQuery], Awaitable[None]]

logger = logging.getLogger("telegram")

# Minimum interval between message edits (Telegram rate limit)
EDIT_INTERVAL = 1.0
LOG_PREVIEW_MAX_LEN = 180


def _get_parse_mode(mode: str | None) -> ParseMode:
    """Convert a parse mode string to ParseMode enum."""
    if not mode:
        return ParseMode.MARKDOWN
    normalized = mode.upper().replace("-", "_")
    try:
        return ParseMode[normalized]
    except KeyError:
        logger.warning("unknown_parse_mode", extra={"telegram.parse_mode": mode})
        return ParseMode.MARKDOWN


def _truncate(text: str, max_len: int = LOG_PREVIEW_MAX_LEN) -> str:
    """Truncate text for logging (first line only, max length)."""
    first_line, *rest = text.split("\n", 1)
    truncated = len(first_line) > max_len or bool(rest)
    return first_line[:max_len] + "..." if truncated else first_line


MAX_SEND_LENGTH = 4000  # Below Telegram's 4096 limit to leave room for formatting


def _find_split_point(text: str, max_length: int) -> int:
    """Find the best point to split text, searching backwards from max_length.

    Priority order: markdown heading > blank line > sentence-ending newline > any newline > hard cut.
    Never splits inside a code block.
    """
    search_region = text[:max_length]

    # Track code block state — don't split inside one
    in_code_block = False
    last_safe_heading = -1
    last_safe_blank = -1
    last_safe_sentence = -1
    last_safe_newline = -1

    i = 0
    while i < len(search_region):
        if search_region[i:].startswith("```"):
            in_code_block = not in_code_block
            i += 3
            continue

        if not in_code_block and search_region[i] == "\n":
            # Check for heading (next line starts with #)
            if i + 1 < len(search_region) and search_region[i + 1] == "#":
                last_safe_heading = i + 1  # Split before the heading
            # Check for blank line
            elif i + 1 < len(search_region) and search_region[i + 1] == "\n":
                last_safe_blank = i + 1
            # Check for sentence end before this newline
            elif i > 0 and search_region[i - 1] in ".!?)":
                last_safe_sentence = i + 1
            else:
                last_safe_newline = i + 1

        i += 1

    # Return best split point in priority order
    for point in (
        last_safe_heading,
        last_safe_blank,
        last_safe_sentence,
        last_safe_newline,
    ):
        if point > 0:
            return point

    return max_length


def split_message(text: str, max_length: int = MAX_SEND_LENGTH) -> list[str]:
    """Split text into chunks at paragraph/heading boundaries."""
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        split_at = _find_split_point(remaining, max_length)
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks


class TelegramProvider(Provider):
    """Telegram provider using aiogram 3.x."""

    def __init__(
        self,
        bot_token: str,
        allowed_users: list[str] | None = None,
        allowed_groups: list[str] | None = None,
        group_mode: str = "mention",
        passive_config: PassiveListeningConfig | None = None,
    ):
        self._token = bot_token
        self._allowed_users = set(allowed_users or [])
        self._allowed_groups = set(allowed_groups or [])
        self._group_mode = group_mode
        self._passive_config = passive_config

        self._bot = Bot(
            token=bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
        )
        self._dp = Dispatcher()
        self._handler: MessageHandler | None = None
        self._passive_handler: MessageHandler | None = None
        self._callback_handler: CallbackHandler | None = None
        self._running = False
        self._bot_username: str | None = None
        self._bot_id: int | None = None

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def bot(self) -> Bot:
        return self._bot

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dp

    @property
    def bot_username(self) -> str | None:
        return self._bot_username

    @property
    def bot_id(self) -> int | None:
        return self._bot_id

    @property
    def passive_config(self) -> PassiveListeningConfig | None:
        return self._passive_config

    def set_callback_handler(self, handler: CallbackHandler) -> None:
        """Set the callback query handler for interactive UI elements."""
        self._callback_handler = handler

    def set_passive_handler(self, handler: MessageHandler) -> None:
        """Set the handler for passive (non-mentioned) messages."""
        self._passive_handler = handler

    def _is_user_allowed(self, user_id: int, username: str | None) -> bool:
        if not self._allowed_users:
            return True
        return str(user_id) in self._allowed_users or (
            username is not None and f"@{username}" in self._allowed_users
        )

    def _is_group_allowed(self, chat_id: int) -> bool:
        if not self._allowed_groups:
            return True
        return str(chat_id) in self._allowed_groups

    def _is_mentioned(self, message: TelegramMessage) -> bool:
        """Check if bot is mentioned in the message."""
        if not self._bot_username:
            return False

        text = message.text or message.caption or ""
        mention = f"@{self._bot_username}"

        if mention.lower() in text.lower():
            return True

        entities = message.entities or message.caption_entities or []
        for entity in entities:
            if entity.type == "mention":
                entity_text = text[entity.offset : entity.offset + entity.length]
                if entity_text.lower() == mention.lower():
                    return True

        return False

    def _is_reply(self, message: TelegramMessage) -> bool:
        """Check if this message is a reply to the bot's own message."""
        if message.reply_to_message is None:
            return False
        if not self._bot_id:
            return True  # Can't determine — fall back to old behavior
        from_user = message.reply_to_message.from_user
        return from_user is not None and from_user.id == self._bot_id

    def _strip_mention(self, text: str) -> str:
        if not self._bot_username:
            return text
        pattern = rf"@{re.escape(self._bot_username)}\b"
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    async def _send_with_fallback(
        self,
        chat_id: int,
        text: str,
        reply_to: int | None = None,
        parse_mode: ParseMode | None = ParseMode.MARKDOWN,
    ) -> TelegramMessage:
        """Send a message with automatic plain-text fallback on parse errors."""
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
                    parse_mode=None,
                )
            if "message to be replied not found" in error_msg and reply_to is not None:
                logger.debug(f"Reply target not found, sending without reply: {e}")
                return await self._bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
            raise

    async def _edit_with_fallback(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: ParseMode | None = ParseMode.MARKDOWN,
    ) -> bool:
        """Edit a message with automatic plain-text fallback on parse errors."""
        try:
            await self._bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=parse_mode,
            )
            return True
        except TelegramBadRequest as e:
            error_msg = str(e).lower()
            if "message is not modified" in error_msg:
                # Content unchanged - not an error, just a no-op
                return True
            if "can't parse" in error_msg and parse_mode is not None:
                logger.debug(f"Markdown parsing failed, editing as plain text: {e}")
                try:
                    await self._bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=text,
                        parse_mode=None,
                    )
                    return True
                except Exception as e2:
                    logger.debug(f"Plain text edit also failed: {e2}")
                    return False
            # Handle other TelegramBadRequest errors gracefully (e.g., message not found)
            logger.debug(f"Edit failed with TelegramBadRequest: {e}")
            return False
        except Exception as e:
            logger.debug(f"Edit failed: {e}")
            return False

    def _should_process_message(self, message: TelegramMessage) -> ProcessingResult:
        """Check if a message should be processed, log and record all incoming messages.

        This method:
        1. Evaluates whether the message should trigger a bot response
        2. Determines processing mode (active vs passive)
        3. Logs ALL incoming messages with metadata about the processing decision
        4. Records ALL incoming messages to per-chat JSONL files for observability

        Returns:
            ("active", user_id, username) for mentioned/replied messages
            ("passive", user_id, username) for group messages eligible for passive listening
            None if message should not be processed
        """

        skip_reason: str | None = None
        processing_mode: Literal["active", "passive"] | None = None
        user_id = message.from_user.id if message.from_user else None
        username = message.from_user.username if message.from_user else None
        display_name = message.from_user.full_name if message.from_user else None

        if not message.from_user:
            skip_reason = "no_user"
        else:
            is_group = message.chat.type in ("group", "supergroup")
            if is_group:
                # Group message decision tree:
                # 1. Check if group is in the allowed list
                # 2. If mention mode: check if bot was @mentioned or replied to
                # 3. If not mentioned but passive listening enabled: route to passive
                # 4. Otherwise: skip the message
                if not self._is_group_allowed(message.chat.id):
                    skip_reason = "group_not_allowed"
                elif self._group_mode == "mention":
                    is_mentioned = self._is_mentioned(message)
                    is_reply = self._is_reply(message)
                    if is_mentioned or is_reply:
                        # Direct engagement: @mentioned or replied to -> active processing
                        processing_mode = "active"
                    elif (
                        self._passive_config
                        and self._passive_config.enabled
                        and self._passive_handler
                    ):
                        # Passive listening: not mentioned, but config allows observing
                        # Message goes through throttling and LLM decision before response
                        processing_mode = "passive"
                    else:
                        # No passive config: silently ignore non-mentioned messages
                        skip_reason = "not_mentioned_or_reply"
                else:
                    # group_mode == "always": respond to all messages in allowed groups
                    processing_mode = "active"
            else:
                # Direct message (DM): just check user allowlist
                if not self._is_user_allowed(user_id, username):  # type: ignore[arg-type]
                    skip_reason = "user_not_allowed"
                else:
                    processing_mode = "active"

        # Log ALL incoming messages with processing decision
        logger.info(
            "incoming_message",
            extra={
                "external_id": str(message.message_id),
                "chat_id": str(message.chat.id),
                "user_id": str(user_id) if user_id else None,
                "username": username,
                "chat_type": message.chat.type,
                "was_processed": processing_mode is not None,
                "processing_mode": processing_mode,
                "skip_reason": skip_reason,
                "input.preview": _truncate(message.text or message.caption or ""),
            },
        )

        # Record ALL incoming user messages to chat-level history.jsonl
        # Format: specs/sessions.md#historyjsonl
        text = message.text or message.caption or ""
        if text:
            history_metadata: dict[str, Any] = {
                "external_id": str(message.message_id),
                "was_processed": processing_mode is not None,
            }
            if skip_reason:
                history_metadata["skip_reason"] = skip_reason
            if processing_mode:
                history_metadata["processing_mode"] = processing_mode

            from ash.chats import ChatHistoryWriter

            writer = ChatHistoryWriter(provider=self.name, chat_id=str(message.chat.id))
            writer.record_user_message(
                content=text,
                created_at=message.date,
                user_id=str(user_id) if user_id else None,
                username=username,
                display_name=display_name,
                metadata=history_metadata,
            )

        if skip_reason or processing_mode is None:
            return None
        return processing_mode, user_id, username  # type: ignore[return-value]

    def _to_incoming_message(
        self,
        message: TelegramMessage,
        user_id: int,
        username: str | None,
        text: str,
        images: list[ImageAttachment] | None = None,
        *,
        was_mentioned: bool = False,
    ) -> IncomingMessage:
        """Convert a Telegram message to an IncomingMessage."""
        metadata = {
            "chat_type": message.chat.type,
            "chat_title": message.chat.title,
            "was_mentioned": was_mentioned,
        }
        # Include thread_id for forum topics (supergroups with topics enabled)
        if message.message_thread_id is not None:
            metadata["thread_id"] = str(message.message_thread_id)

        return IncomingMessage(
            id=str(message.message_id),
            chat_id=str(message.chat.id),
            user_id=str(user_id),
            text=text,
            username=username,
            display_name=message.from_user.full_name if message.from_user else None,
            reply_to_message_id=str(message.reply_to_message.message_id)
            if message.reply_to_message
            else None,
            images=images or [],
            metadata=metadata,
            timestamp=message.date,
        )

    async def start(self, handler: MessageHandler) -> None:
        """Start the Telegram bot."""
        self._handler = handler
        self._setup_handlers()

        # Cache bot username for mention detection
        try:
            bot_info = await self._bot.get_me()
            self._bot_username = bot_info.username
            self._bot_id = bot_info.id
            logger.info(
                "bot_username_resolved",
                extra={"telegram.bot_username": self._bot_username},
            )
        except Exception as e:
            logger.warning("bot_info_failed", extra={"error.message": str(e)})

        self._running = True

        logger.info("telegram_bot_starting")
        await self._bot.delete_webhook(drop_pending_updates=False)
        # Disable aiogram's signal handling - let the app handle SIGINT/SIGTERM
        await self._dp.start_polling(
            self._bot,
            handle_signals=False,
            close_bot_session=False,  # We close it ourselves in stop()
        )

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if not self._running:
            return  # Already stopped
        self._running = False

        # Stop the dispatcher polling
        try:
            await self._dp.stop_polling()
        except Exception as e:
            logger.debug(f"Error stopping polling: {e}")

        try:
            await self._bot.session.close()
        except Exception as e:
            logger.debug(f"Error closing bot session: {e}")

        logger.info("telegram_bot_stopped")

    def _setup_handlers(self) -> None:
        """Set up message handlers on the dispatcher."""

        @self._dp.message(Command("start"))
        async def handle_start(message: TelegramMessage) -> None:
            """Handle /start command."""
            result = self._should_process_message(message)
            if not result or result[0] != "active":
                return

            name = message.from_user.first_name if message.from_user else "there"
            bot_name = (
                self._bot_username.split("_")[0].title()
                if self._bot_username
                else "your personal assistant"
            )
            await message.answer(
                f"Hello, {name}! I'm {bot_name}, your personal assistant.\n\n"
                "Send me a message and I'll help you with tasks, answer questions, "
                "and remember things for you.\n\n"
                "Type /help to see what I can do."
            )

        @self._dp.message(Command("help"))
        async def handle_help(message: TelegramMessage) -> None:
            """Handle /help command."""
            result = self._should_process_message(message)
            if not result or result[0] != "active":
                return

            await message.answer(
                "**What I can do:**\n\n"
                "- Answer questions and have conversations\n"
                "- Remember facts and preferences (say 'remember that...')\n"
                "- Search the web for information\n"
                "- Run commands in a sandboxed environment\n"
                "- Use skills for specialized tasks\n\n"
                "Just send me a message to get started!"
            )

        @self._dp.message(F.photo)
        async def handle_photo(message: TelegramMessage) -> None:
            """Handle photo messages."""
            result = self._should_process_message(message)
            if not result or result[0] != "active":
                return
            _, user_id, username = result

            # Get the largest photo (best quality)
            photo = message.photo[-1] if message.photo else None
            if not photo:
                return

            # Download the photo
            try:
                file = await self._bot.get_file(photo.file_id)
                if not file.file_path:
                    logger.warning("photo_file_no_path")
                    return
                file_data = await self._bot.download_file(file.file_path)
                image_bytes = file_data.read() if file_data else None
            except Exception as e:
                logger.warning("photo_download_failed", extra={"error.message": str(e)})
                image_bytes = None

            # Create image attachment
            image = ImageAttachment(
                file_id=photo.file_id,
                width=photo.width,
                height=photo.height,
                file_size=photo.file_size,
                data=image_bytes,
            )

            # Strip bot mention from caption if in group
            is_group = message.chat.type in ("group", "supergroup")
            was_mentioned = is_group and self._is_mentioned(message)
            caption = message.caption or ""
            if is_group and caption:
                caption = self._strip_mention(caption)

            incoming = self._to_incoming_message(
                message,
                user_id,
                username,
                caption,
                images=[image],
                was_mentioned=was_mentioned,
            )

            if self._handler:
                try:
                    await self._handler(incoming)
                except Exception:
                    logger.exception("Error handling photo message")

        @self._dp.message(F.text)
        async def handle_message(message: TelegramMessage) -> None:
            """Handle text messages."""
            if not message.text:
                return

            result = self._should_process_message(message)
            if not result:
                return

            processing_mode, user_id, username = result

            # Strip bot mention from text if in group
            is_group = message.chat.type in ("group", "supergroup")
            was_mentioned = is_group and self._is_mentioned(message)
            text = self._strip_mention(message.text) if is_group else message.text

            incoming = self._to_incoming_message(
                message, user_id, username, text, was_mentioned=was_mentioned
            )
            # Add processing mode to metadata
            incoming.metadata["processing_mode"] = processing_mode

            if processing_mode == "passive":
                # Handle passive messages (not mentioned)
                if self._passive_handler:
                    try:
                        await self._passive_handler(incoming)
                    except Exception:
                        logger.exception("Error handling passive message")
            elif self._handler:
                try:
                    await self._handler(incoming)
                except Exception:
                    logger.exception("Error handling message")

        @self._dp.callback_query()
        async def handle_callback_query(callback_query: CallbackQuery) -> None:
            """Handle callback queries from inline keyboards."""
            if self._callback_handler:
                try:
                    await self._callback_handler(callback_query)
                except Exception:
                    logger.exception("Error handling callback query")
                    if callback_query.message:
                        await self._bot.answer_callback_query(
                            callback_query.id,
                            text="Error processing your selection",
                            show_alert=True,
                        )

        def _get_group_chat_state(
            message: TelegramMessage,
        ) -> ChatStateManager | None:
            """Get ChatStateManager for group chats, or None for non-groups."""
            from ash.chats import ChatStateManager

            if message.chat.type not in ("group", "supergroup"):
                return None
            return ChatStateManager(
                provider="telegram",
                chat_id=str(message.chat.id),
                thread_id=None,
            )

        @self._dp.message(F.new_chat_members)
        async def handle_new_members(message: TelegramMessage) -> None:
            """Handle new chat members joining."""
            chat_state = _get_group_chat_state(message)
            if not chat_state:
                return

            members = message.new_chat_members or []
            for user in members:
                chat_state.record_member_joined(
                    user_id=str(user.id),
                    username=user.username,
                    display_name=user.full_name,
                    is_bot=user.is_bot,
                )

            logger.debug(
                "Recorded %d new member(s) in chat %s",
                len(members),
                message.chat.id,
            )

        @self._dp.message(F.left_chat_member)
        async def handle_left_member(message: TelegramMessage) -> None:
            """Handle a member leaving the chat."""
            chat_state = _get_group_chat_state(message)
            if not chat_state:
                return

            user = message.left_chat_member
            if not user:
                return

            chat_state.record_member_left(str(user.id))
            logger.debug(
                "member_left_chat",
                extra={
                    "user.id": str(user.id),
                    "messaging.chat_id": str(message.chat.id),
                },
            )

    async def send(self, message: OutgoingMessage) -> str:
        """Send a message via Telegram, splitting long messages into chunks."""
        chunks = split_message(message.text)
        if len(chunks) == 1:
            return await self._send_single(message)

        last_id = ""
        for i, chunk in enumerate(chunks):
            chunk_msg = OutgoingMessage(
                chat_id=message.chat_id,
                text=chunk,
                reply_to_message_id=message.reply_to_message_id if i == 0 else None,
                parse_mode=message.parse_mode,
                reply_markup=message.reply_markup if i == len(chunks) - 1 else None,
            )
            last_id = await self._send_single(chunk_msg)
        return last_id

    async def _send_single(self, message: OutgoingMessage) -> str:
        """Send a single message via Telegram."""
        parse_mode = _get_parse_mode(message.parse_mode)
        reply_to = (
            int(message.reply_to_message_id) if message.reply_to_message_id else None
        )

        try:
            sent = await self._bot.send_message(
                chat_id=int(message.chat_id),
                text=message.text,
                reply_to_message_id=reply_to,
                parse_mode=parse_mode,
                reply_markup=message.reply_markup,
            )
        except TelegramBadRequest as e:
            error_msg = str(e).lower()
            if "can't parse" in error_msg and parse_mode is not None:
                logger.debug(f"Markdown parsing failed, sending as plain text: {e}")
                sent = await self._bot.send_message(
                    chat_id=int(message.chat_id),
                    text=message.text,
                    reply_to_message_id=reply_to,
                    parse_mode=None,
                    reply_markup=message.reply_markup,
                )
            elif (
                "message to be replied not found" in error_msg and reply_to is not None
            ):
                logger.debug(f"Reply target not found, sending without reply: {e}")
                sent = await self._bot.send_message(
                    chat_id=int(message.chat_id),
                    text=message.text,
                    parse_mode=parse_mode,
                    reply_markup=message.reply_markup,
                )
            else:
                raise

        logger.debug(
            "Sent message to chat %s: %s", message.chat_id, _truncate(message.text)
        )
        return str(sent.message_id)

    async def send_message(
        self, chat_id: str, text: str, *, reply_to: str | None = None
    ) -> str:
        """Send a simple text message to a chat."""
        sent = await self._send_with_fallback(
            chat_id=int(chat_id),
            text=text,
            reply_to=int(reply_to) if reply_to else None,
        )
        logger.debug(
            "message_sent",
            extra={
                "messaging.chat_id": chat_id,
                "message.preview": _truncate(text),
            },
        )
        return str(sent.message_id)

    async def send_streaming(
        self,
        chat_id: str,
        stream: AsyncIterator[str],
        *,
        reply_to: str | None = None,
    ) -> str:
        """Send a message with streaming updates."""
        content = ""
        message_id: str | None = None
        last_edit = 0.0
        use_markdown = True

        chat_id_int = int(chat_id)
        reply_to_int = int(reply_to) if reply_to else None

        async for chunk in stream:
            content += chunk
            now = asyncio.get_event_loop().time()

            # Send first message once we have content
            if message_id is None and content.strip():
                parse_mode = ParseMode.MARKDOWN if use_markdown else None
                try:
                    sent = await self._send_with_fallback(
                        chat_id_int, content, reply_to_int, parse_mode
                    )
                    message_id = str(sent.message_id)
                except TelegramBadRequest:
                    # Fallback already tried in helper, disable markdown for future
                    use_markdown = False
                    raise
                last_edit = now

            elif message_id and now - last_edit >= EDIT_INTERVAL:
                # Rate-limited edits during streaming
                parse_mode = ParseMode.MARKDOWN if use_markdown else None
                success = await self._edit_with_fallback(
                    chat_id_int, int(message_id), content, parse_mode
                )
                if success:
                    last_edit = now
                else:
                    # Edit failed, likely markdown issue - disable for future
                    use_markdown = False

        # Final edit with complete content
        if message_id and content:
            parse_mode = ParseMode.MARKDOWN if use_markdown else None
            await self._edit_with_fallback(
                chat_id_int, int(message_id), content, parse_mode
            )
        elif not message_id:
            # No content was streamed, send empty response
            sent = await self._send_with_fallback(
                chat_id_int,
                "I couldn't generate a response.",
                reply_to_int,
                None,
            )
            message_id = str(sent.message_id)

        return message_id  # type: ignore[return-value]

    async def edit(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        pm = _get_parse_mode(parse_mode)
        await self._edit_with_fallback(int(chat_id), int(message_id), text, pm)

    async def delete(self, chat_id: str, message_id: str) -> None:
        await self._bot.delete_message(chat_id=int(chat_id), message_id=int(message_id))

    async def send_typing(self, chat_id: str) -> None:
        await self._bot.send_chat_action(chat_id=int(chat_id), action="typing")

    async def set_reaction(
        self, chat_id: str, message_id: str, emoji: str = "👀"
    ) -> None:
        try:
            await self._bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=int(message_id),
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception as e:
            logger.warning("reaction_failed", extra={"error.message": str(e)})

    async def clear_reaction(self, chat_id: str, message_id: str) -> None:
        try:
            await self._bot.set_message_reaction(
                chat_id=int(chat_id), message_id=int(message_id), reaction=[]
            )
        except Exception as e:
            logger.debug(f"Failed to clear reaction: {e}")
