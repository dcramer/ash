"""Telegram provider using aiogram."""

import asyncio
import logging
import re
from collections.abc import AsyncIterator

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message as TelegramMessage
from aiogram.types import ReactionTypeEmoji

from ash.providers.base import (
    ImageAttachment,
    IncomingMessage,
    MessageHandler,
    OutgoingMessage,
    Provider,
)

logger = logging.getLogger("telegram")

# Minimum interval between message edits (Telegram rate limit)
EDIT_INTERVAL = 1.0


def _get_parse_mode(mode: str | None) -> ParseMode:
    """Convert a parse mode string to ParseMode enum."""
    if not mode:
        return ParseMode.MARKDOWN
    normalized = mode.upper().replace("-", "_")
    try:
        return ParseMode[normalized]
    except KeyError:
        logger.warning(f"Unknown parse mode '{mode}', using MARKDOWN")
        return ParseMode.MARKDOWN


def _truncate(text: str, max_len: int = 40) -> str:
    """Truncate text for logging (first line only, max length)."""
    first_line, *rest = text.split("\n", 1)
    truncated = len(first_line) > max_len or bool(rest)
    return first_line[:max_len] + "..." if truncated else first_line


class TelegramProvider(Provider):
    """Telegram provider using aiogram 3.x."""

    def __init__(
        self,
        bot_token: str,
        allowed_users: list[str] | None = None,
        webhook_url: str | None = None,
        webhook_path: str = "/telegram/webhook",
        allowed_groups: list[str] | None = None,
        group_mode: str = "mention",
    ):
        self._token = bot_token
        self._allowed_users = set(allowed_users or [])
        self._webhook_url = webhook_url
        self._webhook_path = webhook_path
        self._allowed_groups = set(allowed_groups or [])
        self._group_mode = group_mode

        self._bot = Bot(
            token=bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
        )
        self._dp = Dispatcher()
        self._handler: MessageHandler | None = None
        self._running = False
        self._bot_username: str | None = None

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
        return message.reply_to_message is not None

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
            raise
        except Exception as e:
            logger.debug(f"Edit failed: {e}")
            return False

    def _should_process_message(
        self, message: TelegramMessage
    ) -> tuple[int, str | None] | None:
        """Check if a message should be processed (user + group access)."""
        if not message.from_user:
            return None

        user_id = message.from_user.id
        username = message.from_user.username

        is_group = message.chat.type in ("group", "supergroup")

        if is_group:
            # Group authorization implies user authorization within that group
            if not self._is_group_allowed(message.chat.id):
                logger.debug(f"Group not allowed: {message.chat.id}")
                return None
            if self._group_mode == "mention":
                if not self._is_mentioned(message) and not self._is_reply(message):
                    return None
        else:
            # DMs require explicit user authorization
            if not self._is_user_allowed(user_id, username):
                logger.warning(f"Unauthorized user: {user_id} (@{username})")
                return None

        return user_id, username

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
            logger.info(f"Bot username: @{self._bot_username}")
        except Exception as e:
            logger.warning(f"Failed to get bot info: {e}")

        self._running = True

        if self._webhook_url:
            # Webhook mode - just set up the webhook
            full_url = f"{self._webhook_url.rstrip('/')}{self._webhook_path}"
            await self._bot.set_webhook(full_url)
            logger.info(f"Webhook set to: {full_url}")
        else:
            # Polling mode
            logger.info("Starting Telegram bot in polling mode")
            # Don't drop pending updates - we'll check for duplicates in the handler
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

        if self._webhook_url:
            try:
                await self._bot.delete_webhook()
            except Exception as e:
                logger.debug(f"Error deleting webhook: {e}")

        try:
            await self._bot.session.close()
        except Exception as e:
            logger.debug(f"Error closing bot session: {e}")

        logger.info("Telegram bot stopped")

    def _setup_handlers(self) -> None:
        """Set up message handlers on the dispatcher."""

        @self._dp.message(Command("start"))
        async def handle_start(message: TelegramMessage) -> None:
            """Handle /start command."""
            access = self._should_process_message(message)
            if not access:
                return

            name = message.from_user.first_name if message.from_user else "there"
            await message.answer(
                f"Hello, {name}! I'm Ash, your personal assistant.\n\n"
                "Send me a message and I'll help you with tasks, answer questions, "
                "and remember things for you.\n\n"
                "Type /help to see what I can do."
            )

        @self._dp.message(Command("help"))
        async def handle_help(message: TelegramMessage) -> None:
            """Handle /help command."""
            if not self._should_process_message(message):
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
            access = self._should_process_message(message)
            if not access:
                return
            user_id, username = access

            # Get the largest photo (best quality)
            photo = message.photo[-1] if message.photo else None
            if not photo:
                return

            # Download the photo
            try:
                file = await self._bot.get_file(photo.file_id)
                if not file.file_path:
                    logger.warning("Photo file has no file_path")
                    return
                file_data = await self._bot.download_file(file.file_path)
                image_bytes = file_data.read() if file_data else None
            except Exception as e:
                logger.warning(f"Failed to download photo: {e}")
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

            access = self._should_process_message(message)
            if not access:
                return
            user_id, username = access

            # Strip bot mention from text if in group
            is_group = message.chat.type in ("group", "supergroup")
            was_mentioned = is_group and self._is_mentioned(message)
            text = self._strip_mention(message.text) if is_group else message.text

            incoming = self._to_incoming_message(
                message, user_id, username, text, was_mentioned=was_mentioned
            )

            if self._handler:
                try:
                    await self._handler(incoming)
                except Exception:
                    logger.exception("Error handling message")

    async def send(self, message: OutgoingMessage) -> str:
        """Send a message via Telegram."""
        parse_mode = _get_parse_mode(message.parse_mode)
        sent = await self._send_with_fallback(
            chat_id=int(message.chat_id),
            text=message.text,
            reply_to=int(message.reply_to_message_id)
            if message.reply_to_message_id
            else None,
            parse_mode=parse_mode,
        )
        logger.debug(
            "Sent message to chat %s: %s", message.chat_id, _truncate(message.text)
        )
        return str(sent.message_id)

    async def send_message(self, chat_id: str, text: str) -> str:
        """Send a simple text message to a chat."""
        sent = await self._send_with_fallback(
            chat_id=int(chat_id),
            text=text,
        )
        logger.debug("Sent message to chat %s: %s", chat_id, _truncate(text))
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
            logger.warning(f"Failed to set reaction: {e}")

    async def clear_reaction(self, chat_id: str, message_id: str) -> None:
        try:
            await self._bot.set_message_reaction(
                chat_id=int(chat_id), message_id=int(message_id), reaction=[]
            )
        except Exception as e:
            logger.debug(f"Failed to clear reaction: {e}")

    async def process_webhook_update(self, update_data: dict) -> None:
        """Process a webhook update from Telegram."""
        from aiogram.types import Update

        update = Update(**update_data)
        await self._dp.feed_update(self._bot, update)
