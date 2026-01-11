"""Telegram provider using aiogram."""

import asyncio
import logging
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

logger = logging.getLogger(__name__)

# Minimum interval between message edits (Telegram rate limit)
EDIT_INTERVAL = 1.0


class TelegramProvider(Provider):
    """Telegram provider using aiogram 3.x.

    Supports both polling and webhook modes.
    """

    def __init__(
        self,
        bot_token: str,
        allowed_users: list[str] | None = None,
        webhook_url: str | None = None,
        webhook_path: str = "/telegram/webhook",
        allowed_groups: list[str] | None = None,
        group_mode: str = "mention",
    ):
        """Initialize Telegram provider.

        Args:
            bot_token: Telegram bot token from BotFather.
            allowed_users: List of allowed usernames or user IDs.
            webhook_url: Base URL for webhooks (uses polling if None).
            webhook_path: Path for webhook endpoint.
            allowed_groups: List of allowed group IDs (empty = all groups allowed).
            group_mode: How to respond in groups ("mention" or "always").
        """
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
        """Get the aiogram Bot instance."""
        return self._bot

    @property
    def dispatcher(self) -> Dispatcher:
        """Get the aiogram Dispatcher instance."""
        return self._dp

    def _is_user_allowed(self, user_id: int, username: str | None) -> bool:
        """Check if a user is allowed to interact with the bot.

        Args:
            user_id: Telegram user ID.
            username: Telegram username (without @).

        Returns:
            True if user is allowed.
        """
        if not self._allowed_users:
            return True

        if str(user_id) in self._allowed_users:
            return True

        if username and f"@{username}" in self._allowed_users:
            return True

        return False

    def _is_group_allowed(self, chat_id: int) -> bool:
        """Check if a group is allowed.

        Args:
            chat_id: Telegram chat ID.

        Returns:
            True if group is allowed (or if no restrictions set).
        """
        if not self._allowed_groups:
            return True
        return str(chat_id) in self._allowed_groups

    def _is_mentioned(self, message: TelegramMessage) -> bool:
        """Check if bot is mentioned in the message.

        Args:
            message: Telegram message to check.

        Returns:
            True if bot username is mentioned.
        """
        if not self._bot_username:
            return False

        text = message.text or message.caption or ""
        mention = f"@{self._bot_username}"

        # Check for direct text mention
        if mention.lower() in text.lower():
            return True

        # Check entities for mention type
        entities = message.entities or message.caption_entities or []
        for entity in entities:
            if entity.type == "mention":
                entity_text = text[entity.offset : entity.offset + entity.length]
                if entity_text.lower() == mention.lower():
                    return True

        return False

    def _strip_mention(self, text: str) -> str:
        """Remove bot mention from text.

        Args:
            text: Message text.

        Returns:
            Text with bot mention removed.
        """
        if not self._bot_username:
            return text
        # Remove mention (case-insensitive)
        import re

        pattern = rf"@{re.escape(self._bot_username)}\b"
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # --- Message sending helpers ---

    async def _send_with_fallback(
        self,
        chat_id: int,
        text: str,
        reply_to: int | None = None,
        parse_mode: ParseMode | None = ParseMode.MARKDOWN,
    ) -> TelegramMessage:
        """Send a message with automatic plain-text fallback on parse errors.

        Args:
            chat_id: Telegram chat ID.
            text: Message text.
            reply_to: Message ID to reply to.
            parse_mode: Parse mode (falls back to None on error).

        Returns:
            Sent Telegram message.
        """
        try:
            return await self._bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to,
                parse_mode=parse_mode,
            )
        except TelegramBadRequest as e:
            if "can't parse" in str(e).lower() and parse_mode is not None:
                logger.debug(f"Markdown parsing failed, sending as plain text: {e}")
                return await self._bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_to_message_id=reply_to,
                    parse_mode=None,
                )
            raise

    async def _edit_with_fallback(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: ParseMode | None = ParseMode.MARKDOWN,
    ) -> bool:
        """Edit a message with automatic plain-text fallback on parse errors.

        Args:
            chat_id: Telegram chat ID.
            message_id: Message ID to edit.
            text: New message text.
            parse_mode: Parse mode (falls back to None on error).

        Returns:
            True if edit succeeded, False if it failed (e.g., rate limit).
        """
        try:
            await self._bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=parse_mode,
            )
            return True
        except TelegramBadRequest as e:
            if "can't parse" in str(e).lower() and parse_mode is not None:
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

    # --- Message processing helpers ---

    def _should_process_message(
        self, message: TelegramMessage
    ) -> tuple[int, str | None] | None:
        """Check if a message should be processed (user + group access).

        Args:
            message: Telegram message to check.

        Returns:
            (user_id, username) tuple if should process, None otherwise.
        """
        if not message.from_user:
            return None

        user_id = message.from_user.id
        username = message.from_user.username

        # Check user authorization
        if not self._is_user_allowed(user_id, username):
            logger.warning(f"Unauthorized user: {user_id} (@{username})")
            return None

        # Check group access
        is_group = message.chat.type in ("group", "supergroup")
        if is_group:
            if not self._is_group_allowed(message.chat.id):
                logger.debug(f"Group not allowed: {message.chat.id}")
                return None
            if self._group_mode == "mention" and not self._is_mentioned(message):
                return None

        return user_id, username

    def _to_incoming_message(
        self,
        message: TelegramMessage,
        user_id: int,
        username: str | None,
        text: str,
        images: list[ImageAttachment] | None = None,
    ) -> IncomingMessage:
        """Convert a Telegram message to an IncomingMessage.

        Args:
            message: Source Telegram message.
            user_id: User ID (already validated).
            username: Username (already validated).
            text: Processed text (with mentions stripped if needed).
            images: Optional image attachments.

        Returns:
            IncomingMessage for handler processing.
        """
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
            metadata={
                "chat_type": message.chat.type,
                "chat_title": message.chat.title,
            },
            timestamp=message.date,
        )

    async def start(self, handler: MessageHandler) -> None:
        """Start the Telegram bot.

        Args:
            handler: Callback to handle incoming messages.
        """
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
            caption = message.caption or ""
            if is_group and caption:
                caption = self._strip_mention(caption)

            incoming = self._to_incoming_message(
                message, user_id, username, caption, images=[image]
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

            logger.info(
                f"Received text message from @{username} ({user_id}): "
                f"{message.text[:50]}"
            )

            # Strip bot mention from text if in group
            is_group = message.chat.type in ("group", "supergroup")
            text = self._strip_mention(message.text) if is_group else message.text

            incoming = self._to_incoming_message(message, user_id, username, text)

            if self._handler:
                try:
                    await self._handler(incoming)
                except Exception:
                    logger.exception("Error handling message")

    async def send(self, message: OutgoingMessage) -> str:
        """Send a message via Telegram.

        Args:
            message: Message to send.

        Returns:
            Sent message ID.
        """
        parse_mode = (
            ParseMode(message.parse_mode.upper())
            if message.parse_mode
            else ParseMode.MARKDOWN
        )
        sent = await self._send_with_fallback(
            chat_id=int(message.chat_id),
            text=message.text,
            reply_to=int(message.reply_to_message_id)
            if message.reply_to_message_id
            else None,
            parse_mode=parse_mode,
        )
        return str(sent.message_id)

    async def send_streaming(
        self,
        chat_id: str,
        stream: AsyncIterator[str],
        *,
        reply_to: str | None = None,
    ) -> str:
        """Send a message with streaming updates.

        Edits the message as new content arrives, respecting rate limits.

        Args:
            chat_id: Chat to send to.
            stream: Async iterator of text chunks.
            reply_to: Message to reply to.

        Returns:
            Final message ID.
        """
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
        """Edit an existing message.

        Args:
            chat_id: Chat containing the message.
            message_id: Message to edit.
            text: New text content.
            parse_mode: Text parsing mode.
        """
        pm = ParseMode(parse_mode.upper()) if parse_mode else ParseMode.MARKDOWN
        await self._edit_with_fallback(int(chat_id), int(message_id), text, pm)

    async def delete(self, chat_id: str, message_id: str) -> None:
        """Delete a message.

        Args:
            chat_id: Chat containing the message.
            message_id: Message to delete.
        """
        await self._bot.delete_message(
            chat_id=int(chat_id),
            message_id=int(message_id),
        )

    async def send_typing(self, chat_id: str) -> None:
        """Send typing indicator to a chat.

        Args:
            chat_id: Chat to show typing indicator in.
        """
        await self._bot.send_chat_action(
            chat_id=int(chat_id),
            action="typing",
        )

    async def set_reaction(
        self, chat_id: str, message_id: str, emoji: str = "👀"
    ) -> None:
        """Set a reaction on a message.

        Args:
            chat_id: Chat containing the message.
            message_id: Message to react to.
            emoji: Emoji to use for reaction (default: eyes - "looking at it").
        """
        try:
            await self._bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=int(message_id),
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception as e:
            # Reactions may not be available in all chats
            logger.warning(f"Failed to set reaction: {e}")

    async def clear_reaction(self, chat_id: str, message_id: str) -> None:
        """Clear reactions from a message.

        Args:
            chat_id: Chat containing the message.
            message_id: Message to clear reactions from.
        """
        try:
            await self._bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=int(message_id),
                reaction=[],
            )
        except Exception as e:
            logger.debug(f"Failed to clear reaction: {e}")

    async def process_webhook_update(self, update_data: dict) -> None:
        """Process a webhook update.

        Used when running in webhook mode with an external HTTP server.

        Args:
            update_data: Raw update data from Telegram.
        """
        from aiogram.types import Update

        update = Update(**update_data)
        await self._dp.feed_update(self._bot, update)
