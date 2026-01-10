"""Telegram provider using aiogram."""

import asyncio
import logging
from collections.abc import AsyncIterator

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import Message as TelegramMessage

from ash.providers.base import (
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
    ):
        """Initialize Telegram provider.

        Args:
            bot_token: Telegram bot token from BotFather.
            allowed_users: List of allowed usernames or user IDs.
            webhook_url: Base URL for webhooks (uses polling if None).
            webhook_path: Path for webhook endpoint.
        """
        self._token = bot_token
        self._allowed_users = set(allowed_users or [])
        self._webhook_url = webhook_url
        self._webhook_path = webhook_path

        self._bot = Bot(
            token=bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
        )
        self._dp = Dispatcher()
        self._handler: MessageHandler | None = None
        self._running = False

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

    async def start(self, handler: MessageHandler) -> None:
        """Start the Telegram bot.

        Args:
            handler: Callback to handle incoming messages.
        """
        self._handler = handler
        self._setup_handlers()

        self._running = True

        if self._webhook_url:
            # Webhook mode - just set up the webhook
            full_url = f"{self._webhook_url.rstrip('/')}{self._webhook_path}"
            await self._bot.set_webhook(full_url)
            logger.info(f"Webhook set to: {full_url}")
        else:
            # Polling mode
            logger.info("Starting Telegram bot in polling mode")
            await self._bot.delete_webhook(drop_pending_updates=True)
            await self._dp.start_polling(self._bot)

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False

        if self._webhook_url:
            await self._bot.delete_webhook()

        await self._bot.session.close()
        logger.info("Telegram bot stopped")

    def _setup_handlers(self) -> None:
        """Set up message handlers on the dispatcher."""

        @self._dp.message()
        async def handle_message(message: TelegramMessage) -> None:
            if not message.text or not message.from_user:
                return

            user_id = message.from_user.id
            username = message.from_user.username

            # Check if user is allowed
            if not self._is_user_allowed(user_id, username):
                logger.warning(f"Unauthorized user: {user_id} (@{username})")
                return

            # Convert to internal message format
            incoming = IncomingMessage(
                id=str(message.message_id),
                chat_id=str(message.chat.id),
                user_id=str(user_id),
                text=message.text,
                username=username,
                display_name=message.from_user.full_name,
                reply_to_message_id=str(message.reply_to_message.message_id)
                if message.reply_to_message
                else None,
                metadata={
                    "chat_type": message.chat.type,
                    "chat_title": message.chat.title,
                },
            )

            # Call handler
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
        parse_mode = None
        if message.parse_mode:
            parse_mode = ParseMode(message.parse_mode.upper())

        sent = await self._bot.send_message(
            chat_id=int(message.chat_id),
            text=message.text,
            reply_to_message_id=int(message.reply_to_message_id)
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
        # Send initial message
        sent = await self._bot.send_message(
            chat_id=int(chat_id),
            text="...",
            reply_to_message_id=int(reply_to) if reply_to else None,
        )
        message_id = str(sent.message_id)

        content = ""
        last_edit = 0.0

        async for chunk in stream:
            content += chunk

            # Rate limit edits
            now = asyncio.get_event_loop().time()
            if now - last_edit >= EDIT_INTERVAL:
                try:
                    await self._bot.edit_message_text(
                        chat_id=int(chat_id),
                        message_id=int(message_id),
                        text=content or "...",
                    )
                    last_edit = now
                except Exception as e:
                    logger.debug(f"Edit failed (likely rate limit): {e}")

        # Final edit with complete content
        if content:
            try:
                await self._bot.edit_message_text(
                    chat_id=int(chat_id),
                    message_id=int(message_id),
                    text=content,
                )
            except Exception as e:
                logger.warning(f"Final edit failed: {e}")

        return message_id

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
        pm = ParseMode(parse_mode.upper()) if parse_mode else None

        await self._bot.edit_message_text(
            chat_id=int(chat_id),
            message_id=int(message_id),
            text=text,
            parse_mode=pm,
        )

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

    async def process_webhook_update(self, update_data: dict) -> None:
        """Process a webhook update.

        Used when running in webhook mode with an external HTTP server.

        Args:
            update_data: Raw update data from Telegram.
        """
        from aiogram.types import Update

        update = Update(**update_data)
        await self._dp.feed_update(self._bot, update)
