"""Streaming response handling for Telegram provider.

This module provides:
- StreamingHandler: Handles streaming response generation and delivery
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ash.providers.base import IncomingMessage, OutgoingMessage
from ash.providers.telegram.handlers.utils import (
    MIN_EDIT_INTERVAL,
    STREAM_DELAY,
)

if TYPE_CHECKING:
    from ash.core import Agent, SessionState
    from ash.providers.telegram.handlers.session_handler import (
        SessionContext,
        SessionHandler,
    )
    from ash.providers.telegram.handlers.tool_tracker import ToolTracker
    from ash.providers.telegram.provider import TelegramProvider

logger = logging.getLogger("telegram")


class StreamingHandler:
    """Handles streaming response generation and delivery."""

    def __init__(
        self,
        provider: TelegramProvider,
        agent: Agent,
        session_handler: SessionHandler,
        create_tool_tracker: Callable[[IncomingMessage], ToolTracker],
        log_response: Callable[[str | None], None],
    ):
        self._provider = provider
        self._agent = agent
        self._session_handler = session_handler
        self._create_tool_tracker = create_tool_tracker
        self._log_response = log_response

    async def handle_streaming(
        self,
        message: IncomingMessage,
        session: SessionState,
        ctx: SessionContext,
        tracker: ToolTracker | None = None,
    ) -> None:
        """Handle message with streaming response."""
        from ash.providers.telegram.handlers.tool_tracker import ProgressMessageTool

        # Store current message ID so send_message tool can reply to it
        session.metadata["current_message_id"] = message.id
        await self._provider.send_typing(message.chat_id)

        if tracker is None:
            tracker = self._create_tool_tracker(message)
        progress_tool = ProgressMessageTool(tracker)
        response_msg_id: str | None = None
        response_content = ""
        start_time = time.time()
        last_edit_time = 0.0

        async def get_steering_messages() -> list[IncomingMessage]:
            pending = ctx.take_pending()
            if pending:
                logger.info(
                    "Steering: %d new message(s) arrived during processing",
                    len(pending),
                )
            return pending

        try:
            async for chunk in self._agent.process_message_streaming(
                message.text,
                session,
                user_id=message.user_id,
                on_tool_start=tracker.on_tool_start,
                get_steering_messages=get_steering_messages,
                tool_overrides={progress_tool.name: progress_tool},
            ):
                response_content += chunk
                elapsed = time.time() - start_time
                since_last_edit = time.time() - last_edit_time

                if (
                    elapsed > STREAM_DELAY
                    and response_content.strip()
                    and since_last_edit >= MIN_EDIT_INTERVAL
                ):
                    if tracker.thinking_msg_id and response_msg_id is None:
                        await self._provider.edit(
                            message.chat_id,
                            tracker.thinking_msg_id,
                            response_content,
                        )
                        response_msg_id = tracker.thinking_msg_id
                        tracker.thinking_msg_id = None
                    elif response_msg_id is None:
                        response_msg_id = await self._provider.send(
                            OutgoingMessage(
                                chat_id=message.chat_id,
                                text=response_content,
                                reply_to_message_id=message.id,
                            )
                        )
                    else:
                        await self._provider.edit(
                            message.chat_id, response_msg_id, response_content
                        )
                    last_edit_time = time.time()
        except Exception:
            # Clean up dangling thinking message on streaming errors
            if tracker.thinking_msg_id:
                try:
                    await self._provider.delete(
                        message.chat_id, tracker.thinking_msg_id
                    )
                except Exception:
                    logger.debug("Failed to delete thinking message on error")
                tracker.thinking_msg_id = None
            raise

        # Suppress [NO_REPLY] responses (passive engagement, nothing to add)
        if response_content.strip() == "[NO_REPLY]":
            if tracker.thinking_msg_id:
                try:
                    await self._provider.delete(
                        message.chat_id, tracker.thinking_msg_id
                    )
                except Exception:
                    logger.debug("Failed to delete thinking message for NO_REPLY")
            elif response_msg_id:
                try:
                    await self._provider.delete(message.chat_id, response_msg_id)
                except Exception:
                    logger.debug("Failed to delete streamed message for NO_REPLY")
            await self._session_handler.persist_messages(
                message.chat_id,
                message.user_id,
                message.text,
                assistant_message=None,
                external_id=message.id,
                reply_to_external_id=message.reply_to_message_id,
                username=message.username,
                display_name=message.display_name,
                thread_id=message.metadata.get("thread_id"),
                branch_id=session.metadata.get("branch_id"),
            )
            self._log_response("[NO_REPLY]")
            return

        # Build final content: progress messages + response (no stats)
        if tracker.progress_messages:
            parts = tracker.progress_messages + (
                ["", response_content] if response_content else []
            )
            final_content = "\n".join(parts)
        else:
            final_content = response_content

        # Guard against empty content — use fallback message
        if not final_content.strip():
            final_content = "I processed your request but couldn't generate a response."

        # If final content exceeds Telegram's limit and we have an existing message,
        # delete it and send as chunked messages instead
        from ash.providers.telegram.provider import MAX_SEND_LENGTH

        existing_msg_id = tracker.thinking_msg_id or response_msg_id
        if len(final_content) > MAX_SEND_LENGTH and existing_msg_id:
            try:
                await self._provider.delete(message.chat_id, existing_msg_id)
            except Exception:
                logger.debug("Failed to delete message before chunked send")
            sent_message_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text=final_content,
                    reply_to_message_id=message.id,
                )
            )
        elif tracker.thinking_msg_id:
            await self._provider.edit(
                message.chat_id, tracker.thinking_msg_id, final_content
            )
            sent_message_id = tracker.thinking_msg_id
        elif response_msg_id:
            await self._provider.edit(message.chat_id, response_msg_id, final_content)
            sent_message_id = response_msg_id
        else:
            sent_message_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text=final_content,
                    reply_to_message_id=message.id,
                )
            )

        await self._session_handler.persist_messages(
            message.chat_id,
            message.user_id,
            message.text,
            response_content,
            external_id=message.id,
            reply_to_external_id=message.reply_to_message_id,
            bot_response_id=sent_message_id,
            username=message.username,
            display_name=message.display_name,
            thread_id=message.metadata.get("thread_id"),
            branch_id=session.metadata.get("branch_id"),
        )
        self._log_response(response_content)
