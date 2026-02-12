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
        register_progress_tool: Callable[[ToolTracker], None],
        log_response: Callable[[str | None], None],
    ):
        self._provider = provider
        self._agent = agent
        self._session_handler = session_handler
        self._create_tool_tracker = create_tool_tracker
        self._register_progress_tool = register_progress_tool
        self._log_response = log_response

    async def handle_streaming(
        self,
        message: IncomingMessage,
        session: SessionState,
        ctx: SessionContext,
    ) -> None:
        """Handle message with streaming response."""
        # Store current message ID so send_message tool can reply to it
        session.metadata["current_message_id"] = message.id
        await self._provider.send_typing(message.chat_id)

        tracker = self._create_tool_tracker(message)
        self._register_progress_tool(tracker)
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

        async for chunk in self._agent.process_message_streaming(
            message.text,
            session,
            user_id=message.user_id,
            on_tool_start=tracker.on_tool_start,
            get_steering_messages=get_steering_messages,
            session_path=session.metadata.get("session_path"),
        ):
            response_content += chunk
            elapsed = time.time() - start_time
            since_last_edit = time.time() - last_edit_time

            if (
                elapsed > STREAM_DELAY
                and response_content.strip()
                and since_last_edit >= MIN_EDIT_INTERVAL
            ):
                summary_prefix = (
                    tracker.get_summary_prefix() if not response_msg_id else ""
                )
                display_content = summary_prefix + response_content

                if tracker.thinking_msg_id and response_msg_id is None:
                    await self._provider.edit(
                        message.chat_id, tracker.thinking_msg_id, display_content
                    )
                    response_msg_id = tracker.thinking_msg_id
                    tracker.thinking_msg_id = None
                    last_edit_time = time.time()
                elif response_msg_id is None:
                    response_msg_id = await self._provider.send(
                        OutgoingMessage(
                            chat_id=message.chat_id,
                            text=display_content,
                            reply_to_message_id=message.id,
                        )
                    )
                    last_edit_time = time.time()
                else:
                    await self._provider.edit(
                        message.chat_id, response_msg_id, display_content
                    )
                    last_edit_time = time.time()

        summary = tracker.get_summary_prefix()
        if summary or tracker.progress_messages:
            # Final response uses regular MARKDOWN, not MarkdownV2
            final_content = tracker._build_display_message(
                summary, response_content, escape_progress=False
            )
        else:
            final_content = response_content

        if tracker.thinking_msg_id:
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
                    text=response_content,
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
        )
        self._log_response(response_content)
