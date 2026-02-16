"""Synchronous response handling for Telegram provider.

This module provides:
- SyncHandler: Handles synchronous (non-streaming) response generation and delivery
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ash.providers.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from ash.core import Agent, SessionState
    from ash.core.types import AgentResponse
    from ash.providers.telegram.handlers.session_handler import (
        SessionContext,
        SessionHandler,
    )
    from ash.providers.telegram.handlers.tool_tracker import ToolTracker
    from ash.providers.telegram.provider import TelegramProvider

logger = logging.getLogger("telegram")


class SyncHandler:
    """Handles synchronous (non-streaming) response generation and delivery."""

    def __init__(
        self,
        provider: TelegramProvider,
        agent: Agent,
        session_handler: SessionHandler,
        create_tool_tracker: Callable[[IncomingMessage], ToolTracker],
        log_response: Callable[[str | None], None],
        store_checkpoint: Callable[..., str],
    ):
        self._provider = provider
        self._agent = agent
        self._session_handler = session_handler
        self._create_tool_tracker = create_tool_tracker
        self._log_response = log_response
        self._store_checkpoint = store_checkpoint

    async def handle_sync(
        self,
        message: IncomingMessage,
        session: SessionState,
        ctx: SessionContext,
        tracker: ToolTracker | None = None,
    ) -> None:
        """Handle message with synchronous response."""
        await self.handle_sync_with_response(message, session, ctx, tracker=tracker)

    async def handle_sync_with_response(
        self,
        message: IncomingMessage,
        session: SessionState,
        ctx: SessionContext,
        tracker: ToolTracker | None = None,
    ) -> AgentResponse:
        """Handle message with synchronous response, returning the AgentResponse."""
        from ash.providers.telegram.handlers.tool_tracker import ProgressMessageTool

        # Store current message ID so send_message tool can reply to it
        session.context.current_message_id = message.id
        if tracker is None:
            tracker = self._create_tool_tracker(message)
        progress_tool = ProgressMessageTool(tracker)

        async def get_steering_messages() -> list[IncomingMessage]:
            pending = ctx.take_pending()
            if pending:
                logger.info(
                    "Steering: %d new message(s) arrived during processing",
                    len(pending),
                )
            return pending

        typing_task = asyncio.create_task(self._typing_loop(message.chat_id))
        try:
            response = await self._agent.process_message(
                message.text,
                session,
                user_id=message.user_id,
                on_tool_start=tracker.on_tool_start,
                get_steering_messages=get_steering_messages,
                tool_overrides={progress_tool.name: progress_tool},
            )
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        response_text = response.text or ""

        # Suppress [NO_REPLY] responses (passive engagement, nothing to add)
        if response_text.strip() == "[NO_REPLY]":
            if tracker.thinking_msg_id:
                try:
                    await self._provider.delete(
                        message.chat_id, tracker.thinking_msg_id
                    )
                except Exception:
                    logger.debug("Failed to delete thinking message for NO_REPLY")
            thread_id = message.metadata.get("thread_id")
            await self._session_handler.persist_messages(
                message.chat_id,
                message.user_id,
                message.text,
                assistant_message=None,
                external_id=message.id,
                reply_to_external_id=message.reply_to_message_id,
                username=message.username,
                display_name=message.display_name,
                thread_id=thread_id,
                branch_id=session.context.branch_id,
            )
            self._log_response("[NO_REPLY]")
            return response

        # Build final content: progress messages + response (no stats)
        if tracker.progress_messages:
            parts = tracker.progress_messages + (
                ["", response_text] if response_text else []
            )
            final_content = "\n".join(parts)
        else:
            final_content = response_text

        # Check for checkpoint in response and create inline keyboard if present
        reply_markup = None
        if response.checkpoint:
            from ash.providers.telegram.checkpoint_ui import (
                create_checkpoint_keyboard,
                format_checkpoint_message,
            )

            checkpoint = response.checkpoint

            # Extract agent context from the use_agent call that triggered the checkpoint
            agent_name: str | None = None
            original_message: str | None = None
            tool_use_id: str | None = None
            for call in reversed(response.tool_calls):
                if call.get("name") == "use_agent" and call.get("metadata", {}).get(
                    "checkpoint"
                ):
                    agent_name = call["input"].get("agent")
                    original_message = call["input"].get("message")
                    tool_use_id = call["id"]
                    break

            truncated_id = self._store_checkpoint(
                checkpoint,
                message,
                agent_name=agent_name,
                original_message=original_message,
                tool_use_id=tool_use_id,
            )
            reply_markup = create_checkpoint_keyboard(checkpoint)
            checkpoint_msg = format_checkpoint_message(checkpoint)
            # Checkpoint replaces response content
            if tracker.progress_messages:
                parts = tracker.progress_messages + ["", checkpoint_msg]
                final_content = "\n".join(parts)
            else:
                final_content = checkpoint_msg
            logger.info(
                "Checkpoint detected, showing inline keyboard (id=%s, agent=%s)",
                truncated_id,
                agent_name,
            )

        from ash.providers.telegram.provider import MAX_SEND_LENGTH

        # If final content exceeds Telegram's limit and we have an existing message,
        # delete it and send as chunked messages instead
        if (
            tracker.thinking_msg_id
            and final_content.strip()
            and len(final_content) > MAX_SEND_LENGTH
        ):
            try:
                await self._provider.delete(message.chat_id, tracker.thinking_msg_id)
            except Exception:
                logger.debug("Failed to delete thinking message before chunked send")
            sent_message_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text=final_content,
                    reply_to_message_id=message.id,
                    reply_markup=reply_markup,
                )
            )
        elif tracker.thinking_msg_id and final_content.strip():
            await self._provider.edit(
                message.chat_id, tracker.thinking_msg_id, final_content
            )
            sent_message_id = tracker.thinking_msg_id
            # If we have reply_markup, send a new message since we can't add keyboard to edited message
            if reply_markup:
                sent_message_id = await self._provider.send(
                    OutgoingMessage(
                        chat_id=message.chat_id,
                        text=final_content,
                        reply_to_message_id=message.id,
                        reply_markup=reply_markup,
                    )
                )
                # Delete the thinking message since we sent a new one
                try:
                    await self._provider.delete(
                        message.chat_id, tracker.thinking_msg_id
                    )
                except Exception:
                    logger.debug("Failed to delete thinking message")
        elif tracker.thinking_msg_id:
            # Empty response — edit thinking message to fallback instead of silent delete
            fallback = "I processed your request but couldn't generate a response."
            await self._provider.edit(
                message.chat_id, tracker.thinking_msg_id, fallback
            )
            sent_message_id = tracker.thinking_msg_id
        elif final_content.strip():
            sent_message_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text=final_content,
                    reply_to_message_id=message.id,
                    reply_markup=reply_markup,
                )
            )
        else:
            sent_message_id = None

        thread_id = message.metadata.get("thread_id")
        await self._session_handler.persist_messages(
            message.chat_id,
            message.user_id,
            message.text,
            response.text,
            external_id=message.id,
            reply_to_external_id=message.reply_to_message_id,
            bot_response_id=sent_message_id,
            compaction=response.compaction,
            username=message.username,
            display_name=message.display_name,
            thread_id=thread_id,
            branch_id=session.context.branch_id,
        )
        self._log_response(response.text)

        session_manager = self._session_handler.get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        for tool_call in response.tool_calls:
            await session_manager.add_tool_use(
                tool_use_id=tool_call["id"],
                name=tool_call["name"],
                input_data=tool_call["input"],
            )
            await session_manager.add_tool_result(
                tool_use_id=tool_call["id"],
                output=tool_call["result"],
                success=not tool_call.get("is_error", False),
                metadata=tool_call.get("metadata"),
            )

        return response

    async def _typing_loop(self, chat_id: str) -> None:
        """Send typing indicators in a loop (Telegram typing lasts 5 seconds)."""
        while True:
            try:
                await self._provider.send_typing(chat_id)
                await asyncio.sleep(4)
            except asyncio.CancelledError:
                break
            except Exception:
                break
