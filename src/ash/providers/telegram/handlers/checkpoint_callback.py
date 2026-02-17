"""Callback processing helpers for checkpoint handling.

Extracted from checkpoint_handler.py to keep that module focused on
checkpoint storage/retrieval and high-level callback routing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery

logger = logging.getLogger("telegram")


@dataclass
class CallbackContext:
    """Parsed callback query data with routing information."""

    truncated_id: str
    option_index: int
    bot_response_id: str | None
    chat_id: str | None
    user_id: str | None

    @classmethod
    def from_callback_query(
        cls, callback_query: CallbackQuery, truncated_id: str, option_index: int
    ) -> CallbackContext:
        """Create context from a callback query."""
        bot_response_id = (
            str(callback_query.message.message_id) if callback_query.message else None
        )
        chat_id = (
            str(callback_query.message.chat.id) if callback_query.message else None
        )
        user_id = str(callback_query.from_user.id) if callback_query.from_user else None
        return cls(
            truncated_id=truncated_id,
            option_index=option_index,
            bot_response_id=bot_response_id,
            chat_id=chat_id,
            user_id=user_id,
        )


@dataclass
class ValidationResult:
    """Result of callback validation."""

    success: bool
    error_message: str | None = None
    show_alert: bool = False


class CallbackValidator:
    """Validates callback queries for checkpoint processing."""

    @staticmethod
    def parse_callback_data(
        callback_query: CallbackQuery,
    ) -> tuple[CallbackContext | None, str | None]:
        """Parse callback data and return context or error message.

        Returns:
            Tuple of (context, error_message). If parsing succeeds,
            context is set and error_message is None. If parsing fails,
            context is None and error_message describes the issue.
        """
        from ash.providers.telegram.checkpoint_ui import parse_callback_data

        if not callback_query.data:
            logger.warning("callback_query_no_data")
            return None, "Invalid callback data"

        parsed = parse_callback_data(callback_query.data)
        if parsed is None:
            logger.warning(
                "callback_parse_failed", extra={"callback_data": callback_query.data}
            )
            return None, "Invalid callback format"

        truncated_id, option_index = parsed
        context = CallbackContext.from_callback_query(
            callback_query, truncated_id, option_index
        )
        return context, None

    @staticmethod
    def validate_options(option_index: int, options: list[str]) -> ValidationResult:
        """Validate that option index is within bounds."""
        if option_index < 0 or option_index >= len(options):
            logger.warning(
                "invalid_option_index", extra={"checkpoint.option_index": option_index}
            )
            return ValidationResult(
                success=False, error_message="Invalid option selected"
            )
        return ValidationResult(success=True)

    @staticmethod
    def validate_user(
        callback_query: CallbackQuery, expected_user_id: str
    ) -> ValidationResult:
        """Validate that the clicking user is the expected user."""
        from_user = callback_query.from_user
        if not from_user:
            logger.warning("callback_query_no_user")
            return ValidationResult(
                success=False,
                error_message="Unable to verify user.",
                show_alert=True,
            )
        if str(from_user.id) != expected_user_id:
            return ValidationResult(
                success=False,
                error_message="This question was asked to another user.",
                show_alert=True,
            )
        return ValidationResult(success=True)


class ResponseFinalizer:
    """Handles response finalization for checkpoint callbacks."""

    def __init__(
        self,
        provider: Any,
        session_manager: Any,
        thread_index: Any | None,
        chat_id: str,
        thread_id: str | None,
        routing: dict[str, Any],
    ):
        self._provider = provider
        self._session_manager = session_manager
        self._thread_index = thread_index
        self._chat_id = chat_id
        self._thread_id = thread_id
        self._routing = routing

    async def finalize(
        self,
        *,
        result: Any,
        tracker: Any,
        checkpoint_message_id: str | None,
        checkpoint_id: str,
        selected_option: str,
        user_id: str,
        tool_use_id: str,
        tool_input: dict[str, Any],
        agent_name: str | None,
        original_message: str | None,
        store_checkpoint_fn: Any,
    ) -> str | None:
        """Finalize the response, handling nested checkpoints and persistence.

        Returns the sent message ID if a message was sent.
        """
        from ash.providers.base import IncomingMessage
        from ash.providers.telegram.checkpoint_ui import (
            create_checkpoint_keyboard,
            format_checkpoint_message,
        )
        from ash.tools.builtin.agents import CHECKPOINT_METADATA_KEY

        reply_markup = None
        response_text = result.content

        # Check for nested checkpoint in the result
        if CHECKPOINT_METADATA_KEY in result.metadata:
            new_checkpoint = result.metadata[CHECKPOINT_METADATA_KEY]

            synthetic_msg = IncomingMessage(
                id="",
                chat_id=self._chat_id,
                user_id=user_id,
                text="",
                username=self._routing.get("username"),
                display_name=self._routing.get("display_name"),
                metadata={
                    "thread_id": self._thread_id,
                    "chat_type": self._routing.get("chat_type"),
                    "chat_title": self._routing.get("chat_title"),
                },
            )
            new_truncated_id = store_checkpoint_fn(
                new_checkpoint,
                synthetic_msg,
                agent_name=agent_name,
                original_message=original_message,
            )

            reply_markup = create_checkpoint_keyboard(new_checkpoint)
            response_text = format_checkpoint_message(new_checkpoint)
            logger.info(
                "nested_checkpoint_detected",
                extra={"checkpoint.id": new_truncated_id},
            )

        sent_message_id: str | None = None
        if response_text.strip():
            sent_message_id = await self._send_response(
                response_text=response_text,
                reply_markup=reply_markup,
                tracker=tracker,
                checkpoint_message_id=checkpoint_message_id,
            )

            await self._persist_interaction(
                selected_option=selected_option,
                checkpoint_id=checkpoint_id,
                user_id=user_id,
                response_text=response_text,
                sent_message_id=sent_message_id,
            )

            await self._persist_tool_call(
                tool_use_id=tool_use_id,
                tool_input=tool_input,
                result=result,
            )

            if sent_message_id and self._thread_id and self._thread_index:
                self._thread_index.register_message(sent_message_id, self._thread_id)
        else:
            # Still persist tool_use/result even for empty responses
            await self._persist_tool_call(
                tool_use_id=tool_use_id,
                tool_input=tool_input,
                result=result,
            )
            logger.debug("Empty response from resumed agent")

        return sent_message_id

    async def _send_response(
        self,
        *,
        response_text: str,
        reply_markup: Any | None,
        tracker: Any,
        checkpoint_message_id: str | None,
    ) -> str | None:
        """Send the response, handling nested checkpoints differently."""
        from ash.providers.base import OutgoingMessage

        if reply_markup:
            # Nested checkpoint: need keyboard, delete thinking msg and send new
            if tracker.thinking_msg_id:
                try:
                    await self._provider.delete(self._chat_id, tracker.thinking_msg_id)
                except Exception as delete_err:
                    logger.debug("Failed to delete thinking message: %s", delete_err)
            return await self._provider.send(
                OutgoingMessage(
                    chat_id=self._chat_id,
                    text=response_text,
                    reply_to_message_id=checkpoint_message_id,
                    reply_markup=reply_markup,
                )
            )
        else:
            # No nested checkpoint - use tracker finalization
            return await tracker.finalize_response(response_text)

    async def _persist_interaction(
        self,
        *,
        selected_option: str,
        checkpoint_id: str,
        user_id: str,
        response_text: str,
        sent_message_id: str | None,
    ) -> None:
        """Persist user and assistant messages to session."""
        from ash.core.tokens import estimate_tokens

        await self._session_manager.add_user_message(
            content=f"[Checkpoint response: {selected_option}]",
            token_count=estimate_tokens(selected_option),
            metadata={
                "is_checkpoint_response": True,
                "checkpoint.id": checkpoint_id,
            },
            user_id=user_id,
            username=self._routing.get("username"),
            display_name=self._routing.get("display_name"),
        )
        await self._session_manager.add_assistant_message(
            content=response_text,
            token_count=estimate_tokens(response_text),
            metadata={"bot_response_id": sent_message_id} if sent_message_id else None,
        )

    async def _persist_tool_call(
        self,
        *,
        tool_use_id: str,
        tool_input: dict[str, Any],
        result: Any,
    ) -> None:
        """Persist tool_use and tool_result for session consistency."""
        await self._session_manager.add_tool_use(
            tool_use_id=tool_use_id,
            name="use_agent",
            input_data=tool_input,
        )
        await self._session_manager.add_tool_result(
            tool_use_id=tool_use_id,
            output=result.content,
            success=not result.is_error,
            metadata=result.metadata,
        )
