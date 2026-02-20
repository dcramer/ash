"""Message tools for agent-to-chat communication."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ash.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from ash.chats.thread_index import ThreadIndex
    from ash.providers.base import Provider
    from ash.sessions import SessionManager

logger = logging.getLogger(__name__)


class SendMessageTool(Tool):
    """Send a message to the current chat session.

    Use this to provide status updates, intermediate results, or any
    communication that doesn't require user response. Messages are sent
    immediately and the tool returns without waiting for acknowledgment.
    """

    def __init__(
        self,
        provider: "Provider",
        session_manager_factory: "Callable[[str, str, str | None], SessionManager]",
        thread_index_factory: "Callable[[str], ThreadIndex] | None" = None,
    ) -> None:
        """Initialize the tool.

        Args:
            provider: Provider to send messages through.
            session_manager_factory: Factory to create session managers.
                Called with (chat_id, user_id, thread_id) -> SessionManager.
            thread_index_factory: Optional factory to get thread indexes.
                Called with (chat_id) -> ThreadIndex. Used to register sent
                messages so replies to them get routed correctly.
        """
        self._provider = provider
        self._get_session_manager = session_manager_factory
        self._get_thread_index = thread_index_factory

    @property
    def name(self) -> str:
        return "send_message"

    @property
    def description(self) -> str:
        return (
            "Send a message to the user in the current chat. "
            "Use for status updates, intermediate results, or notifications "
            "that don't require a response. The message appears immediately."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send to the user",
                },
            },
            "required": ["message"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        from ash.core.tokens import estimate_tokens
        from ash.providers.base import OutgoingMessage

        message = input_data.get("message", "").strip()
        if not message:
            return ToolResult.error("Message cannot be empty")

        if not context.chat_id:
            return ToolResult.error("No chat context available")

        if not context.user_id:
            return ToolResult.error("No user context available")

        # Use thread anchor if set, otherwise fall back to current message
        reply_to = context.reply_to_message_id

        try:
            sent_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=context.chat_id,
                    text=message,
                    reply_to_message_id=reply_to,
                )
            )
        except Exception as e:
            logger.exception("Failed to send message to chat %s", context.chat_id)
            return ToolResult.error(f"Failed to send message: {e}")

        # Anchor thread to first sent message so subsequent sends reply to it
        if not reply_to and sent_id:
            context.reply_to_message_id = sent_id

        logger.debug(
            "Sent message to chat %s (id=%s): %s",
            context.chat_id,
            sent_id,
            message[:50] + "..." if len(message) > 50 else message,
        )

        # Persist to session log for continuity
        try:
            session_manager = self._get_session_manager(
                context.chat_id, context.user_id, context.thread_id
            )
            await session_manager.add_assistant_message(
                content=message,
                token_count=estimate_tokens(message),
                metadata={
                    "bot_response_id": sent_id,
                    "from_tool": "send_message",
                },
            )
        except Exception as e:
            # Log but don't fail - message was sent successfully
            logger.warning("message_persist_failed", extra={"error.message": str(e)})

        # Register in thread index so replies to this message get routed correctly
        if sent_id and context.thread_id and self._get_thread_index:
            try:
                thread_index = self._get_thread_index(context.chat_id)
                thread_index.register_message(sent_id, context.thread_id)
            except Exception as e:
                logger.warning(
                    "thread_index_register_failed", extra={"error.message": str(e)}
                )

        return ToolResult.success(
            f"Message sent successfully (id: {sent_id})",
            sent_message_id=sent_id,
        )
