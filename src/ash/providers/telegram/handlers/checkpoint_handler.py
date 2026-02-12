"""Checkpoint handling for Telegram inline keyboard callbacks.

This module provides:
- CheckpointHandler: Manages checkpoint storage and resume via callbacks
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from ash.providers.base import IncomingMessage, OutgoingMessage
from ash.providers.telegram.provider import _truncate
from ash.sessions.types import session_key as make_session_key

from .tool_tracker import ProgressMessageTool, ToolTracker

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery

    from ash.agents import AgentRegistry
    from ash.chats import ThreadIndex
    from ash.config import AshConfig
    from ash.providers.telegram.provider import TelegramProvider
    from ash.sessions import SessionManager
    from ash.skills import SkillRegistry
    from ash.tools.registry import ToolRegistry

logger = logging.getLogger("telegram")


class CheckpointHandler:
    """Handles checkpoint storage and resume via inline keyboard callbacks.

    This handler manages the workflow when agents pause for user input:
    1. Store checkpoint routing info for callback lookup
    2. Retrieve checkpoints from cache or session log
    3. Process callback button clicks to resume agents
    """

    def __init__(
        self,
        provider: TelegramProvider,
        get_session_manager: Callable[[str, str, str | None], SessionManager],
        get_session_managers_dict: Callable[[], dict[str, SessionManager]],
        get_thread_index: Callable[[str], ThreadIndex],
        handle_message: Callable[[IncomingMessage], Coroutine[Any, Any, None]],
        config: AshConfig | None = None,
        agent_registry: AgentRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self._provider = provider
        self._get_session_manager = get_session_manager
        self._get_session_managers_dict = get_session_managers_dict
        self._get_thread_index = get_thread_index
        self._handle_message = handle_message
        self._config = config
        self._agent_registry = agent_registry
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry
        self._pending_checkpoints: dict[str, dict[str, Any]] = {}

    def clear_all_checkpoints(self) -> None:
        """Clear all pending checkpoints from memory cache."""
        self._pending_checkpoints.clear()

    def store_checkpoint(
        self,
        checkpoint: dict[str, Any],
        message: IncomingMessage,
        *,
        agent_name: str | None = None,
        original_message: str | None = None,
        tool_use_id: str | None = None,
    ) -> str:
        """Store checkpoint routing info for callback lookup and return its truncated ID.

        Stores routing info in-memory for fast lookup. Full checkpoint data is
        persisted in tool_result metadata in the session log.
        """
        from ash.providers.telegram.checkpoint_ui import MAX_CHECKPOINT_ID_LEN

        truncated_id = checkpoint.get("checkpoint_id", "")[:MAX_CHECKPOINT_ID_LEN]
        thread_id = message.metadata.get("thread_id")
        session_key = make_session_key(
            self._provider.name, message.chat_id, message.user_id, thread_id
        )

        # Store routing info in memory for fast lookup
        # Full checkpoint data is in session log via tool_result metadata
        self._pending_checkpoints[truncated_id] = {
            "session_key": session_key,
            "chat_id": message.chat_id,
            "user_id": message.user_id,
            "thread_id": thread_id,
            "chat_type": message.metadata.get("chat_type"),
            "chat_title": message.metadata.get("chat_title"),
            "username": message.username,
            "display_name": message.display_name,
            "agent_name": agent_name,
            "original_message": original_message,
        }

        return truncated_id

    async def get_checkpoint(
        self,
        truncated_id: str,
        bot_response_id: str | None = None,
        chat_id: str | None = None,
        user_id: str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Get checkpoint, using cache or falling back to session log lookup.

        Returns (routing_info, checkpoint_data) or (None, None).
        routing_info contains session routing info, checkpoint_data contains the full checkpoint.
        """
        # Fast path: check in-memory cache for routing info
        if truncated_id in self._pending_checkpoints:
            routing = self._pending_checkpoints[truncated_id]
            session_manager = self._get_session_manager(
                routing["chat_id"], routing["user_id"], routing.get("thread_id")
            )
            result = await session_manager.get_pending_checkpoint_from_log(truncated_id)
            if result:
                _, _, checkpoint = result
                return routing, checkpoint

        # Slow path (recovery): find session by bot_response_id in loaded sessions
        if bot_response_id:
            for sm in self._get_session_managers_dict().values():
                if await sm.has_bot_response_id(bot_response_id):
                    result = await sm.get_pending_checkpoint_from_log(truncated_id)
                    if result:
                        _, _, checkpoint = result
                        # Build routing info from checkpoint
                        routing = {
                            "session_key": sm.session_key,
                            "chat_id": sm.chat_id,
                            "user_id": sm.user_id,
                            "thread_id": sm.thread_id,
                        }
                        logger.info(
                            "Recovered checkpoint %s from session log",
                            truncated_id[:20],
                        )
                        return routing, checkpoint

        # Disk recovery: try loading session directly from chat/user context
        # This handles server restarts where _session_managers is empty
        if chat_id and user_id:
            # Try without thread_id first (most common case)
            session_manager = self._get_session_manager(chat_id, user_id, None)
            result = await session_manager.get_pending_checkpoint_from_log(truncated_id)
            if result:
                _, _, checkpoint = result
                routing = {
                    "session_key": session_manager.session_key,
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "thread_id": None,
                }
                logger.info(
                    "Recovered checkpoint %s from disk using chat context",
                    truncated_id[:20],
                )
                return routing, checkpoint

        return None, None

    def clear_checkpoint(self, truncated_id: str) -> None:
        """Clear checkpoint routing info from memory cache."""
        self._pending_checkpoints.pop(truncated_id, None)

    async def handle_callback_query(self, callback_query: CallbackQuery) -> None:
        """Handle callback queries from checkpoint inline keyboards.

        When a user clicks a button on a checkpoint keyboard, this method:
        1. Parses the callback data to get checkpoint info
        2. Retrieves the stored checkpoint with agent context
        3. Calls the use_agent tool directly with resume parameters
        4. Formats and sends the result to the user
        5. Handles nested checkpoints if the resumed agent pauses again
        """
        from .checkpoint_callback import CallbackValidator

        # Parse callback data
        context, error = CallbackValidator.parse_callback_data(callback_query)
        if context is None:
            await callback_query.answer(error or "Invalid callback data")
            return

        # Retrieve checkpoint
        routing, checkpoint = await self.get_checkpoint(
            context.truncated_id,
            context.bot_response_id,
            chat_id=context.chat_id,
            user_id=context.user_id,
        )
        if checkpoint is None or routing is None:
            logger.warning("Checkpoint not found: %s", context.truncated_id)
            await callback_query.answer(
                "Checkpoint not found. It may have expired or the session was lost.",
                show_alert=True,
            )
            return

        # Validate option selection
        options = checkpoint.get("options") or ["Proceed", "Cancel"]
        options_result = CallbackValidator.validate_options(
            context.option_index, options
        )
        if not options_result.success:
            await callback_query.answer(
                options_result.error_message or "Invalid option"
            )
            return

        selected_option = options[context.option_index]

        # Validate user authorization
        user_id = routing.get("user_id", "")
        user_result = CallbackValidator.validate_user(callback_query, user_id)
        if not user_result.success:
            await callback_query.answer(
                user_result.error_message or "Unauthorized",
                show_alert=user_result.show_alert,
            )
            return

        # Process with log context for traceability
        from ash.logging import log_context

        chat_id = routing.get("chat_id", "")
        session_key = routing.get("session_key", "")

        with log_context(chat_id=chat_id, session_id=session_key):
            await self._handle_callback_query_inner(
                callback_query=callback_query,
                routing=routing,
                checkpoint=checkpoint,
                selected_option=selected_option,
                truncated_id=context.truncated_id,
            )

    async def _handle_callback_query_inner(
        self,
        callback_query: CallbackQuery,
        routing: dict[str, Any],
        checkpoint: dict[str, Any],
        selected_option: str,
        truncated_id: str,
    ) -> None:
        """Inner implementation of callback query handling (runs with log context)."""
        from ash.tools.base import ToolContext

        from .checkpoint_callback import ResponseFinalizer

        # Extract routing data
        chat_id = routing.get("chat_id", "")
        user_id = routing.get("user_id", "")
        thread_id = routing.get("thread_id")
        agent_name = routing.get("agent_name")
        original_message = routing.get("original_message")
        checkpoint_id = checkpoint.get("checkpoint_id")
        session_key = routing.get("session_key", "")

        # Don't clear checkpoint yet - wait until processing succeeds
        await callback_query.answer(f"Selected: {selected_option}")

        # Store checkpoint message ID for reply threading and update the message
        message = callback_query.message
        checkpoint_message_id = str(message.message_id) if message else None

        if checkpoint_message_id:
            try:
                original_text = getattr(message, "text", None) or "Checkpoint"
                updated_text = f"{original_text}\n\n✓ Selected: {selected_option}"
                await self._provider.edit(chat_id, checkpoint_message_id, updated_text)
            except Exception as e:
                logger.debug("Failed to update message: %s", e)

        # Check if we can use direct tool invocation
        has_agent_context = agent_name and original_message and checkpoint_id
        has_tool_registry = self._tool_registry and self._tool_registry.has("use_agent")

        if not has_agent_context or not has_tool_registry:
            reason = "agent context" if not has_agent_context else "tool registry"
            logger.warning(
                "Missing %s for checkpoint %s, falling back to message flow",
                reason,
                truncated_id,
            )
            # Clear checkpoint before fallback (fallback will create new session context)
            self.clear_checkpoint(truncated_id)
            await self._handle_checkpoint_via_message(
                callback_query, routing, checkpoint, selected_option
            )
            return

        logger.info(
            "Resuming checkpoint via direct tool call: agent=%s, checkpoint=%s, response='%s'",
            agent_name,
            truncated_id[:20],
            selected_option,
        )

        await self._provider.send_typing(chat_id)

        assert self._tool_registry is not None  # Checked above via has_tool_registry

        # Restore CheckpointState to UseAgentTool's cache before calling execute
        from ash.agents.types import CheckpointState
        from ash.tools.builtin.agents import UseAgentTool

        use_agent_tool = self._tool_registry.get("use_agent")
        if not isinstance(use_agent_tool, UseAgentTool):
            logger.error("use_agent tool is not a UseAgentTool instance")
            await self._provider.send(
                OutgoingMessage(
                    chat_id=chat_id,
                    text="Error: use_agent tool is not properly configured.",
                    reply_to_message_id=checkpoint_message_id,
                )
            )
            return

        # checkpoint_id is guaranteed to be non-None here (checked in has_agent_context above)
        assert checkpoint_id is not None
        existing = await use_agent_tool.get_checkpoint(checkpoint_id)
        if existing is None:
            checkpoint_state = CheckpointState.from_dict(checkpoint)
            await use_agent_tool.store_checkpoint(checkpoint_state)
            logger.info(
                "Restored checkpoint %s to UseAgentTool cache",
                truncated_id,
            )

        # Create tracker for resume flow (reply to checkpoint message)
        tracker = ToolTracker(
            provider=self._provider,
            chat_id=chat_id,
            reply_to=checkpoint_message_id or "",
            config=self._config,
            agent_registry=self._agent_registry,
            skill_registry=self._skill_registry,
        )
        progress_tool = ProgressMessageTool(tracker)

        tool_context = ToolContext(
            session_id=session_key,
            user_id=user_id,
            chat_id=chat_id,
            thread_id=thread_id,
            provider=self._provider.name,
            metadata={"current_message_id": checkpoint_message_id},
            tool_overrides={progress_tool.name: progress_tool},
        )

        tool_use_id = f"callback_{uuid.uuid4().hex[:12]}"
        tool_input = {
            "agent": agent_name,
            "message": original_message,
            "resume_checkpoint_id": checkpoint_id,
            "checkpoint_response": selected_option,
        }

        try:
            result = await use_agent_tool.execute(tool_input, tool_context)
        except Exception as e:
            logger.exception("Error calling use_agent tool directly")
            if tracker.thinking_msg_id:
                try:
                    await self._provider.delete(chat_id, tracker.thinking_msg_id)
                except Exception as delete_err:
                    logger.debug("Failed to delete thinking message: %s", delete_err)
            await self._provider.send(
                OutgoingMessage(
                    chat_id=chat_id,
                    text=f"Error resuming agent: {e}. You can try clicking the button again.",
                    reply_to_message_id=checkpoint_message_id,
                )
            )
            return

        # Clear the checkpoint now that processing succeeded
        self.clear_checkpoint(truncated_id)

        # Finalize response using ResponseFinalizer
        session_manager = self._get_session_manager(chat_id, user_id, thread_id)
        thread_index = self._get_thread_index(chat_id) if thread_id else None

        finalizer = ResponseFinalizer(
            provider=self._provider,
            session_manager=session_manager,
            thread_index=thread_index,
            chat_id=chat_id,
            thread_id=thread_id,
            routing=routing,
        )

        sent_message_id = await finalizer.finalize(
            result=result,
            tracker=tracker,
            checkpoint_message_id=checkpoint_message_id,
            checkpoint_id=checkpoint_id,
            selected_option=selected_option,
            user_id=user_id,
            tool_use_id=tool_use_id,
            tool_input=tool_input,
            agent_name=agent_name,
            original_message=original_message,
            store_checkpoint_fn=self.store_checkpoint,
        )

        if sent_message_id and result.content.strip():
            self._log_response(result.content)

    async def _handle_checkpoint_via_message(
        self,
        callback_query: CallbackQuery,
        routing: dict[str, Any],
        checkpoint: dict[str, Any],
        selected_option: str,
    ) -> None:
        """Fall back to synthetic message flow for checkpoint handling.

        Used when agent context is not available for direct tool invocation.
        """
        from_user = callback_query.from_user
        username = from_user.username if from_user else routing.get("username")
        display_name = from_user.full_name if from_user else routing.get("display_name")

        metadata: dict[str, Any] = {
            "is_checkpoint_response": True,
            "checkpoint_id": checkpoint.get("checkpoint_id"),
        }
        for key in ("thread_id", "chat_type", "chat_title"):
            if value := routing.get(key):
                metadata[key] = value

        synthetic_message = IncomingMessage(
            id=f"callback_{callback_query.id}",
            chat_id=routing.get("chat_id", ""),
            user_id=routing.get("user_id", ""),
            text=selected_option,
            username=username,
            display_name=display_name,
            metadata=metadata,
        )

        logger.info(
            "Processing checkpoint callback via message flow: '%s' (session=%s)",
            selected_option,
            routing.get("session_key", ""),
        )

        await self._handle_message(synthetic_message)

    def _log_response(self, text: str | None) -> None:
        bot_name = self._provider.bot_username or "bot"
        logger.info("[cyan]%s:[/cyan] %s", bot_name, _truncate(text or "(no response)"))
