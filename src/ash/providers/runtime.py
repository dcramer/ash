"""Provider runtime composition helpers for CLI/server entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ash.providers.telegram import TelegramProvider
from ash.scheduling.types import ScheduleEntry

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.scheduling.handler import MessagePersister, MessageRegistrar, MessageSender


@dataclass(slots=True)
class ProviderRuntime:
    """Materialized provider runtime wiring."""

    telegram_provider: TelegramProvider | None = None
    senders: dict[str, MessageSender] = field(default_factory=dict)
    registrars: dict[str, MessageRegistrar] = field(default_factory=dict)
    persisters: dict[str, MessagePersister] = field(default_factory=dict)


def build_provider_runtime(config: AshConfig) -> ProviderRuntime:
    """Create provider instances and scheduling routing hooks from config."""
    runtime = ProviderRuntime()

    if config.telegram and config.telegram.bot_token:
        runtime.telegram_provider = TelegramProvider(
            bot_token=config.telegram.bot_token.get_secret_value(),
            allowed_users=config.telegram.allowed_users,
            allowed_groups=config.telegram.allowed_groups,
            group_mode=config.telegram.group_mode,
            passive_config=config.telegram.passive,
        )
        runtime.senders["telegram"] = runtime.telegram_provider.send_message
        runtime.registrars["telegram"] = _telegram_registrar
        runtime.persisters["telegram"] = _telegram_persister

    return runtime


async def _telegram_registrar(chat_id: str, message_id: str) -> None:
    """Register scheduled outbound telegram messages in chat thread index."""
    from ash.chats import ChatStateManager, ThreadIndex

    manager = ChatStateManager(provider="telegram", chat_id=chat_id)
    thread_index = ThreadIndex(manager)
    # Scheduled messages start new threads (message_id is both external_id and thread_id)
    thread_index.register_message(message_id, message_id)


async def _telegram_persister(
    entry: ScheduleEntry, response_text: str, message_id: str
) -> None:
    """Persist scheduled outbound telegram messages into session/chat history."""
    if not entry.chat_id or not entry.user_id:
        return

    from ash.chats import ChatHistoryWriter
    from ash.core.tokens import estimate_tokens
    from ash.sessions import SessionManager

    # Scheduled outbound messages start their own reply thread.
    thread_id = message_id
    session_manager = SessionManager(
        provider="telegram",
        chat_id=entry.chat_id,
        user_id=entry.user_id,
        thread_id=thread_id,
    )
    await session_manager.add_assistant_message(
        content=response_text,
        token_count=estimate_tokens(response_text),
        metadata={"external_id": message_id},
    )

    chat_writer = ChatHistoryWriter("telegram", entry.chat_id)
    chat_writer.record_bot_message(
        content=response_text,
        metadata={"external_id": message_id, "thread_id": thread_id},
    )
