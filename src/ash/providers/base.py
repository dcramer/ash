"""Abstract provider interface for communication channels."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IncomingMessage:
    """Message received from a provider."""

    id: str
    chat_id: str
    user_id: str
    text: str
    username: str | None = None
    display_name: str | None = None
    reply_to_message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutgoingMessage:
    """Message to send via a provider."""

    chat_id: str
    text: str
    reply_to_message_id: str | None = None
    parse_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Type for message handler callback
MessageHandler = Callable[[IncomingMessage], Awaitable[None]]


class Provider(ABC):
    """Abstract interface for communication providers.

    Providers handle receiving messages from and sending messages to
    external services like Telegram, Discord, Slack, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'telegram', 'discord')."""
        ...

    @abstractmethod
    async def start(self, handler: MessageHandler) -> None:
        """Start the provider and begin receiving messages.

        Args:
            handler: Callback to handle incoming messages.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the provider and clean up resources."""
        ...

    @abstractmethod
    async def send(self, message: OutgoingMessage) -> str:
        """Send a message.

        Args:
            message: Message to send.

        Returns:
            Sent message ID.
        """
        ...

    @abstractmethod
    async def send_streaming(
        self,
        chat_id: str,
        stream: AsyncIterator[str],
        *,
        reply_to: str | None = None,
    ) -> str:
        """Send a message with streaming updates.

        Implementations should edit the message as new content arrives.

        Args:
            chat_id: Chat to send to.
            stream: Async iterator of text chunks.
            reply_to: Message to reply to.

        Returns:
            Final message ID.
        """
        ...

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
        raise NotImplementedError("Provider does not support message editing")

    async def delete(self, chat_id: str, message_id: str) -> None:
        """Delete a message.

        Args:
            chat_id: Chat containing the message.
            message_id: Message to delete.
        """
        raise NotImplementedError("Provider does not support message deletion")
