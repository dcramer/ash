"""Chat state manager."""

import json
import logging
from pathlib import Path

from ash.chats.models import ChatInfo, ChatState
from ash.config.paths import get_chat_dir

logger = logging.getLogger(__name__)

STATE_FILENAME = "state.json"


class ChatStateManager:
    """Manages chat state persistence."""

    def __init__(
        self,
        provider: str,
        chat_id: str,
        thread_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.chat_id = chat_id
        self.thread_id = thread_id
        self._chat_dir = get_chat_dir(provider, chat_id, thread_id)
        self._state_path = self._chat_dir / STATE_FILENAME
        self._state: ChatState | None = None

    @property
    def chat_dir(self) -> Path:
        """Get the chat directory path."""
        return self._chat_dir

    @property
    def state_path(self) -> Path:
        """Get the state file path."""
        return self._state_path

    def load(self) -> ChatState:
        """Load state from disk, creating default if missing."""
        if self._state is not None:
            return self._state

        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text())
                self._state = ChatState.model_validate(data)
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Corrupt state.json, creating fresh: %s", self._state_path
                )
                self._state = self._create_default_state()
        else:
            self._state = self._create_default_state()

        return self._state

    def save(self) -> None:
        """Save state to disk."""
        if self._state is None:
            return

        self._chat_dir.mkdir(parents=True, exist_ok=True)
        data = self._state.model_dump(mode="json")
        self._state_path.write_text(json.dumps(data, indent=2, default=str))

    def update_participant(
        self,
        user_id: str,
        username: str | None = None,
        display_name: str | None = None,
    ) -> None:
        """Update participant info and save."""
        state = self.load()
        state.update_participant(user_id, username, display_name)
        self.save()

    def update_chat_info(
        self,
        chat_type: str | None = None,
        title: str | None = None,
    ) -> None:
        """Update chat metadata and save."""
        state = self.load()
        if chat_type is not None:
            state.chat.type = chat_type
        if title is not None:
            state.chat.title = title
        self.save()

    def _create_default_state(self) -> ChatState:
        """Create default state for a new chat."""
        return ChatState(chat=ChatInfo(id=self.chat_id))
