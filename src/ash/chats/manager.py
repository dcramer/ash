"""Chat state manager."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ash.chats.models import ChatInfo, ChatState, Participant
from ash.config.paths import get_chat_dir

if TYPE_CHECKING:
    from ash.store.store import Store

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
                    "corrupt_state_file",
                    extra={"path": str(self._state_path)},
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
        session_id: str | None = None,
    ) -> None:
        """Update participant info and save."""
        state = self.load()
        state.update_participant(user_id, username, display_name, session_id)
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

    def record_member_joined(
        self,
        user_id: str,
        username: str | None = None,
        display_name: str | None = None,
        is_bot: bool = False,
    ) -> None:
        """Record a member joining the chat."""
        state = self.load()
        now = datetime.now(UTC)
        participant = state.get_participant(user_id)

        if participant:
            # They rejoined
            participant.left = False
            participant.joined_at = now
            participant.last_active = now
            if username is not None:
                participant.username = username
            if display_name is not None:
                participant.display_name = display_name
        else:
            # New member
            participant = Participant(
                id=user_id,
                username=username,
                display_name=display_name,
                is_bot=is_bot,
                first_seen=now,
                last_active=now,
                joined_at=now,
            )
            state.participants.append(participant)

        self.save()

    def record_member_left(self, user_id: str) -> None:
        """Record a member leaving the chat."""
        state = self.load()
        participant = state.get_participant(user_id)
        if participant:
            participant.left = True
            participant.last_active = datetime.now(UTC)
            self.save()

    def _create_default_state(self) -> ChatState:
        """Create default state for a new chat."""
        return ChatState(chat=ChatInfo(id=self.chat_id))


async def sync_participates_in_edges(
    state: ChatState,
    store: "Store",
    provider: str,
    chat_id: str,
) -> None:
    """Sync PARTICIPATES_IN edges for chat participants.

    Resolves person IDs from participant usernames and ensures
    edges exist between each person and the graph chat node.
    Called from session handlers after participant updates.

    Args:
        state: Current chat state with participants.
        store: Store instance.
        provider: Chat provider name (e.g. "telegram").
        chat_id: Provider-level chat ID.
    """
    # Resolve graph chat node
    graph_chat_id = await store.get_chat_graph_id(provider, chat_id)
    if not graph_chat_id:
        return

    # Ensure graph_chat_id is recorded in state
    if state.graph_chat_id != graph_chat_id:
        state.graph_chat_id = graph_chat_id

    for participant in state.participants:
        if participant.is_bot or not participant.username:
            continue

        # Resolve person ID from username
        try:
            person_ids = await store.find_person_ids_for_username(participant.username)
        except Exception:
            logger.debug(
                "person_resolve_failed",
                extra={"user.username": participant.username},
            )
            continue

        if not person_ids:
            continue

        person_id = next(iter(person_ids))

        await store.ensure_person_participates_in_chat(person_id, graph_chat_id)

    try:
        await store.flush_graph()
    except Exception:
        logger.debug("Failed to flush PARTICIPATES_IN edges", exc_info=True)
