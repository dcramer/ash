"""Chat state models."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Participant(BaseModel):
    """A participant in a chat."""

    id: str
    username: str | None = None
    display_name: str | None = None
    session_id: str | None = None  # Reference to session key
    is_bot: bool = False
    first_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_active: datetime = Field(default_factory=lambda: datetime.now(UTC))
    joined_at: datetime | None = None  # When they joined (if we saw the event)
    left: bool = False  # True if they left the chat
    graph_user_id: str | None = None  # Reference to graph UserEntry.id


class ChatInfo(BaseModel):
    """Chat metadata."""

    id: str
    type: str | None = None  # "private", "group", "supergroup", "channel"
    title: str | None = None


class ChatState(BaseModel):
    """State for a chat, stored in state.json."""

    chat: ChatInfo
    participants: list[Participant] = Field(default_factory=list)
    thread_index: dict[str, str] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    graph_chat_id: str | None = None  # Reference to graph ChatEntry.id

    def get_participant(self, user_id: str) -> Participant | None:
        """Get a participant by ID."""
        return next((p for p in self.participants if p.id == user_id), None)

    def update_participant(
        self,
        user_id: str,
        username: str | None = None,
        display_name: str | None = None,
        session_id: str | None = None,
    ) -> Participant:
        """Update or add a participant, returns the participant."""
        now = datetime.now(UTC)
        participant = self.get_participant(user_id)

        if participant:
            participant.last_active = now
            if username is not None:
                participant.username = username
            if display_name is not None:
                participant.display_name = display_name
            if session_id is not None:
                participant.session_id = session_id
        else:
            participant = Participant(
                id=user_id,
                username=username,
                display_name=display_name,
                session_id=session_id,
                first_seen=now,
                last_active=now,
            )
            self.participants.append(participant)

        self.updated_at = now
        return participant
