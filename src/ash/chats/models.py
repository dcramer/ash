"""Chat state models."""

from datetime import UTC, datetime, timedelta

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


class MutationConfirmation(BaseModel):
    """Chat-scoped proof that a mutating operation was shown and confirmed."""

    plan_id: str
    capability_id: str
    operation: str
    status: str = "presented"  # presented | confirmed | executed
    presented_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime
    confirmed_at: datetime | None = None
    executed_at: datetime | None = None
    target_fingerprint: str | None = None
    thread_id: str | None = None
    summary: str | None = None


class ChatState(BaseModel):
    """State for a chat, stored in state.json."""

    chat: ChatInfo
    participants: list[Participant] = Field(default_factory=list)
    thread_index: dict[str, str] = Field(default_factory=dict)
    active_thread_id: str | None = None
    active_thread_updated_at: datetime | None = None
    active_thread_reason: str | None = None
    mutation_confirmations: list[MutationConfirmation] = Field(default_factory=list)
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

    def set_active_thread(
        self,
        thread_id: str,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> None:
        """Record the active thread for chat-scoped DM routing."""
        ts = now or datetime.now(UTC)
        self.active_thread_id = str(thread_id)
        self.active_thread_updated_at = ts
        self.active_thread_reason = reason
        self.updated_at = ts

    def get_active_thread(
        self,
        *,
        max_age_minutes: int,
        now: datetime | None = None,
    ) -> str | None:
        """Return active thread_id if it is still within the freshness window."""
        if not self.active_thread_id or not self.active_thread_updated_at:
            return None
        ts = now or datetime.now(UTC)
        max_age = max(1, int(max_age_minutes))
        if ts - self.active_thread_updated_at > timedelta(minutes=max_age):
            return None
        return self.active_thread_id

    def add_mutation_confirmation(
        self,
        *,
        plan_id: str,
        capability_id: str,
        operation: str,
        target_fingerprint: str | None = None,
        thread_id: str | None = None,
        summary: str | None = None,
        ttl_hours: int = 24,
        now: datetime | None = None,
    ) -> MutationConfirmation:
        """Store a mutation confirmation prompt shown to the user."""
        ts = now or datetime.now(UTC)
        self.prune_expired_mutation_confirmations(now=ts)
        confirmation = MutationConfirmation(
            plan_id=plan_id,
            capability_id=capability_id,
            operation=operation,
            expires_at=ts + timedelta(hours=max(1, int(ttl_hours))),
            target_fingerprint=target_fingerprint,
            thread_id=thread_id,
            summary=summary,
        )
        self.mutation_confirmations.append(confirmation)
        self.updated_at = ts
        return confirmation

    def confirm_latest_mutation(
        self,
        *,
        now: datetime | None = None,
        thread_id: str | None = None,
    ) -> MutationConfirmation | None:
        """Confirm the latest non-expired presented mutation plan."""
        ts = now or datetime.now(UTC)
        self.prune_expired_mutation_confirmations(now=ts)
        for confirmation in reversed(self.mutation_confirmations):
            if confirmation.status != "presented":
                continue
            if (
                thread_id
                and confirmation.thread_id
                and confirmation.thread_id != thread_id
            ):
                continue
            confirmation.status = "confirmed"
            confirmation.confirmed_at = ts
            self.updated_at = ts
            return confirmation
        return None

    def find_confirmed_mutation(
        self,
        *,
        capability_id: str,
        operation: str,
        target_fingerprint: str | None = None,
        thread_id: str | None = None,
        now: datetime | None = None,
    ) -> MutationConfirmation | None:
        """Find a non-expired confirmed mutation authorization."""
        ts = now or datetime.now(UTC)
        self.prune_expired_mutation_confirmations(now=ts)
        for confirmation in reversed(self.mutation_confirmations):
            if confirmation.status != "confirmed":
                continue
            if confirmation.capability_id != capability_id:
                continue
            if confirmation.operation != operation:
                continue
            if target_fingerprint and confirmation.target_fingerprint:
                if confirmation.target_fingerprint != target_fingerprint:
                    continue
            if (
                thread_id
                and confirmation.thread_id
                and confirmation.thread_id != thread_id
            ):
                continue
            return confirmation
        return None

    def mark_mutation_executed(
        self,
        *,
        plan_id: str,
        now: datetime | None = None,
    ) -> bool:
        """Mark a confirmed mutation plan as executed."""
        ts = now or datetime.now(UTC)
        for confirmation in self.mutation_confirmations:
            if confirmation.plan_id != plan_id:
                continue
            confirmation.status = "executed"
            confirmation.executed_at = ts
            self.updated_at = ts
            return True
        return False

    def prune_expired_mutation_confirmations(
        self,
        *,
        now: datetime | None = None,
    ) -> None:
        """Remove expired mutation confirmation entries."""
        ts = now or datetime.now(UTC)
        kept = [item for item in self.mutation_confirmations if item.expires_at > ts]
        if len(kept) != len(self.mutation_confirmations):
            self.mutation_confirmations = kept
            self.updated_at = ts
