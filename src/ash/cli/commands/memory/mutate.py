"""Mutation commands for memory entries (add, remove, clear)."""

from datetime import UTC, datetime, timedelta

import typer

from ash.cli.commands.memory._helpers import get_memory_store
from ash.cli.console import dim, error, success, warning


async def memory_add(
    session, content: str, source: str | None, expires_days: int | None
) -> None:
    """Add a memory entry."""
    import os

    from ash.memory.types import MemoryType

    store = get_memory_store()

    expires_at = None
    if expires_days:
        expires_at = datetime.now(UTC) + timedelta(days=expires_days)

    # Read user attribution from environment (set by sandbox)
    # ASH_USERNAME is the user's handle (e.g., "notzeeg")
    # ASH_DISPLAY_NAME is the user's display name (e.g., "David Cramer")
    source_user_id = os.environ.get("ASH_USERNAME") or None
    source_user_name = os.environ.get("ASH_DISPLAY_NAME") or None

    entry = await store.add_memory(
        content=content,
        source=source or "cli",
        memory_type=MemoryType.KNOWLEDGE,
        expires_at=expires_at,
        source_user_id=source_user_id,
        source_user_name=source_user_name,
    )

    success(f"Added memory entry: {entry.id[:8]}")
    dim(f"Type: {entry.memory_type.value}")
    if source_user_id:
        dim(f"Source: @{source_user_id}")
    if expires_at:
        dim(f"Expires: {expires_at.strftime('%Y-%m-%d')}")


async def memory_remove(
    session,
    entry_id: str | None,
    all_entries: bool,
    force: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
) -> None:
    """Remove memory entries."""
    from sqlalchemy import text

    if not entry_id and not all_entries:
        error("--id or --all is required to remove entries")
        raise typer.Exit(1)

    store = get_memory_store()

    if all_entries:
        filter_desc = [
            f"{k}={v}"
            for k, v in [("user", user_id), ("chat", chat_id), ("scope", scope)]
            if v
        ]
        scope_msg = f" matching [{', '.join(filter_desc)}]" if filter_desc else ""

        if not force:
            warning(f"This will remove ALL memory entries{scope_msg}.")
            if not typer.confirm("Are you sure?"):
                dim("Cancelled")
                return

        # Get all entries and filter
        entries = await store.get_all_memories()
        to_remove = []

        for entry in entries:
            if scope == "personal" and not entry.owner_user_id:
                continue
            if scope == "shared" and entry.owner_user_id:
                continue
            if scope == "global" and (entry.owner_user_id or entry.chat_id):
                continue
            if user_id and entry.owner_user_id != user_id:
                continue
            if chat_id and entry.chat_id != chat_id:
                continue
            to_remove.append(entry)

        for entry in to_remove:
            await store.delete_memory(entry.id)
            try:
                await session.execute(
                    text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                    {"id": entry.id},
                )
            except Exception:  # noqa: S110
                pass

        await session.commit()
        success(f"Removed {len(to_remove)} memory entries")
    else:
        # Find entry by ID prefix
        assert entry_id is not None  # Guaranteed by check above
        entry = await store.get_memory_by_prefix(entry_id)

        if not entry:
            error(f"No memory entry found with ID: {entry_id}")
            raise typer.Exit(1)

        if not force:
            warning(f"Content: {entry.content[:100]}...")
            confirm = typer.confirm("Remove this entry?")
            if not confirm:
                dim("Cancelled")
                return

        # Delete the memory entry
        deleted = await store.delete_memory(entry.id)

        if deleted:
            # Delete embedding if exists
            try:
                await session.execute(
                    text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                    {"id": entry.id},
                )
                await session.commit()
            except Exception:  # noqa: S110
                pass

            success(f"Removed memory entry: {entry.id[:8]}")
        else:
            error(f"Failed to remove memory entry: {entry.id[:8]}")


async def memory_clear(session, force: bool) -> None:
    """Clear all memory entries."""
    from sqlalchemy import text

    if not force:
        warning("This will delete ALL memory entries.")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            dim("Cancelled")
            return

    store = get_memory_store()
    entries = await store.get_all_memories()

    # Delete all entries
    for entry in entries:
        await store.delete_memory(entry.id)

    # Clear embeddings
    try:
        await session.execute(text("DELETE FROM memory_embeddings"))
        await session.commit()
    except Exception:  # noqa: S110
        pass

    success(f"Cleared {len(entries)} memory entries")
