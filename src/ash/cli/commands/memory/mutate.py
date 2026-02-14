"""Mutation commands for memory entries (add, remove, clear)."""

import logging
from typing import TYPE_CHECKING

import typer

from ash.cli.commands.memory._helpers import get_memory_store
from ash.cli.console import dim, error, success, warning

if TYPE_CHECKING:
    from ash.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


async def memory_add(
    manager: "MemoryManager",
    content: str,
    source: str | None,
    expires_days: int | None,
) -> None:
    """Add a memory entry via MemoryManager (with embedding + supersession)."""
    import os

    # Read user attribution from environment (set by sandbox)
    source_username = os.environ.get("ASH_USERNAME") or None
    source_display_name = os.environ.get("ASH_DISPLAY_NAME") or None

    try:
        entry = await manager.add_memory(
            content=content,
            source=source or "cli",
            expires_in_days=expires_days,
            source_username=source_username,
            source_display_name=source_display_name,
        )
    except ValueError as e:
        error(str(e))
        raise typer.Exit(1) from None

    success(f"Added memory entry: {entry.id[:8]}")
    dim(f"Type: {entry.memory_type.value}")
    if source_username:
        dim(f"Source: @{source_username}")
    if entry.expires_at:
        dim(f"Expires: {entry.expires_at.strftime('%Y-%m-%d')}")


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

        # Get active entries only and filter
        entries = await store.get_memories(
            limit=10000, include_expired=True, include_superseded=True
        )
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

        removed_count = 0
        for entry in to_remove:
            deleted = await store.delete_memory(entry.id)
            if deleted:
                removed_count += 1
            try:
                await session.execute(
                    text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                    {"id": entry.id},
                )
            except Exception:
                logger.debug(
                    "Failed to delete embedding for memory %s", entry.id, exc_info=True
                )

        await session.commit()
        success(f"Removed {removed_count} memory entries")
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
            except Exception:
                logger.debug(
                    "Failed to delete embedding for memory %s", entry.id, exc_info=True
                )

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

    # Physically wipe memories and embeddings JSONL files
    count = await store.clear()

    # Clear vector index
    try:
        await session.execute(text("DELETE FROM memory_embeddings"))
        await session.commit()
    except Exception:
        logger.debug("Failed to clear embeddings table", exc_info=True)

    success(f"Cleared {count} memory entries")
