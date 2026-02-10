"""Memory management commands."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import click
import typer

from ash.cli.console import console, dim, error, success, warning
from ash.cli.context import get_config, get_database

if TYPE_CHECKING:
    from ash.memory.file_store import FileMemoryStore


def _get_memory_store() -> "FileMemoryStore":
    """Get a configured FileMemoryStore instance.

    This factory function allows for easier testing and future
    path injection if needed.
    """
    from ash.memory.file_store import FileMemoryStore

    return FileMemoryStore()


def _matches_person(source_id: str, person) -> bool:
    """Check if source_id matches a person's name or aliases."""
    if person.name.lower() == source_id:
        return True
    for alias in person.aliases or []:
        if alias.lower() == source_id:
            return True
    return False


def _is_source_self_reference(
    source_user_id: str | None,
    owner_user_id: str | None,
    subject_person_ids: list[str] | None,
    all_people: list,
    people_by_id: dict,
) -> bool:
    """Determine if the source user is speaking about themselves.

    Returns True if:
    - source matches a self-person (owner speaking about themselves)
    - source matches any subject person (source is the subject)
    """
    if not source_user_id:
        return False

    source_id = source_user_id.lower()

    # Check if source matches a self-person
    for person in all_people:
        if person.owner_user_id == owner_user_id and person.relationship == "self":
            if _matches_person(source_id, person):
                return True

    # Check if source matches any subject person
    if subject_person_ids:
        for person_id in subject_person_ids:
            person = people_by_id.get(person_id)
            if person and _matches_person(source_id, person):
                return True

    return False


def register(app: typer.Typer) -> None:
    """Register the memory command."""

    @app.command()
    def memory(
        action: Annotated[
            str | None,
            typer.Argument(
                help="Action: list, show, add, remove, clear, gc, rebuild-index, history, doctor"
            ),
        ] = None,
        target: Annotated[
            str | None,
            typer.Argument(
                help="Target ID for show/history commands",
            ),
        ] = None,
        query: Annotated[
            str | None,
            typer.Option(
                "--query",
                "-q",
                help="Search query or content to add",
            ),
        ] = None,
        entry_id: Annotated[
            str | None,
            typer.Option(
                "--id",
                help="Memory entry ID (for remove, deprecated: use positional arg)",
            ),
        ] = None,
        source: Annotated[
            str | None,
            typer.Option(
                "--source",
                "-s",
                help="Source label for new entry",
            ),
        ] = "cli",
        expires_days: Annotated[
            int | None,
            typer.Option(
                "--expires",
                "-e",
                help="Days until expiration (for add)",
            ),
        ] = None,
        include_expired: Annotated[
            bool,
            typer.Option(
                "--include-expired",
                help="Include expired entries",
            ),
        ] = False,
        limit: Annotated[
            int,
            typer.Option(
                "--limit",
                "-n",
                help="Maximum entries to show",
            ),
        ] = 20,
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                "-f",
                help="Force action without confirmation",
            ),
        ] = False,
        all_entries: Annotated[
            bool,
            typer.Option(
                "--all",
                help="Remove all entries (for remove action)",
            ),
        ] = False,
        user_id: Annotated[
            str | None,
            typer.Option(
                "--user",
                "-u",
                help="Filter by owner user ID",
            ),
        ] = None,
        chat_id: Annotated[
            str | None,
            typer.Option(
                "--chat",
                help="Filter by chat ID",
            ),
        ] = None,
        scope: Annotated[
            str | None,
            typer.Option(
                "--scope",
                help="Filter by scope: personal, shared, or global",
            ),
        ] = None,
    ) -> None:
        """Manage memory entries.

        Examples:
            ash memory list                    # List all memories
            ash memory list -q "api keys"      # Filter memories by content
            ash memory list --scope personal   # List personal memories only
            ash memory list --scope shared     # List shared/group memories
            ash memory list --user bob         # List memories owned by bob
            ash memory show <id>               # Show full details of a memory
            ash memory add -q "User prefers dark mode"
            ash memory remove <id>             # Remove specific entry
            ash memory remove --all            # Remove all entries
            ash memory clear                   # Clear all memory entries
            ash memory gc                      # Garbage collect expired/superseded
            ash memory rebuild-index           # Rebuild vector index from JSONL
            ash memory history <id>            # Show supersession chain
        """
        if action is None:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            raise typer.Exit(0)

        # Use target (positional) or entry_id (--id flag) for ID-based commands
        resolved_id = target or entry_id

        try:
            asyncio.run(
                _run_memory_action(
                    action=action,
                    query=query,
                    entry_id=resolved_id,
                    source=source,
                    expires_days=expires_days,
                    include_expired=include_expired,
                    limit=limit,
                    config_path=config_path,
                    force=force,
                    all_entries=all_entries,
                    user_id=user_id,
                    chat_id=chat_id,
                    scope=scope,
                )
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled[/dim]")


async def _run_memory_action(
    action: str,
    query: str | None,
    entry_id: str | None,
    source: str | None,
    expires_days: int | None,
    include_expired: bool,
    limit: int,
    config_path: Path | None,
    force: bool,
    all_entries: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
) -> None:
    """Run memory action asynchronously."""
    if scope and scope not in ("personal", "shared", "global"):
        error("--scope must be: personal, shared, or global")
        raise typer.Exit(1)

    config = get_config(config_path)
    database = await get_database(config)

    try:
        async with database.session() as session:
            if action == "list":
                await _memory_list(
                    session, query, limit, include_expired, user_id, chat_id, scope
                )
            elif action == "add":
                if not query:
                    error("--query is required to specify content to add")
                    raise typer.Exit(1)
                await _memory_add(session, query, source, expires_days)
            elif action == "remove":
                await _memory_remove(
                    session, entry_id, all_entries, force, user_id, chat_id, scope
                )
            elif action == "clear":
                await _memory_clear(session, force)
            elif action == "gc":
                await _memory_gc()
            elif action == "rebuild-index":
                await _memory_rebuild_index(session)
            elif action == "show":
                if not entry_id:
                    error("Usage: ash memory show <id>")
                    raise typer.Exit(1)
                await _memory_show(entry_id)
            elif action == "history":
                if not entry_id:
                    error("Usage: ash memory history <id>")
                    raise typer.Exit(1)
                await _memory_history(entry_id)
            elif action == "doctor":
                await _memory_doctor(config, force)
            else:
                error(f"Unknown action: {action}")
                console.print(
                    "Valid actions: list, show, add, remove, clear, gc, rebuild-index, history, doctor"
                )
                raise typer.Exit(1)
    finally:
        await database.disconnect()


async def _memory_list(
    session,
    query: str | None,
    limit: int,
    include_expired: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
) -> None:
    """List memory entries."""
    from rich.table import Table

    store = _get_memory_store()

    # Apply scope-based filters
    owner_user_id = None
    filter_chat_id = None

    if scope == "personal":
        if user_id:
            owner_user_id = user_id
        else:
            # Need to show all personal memories - get all and filter
            pass
    elif scope == "shared":
        if chat_id:
            filter_chat_id = chat_id
    elif user_id:
        owner_user_id = user_id
    elif chat_id:
        filter_chat_id = chat_id

    entries = await store.get_memories(
        limit=limit * 2,  # Get more to filter
        include_expired=include_expired,
        include_superseded=False,
        owner_user_id=owner_user_id,
        chat_id=filter_chat_id,
    )

    # Apply additional filters
    filtered_entries = []
    for entry in entries:
        # Scope filter
        if scope == "personal" and not entry.owner_user_id:
            continue
        if scope == "shared" and entry.owner_user_id:
            continue
        if scope == "global" and (entry.owner_user_id or entry.chat_id):
            continue

        # Content filter
        if query and query.lower() not in entry.content.lower():
            continue

        filtered_entries.append(entry)
        if len(filtered_entries) >= limit:
            break

    # Load people for name lookup
    from ash.config.paths import get_people_jsonl_path
    from ash.memory.jsonl import PersonJSONL

    people_jsonl = PersonJSONL(get_people_jsonl_path())
    people = await people_jsonl.load_all()
    people_by_id = {p.id: p for p in people}

    if not filtered_entries:
        if query:
            warning(f"No memories found matching '{query}'")
        else:
            warning("No memory entries found")
        return

    title = f"Memory Search: '{query}'" if query else "Memory Entries"
    table = Table(title=title)
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Type", style="blue", max_width=10)
    table.add_column("About", style="green", max_width=12)
    table.add_column("Source", style="cyan", max_width=12)
    table.add_column("Trust", style="yellow", max_width=8)
    table.add_column("Created", style="dim")
    table.add_column("Content", style="white", max_width=40)

    for entry in filtered_entries:
        content = (
            entry.content[:60] + "..." if len(entry.content) > 60 else entry.content
        )
        content = content.replace("\n", " ")

        # Resolve subject person names
        subject_names = []
        for person_id in entry.subject_person_ids or []:
            person = people_by_id.get(person_id)
            if person:
                subject_names.append(person.name)
            else:
                subject_names.append(person_id[:8])

        # About: subjects if present, otherwise show username for self-referential facts
        if subject_names:
            about = ", ".join(subject_names)
        elif entry.source_user_id:
            # Self-referential fact - show username to match source column
            about = f"@{entry.source_user_id}"
        elif entry.source_user_name:
            about = entry.source_user_name
        else:
            about = "[dim]-[/dim]"

        # Source: who provided this information
        if entry.source_user_id:
            source_display = f"@{entry.source_user_id}"
        elif entry.source:
            source_display = entry.source
        else:
            source_display = "[dim]-[/dim]"

        # Trust: fact (speaking about self) vs hearsay (speaking about others)
        is_self_reference = _is_source_self_reference(
            entry.source_user_id,
            entry.owner_user_id,
            entry.subject_person_ids,
            people,
            people_by_id,
        )
        if not subject_names or is_self_reference:
            trust = "[green]fact[/green]"
        else:
            trust = "[yellow]hearsay[/yellow]"

        created = entry.created_at.strftime("%Y-%m-%d") if entry.created_at else "-"

        table.add_row(
            entry.id[:8],
            entry.memory_type.value,
            about,
            source_display,
            trust,
            created,
            content,
        )

    console.print(table)
    dim(f"\nShowing {len(filtered_entries)} entries")


async def _memory_add(
    session, content: str, source: str | None, expires_days: int | None
) -> None:
    """Add a memory entry."""
    import os

    from ash.memory.types import MemoryType

    store = _get_memory_store()

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


async def _memory_remove(
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

    store = _get_memory_store()

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


async def _memory_clear(session, force: bool) -> None:
    """Clear all memory entries."""
    from sqlalchemy import text

    if not force:
        warning("This will delete ALL memory entries.")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            dim("Cancelled")
            return

    store = _get_memory_store()
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


async def _memory_gc() -> None:
    """Garbage collect expired and superseded memories."""
    store = _get_memory_store()
    result = await store.gc()

    if result.removed_count == 0:
        dim("No memories to clean up")
    else:
        success(f"Archived and removed {result.removed_count} memories")
        if result.archived_ids:
            dim(f"Archived IDs: {', '.join(id[:8] for id in result.archived_ids[:5])}")
            if len(result.archived_ids) > 5:
                dim(f"  ... and {len(result.archived_ids) - 5} more")


async def _memory_rebuild_index(session) -> None:
    """Rebuild vector index from JSONL source of truth."""
    from ash.config.paths import get_memories_jsonl_path
    from ash.memory.index import rebuild_vector_index_from_jsonl

    memories_path = get_memories_jsonl_path()

    if not memories_path.exists():
        warning("No memories.jsonl file found")
        return

    dim(f"Rebuilding index from {memories_path}")
    count = await rebuild_vector_index_from_jsonl(memories_path)

    success(f"Rebuilt index with {count} embeddings")


async def _memory_show(memory_id: str) -> None:
    """Show full details of a memory entry."""
    from rich.panel import Panel
    from rich.table import Table

    from ash.config.paths import get_people_jsonl_path
    from ash.memory.jsonl import PersonJSONL

    store = _get_memory_store()

    # Find the memory by prefix
    memory = await store.get_memory_by_prefix(memory_id)
    if not memory:
        error(f"No memory found with ID: {memory_id}")
        raise typer.Exit(1)

    # Load people for name lookup
    people_jsonl = PersonJSONL(get_people_jsonl_path())
    people = await people_jsonl.load_all()
    people_by_id = {p.id: p for p in people}

    # Build details table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("ID", memory.id)
    table.add_row("Type", memory.memory_type.value)

    # Source user attribution
    if memory.source_user_id and memory.source_user_name:
        table.add_row(
            "Source User", f"@{memory.source_user_id} ({memory.source_user_name})"
        )
    elif memory.source_user_id:
        table.add_row("Source User", f"@{memory.source_user_id}")
    else:
        table.add_row("Source User", "-")

    table.add_row("Source Method", memory.source or "-")

    # Scope
    if memory.owner_user_id:
        table.add_row("Scope", f"Personal ({memory.owner_user_id})")
    elif memory.chat_id:
        table.add_row("Scope", f"Group ({memory.chat_id})")
    else:
        table.add_row("Scope", "Global")

    # Subjects (who this memory is about)
    subject_names = []
    if memory.subject_person_ids:
        for person_id in memory.subject_person_ids:
            person = people_by_id.get(person_id)
            if person:
                subject_names.append(f"{person.name} ({person_id[:8]})")
            else:
                subject_names.append(person_id)
        table.add_row("About", ", ".join(subject_names))
    elif memory.source_user_name:
        # No subjects means speaking about self
        table.add_row("About", f"{memory.source_user_name} (self)")
    elif memory.source_user_id:
        table.add_row("About", f"@{memory.source_user_id} (self)")
    else:
        table.add_row("About", "-")

    # Trust level - check if source user matches any subject person (self-reference)
    all_people = await store.get_people_for_user(memory.owner_user_id or "")
    people_by_id_show = {p.id: p for p in all_people}
    is_self_reference = _is_source_self_reference(
        memory.source_user_id,
        memory.owner_user_id,
        memory.subject_person_ids,
        all_people,
        people_by_id_show,
    )
    if not subject_names or is_self_reference:
        table.add_row("Trust", "fact (source speaking about themselves)")
    else:
        table.add_row("Trust", "hearsay (source speaking about others)")

    # Timestamps
    if memory.created_at:
        table.add_row("Created", memory.created_at.isoformat())
    if memory.observed_at and memory.observed_at != memory.created_at:
        table.add_row("Observed", memory.observed_at.isoformat())
    if memory.expires_at:
        table.add_row("Expires", memory.expires_at.isoformat())
    if memory.superseded_at:
        table.add_row("Superseded", memory.superseded_at.isoformat())
    if memory.superseded_by_id:
        table.add_row("Superseded By", memory.superseded_by_id)

    # Source attribution
    if memory.source_session_id:
        table.add_row("Session", memory.source_session_id)
    if memory.source_message_id:
        table.add_row("Message", memory.source_message_id)
    if memory.extraction_confidence is not None:
        table.add_row("Confidence", f"{memory.extraction_confidence:.2f}")

    # Embedding info
    if memory.embedding:
        table.add_row("Embedding", f"{len(memory.embedding)} chars (base64)")

    console.print(Panel(table, title=f"Memory {memory.id[:8]}"))
    console.print()
    console.print(Panel(memory.content, title="Content"))


async def _memory_history(memory_id: str) -> None:
    """Show supersession chain for a memory."""
    from rich.table import Table

    store = _get_memory_store()

    # First, find the memory
    memory = await store.get_memory_by_prefix(memory_id)
    if not memory:
        error(f"No memory found with ID: {memory_id}")
        raise typer.Exit(1)

    # Get the supersession chain
    chain = await store.get_supersession_chain(memory.id)

    if not chain:
        dim("No supersession history for this memory")
        console.print(f"\nCurrent: {memory.content[:100]}")
        return

    table = Table(title=f"Supersession Chain for {memory.id[:8]}")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Created", style="dim")
    table.add_column("Archived", style="yellow")
    table.add_column("Reason", style="cyan")
    table.add_column("Content", style="white", max_width=50)

    for entry in chain:
        content = (
            entry.content[:50] + "..." if len(entry.content) > 50 else entry.content
        )
        content = content.replace("\n", " ")

        archived_at = (
            entry.archived_at.strftime("%Y-%m-%d") if entry.archived_at else "-"
        )

        table.add_row(
            entry.id[:8],
            entry.created_at.strftime("%Y-%m-%d") if entry.created_at else "-",
            archived_at,
            entry.archive_reason or "-",
            content,
        )

    # Add current memory at the end
    current_content = (
        memory.content[:50] + "..." if len(memory.content) > 50 else memory.content
    )
    current_content = current_content.replace("\n", " ")
    table.add_row(
        memory.id[:8],
        memory.created_at.strftime("%Y-%m-%d") if memory.created_at else "-",
        "[green]current[/green]",
        "-",
        f"[green]{current_content}[/green]",
    )

    console.print(table)
    dim(f"\n{len(chain)} superseded entries")


# Classification prompt for doctor command
CLASSIFY_PROMPT = """Classify each memory into the correct type based on its content.

## Memory Types:
Long-lived (no automatic expiration):
- preference: likes, dislikes, habits (e.g., "prefers dark mode", "hates olives")
- identity: facts about the user themselves (e.g., "works as engineer", "lives in SF", "is 52 years old")
- relationship: people in user's life (e.g., "Sarah is my wife", "boss is John")
- knowledge: factual info about external things (e.g., "project uses Python", "company uses Slack")

Ephemeral (decay over time):
- context: current situation/state (e.g., "working on project X", "feeling stressed")
- event: past occurrences with dates (e.g., "had dinner with Sarah Tuesday")
- task: things to do (e.g., "needs to call dentist")
- observation: fleeting observations (e.g., "seemed tired today")

## Memories to classify:
{memories}

Return a JSON object mapping memory ID to new type. Only include memories that need reclassification.
Example: {{"abc123": "preference", "def456": "identity"}}

If all memories are correctly classified, return: {{}}"""


async def _memory_doctor(config, force: bool) -> None:
    """Reclassify memory types using LLM."""
    import json

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from ash.config.models import AshConfig
    from ash.llm import create_llm_provider
    from ash.llm.types import Message, Role
    from ash.memory.types import MemoryType

    config: AshConfig

    store = _get_memory_store()
    memories = await store.get_all_memories()

    if not memories:
        warning("No memories to process")
        return

    # Filter to only KNOWLEDGE type (most likely to be misclassified)
    knowledge_memories = [m for m in memories if m.memory_type == MemoryType.KNOWLEDGE]

    if not knowledge_memories:
        success("All memories already have specific types (no KNOWLEDGE to reclassify)")
        return

    console.print(
        f"Found {len(knowledge_memories)} memories with KNOWLEDGE type to review"
    )

    if not force:
        if not typer.confirm("Proceed with reclassification?"):
            dim("Cancelled")
            return

    # Create LLM provider
    llm = create_llm_provider(config.default_llm)

    # Process in batches of 20
    batch_size = 20
    total_reclassified = 0
    changes: list[tuple[str, str, str]] = []  # (id, old_type, new_type)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Classifying memories...", total=len(knowledge_memories)
        )

        for i in range(0, len(knowledge_memories), batch_size):
            batch = knowledge_memories[i : i + batch_size]

            # Format memories for the prompt
            memory_text = "\n".join(f"- {m.id[:8]}: {m.content[:200]}" for m in batch)

            prompt = CLASSIFY_PROMPT.format(memories=memory_text)

            try:
                response = await llm.complete(
                    messages=[Message(role=Role.USER, content=prompt)],
                    max_tokens=1024,
                    temperature=0.1,
                )

                # Parse response
                text = response.message.get_text().strip()
                # Handle markdown code blocks
                if text.startswith("```"):
                    lines = text.split("\n")
                    json_lines = []
                    in_block = False
                    for line in lines:
                        if line.startswith("```"):
                            in_block = not in_block
                            continue
                        if in_block:
                            json_lines.append(line)
                    text = "\n".join(json_lines)

                classifications = json.loads(text)

                # Apply reclassifications
                for memory in batch:
                    short_id = memory.id[:8]
                    if short_id in classifications:
                        new_type_str = classifications[short_id]
                        try:
                            new_type = MemoryType(new_type_str)
                            if new_type != memory.memory_type:
                                old_type = memory.memory_type.value
                                memory.memory_type = new_type
                                await store.update_memory(memory)
                                changes.append((short_id, old_type, new_type.value))
                                total_reclassified += 1
                        except ValueError:
                            pass  # Invalid type, skip

            except Exception as e:
                dim(f"Batch failed: {e}")

            progress.advance(task, len(batch))

    # Report results
    if changes:
        from rich.table import Table

        table = Table(title="Reclassified Memories")
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("Old Type", style="yellow")
        table.add_column("New Type", style="green")

        for short_id, old_type, new_type in changes:
            table.add_row(short_id, old_type, new_type)

        console.print(table)
        success(f"Reclassified {total_reclassified} memories")
    else:
        dim("No memories needed reclassification")
