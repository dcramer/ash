"""Forget-person command for memory entries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import typer

from ash.cli.console import dim, error, success, warning

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


async def memory_forget(
    store: Store,
    person_id: str,
    delete_person: bool,
    force: bool,
) -> None:
    """Archive all memories about a person."""
    people = await store.list_people()
    people_by_id = {p.id: p for p in people}

    # Find person by exact or prefix match
    person = people_by_id.get(person_id)
    if not person:
        matches = [p for p in people if p.id.startswith(person_id)]
        if len(matches) == 1:
            person = matches[0]
        elif len(matches) > 1:
            error(
                f"Ambiguous person ID prefix: {person_id} matches {len(matches)} people"
            )
            raise typer.Exit(1)
        else:
            error(f"No person found with ID: {person_id}")
            raise typer.Exit(1)

    if not force:
        warning(
            f"This will archive all memories about: {person.name} ({person.id[:8]})"
        )
        if delete_person:
            warning("The person record will also be deleted.")
        if not typer.confirm("Are you sure?"):
            dim("Cancelled")
            return

    archived_count = await store.forget_person(
        person_id=person.id,
        delete_person_record=delete_person,
    )

    success(f"Archived {archived_count} memories about {person.name}")
    if delete_person:
        dim("Person record deleted")
