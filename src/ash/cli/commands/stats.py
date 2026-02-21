"""Operational stats command for Ash home directory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import typer

from ash.cli.console import console, create_table, dim
from ash.config.paths import (
    get_ash_home,
    get_auth_path,
    get_config_path,
    get_schedule_file,
)

DIR_PURPOSES: dict[str, str] = {
    "graph": "Memory graph store (memories, people, vectors)",
    "sessions": "Session transcripts (context/history JSONL)",
    "chats": "Chat-level history and chat state",
    "logs": "Structured runtime logs",
    "run": "Runtime files (PID, sockets)",
    "workspace": "User workspace and skill files",
    "skills": "User-defined skills and skill state",
    "skills.installed": "Installed external skill sources",
    "cache": "Runtime caches (including uv cache)",
}

FILE_PURPOSES: dict[str, str] = {
    "config.toml": "Main Ash configuration",
    "schedule.jsonl": "Scheduled task queue",
    "auth.json": "OAuth/provider credentials",
}


@dataclass
class DirStats:
    """Recursive stats for a directory."""

    file_count: int
    dir_count: int
    total_bytes: int
    latest_mtime: datetime | None


def register(app: typer.Typer) -> None:
    """Register stats/info commands."""

    @app.command("stats")
    def stats() -> None:
        """Show operational stats for Ash home."""
        _render_stats()

    @app.command("info")
    def info() -> None:
        """Alias for `ash stats`."""
        _render_stats()


def _render_stats() -> None:
    home = get_ash_home()

    console.print(f"[bold]Ash Home[/bold]: [cyan]{home}[/cyan]")
    if not home.exists():
        dim("Home directory does not exist yet.")
        return

    dir_table = create_table(
        "Directory Stats",
        [
            ("Directory", "cyan"),
            ("Purpose", "white"),
            ("Status", "white"),
            ("Files", "magenta"),
            ("Dirs", "magenta"),
            ("Size", "green"),
            ("Last Modified", "white"),
        ],
    )

    for path in sorted((p for p in home.iterdir() if p.is_dir()), key=lambda p: p.name):
        stats = _collect_dir_stats(path)
        purpose = DIR_PURPOSES.get(path.name, "Untracked directory")
        status = "ok"
        if stats.file_count == 0 and stats.dir_count == 0:
            status = "empty"
        dir_table.add_row(
            path.name,
            purpose,
            status,
            str(stats.file_count),
            str(stats.dir_count),
            _format_bytes(stats.total_bytes),
            _format_dt(stats.latest_mtime),
        )

    console.print(dir_table)

    file_table = create_table(
        "Core Files",
        [
            ("File", "cyan"),
            ("Purpose", "white"),
            ("Exists", "white"),
            ("Size", "green"),
            ("Last Modified", "white"),
        ],
    )

    for path, purpose in (
        (get_config_path(), FILE_PURPOSES["config.toml"]),
        (get_schedule_file(), FILE_PURPOSES["schedule.jsonl"]),
        (get_auth_path(), FILE_PURPOSES["auth.json"]),
    ):
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC) if exists else None
        file_table.add_row(
            path.name,
            purpose,
            "yes" if exists else "no",
            _format_bytes(size) if exists else "-",
            _format_dt(mtime),
        )
    console.print(file_table)


def _collect_dir_stats(path: Path) -> DirStats:
    file_count = 0
    dir_count = 0
    total_bytes = 0
    latest_mtime: datetime | None = None

    try:
        for child in path.rglob("*"):
            try:
                stat = child.stat()
            except OSError:
                continue
            if child.is_file():
                file_count += 1
                total_bytes += stat.st_size
            elif child.is_dir():
                dir_count += 1

            mtime = datetime.fromtimestamp(stat.st_mtime, UTC)
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
    except OSError:
        pass

    return DirStats(
        file_count=file_count,
        dir_count=dir_count,
        total_bytes=total_bytes,
        latest_mtime=latest_mtime,
    )


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if size < 1024 or candidate == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.1f} {unit}"


def _format_dt(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
