"""Operational stats command for Ash home directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import typer

from ash.cli.console import console, create_table, dim
from ash.config.paths import (
    get_ash_home,
    get_auth_path,
    get_config_path,
    get_logs_path,
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
    "browser": "Browser sessions, profiles, and artifacts",
    "cache": "Runtime caches (including uv cache)",
}

FILE_PURPOSES: dict[str, str] = {
    "config.toml": "Main Ash configuration",
    "auth.json": "OAuth/provider credentials",
}


@dataclass
class DirStats:
    """Recursive stats for a directory."""

    file_count: int
    dir_count: int
    total_bytes: int
    latest_mtime: datetime | None


@dataclass
class MemoryQualityStats:
    """Aggregated memory extraction/verification counters from logs."""

    extraction_runs: int = 0
    extraction_candidates: int = 0
    extraction_accepted: int = 0
    extraction_dropped_low_confidence: int = 0
    extraction_dropped_secret: int = 0
    verification_runs: int = 0
    verification_candidates: int = 0
    verification_accepted: int = 0
    verification_rewritten: int = 0
    verification_dropped_ambiguous: int = 0
    verification_dropped_meta_system: int = 0
    verification_dropped_stale_status: int = 0
    verification_dropped_low_utility: int = 0
    last_seen: datetime | None = None


@dataclass
class VisionStats:
    """Aggregated image preprocessing counters from logs."""

    runs_started: int = 0
    runs_succeeded: int = 0
    runs_failed: int = 0
    runs_skipped: int = 0
    images_started: int = 0
    images_succeeded: int = 0
    success_duration_ms_total: int = 0
    skipped_no_usable_images: int = 0
    skipped_missing_openai_api_key: int = 0
    skipped_unsupported_provider: int = 0
    skipped_invalid_image_model_config: int = 0
    skipped_other: int = 0
    last_seen: datetime | None = None


@dataclass
class BrowserStats:
    """Aggregated browser action/session counters from logs."""

    actions_started: int = 0
    actions_succeeded: int = 0
    actions_failed: int = 0
    sessions_started: int = 0
    sessions_archived: int = 0
    sessions_closed: int = 0
    provider_sandbox: int = 0
    provider_kernel: int = 0
    last_seen: datetime | None = None


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

    quality = _collect_memory_quality_stats()
    quality_table = create_table(
        "Memory Quality (from logs)",
        [
            ("Metric", "cyan"),
            ("Value", "green"),
        ],
    )
    quality_table.add_row("Extraction runs", str(quality.extraction_runs))
    quality_table.add_row("Extraction candidates", str(quality.extraction_candidates))
    quality_table.add_row("Extraction accepted", str(quality.extraction_accepted))
    quality_table.add_row(
        "Extraction dropped (low confidence)",
        str(quality.extraction_dropped_low_confidence),
    )
    quality_table.add_row(
        "Extraction dropped (secret)",
        str(quality.extraction_dropped_secret),
    )
    quality_table.add_row("Verification runs", str(quality.verification_runs))
    quality_table.add_row(
        "Verification candidates", str(quality.verification_candidates)
    )
    quality_table.add_row("Verification accepted", str(quality.verification_accepted))
    quality_table.add_row("Verification rewritten", str(quality.verification_rewritten))
    quality_table.add_row(
        "Verification dropped (ambiguous)",
        str(quality.verification_dropped_ambiguous),
    )
    quality_table.add_row(
        "Verification dropped (meta/system)",
        str(quality.verification_dropped_meta_system),
    )
    quality_table.add_row(
        "Verification dropped (stale status)",
        str(quality.verification_dropped_stale_status),
    )
    quality_table.add_row(
        "Verification dropped (low utility)",
        str(quality.verification_dropped_low_utility),
    )
    quality_table.add_row("Last quality event", _format_dt(quality.last_seen))
    console.print(quality_table)

    vision = _collect_vision_stats()
    vision_table = create_table(
        "Vision (from logs)",
        [
            ("Metric", "cyan"),
            ("Value", "green"),
        ],
    )
    vision_table.add_row("Runs started", str(vision.runs_started))
    vision_table.add_row("Runs succeeded", str(vision.runs_succeeded))
    vision_table.add_row("Runs failed", str(vision.runs_failed))
    vision_table.add_row("Runs skipped", str(vision.runs_skipped))
    vision_table.add_row("Images started", str(vision.images_started))
    vision_table.add_row("Images succeeded", str(vision.images_succeeded))
    avg_success_ms = (
        f"{vision.success_duration_ms_total / vision.runs_succeeded:.1f} ms"
        if vision.runs_succeeded
        else "-"
    )
    vision_table.add_row("Avg success latency", avg_success_ms)
    vision_table.add_row(
        "Skipped (no usable images)", str(vision.skipped_no_usable_images)
    )
    vision_table.add_row(
        "Skipped (missing OpenAI key)", str(vision.skipped_missing_openai_api_key)
    )
    vision_table.add_row(
        "Skipped (unsupported provider)", str(vision.skipped_unsupported_provider)
    )
    vision_table.add_row(
        "Skipped (invalid image model config)",
        str(vision.skipped_invalid_image_model_config),
    )
    vision_table.add_row("Skipped (other)", str(vision.skipped_other))
    vision_table.add_row("Last vision event", _format_dt(vision.last_seen))
    console.print(vision_table)

    browser = _collect_browser_stats()
    browser_table = create_table(
        "Browser (from logs)",
        [
            ("Metric", "cyan"),
            ("Value", "green"),
        ],
    )
    browser_table.add_row("Actions started", str(browser.actions_started))
    browser_table.add_row("Actions succeeded", str(browser.actions_succeeded))
    browser_table.add_row("Actions failed", str(browser.actions_failed))
    browser_table.add_row("Sessions started", str(browser.sessions_started))
    browser_table.add_row("Sessions archived", str(browser.sessions_archived))
    browser_table.add_row("Sessions closed", str(browser.sessions_closed))
    browser_table.add_row("Provider usage (sandbox)", str(browser.provider_sandbox))
    browser_table.add_row("Provider usage (kernel)", str(browser.provider_kernel))
    browser_table.add_row("Last browser event", _format_dt(browser.last_seen))
    console.print(browser_table)


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


def _collect_memory_quality_stats() -> MemoryQualityStats:
    stats = MemoryQualityStats()
    logs_dir = get_logs_path()
    if not logs_dir.exists():
        return stats

    log_files = sorted(logs_dir.glob("*.jsonl"), reverse=True)[:7]
    for path in log_files:
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue

                    message = record.get("message")
                    if message == "memory_extraction_filter_stats":
                        stats.extraction_runs += 1
                        stats.extraction_candidates += int(
                            record.get("fact.total_candidates", 0) or 0
                        )
                        stats.extraction_accepted += int(
                            record.get("fact.accepted_count", 0) or 0
                        )
                        stats.extraction_dropped_low_confidence += int(
                            record.get("fact.dropped_low_confidence", 0) or 0
                        )
                        stats.extraction_dropped_secret += int(
                            record.get("fact.dropped_secret", 0) or 0
                        )
                    elif message == "memory_verification_stats":
                        stats.verification_runs += 1
                        stats.verification_candidates += int(
                            record.get("fact.total_candidates", 0) or 0
                        )
                        stats.verification_accepted += int(
                            record.get("fact.accepted_count", 0) or 0
                        )
                        stats.verification_rewritten += int(
                            record.get("fact.rewritten_count", 0) or 0
                        )
                        stats.verification_dropped_ambiguous += int(
                            record.get("fact.dropped_ambiguous", 0) or 0
                        )
                        stats.verification_dropped_meta_system += int(
                            record.get("fact.dropped_meta_system", 0) or 0
                        )
                        stats.verification_dropped_stale_status += int(
                            record.get("fact.dropped_stale_status", 0) or 0
                        )
                        stats.verification_dropped_low_utility += int(
                            record.get("fact.dropped_low_utility", 0) or 0
                        )

                    ts = record.get("ts")
                    if isinstance(ts, str):
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except ValueError:
                            continue
                        if stats.last_seen is None or dt > stats.last_seen:
                            stats.last_seen = dt
        except OSError:
            continue

    return stats


def _collect_vision_stats() -> VisionStats:
    stats = VisionStats()
    logs_dir = get_logs_path()
    if not logs_dir.exists():
        return stats

    log_files = sorted(logs_dir.glob("*.jsonl"), reverse=True)[:7]
    for path in log_files:
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue

                    message = record.get("message")
                    if message == "image_preprocess_started":
                        stats.runs_started += 1
                        stats.images_started += int(record.get("image.count", 0) or 0)
                    elif message == "image_preprocess_succeeded":
                        stats.runs_succeeded += 1
                        stats.images_succeeded += int(record.get("image.count", 0) or 0)
                        stats.success_duration_ms_total += int(
                            record.get("duration_ms", 0) or 0
                        )
                    elif message == "image_preprocess_failed":
                        stats.runs_failed += 1
                    elif message == "image_preprocess_skipped":
                        stats.runs_skipped += 1
                        skip_reason = str(record.get("skip_reason", "") or "").strip()
                        if skip_reason == "no_usable_images":
                            stats.skipped_no_usable_images += 1
                        elif skip_reason == "missing_openai_api_key":
                            stats.skipped_missing_openai_api_key += 1
                        elif skip_reason == "unsupported_provider":
                            stats.skipped_unsupported_provider += 1
                        elif skip_reason == "invalid_image_model_config":
                            stats.skipped_invalid_image_model_config += 1
                        else:
                            stats.skipped_other += 1
                    else:
                        continue

                    ts = record.get("ts")
                    if isinstance(ts, str):
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except ValueError:
                            continue
                        if stats.last_seen is None or dt > stats.last_seen:
                            stats.last_seen = dt
        except OSError:
            continue

    return stats


def _collect_browser_stats() -> BrowserStats:
    stats = BrowserStats()
    logs_dir = get_logs_path()
    if not logs_dir.exists():
        return stats

    log_files = sorted(logs_dir.glob("*.jsonl"), reverse=True)[:7]
    for path in log_files:
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue

                    message = record.get("message")
                    if message == "browser_action_started":
                        stats.actions_started += 1
                    elif message == "browser_action_succeeded":
                        stats.actions_succeeded += 1
                    elif message == "browser_action_failed":
                        stats.actions_failed += 1
                    elif message == "browser_session_started":
                        stats.sessions_started += 1
                    elif message == "browser_session_closed":
                        stats.sessions_closed += 1
                    elif message == "browser_session_archived":
                        stats.sessions_archived += 1
                    else:
                        continue

                    provider = str(record.get("browser.provider", "") or "").lower()
                    if provider == "sandbox":
                        stats.provider_sandbox += 1
                    elif provider == "kernel":
                        stats.provider_kernel += 1

                    ts = record.get("ts")
                    if isinstance(ts, str):
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except ValueError:
                            continue
                        if stats.last_seen is None or dt > stats.last_seen:
                            stats.last_seen = dt
        except OSError:
            continue
    return stats


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
