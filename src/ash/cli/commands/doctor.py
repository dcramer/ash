"""System health checks for Ash runtime and data directories."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import typer

from ash.cli.console import console, create_table, dim, error, success, warning
from ash.config.paths import (
    get_ash_home,
    get_graph_dir,
    get_logs_path,
    get_run_path,
    get_schedule_file,
    get_sessions_path,
)

SESSION_VERSION = "2"


@dataclass
class Finding:
    level: Literal["ok", "warning", "error"]
    check: str
    detail: str
    repair: str | None = None


def register(app: typer.Typer) -> None:
    """Register the top-level doctor command."""

    @app.command()
    def doctor() -> None:
        """Run read-only operational health checks."""
        findings = run_doctor_checks()
        _render_doctor_report(findings)

        error_count = sum(1 for f in findings if f.level == "error")
        if error_count:
            raise typer.Exit(1)


def run_doctor_checks() -> list[Finding]:
    """Run all read-only doctor checks."""
    findings: list[Finding] = []

    findings.extend(_check_home())
    findings.extend(_check_runtime_artifacts())
    findings.extend(_check_schedule_file())
    findings.extend(_check_sessions_jsonl())
    findings.extend(_check_graph_state())
    findings.extend(_check_logs_dir())

    return findings


def _render_doctor_report(findings: list[Finding]) -> None:
    home = get_ash_home()
    ok_count = sum(1 for f in findings if f.level == "ok")
    warning_count = sum(1 for f in findings if f.level == "warning")
    error_count = sum(1 for f in findings if f.level == "error")

    console.print(f"[bold]Ash Doctor[/bold] [cyan]{home}[/cyan]")
    table = create_table(
        "Doctor Findings",
        [
            ("Level", "white"),
            ("Check", "cyan"),
            ("Detail", "white"),
            ("Repair", "green"),
        ],
    )

    level_label = {
        "ok": "[green]OK[/green]",
        "warning": "[yellow]WARN[/yellow]",
        "error": "[red]ERROR[/red]",
    }
    for finding in findings:
        table.add_row(
            level_label[finding.level],
            finding.check,
            finding.detail,
            finding.repair or "-",
        )
    console.print(table)

    summary = f"checks={len(findings)} ok={ok_count} warnings={warning_count} errors={error_count}"
    console.print(f"[bold]Summary:[/bold] {summary}")
    if error_count:
        error("Doctor found blocking issues")
    elif warning_count:
        warning("Doctor found non-blocking issues")
    else:
        success("Doctor checks passed")

    console.print("\n[bold]Doctor Commands[/bold]")
    console.print(
        "- [cyan]ash doctor[/cyan]: system/runtime/data integrity checks (read-only)"
    )
    console.print(
        "- [cyan]ash memory doctor[/cyan]: memory repair flows (preview by default)"
    )
    console.print(
        "- [cyan]ash people doctor[/cyan]: people repair flows (preview by default)"
    )
    dim("Read-only checks. No changes were made.")


def _check_home() -> list[Finding]:
    home = get_ash_home()
    if not home.exists():
        return [
            Finding(
                level="warning",
                check="home.exists",
                detail=f"ASH_HOME does not exist: {home}",
                repair="Run any ash command (or `ash init`) to bootstrap",
            )
        ]

    findings: list[Finding] = [
        Finding(
            level="ok", check="home.exists", detail=f"home directory exists: {home}"
        )
    ]

    expected_dirs = (
        "graph",
        "sessions",
        "chats",
        "logs",
        "run",
        "workspace",
    )
    for dirname in expected_dirs:
        path = home / dirname
        if path.exists():
            findings.append(
                Finding(
                    level="ok",
                    check=f"dir.{dirname}",
                    detail=f"{dirname} exists",
                )
            )
        else:
            findings.append(
                Finding(
                    level="warning",
                    check=f"dir.{dirname}",
                    detail=f"{dirname} missing",
                    repair=f"Create {path}",
                )
            )
    return findings


def _check_runtime_artifacts() -> list[Finding]:
    findings: list[Finding] = []
    run_dir = get_run_path()
    pid_path = run_dir / "ash.pid"
    sock_path = run_dir / "rpc.sock"

    if not pid_path.exists():
        findings.append(Finding(level="ok", check="run.pid", detail="no pid file"))
    else:
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            findings.append(
                Finding(
                    level="ok",
                    check="run.pid",
                    detail=f"pid file references running process: {pid}",
                )
            )
        except ProcessLookupError:
            findings.append(
                Finding(
                    level="warning",
                    check="run.pid",
                    detail=f"stale pid file: {pid_path}",
                    repair=f"Remove {pid_path}",
                )
            )
        except (PermissionError, ValueError, OSError):
            findings.append(
                Finding(
                    level="warning",
                    check="run.pid",
                    detail=f"unreadable or invalid pid file: {pid_path}",
                    repair=f"Remove {pid_path} and restart service",
                )
            )

    if not sock_path.exists():
        findings.append(
            Finding(level="ok", check="run.rpc_socket", detail="rpc socket not present")
        )
    else:
        try:
            mode = sock_path.stat().st_mode
            if stat.S_ISSOCK(mode):
                findings.append(
                    Finding(
                        level="ok", check="run.rpc_socket", detail="rpc socket exists"
                    )
                )
            else:
                findings.append(
                    Finding(
                        level="warning",
                        check="run.rpc_socket",
                        detail=f"path exists but is not a socket: {sock_path}",
                        repair=f"Remove {sock_path} and restart service",
                    )
                )
        except OSError:
            findings.append(
                Finding(
                    level="warning",
                    check="run.rpc_socket",
                    detail=f"failed to stat socket path: {sock_path}",
                )
            )

    return findings


def _check_schedule_file() -> list[Finding]:
    schedule_file = get_schedule_file()
    if not schedule_file.exists():
        return [
            Finding(
                level="ok", check="schedule.file", detail="schedule file not present"
            )
        ]

    invalid_lines = _count_invalid_jsonl_lines(schedule_file)
    if invalid_lines:
        return [
            Finding(
                level="warning",
                check="schedule.jsonl",
                detail=f"{invalid_lines} invalid JSONL lines in {schedule_file}",
                repair=f"Inspect and repair {schedule_file}",
            )
        ]
    return [
        Finding(
            level="ok", check="schedule.jsonl", detail="schedule file is valid JSONL"
        )
    ]


def _check_sessions_jsonl() -> list[Finding]:
    sessions_path = get_sessions_path()
    if not sessions_path.exists():
        return [
            Finding(
                level="ok",
                check="sessions.dir",
                detail="sessions directory not present",
            )
        ]

    context_files = list(sessions_path.glob("*/context.jsonl"))
    history_files = list(sessions_path.glob("*/history.jsonl"))

    invalid_context_lines = 0
    invalid_history_lines = 0
    legacy_session_headers = 0

    for path in context_files:
        invalid_count, legacy_count = _scan_session_context_file(path)
        invalid_context_lines += invalid_count
        legacy_session_headers += legacy_count

    for path in history_files:
        invalid_history_lines += _count_invalid_jsonl_lines(path)

    findings: list[Finding] = [
        Finding(
            level="ok",
            check="sessions.files",
            detail=f"scanned {len(context_files)} context and {len(history_files)} history files",
        )
    ]
    if invalid_context_lines:
        findings.append(
            Finding(
                level="warning",
                check="sessions.context_jsonl",
                detail=f"invalid lines in context.jsonl files: {invalid_context_lines}",
                repair="Archive or fix corrupted session files",
            )
        )
    else:
        findings.append(
            Finding(
                level="ok",
                check="sessions.context_jsonl",
                detail="all context.jsonl lines parse as JSON",
            )
        )

    if invalid_history_lines:
        findings.append(
            Finding(
                level="warning",
                check="sessions.history_jsonl",
                detail=f"invalid lines in history.jsonl files: {invalid_history_lines}",
                repair="Archive or fix corrupted history files",
            )
        )
    else:
        findings.append(
            Finding(
                level="ok",
                check="sessions.history_jsonl",
                detail="all history.jsonl lines parse as JSON",
            )
        )

    if legacy_session_headers:
        findings.append(
            Finding(
                level="warning",
                check="sessions.version",
                detail=f"legacy session headers found (version != {SESSION_VERSION}): {legacy_session_headers}",
                repair="Delete/recreate legacy sessions or migrate them",
            )
        )
    else:
        findings.append(
            Finding(
                level="ok",
                check="sessions.version",
                detail=f"no legacy session headers found (version={SESSION_VERSION})",
            )
        )

    return findings


def _check_graph_state() -> list[Finding]:
    graph_dir = get_graph_dir()
    state_path = graph_dir / "state.json"
    if not state_path.exists():
        return [
            Finding(
                level="ok", check="graph.state", detail="graph state file not present"
            )
        ]

    try:
        state = json.loads(state_path.read_text())
    except Exception:
        return [
            Finding(
                level="warning",
                check="graph.state",
                detail=f"failed to parse {state_path}",
                repair=f"Repair or remove {state_path}",
            )
        ]

    findings: list[Finding] = [
        Finding(
            level="ok", check="graph.state", detail="graph state parsed successfully"
        )
    ]

    vector_missing = _as_non_negative_int(state.get("vector_missing_count"))
    provenance_missing = _as_non_negative_int(state.get("provenance_missing_count"))

    if vector_missing > 0:
        findings.append(
            Finding(
                level="warning",
                check="graph.vector_consistency",
                detail=f"vector_missing_count={vector_missing}",
                repair="Run `ash memory doctor embed-missing --force`",
            )
        )
    else:
        findings.append(
            Finding(
                level="ok",
                check="graph.vector_consistency",
                detail="vector_missing_count=0",
            )
        )

    if provenance_missing > 0:
        findings.append(
            Finding(
                level="warning",
                check="graph.provenance",
                detail=f"provenance_missing_count={provenance_missing}",
                repair="Run `ash memory doctor prune-missing-provenance --force`",
            )
        )
    else:
        findings.append(
            Finding(
                level="ok",
                check="graph.provenance",
                detail="provenance_missing_count=0",
            )
        )

    return findings


def _check_logs_dir() -> list[Finding]:
    logs_path = get_logs_path()
    if not logs_path.exists():
        return [
            Finding(level="ok", check="logs.dir", detail="logs directory not present")
        ]

    log_files = list(logs_path.glob("*.jsonl"))
    if not log_files:
        return [Finding(level="ok", check="logs.files", detail="no log files found")]

    invalid_lines = 0
    for path in log_files:
        invalid_lines += _count_invalid_jsonl_lines(path)

    if invalid_lines:
        return [
            Finding(
                level="warning",
                check="logs.jsonl",
                detail=f"invalid lines across log files: {invalid_lines}",
                repair="Rotate or repair corrupted log files",
            )
        ]
    return [
        Finding(
            level="ok",
            check="logs.jsonl",
            detail=f"log files parse as JSONL ({len(log_files)} files)",
        )
    ]


def _scan_session_context_file(path: Path) -> tuple[int, int]:
    invalid = 0
    legacy_headers = 0
    for line in _iter_lines(path):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(payload, dict):
            invalid += 1
            continue
        if payload.get("type") == "session":
            version = payload.get("version")
            if version != SESSION_VERSION:
                legacy_headers += 1
    return invalid, legacy_headers


def _count_invalid_jsonl_lines(path: Path) -> int:
    invalid = 0
    for line in _iter_lines(path):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(payload, dict):
            invalid += 1
    return invalid


def _iter_lines(path: Path) -> list[str]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines if line.strip()]


def _as_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value >= 0 else 0
    return 0
