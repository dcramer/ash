"""System health checks for Ash runtime and data directories."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from ash.cli.console import console, create_table, dim, error, success, warning
from ash.cli.doctor_utils import DoctorFinding, DoctorResult
from ash.config import load_config
from ash.config.paths import (
    get_ash_home,
    get_config_path,
    get_graph_dir,
    get_logs_path,
    get_run_path,
    get_schedule_file,
    get_sessions_path,
)
from ash.images.service import _resolve_image_model

SESSION_VERSION = "2"

if TYPE_CHECKING:
    from ash.config import AshConfig


def register(app: typer.Typer) -> None:
    """Register the top-level doctor command."""

    @app.command()
    def doctor() -> None:
        """Run read-only operational health checks."""
        result = run_doctor_checks()
        _render_doctor_report(result)
        if result.has_errors:
            raise typer.Exit(1)


def run_doctor_checks() -> DoctorResult:
    """Run all read-only doctor checks."""
    findings: list[DoctorFinding] = []

    findings.extend(_check_home())
    findings.extend(_check_runtime_artifacts())
    findings.extend(_check_config())
    findings.extend(_check_schedule_file())
    findings.extend(_check_sessions_jsonl())
    findings.extend(_check_graph_state())
    findings.extend(_check_logs_dir())

    return DoctorResult(findings=findings)


def _render_doctor_report(result: DoctorResult) -> None:
    home = get_ash_home()

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
    for finding in result.findings:
        table.add_row(
            level_label[finding.level],
            finding.check,
            finding.detail,
            finding.repair or "-",
        )
    console.print(table)

    console.print(f"[bold]Summary:[/bold] {result.summary_text()}")
    if result.has_errors:
        error("Doctor found blocking issues")
    elif result.warning_count:
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


def _check_home() -> list[DoctorFinding]:
    home = get_ash_home()
    if not home.exists():
        return [
            DoctorFinding(
                level="warning",
                check="home.exists",
                detail=f"ASH_HOME does not exist: {home}",
                repair="Run any ash command (or `ash init`) to bootstrap",
            )
        ]

    findings: list[DoctorFinding] = [
        DoctorFinding(
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
                DoctorFinding(
                    level="ok",
                    check=f"dir.{dirname}",
                    detail=f"{dirname} exists",
                )
            )
        else:
            findings.append(
                DoctorFinding(
                    level="warning",
                    check=f"dir.{dirname}",
                    detail=f"{dirname} missing",
                    repair=f"Create {path}",
                )
            )
    return findings


def _check_runtime_artifacts() -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    run_dir = get_run_path()
    pid_path = run_dir / "ash.pid"
    sock_path = run_dir / "rpc.sock"

    if not pid_path.exists():
        findings.append(
            DoctorFinding(level="ok", check="run.pid", detail="no pid file")
        )
    else:
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            findings.append(
                DoctorFinding(
                    level="ok",
                    check="run.pid",
                    detail=f"pid file references running process: {pid}",
                )
            )
        except ProcessLookupError:
            findings.append(
                DoctorFinding(
                    level="warning",
                    check="run.pid",
                    detail=f"stale pid file: {pid_path}",
                    repair=f"Remove {pid_path}",
                )
            )
        except (PermissionError, ValueError, OSError):
            findings.append(
                DoctorFinding(
                    level="warning",
                    check="run.pid",
                    detail=f"unreadable or invalid pid file: {pid_path}",
                    repair=f"Remove {pid_path} and restart service",
                )
            )

    if not sock_path.exists():
        findings.append(
            DoctorFinding(
                level="ok", check="run.rpc_socket", detail="rpc socket not present"
            )
        )
    else:
        try:
            mode = sock_path.stat().st_mode
            if stat.S_ISSOCK(mode):
                findings.append(
                    DoctorFinding(
                        level="ok", check="run.rpc_socket", detail="rpc socket exists"
                    )
                )
            else:
                findings.append(
                    DoctorFinding(
                        level="warning",
                        check="run.rpc_socket",
                        detail=f"path exists but is not a socket: {sock_path}",
                        repair=f"Remove {sock_path} and restart service",
                    )
                )
        except OSError:
            findings.append(
                DoctorFinding(
                    level="warning",
                    check="run.rpc_socket",
                    detail=f"failed to stat socket path: {sock_path}",
                )
            )

    return findings


def _check_config() -> list[DoctorFinding]:
    config_path = get_config_path()
    if not config_path.exists():
        return [
            DoctorFinding(
                level="ok",
                check="config.file",
                detail=f"config file not present: {config_path}",
                repair="Run `ash init` to create a starter config",
            )
        ]

    try:
        config = load_config(config_path)
    except Exception:
        return [
            DoctorFinding(
                level="warning",
                check="config.parse",
                detail=f"failed to parse/validate config: {config_path}",
                repair=f"Run `ash config validate --path {config_path}`",
            )
        ]

    findings: list[DoctorFinding] = [
        DoctorFinding(
            level="ok",
            check="config.parse",
            detail=f"config parsed successfully: {config_path}",
        )
    ]
    findings.extend(_check_browser_config(config))
    findings.extend(_check_image_config(config))
    return findings


def _check_browser_config(config: AshConfig) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    browser_cfg = config.browser

    if not browser_cfg.enabled:
        findings.append(
            DoctorFinding(
                level="ok",
                check="config.browser.enabled",
                detail="browser integration disabled",
            )
        )
        return findings

    if browser_cfg.provider not in {"sandbox", "kernel"}:
        findings.append(
            DoctorFinding(
                level="warning",
                check="config.browser.provider",
                detail=f"unsupported browser provider: {browser_cfg.provider}",
                repair='Set `[browser].provider` to "sandbox" or "kernel"',
            )
        )
        return findings

    if browser_cfg.timeout_seconds <= 0:
        findings.append(
            DoctorFinding(
                level="warning",
                check="config.browser.timeout_seconds",
                detail=f"invalid value: {browser_cfg.timeout_seconds}",
                repair="Set `[browser].timeout_seconds` to > 0",
            )
        )
    if browser_cfg.max_session_minutes <= 0:
        findings.append(
            DoctorFinding(
                level="warning",
                check="config.browser.max_session_minutes",
                detail=f"invalid value: {browser_cfg.max_session_minutes}",
                repair="Set `[browser].max_session_minutes` to > 0",
            )
        )
    if browser_cfg.artifacts_retention_days < 0:
        findings.append(
            DoctorFinding(
                level="warning",
                check="config.browser.artifacts_retention_days",
                detail=f"invalid value: {browser_cfg.artifacts_retention_days}",
                repair="Set `[browser].artifacts_retention_days` to >= 0",
            )
        )

    if browser_cfg.provider == "kernel":
        if not browser_cfg.kernel.api_key and not os.environ.get("KERNEL_API_KEY"):
            findings.append(
                DoctorFinding(
                    level="warning",
                    check="config.browser.kernel.api_key",
                    detail="kernel provider selected but KERNEL_API_KEY is missing",
                    repair="Set `[browser.kernel].api_key` or `KERNEL_API_KEY`",
                )
            )
        else:
            findings.append(
                DoctorFinding(
                    level="ok",
                    check="config.browser.kernel.api_key",
                    detail="Kernel API key configured",
                )
            )

    if browser_cfg.provider == "sandbox":
        if importlib.util.find_spec("playwright") is None:
            findings.append(
                DoctorFinding(
                    level="warning",
                    check="config.browser.sandbox.playwright",
                    detail="sandbox browser provider requires playwright package",
                    repair="Run `uv sync --all-groups` and `uv run playwright install chromium`",
                )
            )
        else:
            findings.append(
                DoctorFinding(
                    level="ok",
                    check="config.browser.sandbox.playwright",
                    detail="playwright package is available",
                )
            )

    if not findings:
        findings.append(
            DoctorFinding(
                level="ok",
                check="config.browser",
                detail=f"browser config valid (provider={browser_cfg.provider})",
            )
        )
    return findings


def _check_image_config(config: AshConfig) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    image_cfg = config.image

    if not image_cfg.enabled:
        findings.append(
            DoctorFinding(
                level="ok",
                check="config.image.enabled",
                detail="image understanding disabled",
            )
        )
        return findings

    if image_cfg.provider != "openai":
        findings.append(
            DoctorFinding(
                level="warning",
                check="config.image.provider",
                detail=f"unsupported image provider: {image_cfg.provider}",
                repair='Set `[image].provider = "openai"`',
            )
        )
        return findings

    api_key = config.resolve_provider_api_key("openai")
    if not api_key or not api_key.get_secret_value().strip():
        findings.append(
            DoctorFinding(
                level="warning",
                check="config.image.openai_api_key",
                detail="image enabled but OpenAI API key is missing",
                repair="Set `[openai].api_key` or `OPENAI_API_KEY`",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                level="ok",
                check="config.image.openai_api_key",
                detail="OpenAI API key configured for image understanding",
            )
        )

    image_model = (image_cfg.model or "").strip()
    if image_model:
        if "/" in image_model:
            provider, model = image_model.split("/", 1)
            if provider.lower() != "openai" or not model.strip():
                findings.append(
                    DoctorFinding(
                        level="warning",
                        check="config.image.model",
                        detail=f"invalid image model reference: {image_model}",
                        repair="Use `openai/<model>` or an alias that resolves to OpenAI",
                    )
                )
        elif image_model in config.models:
            alias_model = config.get_model(image_model)
            if alias_model.provider != "openai":
                findings.append(
                    DoctorFinding(
                        level="warning",
                        check="config.image.model",
                        detail=f"image model alias is not OpenAI-backed: {image_model}",
                        repair="Point `[image].model` to an OpenAI model/alias",
                    )
                )

    if image_cfg.max_images_per_message <= 0:
        findings.append(
            DoctorFinding(
                level="warning",
                check="config.image.max_images_per_message",
                detail=f"invalid value: {image_cfg.max_images_per_message}",
                repair="Set `[image].max_images_per_message` to >= 1",
            )
        )
    if image_cfg.max_image_bytes <= 0:
        findings.append(
            DoctorFinding(
                level="warning",
                check="config.image.max_image_bytes",
                detail=f"invalid value: {image_cfg.max_image_bytes}",
                repair="Set `[image].max_image_bytes` to a positive integer",
            )
        )
    if image_cfg.request_timeout_seconds <= 0:
        findings.append(
            DoctorFinding(
                level="warning",
                check="config.image.request_timeout_seconds",
                detail=f"invalid value: {image_cfg.request_timeout_seconds}",
                repair="Set `[image].request_timeout_seconds` to > 0",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                level="ok",
                check="config.image.request_timeout_seconds",
                detail=f"request timeout is valid: {image_cfg.request_timeout_seconds}s",
            )
        )

    try:
        resolved_model = _resolve_image_model(config)
    except ValueError as e:
        findings.append(
            DoctorFinding(
                level="warning",
                check="config.image.model_resolution",
                detail=f"failed to resolve image model: {e}",
                repair="Fix `[image].model` to a valid OpenAI model or alias",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                level="ok",
                check="config.image.model_resolution",
                detail=f"resolved image model: {resolved_model}",
            )
        )

    return findings


def _check_schedule_file() -> list[DoctorFinding]:
    schedule_file = get_schedule_file()
    if not schedule_file.exists():
        return [
            DoctorFinding(
                level="ok", check="schedule.file", detail="schedule file not present"
            )
        ]

    invalid_lines = _count_invalid_jsonl_lines(schedule_file)
    if invalid_lines:
        return [
            DoctorFinding(
                level="warning",
                check="schedule.jsonl",
                detail=f"{invalid_lines} invalid JSONL lines in {schedule_file}",
                repair=f"Inspect and repair {schedule_file}",
            )
        ]
    return [
        DoctorFinding(
            level="ok", check="schedule.jsonl", detail="schedule file is valid JSONL"
        )
    ]


def _check_sessions_jsonl() -> list[DoctorFinding]:
    sessions_path = get_sessions_path()
    if not sessions_path.exists():
        return [
            DoctorFinding(
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

    findings: list[DoctorFinding] = [
        DoctorFinding(
            level="ok",
            check="sessions.files",
            detail=f"scanned {len(context_files)} context and {len(history_files)} history files",
        )
    ]
    if invalid_context_lines:
        findings.append(
            DoctorFinding(
                level="warning",
                check="sessions.context_jsonl",
                detail=f"invalid lines in context.jsonl files: {invalid_context_lines}",
                repair="Archive or fix corrupted session files",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                level="ok",
                check="sessions.context_jsonl",
                detail="all context.jsonl lines parse as JSON",
            )
        )

    if invalid_history_lines:
        findings.append(
            DoctorFinding(
                level="warning",
                check="sessions.history_jsonl",
                detail=f"invalid lines in history.jsonl files: {invalid_history_lines}",
                repair="Archive or fix corrupted history files",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                level="ok",
                check="sessions.history_jsonl",
                detail="all history.jsonl lines parse as JSON",
            )
        )

    if legacy_session_headers:
        findings.append(
            DoctorFinding(
                level="warning",
                check="sessions.version",
                detail=f"legacy session headers found (version != {SESSION_VERSION}): {legacy_session_headers}",
                repair="Delete/recreate legacy sessions or migrate them",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                level="ok",
                check="sessions.version",
                detail=f"no legacy session headers found (version={SESSION_VERSION})",
            )
        )

    return findings


def _check_graph_state() -> list[DoctorFinding]:
    graph_dir = get_graph_dir()
    state_path = graph_dir / "state.json"
    if not state_path.exists():
        return [
            DoctorFinding(
                level="ok", check="graph.state", detail="graph state file not present"
            )
        ]

    try:
        state = json.loads(state_path.read_text())
    except Exception:
        return [
            DoctorFinding(
                level="warning",
                check="graph.state",
                detail=f"failed to parse {state_path}",
                repair=f"Repair or remove {state_path}",
            )
        ]

    findings: list[DoctorFinding] = [
        DoctorFinding(
            level="ok", check="graph.state", detail="graph state parsed successfully"
        )
    ]

    vector_missing = _as_non_negative_int(state.get("vector_missing_count"))
    provenance_missing = _as_non_negative_int(state.get("provenance_missing_count"))

    if vector_missing > 0:
        findings.append(
            DoctorFinding(
                level="warning",
                check="graph.vector_consistency",
                detail=f"vector_missing_count={vector_missing}",
                repair="Run `ash memory doctor embed-missing --force`",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                level="ok",
                check="graph.vector_consistency",
                detail="vector_missing_count=0",
            )
        )

    if provenance_missing > 0:
        findings.append(
            DoctorFinding(
                level="warning",
                check="graph.provenance",
                detail=f"provenance_missing_count={provenance_missing}",
                repair="Run `ash memory doctor prune-missing-provenance --force`",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                level="ok",
                check="graph.provenance",
                detail="provenance_missing_count=0",
            )
        )

    return findings


def _check_logs_dir() -> list[DoctorFinding]:
    logs_path = get_logs_path()
    if not logs_path.exists():
        return [
            DoctorFinding(
                level="ok", check="logs.dir", detail="logs directory not present"
            )
        ]

    log_files = list(logs_path.glob("*.jsonl"))
    if not log_files:
        return [
            DoctorFinding(level="ok", check="logs.files", detail="no log files found")
        ]

    invalid_lines = 0
    for path in log_files:
        invalid_lines += _count_invalid_jsonl_lines(path)

    if invalid_lines:
        return [
            DoctorFinding(
                level="warning",
                check="logs.jsonl",
                detail=f"invalid lines across log files: {invalid_lines}",
                repair="Rotate or repair corrupted log files",
            )
        ]
    return [
        DoctorFinding(
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
