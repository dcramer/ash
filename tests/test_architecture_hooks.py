from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _find_call_sites(pattern: str, paths: list[Path]) -> set[Path]:
    regex = re.compile(pattern)
    matches: set[Path] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if regex.search(text):
            matches.add(path.relative_to(ROOT))
    return matches


def _python_files_under(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend((ROOT / root).rglob("*.py"))
    return files


def test_register_memory_methods_wiring_is_constrained() -> None:
    files = _python_files_under("src/ash")
    files.append(ROOT / "evals/harness.py")

    call_sites = _find_call_sites(r"\bregister_memory_methods\(", files)
    assert call_sites == {
        Path("src/ash/integrations/builtin.py"),
        Path("src/ash/rpc/methods/memory.py"),
    }


def test_register_schedule_methods_wiring_is_constrained() -> None:
    files = _python_files_under("src/ash")
    files.append(ROOT / "evals/harness.py")

    call_sites = _find_call_sites(r"\bregister_schedule_methods\(", files)
    assert call_sites == {
        Path("src/ash/integrations/builtin.py"),
        Path("src/ash/rpc/methods/schedule.py"),
    }


def test_harness_boundaries_reference_integration_hooks_spec() -> None:
    expected_comment = "specs/subsystems.md (Integration Hooks)"
    boundary_files = [
        ROOT / "src/ash/core/agent.py",
        ROOT / "src/ash/cli/commands/serve.py",
        ROOT / "src/ash/cli/commands/chat.py",
        ROOT / "evals/harness.py",
    ]
    for path in boundary_files:
        text = path.read_text(encoding="utf-8")
        assert expected_comment in text, (
            f"Missing integration hooks spec reference in {path.relative_to(ROOT)}"
        )


def test_entrypoints_use_shared_create_agent_composition_path() -> None:
    entrypoint_files = [
        ROOT / "src/ash/cli/commands/serve.py",
        ROOT / "src/ash/cli/commands/chat.py",
        ROOT / "evals/harness.py",
    ]
    for path in entrypoint_files:
        text = path.read_text(encoding="utf-8")
        assert "create_agent(" in text, (
            f"Expected shared create_agent composition path in {path.relative_to(ROOT)}"
        )


def test_entrypoints_compose_integrations_via_runtime() -> None:
    entrypoint_files = [
        ROOT / "src/ash/cli/commands/serve.py",
        ROOT / "src/ash/cli/commands/chat.py",
        ROOT / "evals/harness.py",
    ]
    for path in entrypoint_files:
        text = path.read_text(encoding="utf-8")
        assert "IntegrationRuntime(" in text, (
            f"Expected IntegrationRuntime composition in {path.relative_to(ROOT)}"
        )
