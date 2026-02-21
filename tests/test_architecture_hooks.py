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


def test_register_config_methods_wiring_is_constrained() -> None:
    files = _python_files_under("src/ash")

    call_sites = _find_call_sites(r"\bregister_config_methods\(", files)
    assert call_sites == {
        Path("src/ash/integrations/builtin.py"),
        Path("src/ash/rpc/methods/config.py"),
    }


def test_register_log_methods_wiring_is_constrained() -> None:
    files = _python_files_under("src/ash")

    call_sites = _find_call_sites(r"\bregister_log_methods\(", files)
    assert call_sites == {
        Path("src/ash/integrations/builtin.py"),
        Path("src/ash/rpc/methods/logs.py"),
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
        assert "create_agent(" in text or "bootstrap_runtime(" in text, (
            f"Expected shared agent composition path in {path.relative_to(ROOT)}"
        )


def test_entrypoints_compose_integrations_via_runtime() -> None:
    entrypoint_files = [
        ROOT / "src/ash/cli/commands/serve.py",
        ROOT / "src/ash/cli/commands/chat.py",
        ROOT / "evals/harness.py",
    ]
    for path in entrypoint_files:
        text = path.read_text(encoding="utf-8")
        assert "compose_integrations(" in text or "active_integrations(" in text, (
            f"Expected shared integration composition path in {path.relative_to(ROOT)}"
        )


def test_entrypoints_use_default_integration_builder() -> None:
    entrypoint_files = [
        ROOT / "src/ash/cli/commands/serve.py",
        ROOT / "src/ash/cli/commands/chat.py",
        ROOT / "evals/harness.py",
    ]
    for path in entrypoint_files:
        text = path.read_text(encoding="utf-8")
        assert "create_default_integrations(" in text, (
            f"Expected shared default integration builder in {path.relative_to(ROOT)}"
        )


def test_entrypoints_use_shared_rpc_lifecycle_helper() -> None:
    entrypoint_files = [
        ROOT / "src/ash/cli/commands/serve.py",
        ROOT / "src/ash/cli/commands/chat.py",
        ROOT / "evals/harness.py",
    ]
    for path in entrypoint_files:
        text = path.read_text(encoding="utf-8")
        assert "active_rpc_server(" in text, (
            f"Expected shared RPC lifecycle helper in {path.relative_to(ROOT)}"
        )


def test_chat_and_serve_use_shared_runtime_bootstrap_helper() -> None:
    entrypoint_files = [
        ROOT / "src/ash/cli/commands/serve.py",
        ROOT / "src/ash/cli/commands/chat.py",
    ]
    for path in entrypoint_files:
        text = path.read_text(encoding="utf-8")
        assert "bootstrap_runtime(" in text, (
            f"Expected shared runtime bootstrap helper in {path.relative_to(ROOT)}"
        )


def test_serve_uses_provider_runtime_adapter() -> None:
    path = ROOT / "src/ash/cli/commands/serve.py"
    text = path.read_text(encoding="utf-8")
    assert "build_provider_runtime(" in text, (
        f"Expected provider runtime adapter in {path.relative_to(ROOT)}"
    )
