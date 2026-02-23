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


def _find_import_sites(pattern: str, paths: list[Path]) -> set[Path]:
    regex = re.compile(pattern)
    matches: set[Path] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if regex.search(line):
                matches.add(path.relative_to(ROOT))
                break
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
        Path("src/ash/integrations/memory.py"),
        Path("src/ash/rpc/methods/memory.py"),
    }


def test_register_schedule_methods_wiring_is_constrained() -> None:
    files = _python_files_under("src/ash")
    files.append(ROOT / "evals/harness.py")

    call_sites = _find_call_sites(r"\bregister_schedule_methods\(", files)
    assert call_sites == {
        Path("src/ash/integrations/scheduling.py"),
        Path("src/ash/rpc/methods/schedule.py"),
    }


def test_create_store_wiring_is_constrained() -> None:
    files = _python_files_under("src/ash")

    call_sites = _find_call_sites(r"\bcreate_store\(", files)
    assert call_sites == {
        Path("src/ash/memory/runtime.py"),
        Path("src/ash/store/store.py"),
    }


def test_register_config_methods_wiring_is_constrained() -> None:
    files = _python_files_under("src/ash")

    call_sites = _find_call_sites(r"\bregister_config_methods\(", files)
    assert call_sites == {
        Path("src/ash/integrations/runtime_rpc.py"),
        Path("src/ash/rpc/methods/config.py"),
    }


def test_register_log_methods_wiring_is_constrained() -> None:
    files = _python_files_under("src/ash")

    call_sites = _find_call_sites(r"\bregister_log_methods\(", files)
    assert call_sites == {
        Path("src/ash/integrations/runtime_rpc.py"),
        Path("src/ash/rpc/methods/logs.py"),
    }


def test_rpc_method_registrar_imports_are_constrained() -> None:
    files = _python_files_under("src/ash")
    files.append(ROOT / "evals/harness.py")

    memory_imports = _find_import_sites(
        r"from ash\.rpc\.methods\.memory import register_memory_methods",
        files,
    )
    assert memory_imports == {
        Path("src/ash/integrations/memory.py"),
        Path("src/ash/rpc/methods/__init__.py"),
    }

    schedule_imports = _find_import_sites(
        r"from ash\.rpc\.methods\.schedule import register_schedule_methods",
        files,
    )
    assert schedule_imports == {
        Path("src/ash/integrations/scheduling.py"),
        Path("src/ash/rpc/methods/__init__.py"),
    }

    config_imports = _find_import_sites(
        r"from ash\.rpc\.methods\.config import register_config_methods",
        files,
    )
    assert config_imports == {
        Path("src/ash/integrations/runtime_rpc.py"),
        Path("src/ash/rpc/methods/__init__.py"),
    }

    log_imports = _find_import_sites(
        r"from ash\.rpc\.methods\.logs import register_log_methods",
        files,
    )
    assert log_imports == {
        Path("src/ash/integrations/runtime_rpc.py"),
        Path("src/ash/rpc/methods/__init__.py"),
    }


def test_integration_module_import_direction_is_constrained() -> None:
    files = _python_files_under("src/ash")

    memory_imports = _find_import_sites(
        r"(from ash\.integrations\.memory import|import ash\.integrations\.memory)",
        files,
    )
    assert memory_imports == {
        Path("src/ash/integrations/__init__.py"),
        Path("src/ash/integrations/defaults.py"),
    }

    scheduling_imports = _find_import_sites(
        r"(from ash\.integrations\.scheduling import|import ash\.integrations\.scheduling)",
        files,
    )
    assert scheduling_imports == {
        Path("src/ash/integrations/__init__.py"),
        Path("src/ash/integrations/defaults.py"),
    }

    runtime_rpc_imports = _find_import_sites(
        r"(from ash\.integrations\.runtime_rpc import|import ash\.integrations\.runtime_rpc)",
        files,
    )
    assert runtime_rpc_imports == {
        Path("src/ash/integrations/__init__.py"),
        Path("src/ash/integrations/defaults.py"),
    }


def test_scheduling_lifecycle_wiring_is_owned_by_scheduling_integration() -> None:
    files = _python_files_under("src/ash")

    schedule_runtime_imports = _find_import_sites(
        r"from ash\.scheduling import .*?(ScheduleWatcher|ScheduledTaskHandler)",
        files,
    )
    assert schedule_runtime_imports == {
        Path("src/ash/integrations/scheduling.py"),
    }


def test_scheduling_runtime_instantiation_is_constrained() -> None:
    files = _python_files_under("src/ash")

    watcher_call_sites = _find_call_sites(r"\bScheduleWatcher\(", files)
    assert watcher_call_sites == {
        Path("src/ash/integrations/scheduling.py"),
        Path("src/ash/scheduling/watcher.py"),
    }

    handler_call_sites = _find_call_sites(r"\bScheduledTaskHandler\(", files)
    assert handler_call_sites == {
        Path("src/ash/integrations/scheduling.py"),
    }


def test_memory_postprocess_service_wiring_is_integration_owned() -> None:
    files = _python_files_under("src/ash")

    call_sites = _find_call_sites(r"\bMemoryPostprocessService\(", files)
    assert call_sites == {
        Path("src/ash/integrations/memory.py"),
    }


def test_browser_runtime_wiring_is_integration_owned() -> None:
    files = _python_files_under("src/ash")

    manager_call_sites = _find_call_sites(r"\bcreate_browser_manager\(", files)
    assert manager_call_sites == {
        Path("src/ash/browser/manager.py"),
        Path("src/ash/cli/commands/browser.py"),
        Path("src/ash/integrations/browser.py"),
    }

    warmup_call_sites = _find_call_sites(r"\bwarmup_default_provider\(", files)
    assert warmup_call_sites == {
        Path("src/ash/browser/manager.py"),
        Path("src/ash/integrations/browser.py"),
    }


def test_browser_prompt_guidance_is_integration_owned() -> None:
    prompt_text = (ROOT / "src/ash/core/prompt.py").read_text(encoding="utf-8")
    assert "Web/Search/Browser Routing" not in prompt_text
    assert "screenshot/image from the browser" not in prompt_text
    assert "`web_search` -> `web_fetch` -> `browser`" not in prompt_text

    integration_text = (ROOT / "src/ash/integrations/browser.py").read_text(
        encoding="utf-8"
    )
    assert "tool_routing_rules" in integration_text
    assert "core_principles_rules" in integration_text


def test_ensure_self_person_wiring_avoids_core_agent() -> None:
    files = _python_files_under("src/ash")

    call_sites = _find_call_sites(r"\bensure_self_person\(", files)
    assert call_sites == {
        Path("src/ash/memory/processing.py"),
        Path("src/ash/memory/postprocess.py"),
        Path("src/ash/providers/telegram/passive.py"),
        Path("src/ash/rpc/methods/memory.py"),
    }


def test_harness_boundaries_reference_integration_hooks_spec() -> None:
    expected_comment = "specs/subsystems.md (Integration Hooks)"
    boundary_files = [
        ROOT / "src/ash/core/agent.py",
        ROOT / "src/ash/cli/commands/serve.py",
        ROOT / "src/ash/cli/commands/chat.py",
        ROOT / "src/ash/integrations/runtime.py",
        ROOT / "src/ash/integrations/composer.py",
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


def test_entrypoints_import_integrations_from_package_root() -> None:
    entrypoint_files = [
        ROOT / "src/ash/cli/commands/serve.py",
        ROOT / "src/ash/cli/commands/chat.py",
        ROOT / "evals/harness.py",
    ]
    disallowed = (
        "from ash.integrations.defaults import",
        "from ash.integrations.memory import",
        "from ash.integrations.scheduling import",
        "from ash.integrations.runtime_rpc import",
        "from ash.integrations.composer import",
        "from ash.integrations.rpc import",
    )
    for path in entrypoint_files:
        text = path.read_text(encoding="utf-8")
        assert "from ash.integrations import " in text, (
            f"Expected root integration imports in {path.relative_to(ROOT)}"
        )
        for pattern in disallowed:
            assert pattern not in text, (
                f"Disallowed direct integration module import in {path.relative_to(ROOT)}: {pattern}"
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


def test_serve_uses_shared_server_runner() -> None:
    path = ROOT / "src/ash/cli/commands/serve.py"
    text = path.read_text(encoding="utf-8")
    assert "ServerRunner(" in text, (
        f"Expected shared server runner in {path.relative_to(ROOT)}"
    )
