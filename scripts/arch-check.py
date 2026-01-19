#!/usr/bin/env python
"""Architecture verification for role-arch.

Checks for:
- Circular imports
- Cross-subsystem imports (violations of subsystem boundaries)
- Public API usage (should import from __init__.py, not internal modules)
- Dependency direction violations

Run with: uv run python scripts/arch-check.py
"""

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Define the expected subsystems and their allowed dependencies
SUBSYSTEMS = {
    "memory",
    "sessions",
    "skills",
    "agents",
    "chats",
    "cli",
    "config",
    "core",
    "db",
    "events",
    "llm",
    "observability",
    "providers",
    "rpc",
    "sandbox",
    "server",
    "service",
    "tools",
}

# Foundational layers that all subsystems may depend on
FOUNDATIONAL = {"db", "llm", "config", "logging"}

# Subsystems should not depend on these (they're orchestrators)
ORCHESTRATORS = {"core", "cli", "server", "service"}


@dataclass
class ImportInfo:
    """Information about an import."""

    source_file: str
    source_subsystem: str
    target_module: str
    target_subsystem: str
    is_internal: bool  # Importing from internal module instead of __init__
    line_number: int


@dataclass
class ArchReport:
    """Architecture analysis report."""

    circular_imports: list[list[str]] = field(default_factory=list)
    cross_subsystem_violations: list[ImportInfo] = field(default_factory=list)
    internal_import_violations: list[ImportInfo] = field(default_factory=list)
    orchestrator_violations: list[ImportInfo] = field(default_factory=list)


def get_subsystem(filepath: str | Path) -> str | None:
    """Extract subsystem name from file path."""
    path = Path(filepath)
    parts = path.parts
    # Find the index of 'ash' in the path
    try:
        ash_idx = parts.index("ash")
        if ash_idx + 1 < len(parts):
            potential = parts[ash_idx + 1]
            if potential in SUBSYSTEMS:
                return potential
    except ValueError:
        pass
    return None


def get_module_subsystem(module: str) -> str | None:
    """Extract subsystem from module path like 'ash.memory.store'."""
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "ash":
        potential = parts[1]
        if potential in SUBSYSTEMS:
            return potential
    return None


def is_internal_import(module: str) -> bool:
    """Check if this imports from an internal module rather than __init__."""
    parts = module.split(".")
    # ash.subsystem is fine (imports from __init__)
    # ash.subsystem.internal is a violation
    return len(parts) > 2 and parts[0] == "ash" and parts[1] in SUBSYSTEMS


def collect_imports(filepath: Path) -> list[tuple[str, int]]:
    """Collect all import statements from a Python file.

    Returns list of (module_path, line_number) tuples.
    Excludes imports inside TYPE_CHECKING blocks (type-only imports).
    """
    imports: list[tuple[str, int]] = []
    try:
        with filepath.open() as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return imports

    # Find TYPE_CHECKING block line ranges
    type_checking_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check if condition is TYPE_CHECKING
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                start = node.body[0].lineno if node.body else node.lineno
                end = node.body[-1].end_lineno if node.body else node.lineno
                type_checking_ranges.append((start, end or start))

    def in_type_checking(lineno: int) -> bool:
        return any(start <= lineno <= end for start, end in type_checking_ranges)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if not in_type_checking(node.lineno):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module and not in_type_checking(node.lineno):
                imports.append((node.module, node.lineno))

    return imports


def find_circular_imports(project_root: Path) -> list[list[str]]:
    """Detect circular import chains.

    Uses depth-first search to find cycles in the import graph.
    """
    src_dir = project_root / "src" / "ash"

    # Build import graph
    graph: dict[str, set[str]] = defaultdict(set)

    for py_file in src_dir.rglob("*.py"):
        rel_path = py_file.relative_to(project_root / "src")
        # Convert path to module: src/ash/memory/store.py -> ash.memory.store
        module = str(rel_path).replace("/", ".").replace(".py", "")
        if module.endswith(".__init__"):
            module = module[:-9]

        for imp, _ in collect_imports(py_file):
            if imp.startswith("ash."):
                graph[module].add(imp)

    # Find cycles using DFS
    cycles: list[list[str]] = []
    visited: set[str] = set()
    path: list[str] = []
    path_set: set[str] = set()

    def dfs(node: str) -> None:
        if node in path_set:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            # Skip self-loops (__init__.py re-export patterns)
            if len(cycle) == 2 and cycle[0] == cycle[1]:
                return
            # Normalize cycle (start from smallest element)
            min_idx = cycle.index(min(cycle[:-1]))
            normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
            if normalized not in cycles:
                cycles.append(normalized)
            return

        if node in visited:
            return

        visited.add(node)
        path.append(node)
        path_set.add(node)

        for neighbor in graph.get(node, set()):
            dfs(neighbor)

        path.pop()
        path_set.remove(node)

    for module in graph:
        dfs(module)

    return cycles


def analyze_imports(project_root: Path) -> ArchReport:
    """Analyze all imports for architecture violations."""
    src_dir = project_root / "src" / "ash"
    report = ArchReport()

    for py_file in src_dir.rglob("*.py"):
        rel_path = str(py_file.relative_to(project_root))
        source_subsystem = get_subsystem(py_file)

        if not source_subsystem:
            continue

        for imp, line in collect_imports(py_file):
            if not imp.startswith("ash."):
                continue

            target_subsystem = get_module_subsystem(imp)
            if not target_subsystem:
                continue

            info = ImportInfo(
                source_file=rel_path,
                source_subsystem=source_subsystem,
                target_module=imp,
                target_subsystem=target_subsystem,
                is_internal=is_internal_import(imp),
                line_number=line,
            )

            # Check for internal imports (should use public API)
            if info.is_internal and target_subsystem != source_subsystem:
                report.internal_import_violations.append(info)

            # Check for cross-subsystem imports (should go through agent/events)
            if (
                target_subsystem != source_subsystem
                and target_subsystem not in FOUNDATIONAL
                and source_subsystem not in ORCHESTRATORS
            ):
                report.cross_subsystem_violations.append(info)

            # Check for imports from orchestrator layers
            if (
                target_subsystem in ORCHESTRATORS
                and source_subsystem not in ORCHESTRATORS
            ):
                report.orchestrator_violations.append(info)

    # Check for circular imports
    report.circular_imports = find_circular_imports(project_root)

    return report


def print_report(report: ArchReport, verbose: bool = False) -> None:
    """Print the architecture report."""
    print("=" * 60)
    print("Architecture Analysis")
    print("=" * 60)

    # Circular imports
    if report.circular_imports:
        print(f"\n[FAIL] Circular Imports: {len(report.circular_imports)} found")
        for cycle in report.circular_imports[:5]:
            print(f"  {' -> '.join(cycle)}")
        if len(report.circular_imports) > 5:
            print(f"  ... and {len(report.circular_imports) - 5} more")
    else:
        print("\n[OK] Circular Imports: None detected")

    # Internal imports
    if report.internal_import_violations:
        print(
            f"\n[WARN] Internal Imports: {len(report.internal_import_violations)} violations"
        )
        print("  (Should import from subsystem root, not internal modules)")
        for v in report.internal_import_violations[:5]:
            print(f"  {v.source_file}:{v.line_number}")
            print(f"    imports {v.target_module}")
        if len(report.internal_import_violations) > 5:
            print(f"  ... and {len(report.internal_import_violations) - 5} more")
    else:
        print("\n[OK] Internal Imports: All using public API")

    # Cross-subsystem
    if report.cross_subsystem_violations:
        print(
            f"\n[WARN] Cross-Subsystem: {len(report.cross_subsystem_violations)} direct imports"
        )
        print("  (Subsystems should communicate via agent/events, not direct imports)")
        for v in report.cross_subsystem_violations[:5]:
            print(f"  {v.source_subsystem} -> {v.target_subsystem}")
            print(f"    {v.source_file}:{v.line_number}")
        if len(report.cross_subsystem_violations) > 5:
            print(f"  ... and {len(report.cross_subsystem_violations) - 5} more")
    else:
        print("\n[OK] Cross-Subsystem: No violations")

    # Orchestrator imports
    if report.orchestrator_violations:
        print(
            f"\n[FAIL] Orchestrator Imports: {len(report.orchestrator_violations)} violations"
        )
        print("  (Subsystems should not import from core/cli/server)")
        for v in report.orchestrator_violations[:5]:
            print(f"  {v.source_file}:{v.line_number}")
            print(f"    imports {v.target_module}")
    else:
        print("\n[OK] Orchestrator Imports: None detected")

    print("\n" + "=" * 60)

    # Summary
    total_issues = len(report.circular_imports) + len(report.orchestrator_violations)
    if total_issues == 0:
        print("Overall: CLEAN ARCHITECTURE")
    else:
        print(f"Overall: {total_issues} critical issues")
    print("=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Architecture verification")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of text"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    report = analyze_imports(project_root)

    if args.json:
        output = {
            "circular_imports": report.circular_imports,
            "internal_import_violations": [
                {
                    "source": v.source_file,
                    "line": v.line_number,
                    "imports": v.target_module,
                }
                for v in report.internal_import_violations
            ],
            "cross_subsystem_violations": [
                {
                    "from": v.source_subsystem,
                    "to": v.target_subsystem,
                    "file": v.source_file,
                    "line": v.line_number,
                }
                for v in report.cross_subsystem_violations
            ],
            "orchestrator_violations": [
                {
                    "source": v.source_file,
                    "line": v.line_number,
                    "imports": v.target_module,
                }
                for v in report.orchestrator_violations
            ],
            "clean": len(report.circular_imports) == 0
            and len(report.orchestrator_violations) == 0,
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Exit with error only for critical issues
    if report.circular_imports or report.orchestrator_violations:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
