#!/usr/bin/env python
"""Eval coverage analysis for role-eval.

Analyzes eval coverage to identify:
- Behaviors without evals
- Eval pass rates
- Judge consistency across runs
- Missing eval cases

Run with: uv run python scripts/eval-coverage.py
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EvalCase:
    """Summary of an eval case."""

    id: str
    suite: str
    description: str
    behaviors: list[str] = field(default_factory=list)


@dataclass
class AgentCapability:
    """A capability/behavior of an agent that should be tested."""

    agent: str
    behavior: str
    description: str


@dataclass
class EvalCoverageReport:
    """Eval coverage analysis report."""

    total_cases: int
    total_suites: int
    cases_by_suite: dict[str, list[EvalCase]] = field(default_factory=dict)
    covered_behaviors: list[str] = field(default_factory=list)
    uncovered_behaviors: list[AgentCapability] = field(default_factory=list)
    missing_agents: list[str] = field(default_factory=list)


def load_eval_suites(cases_dir: Path) -> tuple[dict[str, list[EvalCase]], set[str]]:
    """Load all eval suite YAML files.

    Returns:
        Tuple of (suites dict keyed by display name, set of file stems for matching)
    """
    suites: dict[str, list[EvalCase]] = {}
    file_stems: set[str] = set()

    for suite_file in cases_dir.glob("*.yaml"):
        try:
            with suite_file.open() as f:
                data = yaml.safe_load(f)

            suite_name = data.get("name", suite_file.stem)
            file_stems.add(suite_file.stem)  # Track file stem for agent matching
            cases: list[EvalCase] = []

            for case_data in data.get("cases", []):
                case = EvalCase(
                    id=case_data.get("id", "unknown"),
                    suite=suite_name,
                    description=case_data.get("description", ""),
                    behaviors=case_data.get("behaviors", []),
                )
                cases.append(case)

            suites[suite_name] = cases
        except Exception as e:
            print(f"Warning: Could not load {suite_file}: {e}", file=sys.stderr)

    return suites, file_stems


def discover_agent_capabilities(agents_dir: Path) -> list[AgentCapability]:
    """Discover agent capabilities from source code.

    Parses agent files to extract:
    - Tool definitions
    - Documented behaviors
    - Agent types
    """
    import ast

    capabilities: list[AgentCapability] = []

    for agent_file in agents_dir.rglob("*.py"):
        if agent_file.name.startswith("_"):
            continue

        agent_name = agent_file.stem
        try:
            with agent_file.open() as f:
                content = f.read()
                tree = ast.parse(content)

            # Look for class docstrings that mention behaviors
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Extract behaviors from docstring
                        for line in docstring.split("\n"):
                            line = line.strip()
                            if line.startswith("-") or line.startswith("*"):
                                behavior = line.lstrip("-* ").strip()
                                if behavior and len(behavior) > 10:
                                    capabilities.append(
                                        AgentCapability(
                                            agent=agent_name,
                                            behavior=behavior[:100],
                                            description=f"From {node.name} docstring",
                                        )
                                    )

                # Look for method names that indicate behaviors
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("handle_") or node.name.startswith(
                        "process_"
                    ):
                        behavior = node.name.replace("_", " ").title()
                        capabilities.append(
                            AgentCapability(
                                agent=agent_name,
                                behavior=behavior,
                                description=f"Method: {node.name}",
                            )
                        )
        except Exception:  # noqa: S110
            pass  # Skip files that can't be parsed

    return capabilities


def analyze_coverage(project_root: Path) -> EvalCoverageReport:
    """Analyze eval coverage."""
    cases_dir = project_root / "evals" / "cases"
    agents_dir = project_root / "src" / "ash" / "agents"

    # Load eval suites
    suites, file_stems = load_eval_suites(cases_dir)

    # Count totals
    total_cases = sum(len(cases) for cases in suites.values())
    total_suites = len(suites)

    # Collect covered behaviors from eval cases
    covered_behaviors: set[str] = set()
    for cases in suites.values():
        for case in cases:
            covered_behaviors.update(case.behaviors)
            # Also add case description as a covered behavior
            if case.description:
                covered_behaviors.add(case.description.lower())

    # Find agents with evals (use file stems, not suite display names)
    agents_with_evals: set[str] = set()
    for stem in file_stems:
        agents_with_evals.add(stem.replace("_", "-"))

    # Find all agents
    all_agents: set[str] = set()
    if agents_dir.exists():
        for agent_file in agents_dir.glob("*.py"):
            if not agent_file.name.startswith("_"):
                all_agents.add(agent_file.stem.replace("_", "-"))

    # Find missing agents (no eval suite)
    missing_agents = list(all_agents - agents_with_evals - {"base", "executor"})

    # Discover capabilities and find uncovered ones
    capabilities = discover_agent_capabilities(agents_dir)
    uncovered: list[AgentCapability] = []
    for cap in capabilities:
        # Check if this behavior is covered by any eval
        behavior_lower = cap.behavior.lower()
        is_covered = any(
            behavior_lower in cov.lower() or cov.lower() in behavior_lower
            for cov in covered_behaviors
        )
        if not is_covered:
            uncovered.append(cap)

    return EvalCoverageReport(
        total_cases=total_cases,
        total_suites=total_suites,
        cases_by_suite=suites,
        covered_behaviors=list(covered_behaviors),
        uncovered_behaviors=uncovered,
        missing_agents=sorted(missing_agents),
    )


def print_report(report: EvalCoverageReport, verbose: bool = False) -> None:
    """Print the eval coverage report."""
    print("=" * 60)
    print("Eval Coverage Analysis")
    print("=" * 60)

    print(f"\nTotal Suites: {report.total_suites}")
    print(f"Total Cases: {report.total_cases}")

    # Cases by suite
    print("\n--- Cases by Suite ---")
    for suite_name, cases in sorted(report.cases_by_suite.items()):
        print(f"  {suite_name}: {len(cases)} cases")
        if verbose:
            for case in cases:
                print(f"    - {case.id}: {case.description[:50]}...")

    # Missing agents (no eval suite at all)
    if report.missing_agents:
        print(f"\n--- Agents Without Eval Suites ({len(report.missing_agents)}) ---")
        for agent in report.missing_agents:
            print(f"  [MISSING] {agent}")
    else:
        print("\n[OK] All agents have eval suites")

    # Uncovered behaviors
    if report.uncovered_behaviors:
        print(
            f"\n--- Potentially Uncovered Behaviors ({len(report.uncovered_behaviors)}) ---"
        )
        # Group by agent
        by_agent: dict[str, list[AgentCapability]] = {}
        for cap in report.uncovered_behaviors:
            if cap.agent not in by_agent:
                by_agent[cap.agent] = []
            by_agent[cap.agent].append(cap)

        for agent, caps in sorted(by_agent.items()):
            print(f"  {agent}:")
            for cap in caps[:3]:  # Limit per agent
                print(f"    - {cap.behavior}")
            if len(caps) > 3:
                print(f"    ... and {len(caps) - 3} more")

    print("\n" + "=" * 60)

    # Summary
    if not report.missing_agents and not report.uncovered_behaviors:
        print("Overall: GOOD COVERAGE")
    else:
        issues = []
        if report.missing_agents:
            issues.append(f"{len(report.missing_agents)} agents missing evals")
        if report.uncovered_behaviors:
            issues.append(f"{len(report.uncovered_behaviors)} behaviors uncovered")
        print(f"Overall: {', '.join(issues)}")
    print("=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Eval coverage analysis")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of text"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    report = analyze_coverage(project_root)

    if args.json:
        output = {
            "total_suites": report.total_suites,
            "total_cases": report.total_cases,
            "suites": {
                name: [{"id": c.id, "description": c.description} for c in cases]
                for name, cases in report.cases_by_suite.items()
            },
            "missing_agents": report.missing_agents,
            "uncovered_behaviors": [
                {"agent": c.agent, "behavior": c.behavior}
                for c in report.uncovered_behaviors
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Exit with error if missing agents
    if report.missing_agents:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
