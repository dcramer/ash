#!/usr/bin/env python
"""Project health dashboard for role-master.

Reports on the overall health of the Ash project:
- Test status (pass/fail count)
- Lint status (error count)
- Type check status (error count)
- Coverage percentage
- Open TODOs in codebase

Run with: uv run python scripts/role-health.py
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HealthReport:
    """Project health status."""

    tests_passed: int
    tests_failed: int
    tests_errors: int
    lint_errors: int
    lint_warnings: int
    type_errors: int
    coverage_percent: float | None
    todo_count: int


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)  # noqa: S603
    return result.returncode, result.stdout, result.stderr


def check_tests(project_root: Path) -> tuple[int, int, int]:
    """Run pytest and return (passed, failed, errors) counts."""
    code, stdout, stderr = run_command(
        ["uv", "run", "pytest", "--tb=no", "-q"], cwd=project_root
    )

    # Parse pytest output for summary line like "5 passed, 2 failed"
    output = stdout + stderr
    passed = failed = errors = 0

    for line in output.split("\n"):
        line = line.strip()
        if "passed" in line or "failed" in line or "error" in line:
            # Look for patterns like "5 passed" or "2 failed"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "passed" and i > 0:
                    try:
                        passed = int(parts[i - 1])
                    except ValueError:
                        pass
                elif part == "failed" and i > 0:
                    try:
                        failed = int(parts[i - 1])
                    except ValueError:
                        pass
                elif part in ("error", "errors") and i > 0:
                    try:
                        errors = int(parts[i - 1])
                    except ValueError:
                        pass

    return passed, failed, errors


def check_lint(project_root: Path) -> tuple[int, int]:
    """Run ruff check and return (errors, warnings) counts."""
    code, stdout, stderr = run_command(
        ["uv", "run", "ruff", "check", "--output-format=json", "."], cwd=project_root
    )

    errors = warnings = 0
    if stdout.strip():
        try:
            issues = json.loads(stdout)
            for issue in issues:
                # E and F codes are errors, W codes are warnings
                if issue.get("code", "").startswith(("E", "F")):
                    errors += 1
                else:
                    warnings += 1
        except json.JSONDecodeError:
            # Count lines as rough estimate
            errors = len(stdout.strip().split("\n"))

    return errors, warnings


def check_types(project_root: Path) -> int:
    """Run ty check and return error count."""
    code, stdout, stderr = run_command(["uv", "run", "ty", "check"], cwd=project_root)

    # ty outputs errors to stderr typically
    output = stdout + stderr
    error_count = 0

    for line in output.split("\n"):
        if "error" in line.lower() and ":" in line:
            error_count += 1

    return error_count


def check_coverage(project_root: Path) -> float | None:
    """Run pytest with coverage and return coverage percentage."""
    # Check if coverage data exists or run quick coverage
    code, stdout, stderr = run_command(
        [
            "uv",
            "run",
            "pytest",
            "--cov=src/ash",
            "--cov-report=json",
            "--cov-report=term:skip-covered",
            "-q",
            "--tb=no",
        ],
        cwd=project_root,
    )

    coverage_file = project_root / "coverage.json"
    if coverage_file.exists():
        try:
            with coverage_file.open() as f:
                data = json.load(f)
            return data.get("totals", {}).get("percent_covered", 0.0)
        except (json.JSONDecodeError, KeyError):
            pass

    return None


def count_todos(project_root: Path) -> int:
    """Count TODO comments in the codebase."""
    code, stdout, stderr = run_command(
        ["grep", "-r", "-c", "TODO", "src/ash", "--include=*.py"], cwd=project_root
    )

    total = 0
    for line in stdout.strip().split("\n"):
        if ":" in line:
            try:
                count = int(line.split(":")[-1])
                total += count
            except ValueError:
                pass

    return total


def get_health_report(project_root: Path, skip_slow: bool = False) -> HealthReport:
    """Generate a complete health report."""
    tests_passed, tests_failed, tests_errors = (0, 0, 0)
    coverage = None

    if not skip_slow:
        tests_passed, tests_failed, tests_errors = check_tests(project_root)
        coverage = check_coverage(project_root)

    lint_errors, lint_warnings = check_lint(project_root)
    type_errors = check_types(project_root)
    todo_count = count_todos(project_root)

    return HealthReport(
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_errors=tests_errors,
        lint_errors=lint_errors,
        lint_warnings=lint_warnings,
        type_errors=type_errors,
        coverage_percent=coverage,
        todo_count=todo_count,
    )


def format_status(ok: bool) -> str:
    """Format pass/fail status."""
    return "[OK]" if ok else "[FAIL]"


def print_report(report: HealthReport, verbose: bool = False) -> None:
    """Print the health report."""
    print("=" * 60)
    print("Project Health Dashboard")
    print("=" * 60)

    # Tests
    tests_ok = report.tests_failed == 0 and report.tests_errors == 0
    test_total = report.tests_passed + report.tests_failed
    print(
        f"\nTests:    {format_status(tests_ok)} {report.tests_passed}/{test_total} passed"
    )
    if report.tests_errors > 0:
        print(f"          {report.tests_errors} collection errors")

    # Lint
    lint_ok = report.lint_errors == 0
    print(f"Lint:     {format_status(lint_ok)} {report.lint_errors} errors")
    if report.lint_warnings > 0:
        print(f"          {report.lint_warnings} warnings")

    # Types
    types_ok = report.type_errors == 0
    print(f"Types:    {format_status(types_ok)} {report.type_errors} errors")

    # Coverage
    if report.coverage_percent is not None:
        cov_ok = report.coverage_percent >= 80
        print(f"Coverage: {format_status(cov_ok)} {report.coverage_percent:.1f}%")
    else:
        print("Coverage: [SKIP] Not measured")

    # TODOs
    print(f"\nOpen TODOs: {report.todo_count}")

    print("\n" + "=" * 60)

    # Summary
    all_ok = tests_ok and lint_ok and types_ok
    if all_ok:
        print("Overall: HEALTHY")
    else:
        issues = []
        if not tests_ok:
            issues.append("tests")
        if not lint_ok:
            issues.append("lint")
        if not types_ok:
            issues.append("types")
        print(f"Overall: NEEDS ATTENTION ({', '.join(issues)})")

    print("=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Project health dashboard")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of text"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow checks (tests, coverage)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    report = get_health_report(project_root, skip_slow=args.fast)

    if args.json:
        output = {
            "tests": {
                "passed": report.tests_passed,
                "failed": report.tests_failed,
                "errors": report.tests_errors,
            },
            "lint": {"errors": report.lint_errors, "warnings": report.lint_warnings},
            "types": {"errors": report.type_errors},
            "coverage": report.coverage_percent,
            "todos": report.todo_count,
            "healthy": (
                report.tests_failed == 0
                and report.tests_errors == 0
                and report.lint_errors == 0
                and report.type_errors == 0
            ),
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Exit with error if unhealthy
    if (
        report.tests_failed > 0
        or report.tests_errors > 0
        or report.lint_errors > 0
        or report.type_errors > 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
