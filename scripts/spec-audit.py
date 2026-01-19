#!/usr/bin/env python
"""Spec health check for role-spec.

Analyzes specs to identify:
- Specs without corresponding tests
- Stale specs (implementation drift)
- Unimplemented requirements
- Missing spec sections

Run with: uv run python scripts/spec-audit.py
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Requirement:
    """A single requirement from a spec."""

    text: str
    level: str  # MUST, SHOULD, MAY
    has_test: bool = False


@dataclass
class SpecHealth:
    """Health assessment for a single spec."""

    name: str
    path: str
    has_files_section: bool
    has_requirements: bool
    has_interface: bool
    has_verification: bool
    must_requirements: list[Requirement] = field(default_factory=list)
    should_requirements: list[Requirement] = field(default_factory=list)
    may_requirements: list[Requirement] = field(default_factory=list)
    referenced_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def total_requirements(self) -> int:
        return (
            len(self.must_requirements)
            + len(self.should_requirements)
            + len(self.may_requirements)
        )

    @property
    def tested_must_requirements(self) -> int:
        return sum(1 for r in self.must_requirements if r.has_test)


@dataclass
class SpecAuditReport:
    """Full spec audit report."""

    total_specs: int
    specs: list[SpecHealth] = field(default_factory=list)
    specs_without_tests: list[str] = field(default_factory=list)
    stale_specs: list[str] = field(default_factory=list)
    incomplete_specs: list[str] = field(default_factory=list)


def parse_spec(filepath: Path) -> SpecHealth:
    """Parse a spec file and extract structure."""
    with filepath.open() as f:
        content = f.read()

    name = filepath.stem
    health = SpecHealth(
        name=name,
        path=str(filepath),
        has_files_section=bool(re.search(r"^Files:", content, re.MULTILINE)),
        has_requirements=bool(re.search(r"^## Requirements", content, re.MULTILINE)),
        has_interface=bool(re.search(r"^## Interface", content, re.MULTILINE)),
        has_verification=bool(re.search(r"^## Verification", content, re.MULTILINE)),
    )

    # Extract referenced files
    files_match = re.search(r"^Files:\s*(.+)$", content, re.MULTILINE)
    if files_match:
        files_str = files_match.group(1)
        health.referenced_files = [f.strip() for f in files_str.split(",") if f.strip()]

    # Extract MUST requirements
    must_section = re.search(
        r"### MUST\s*\n(.*?)(?=###|\Z)", content, re.DOTALL | re.MULTILINE
    )
    if must_section:
        requirements = re.findall(r"^-\s+(.+)$", must_section.group(1), re.MULTILINE)
        health.must_requirements = [
            Requirement(text=r, level="MUST") for r in requirements
        ]

    # Extract SHOULD requirements
    should_section = re.search(
        r"### SHOULD\s*\n(.*?)(?=###|\Z)", content, re.DOTALL | re.MULTILINE
    )
    if should_section:
        requirements = re.findall(r"^-\s+(.+)$", should_section.group(1), re.MULTILINE)
        health.should_requirements = [
            Requirement(text=r, level="SHOULD") for r in requirements
        ]

    # Extract MAY requirements
    may_section = re.search(
        r"### MAY\s*\n(.*?)(?=###|\Z)", content, re.DOTALL | re.MULTILINE
    )
    if may_section:
        requirements = re.findall(r"^-\s+(.+)$", may_section.group(1), re.MULTILINE)
        health.may_requirements = [
            Requirement(text=r, level="MAY") for r in requirements
        ]

    # Check for issues
    if not health.has_files_section:
        health.issues.append("Missing Files section")
    if not health.has_requirements:
        health.issues.append("Missing Requirements section")
    if not health.has_interface:
        health.issues.append("Missing Interface section")
    if not health.has_verification:
        health.issues.append("Missing Verification section")

    return health


def check_files_exist(health: SpecHealth, project_root: Path) -> None:
    """Check if referenced files exist."""
    for filepath in health.referenced_files:
        full_path = project_root / filepath
        if not full_path.exists():
            # Try with src/ prefix
            src_path = project_root / "src" / filepath
            if not src_path.exists():
                health.missing_files.append(filepath)
                health.issues.append(f"Referenced file not found: {filepath}")


def check_tests_exist(health: SpecHealth, project_root: Path) -> None:
    """Check if spec has corresponding tests."""
    tests_dir = project_root / "tests"
    spec_name = health.name.replace("-", "_")

    # Look for test files that match the spec name
    test_patterns = [
        f"test_{spec_name}.py",
        f"test_{spec_name}_*.py",
        f"*{spec_name}*test*.py",
    ]

    has_tests = False
    for pattern in test_patterns:
        if list(tests_dir.glob(f"**/{pattern}")):
            has_tests = True
            break

    if not has_tests:
        health.issues.append("No test file found for this spec")

    # Try to match MUST requirements to test functions
    if has_tests:
        for test_file in tests_dir.glob("**/*.py"):
            try:
                test_content = test_file.read_text().lower()
                for req in health.must_requirements:
                    # Simple heuristic: check if key words appear in tests
                    keywords = [
                        w.lower()
                        for w in req.text.split()
                        if len(w) > 4 and w.isalpha()
                    ]
                    if keywords and any(kw in test_content for kw in keywords[:3]):
                        req.has_test = True
            except Exception:  # noqa: S110
                pass  # Skip files that can't be parsed


def audit_specs(project_root: Path) -> SpecAuditReport:
    """Audit all specs in the project."""
    specs_dir = project_root / "specs"
    if not specs_dir.exists():
        return SpecAuditReport(total_specs=0)

    specs: list[SpecHealth] = []
    specs_without_tests: list[str] = []
    stale_specs: list[str] = []
    incomplete_specs: list[str] = []

    for spec_file in sorted(specs_dir.glob("*.md")):
        health = parse_spec(spec_file)
        check_files_exist(health, project_root)
        check_tests_exist(health, project_root)
        specs.append(health)

        # Categorize issues
        if "No test file found" in str(health.issues):
            specs_without_tests.append(health.name)

        if health.missing_files:
            stale_specs.append(health.name)

        if not all(
            [
                health.has_files_section,
                health.has_requirements,
                health.has_interface,
                health.has_verification,
            ]
        ):
            incomplete_specs.append(health.name)

    return SpecAuditReport(
        total_specs=len(specs),
        specs=specs,
        specs_without_tests=specs_without_tests,
        stale_specs=stale_specs,
        incomplete_specs=incomplete_specs,
    )


def print_report(report: SpecAuditReport, verbose: bool = False) -> None:
    """Print the spec audit report."""
    print("=" * 60)
    print("Spec Audit Report")
    print("=" * 60)

    print(f"\nTotal Specs: {report.total_specs}")

    # Summary
    healthy = report.total_specs - len(
        set(report.specs_without_tests)
        | set(report.stale_specs)
        | set(report.incomplete_specs)
    )
    print(f"Healthy Specs: {healthy}/{report.total_specs}")

    # Incomplete specs
    if report.incomplete_specs:
        print(f"\n--- Incomplete Specs ({len(report.incomplete_specs)}) ---")
        for name in report.incomplete_specs:
            spec = next(s for s in report.specs if s.name == name)
            missing = []
            if not spec.has_files_section:
                missing.append("Files")
            if not spec.has_requirements:
                missing.append("Requirements")
            if not spec.has_interface:
                missing.append("Interface")
            if not spec.has_verification:
                missing.append("Verification")
            print(f"  {name}: missing {', '.join(missing)}")

    # Specs without tests
    if report.specs_without_tests:
        print(f"\n--- Specs Without Tests ({len(report.specs_without_tests)}) ---")
        for name in report.specs_without_tests:
            print(f"  {name}")

    # Stale specs
    if report.stale_specs:
        print(f"\n--- Stale Specs (missing files) ({len(report.stale_specs)}) ---")
        for name in report.stale_specs:
            spec = next(s for s in report.specs if s.name == name)
            print(f"  {name}: {', '.join(spec.missing_files)}")

    # Requirements coverage
    if verbose:
        print("\n--- Requirements Coverage ---")
        for spec in report.specs:
            if spec.must_requirements:
                tested = spec.tested_must_requirements
                total = len(spec.must_requirements)
                print(f"  {spec.name}: {tested}/{total} MUST requirements tested")

    print("\n" + "=" * 60)

    # Overall status
    if (
        not report.incomplete_specs
        and not report.stale_specs
        and not report.specs_without_tests
    ):
        print("Overall: SPECS HEALTHY")
    else:
        issues = []
        if report.incomplete_specs:
            issues.append(f"{len(report.incomplete_specs)} incomplete")
        if report.stale_specs:
            issues.append(f"{len(report.stale_specs)} stale")
        if report.specs_without_tests:
            issues.append(f"{len(report.specs_without_tests)} untested")
        print(f"Overall: {', '.join(issues)}")
    print("=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Spec audit")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of text"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    report = audit_specs(project_root)

    if args.json:
        output = {
            "total_specs": report.total_specs,
            "specs": [
                {
                    "name": s.name,
                    "path": s.path,
                    "has_files_section": s.has_files_section,
                    "has_requirements": s.has_requirements,
                    "has_interface": s.has_interface,
                    "has_verification": s.has_verification,
                    "must_requirements": len(s.must_requirements),
                    "tested_must_requirements": s.tested_must_requirements,
                    "issues": s.issues,
                }
                for s in report.specs
            ],
            "specs_without_tests": report.specs_without_tests,
            "stale_specs": report.stale_specs,
            "incomplete_specs": report.incomplete_specs,
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Exit with error if issues
    if report.incomplete_specs or report.stale_specs:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
