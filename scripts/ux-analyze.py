#!/usr/bin/env python
"""Session/conversation UX analysis for role-ux.

Analyzes session history to identify:
- Response length patterns
- Checkpoint usage
- Error patterns
- Conversation flow issues

Run with: uv run python scripts/ux-analyze.py
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class SessionMetrics:
    """Metrics for a single session."""

    session_id: str
    message_count: int
    user_messages: int
    assistant_messages: int
    avg_response_length: float
    max_response_length: int
    error_count: int
    checkpoint_count: int
    duration_minutes: float | None


@dataclass
class UXReport:
    """UX analysis report."""

    sessions_analyzed: int
    total_messages: int
    avg_response_length: float
    response_length_distribution: dict[str, int] = field(default_factory=dict)
    error_patterns: list[tuple[str, int]] = field(default_factory=list)
    checkpoint_usage: int = 0
    issues: list[str] = field(default_factory=list)
    session_metrics: list[SessionMetrics] = field(default_factory=list)


def parse_history_file(filepath: Path) -> list[dict]:
    """Parse a history.jsonl file."""
    messages = []
    try:
        with filepath.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:  # noqa: S110
        pass  # Skip files that can't be read
    return messages


def analyze_session(session_dir: Path) -> SessionMetrics | None:
    """Analyze a single session directory."""
    history_file = session_dir / "history.jsonl"
    if not history_file.exists():
        return None

    messages = parse_history_file(history_file)
    if not messages:
        return None

    session_id = session_dir.name
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

    # Calculate response lengths
    response_lengths = [len(m.get("content", "")) for m in assistant_msgs]
    avg_length = (
        sum(response_lengths) / len(response_lengths) if response_lengths else 0
    )
    max_length = max(response_lengths) if response_lengths else 0

    # Count errors (look for error indicators in assistant messages)
    error_patterns = ["error", "failed", "couldn't", "can't", "unable", "issue"]
    error_count = 0
    for msg in assistant_msgs:
        content = msg.get("content", "").lower()
        if any(pattern in content for pattern in error_patterns):
            error_count += 1

    # Count checkpoints (look for checkpoint indicators)
    checkpoint_patterns = ["confirm", "proceed", "continue", "approve", "checkpoint"]
    checkpoint_count = 0
    for msg in assistant_msgs:
        content = msg.get("content", "").lower()
        if any(pattern in content for pattern in checkpoint_patterns):
            checkpoint_count += 1

    # Calculate duration
    duration = None
    if messages:
        try:
            first_ts = messages[0].get("created_at")
            last_ts = messages[-1].get("created_at")
            if first_ts and last_ts:
                first = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                last = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                duration = (last - first).total_seconds() / 60
        except Exception:  # noqa: S110
            pass  # Skip if timestamp parsing fails

    return SessionMetrics(
        session_id=session_id,
        message_count=len(messages),
        user_messages=len(user_msgs),
        assistant_messages=len(assistant_msgs),
        avg_response_length=avg_length,
        max_response_length=max_length,
        error_count=error_count,
        checkpoint_count=checkpoint_count,
        duration_minutes=duration,
    )


def find_error_patterns(sessions_dir: Path) -> list[tuple[str, int]]:
    """Find common error patterns across all sessions."""
    error_phrases: Counter[str] = Counter()

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        history_file = session_dir / "history.jsonl"
        if not history_file.exists():
            continue

        messages = parse_history_file(history_file)
        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "").lower()

            # Look for error-like phrases
            patterns = [
                r"(couldn't|can't|cannot) [\w\s]+",
                r"error:? [\w\s]+",
                r"failed to [\w\s]+",
                r"unable to [\w\s]+",
                r"(not|isn't|wasn't) [\w\s]+ (found|available|set)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple):
                        match = " ".join(match)
                    phrase = match.strip()[:50]
                    if len(phrase) > 10:
                        error_phrases[phrase] += 1

    return error_phrases.most_common(10)


def identify_ux_issues(metrics: list[SessionMetrics]) -> list[str]:
    """Identify UX issues from session metrics."""
    issues = []

    # Check for very long responses
    long_response_sessions = [m for m in metrics if m.avg_response_length > 500]
    if long_response_sessions:
        issues.append(
            f"{len(long_response_sessions)} sessions have avg response > 500 chars (may be too verbose)"
        )

    # Check for high error rates
    high_error_sessions = [
        m for m in metrics if m.error_count > m.assistant_messages * 0.3
    ]
    if high_error_sessions:
        issues.append(
            f"{len(high_error_sessions)} sessions have >30% error rate in responses"
        )

    # Check for sessions with no checkpoints where they might be expected
    long_sessions = [m for m in metrics if m.message_count > 10]
    no_checkpoint_long = [s for s in long_sessions if s.checkpoint_count == 0]
    if len(no_checkpoint_long) > len(long_sessions) * 0.5:
        issues.append(
            f"{len(no_checkpoint_long)} long sessions lack checkpoints (may need confirmation points)"
        )

    return issues


def analyze_sessions(sessions_dir: Path, limit: int = 50) -> UXReport:
    """Analyze all sessions in the directory."""
    session_metrics: list[SessionMetrics] = []

    # Get session directories sorted by modification time (most recent first)
    session_dirs = sorted(
        [d for d in sessions_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )[:limit]

    for session_dir in session_dirs:
        metrics = analyze_session(session_dir)
        if metrics:
            session_metrics.append(metrics)

    if not session_metrics:
        return UXReport(
            sessions_analyzed=0,
            total_messages=0,
            avg_response_length=0,
        )

    # Aggregate metrics
    total_messages = sum(m.message_count for m in session_metrics)
    total_response_length = sum(
        m.avg_response_length * m.assistant_messages for m in session_metrics
    )
    total_assistant = sum(m.assistant_messages for m in session_metrics)
    avg_response = total_response_length / total_assistant if total_assistant else 0

    # Response length distribution
    distribution = {"short (<100)": 0, "medium (100-300)": 0, "long (>300)": 0}
    for m in session_metrics:
        if m.avg_response_length < 100:
            distribution["short (<100)"] += 1
        elif m.avg_response_length < 300:
            distribution["medium (100-300)"] += 1
        else:
            distribution["long (>300)"] += 1

    # Checkpoint usage
    total_checkpoints = sum(m.checkpoint_count for m in session_metrics)

    # Error patterns
    error_patterns = find_error_patterns(sessions_dir)

    # Identify issues
    issues = identify_ux_issues(session_metrics)

    return UXReport(
        sessions_analyzed=len(session_metrics),
        total_messages=total_messages,
        avg_response_length=avg_response,
        response_length_distribution=distribution,
        error_patterns=error_patterns,
        checkpoint_usage=total_checkpoints,
        issues=issues,
        session_metrics=session_metrics,
    )


def print_report(report: UXReport, verbose: bool = False) -> None:
    """Print the UX report."""
    print("=" * 60)
    print("UX Analysis Report")
    print("=" * 60)

    print(f"\nSessions Analyzed: {report.sessions_analyzed}")
    print(f"Total Messages: {report.total_messages}")
    print(f"Avg Response Length: {report.avg_response_length:.0f} chars")

    # Response length distribution
    print("\n--- Response Length Distribution ---")
    for bucket, count in report.response_length_distribution.items():
        pct = (
            (count / report.sessions_analyzed * 100) if report.sessions_analyzed else 0
        )
        print(f"  {bucket}: {count} ({pct:.0f}%)")

    # Checkpoint usage
    print(f"\nCheckpoint Messages: {report.checkpoint_usage}")

    # Error patterns
    if report.error_patterns:
        print("\n--- Common Error Patterns ---")
        for pattern, count in report.error_patterns[:5]:
            print(f"  [{count}x] {pattern}")

    # Issues
    if report.issues:
        print("\n--- UX Issues Found ---")
        for issue in report.issues:
            print(f"  [WARN] {issue}")
    else:
        print("\n[OK] No major UX issues detected")

    if verbose and report.session_metrics:
        print("\n--- Session Details ---")
        for m in report.session_metrics[:10]:
            print(
                f"  {m.session_id}: {m.message_count} msgs, avg {m.avg_response_length:.0f} chars"
            )

    print("\n" + "=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Session UX analysis")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of text"
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path.home() / ".ash" / "sessions",
        help="Sessions directory to analyze",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of sessions to analyze",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    args = parser.parse_args()

    if not args.sessions_dir.exists():
        print(f"Sessions directory not found: {args.sessions_dir}")
        return 1

    report = analyze_sessions(args.sessions_dir, limit=args.limit)

    if args.json:
        output = {
            "sessions_analyzed": report.sessions_analyzed,
            "total_messages": report.total_messages,
            "avg_response_length": report.avg_response_length,
            "response_length_distribution": report.response_length_distribution,
            "checkpoint_usage": report.checkpoint_usage,
            "error_patterns": [
                {"pattern": p, "count": c} for p, c in report.error_patterns
            ],
            "issues": report.issues,
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Exit with error if issues found
    if report.issues:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
