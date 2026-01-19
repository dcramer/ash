#!/usr/bin/env python
"""Session debugging analysis for role-debug.

Deep analysis of a single chat session to identify:
- Tool failures and misuse
- Behavioral gaps
- Missed opportunities
- Prompt improvement suggestions

Run with: uv run python scripts/session-debug.py <session_id>
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolCall:
    """A tool call from the session."""

    name: str
    input: dict
    result: str | None = None
    error: bool = False
    duration_hint: str | None = None  # If available


@dataclass
class Turn:
    """A conversation turn (user message + assistant response)."""

    user_message: str
    assistant_response: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turn_number: int = 0


@dataclass
class DebugFinding:
    """A debugging finding."""

    category: str  # tool_failure, missed_tool, behavior_gap, prompt_issue
    severity: str  # error, warning, suggestion
    turn: int
    description: str
    suggestion: str | None = None


@dataclass
class SessionDebugReport:
    """Full session debug report."""

    session_id: str
    total_turns: int
    total_tool_calls: int
    tool_failures: int
    findings: list[DebugFinding] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    tool_usage: dict[str, int] = field(default_factory=dict)


def load_session_context(session_dir: Path) -> list[dict]:
    """Load context.jsonl from a session directory."""
    context_file = session_dir / "context.jsonl"
    if not context_file.exists():
        return []

    entries = []
    with context_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def load_session_history(session_dir: Path) -> list[dict]:
    """Load history.jsonl from a session directory."""
    history_file = session_dir / "history.jsonl"
    if not history_file.exists():
        return []

    entries = []
    with history_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def extract_tool_calls(context: list[dict]) -> list[ToolCall]:
    """Extract tool calls from context entries."""
    tool_calls = []

    for entry in context:
        if entry.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    name=entry.get("name", "unknown"),
                    input=entry.get("input", {}),
                )
            )
        elif entry.get("type") == "tool_result":
            # Match with previous tool call
            content = entry.get("content", "")
            is_error = entry.get("is_error", False)
            if tool_calls:
                tool_calls[-1].result = content[:500] if content else None
                tool_calls[-1].error = is_error

    return tool_calls


def parse_turns(history: list[dict], context: list[dict]) -> list[Turn]:
    """Parse conversation into turns."""
    turns = []
    current_user = None
    turn_number = 0

    for entry in history:
        role = entry.get("role")
        content = entry.get("content", "")

        if role == "user":
            current_user = content
        elif role == "assistant" and current_user:
            turn_number += 1
            turns.append(
                Turn(
                    user_message=current_user,
                    assistant_response=content,
                    turn_number=turn_number,
                )
            )
            current_user = None

    # Attach tool calls to turns (simplified - based on order)
    all_tools = extract_tool_calls(context)
    tools_per_turn = len(all_tools) // len(turns) if turns else 0
    for i, turn in enumerate(turns):
        start = i * tools_per_turn
        end = start + tools_per_turn if i < len(turns) - 1 else len(all_tools)
        turn.tool_calls = all_tools[start:end]

    return turns


def analyze_tool_failures(turns: list[Turn]) -> list[DebugFinding]:
    """Find tool failures and issues."""
    findings = []

    for turn in turns:
        for tc in turn.tool_calls:
            if tc.error:
                findings.append(
                    DebugFinding(
                        category="tool_failure",
                        severity="error",
                        turn=turn.turn_number,
                        description=f"Tool '{tc.name}' failed: {tc.result[:100] if tc.result else 'unknown error'}",
                        suggestion="Check tool input parameters or handle this error case",
                    )
                )

            # Check for repeated tool calls (possible retry loop)
            same_tool_count = sum(1 for t in turn.tool_calls if t.name == tc.name)
            if same_tool_count > 3:
                findings.append(
                    DebugFinding(
                        category="behavior_gap",
                        severity="warning",
                        turn=turn.turn_number,
                        description=f"Tool '{tc.name}' called {same_tool_count} times in one turn (possible retry loop)",
                        suggestion="Consider adding backoff or early exit logic",
                    )
                )

    return findings


def analyze_behavior_gaps(turns: list[Turn]) -> list[DebugFinding]:
    """Find behavioral issues and gaps."""
    findings = []

    for turn in turns:
        assistant_msg = turn.assistant_response.lower()

        # Check for unanswered questions
        if "?" in turn.user_message and len(turn.assistant_response) < 50:
            findings.append(
                DebugFinding(
                    category="behavior_gap",
                    severity="warning",
                    turn=turn.turn_number,
                    description="User asked a question but response was very short",
                    suggestion="Ensure questions are fully addressed",
                )
            )

        # Check for "I can't" or "I don't know" without trying tools
        if (
            "can't" in assistant_msg
            or "cannot" in assistant_msg
            or "don't know" in assistant_msg
        ):
            if not turn.tool_calls:
                findings.append(
                    DebugFinding(
                        category="missed_tool",
                        severity="suggestion",
                        turn=turn.turn_number,
                        description="Agent said it can't/doesn't know without trying tools",
                        suggestion="Consider using search or other tools before giving up",
                    )
                )

        # Check for very long responses (verbosity)
        if len(turn.assistant_response) > 1500:
            findings.append(
                DebugFinding(
                    category="prompt_issue",
                    severity="suggestion",
                    turn=turn.turn_number,
                    description=f"Response very long ({len(turn.assistant_response)} chars)",
                    suggestion="Consider more concise responses for simple queries",
                )
            )

        # Check for error keywords in response
        error_indicators = ["error", "failed", "sorry", "apologize", "mistake"]
        if any(ind in assistant_msg for ind in error_indicators):
            if not any(tc.error for tc in turn.tool_calls):
                findings.append(
                    DebugFinding(
                        category="behavior_gap",
                        severity="warning",
                        turn=turn.turn_number,
                        description="Response contains error/apology language",
                        suggestion="Review what went wrong in this interaction",
                    )
                )

    return findings


def analyze_missed_opportunities(turns: list[Turn]) -> list[DebugFinding]:
    """Find missed opportunities for better responses."""
    findings = []

    for turn in turns:
        user_msg = turn.user_message.lower()

        # Check for memory-related requests without memory tool
        memory_keywords = ["remember", "don't forget", "keep in mind", "note that"]
        if any(kw in user_msg for kw in memory_keywords):
            memory_tools = [tc for tc in turn.tool_calls if "memory" in tc.name.lower()]
            if not memory_tools:
                findings.append(
                    DebugFinding(
                        category="missed_tool",
                        severity="suggestion",
                        turn=turn.turn_number,
                        description="User asked to remember something but no memory tool was used",
                        suggestion="Use memory tools when user asks to remember things",
                    )
                )

        # Check for search-related requests without search
        search_keywords = [
            "search",
            "look up",
            "find out",
            "what is",
            "who is",
            "latest",
            "current",
        ]
        if any(kw in user_msg for kw in search_keywords):
            search_tools = [
                tc
                for tc in turn.tool_calls
                if "search" in tc.name.lower() or "web" in tc.name.lower()
            ]
            if not search_tools and "?" in turn.user_message:
                findings.append(
                    DebugFinding(
                        category="missed_tool",
                        severity="suggestion",
                        turn=turn.turn_number,
                        description="User asked for information lookup but no search tool was used",
                        suggestion="Consider using web_search for current information requests",
                    )
                )

    return findings


def analyze_session(session_dir: Path) -> SessionDebugReport:
    """Perform full session analysis."""
    session_id = session_dir.name

    context = load_session_context(session_dir)
    history = load_session_history(session_dir)

    if not history:
        return SessionDebugReport(
            session_id=session_id,
            total_turns=0,
            total_tool_calls=0,
            tool_failures=0,
        )

    turns = parse_turns(history, context)
    all_tool_calls = [tc for turn in turns for tc in turn.tool_calls]

    # Count tool usage
    tool_usage: dict[str, int] = {}
    for tc in all_tool_calls:
        tool_usage[tc.name] = tool_usage.get(tc.name, 0) + 1

    # Gather findings
    findings: list[DebugFinding] = []
    findings.extend(analyze_tool_failures(turns))
    findings.extend(analyze_behavior_gaps(turns))
    findings.extend(analyze_missed_opportunities(turns))

    # Sort by turn number, then severity
    severity_order = {"error": 0, "warning": 1, "suggestion": 2}
    findings.sort(key=lambda f: (f.turn, severity_order.get(f.severity, 3)))

    return SessionDebugReport(
        session_id=session_id,
        total_turns=len(turns),
        total_tool_calls=len(all_tool_calls),
        tool_failures=sum(1 for tc in all_tool_calls if tc.error),
        findings=findings,
        turns=turns,
        tool_usage=tool_usage,
    )


def print_report(report: SessionDebugReport, verbose: bool = False) -> None:
    """Print the debug report."""
    print("=" * 60)
    print(f"Session Debug: {report.session_id}")
    print("=" * 60)

    print(f"\nTurns: {report.total_turns}")
    print(f"Tool Calls: {report.total_tool_calls}")
    print(f"Tool Failures: {report.tool_failures}")

    if report.tool_usage:
        print("\n--- Tool Usage ---")
        for tool, count in sorted(report.tool_usage.items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")

    if report.findings:
        print(f"\n--- Findings ({len(report.findings)}) ---")

        errors = [f for f in report.findings if f.severity == "error"]
        warnings = [f for f in report.findings if f.severity == "warning"]
        suggestions = [f for f in report.findings if f.severity == "suggestion"]

        if errors:
            print(f"\nErrors ({len(errors)}):")
            for f in errors:
                print(f"  [Turn {f.turn}] {f.description}")
                if f.suggestion:
                    print(f"    → {f.suggestion}")

        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for f in warnings:
                print(f"  [Turn {f.turn}] {f.description}")
                if f.suggestion:
                    print(f"    → {f.suggestion}")

        if verbose and suggestions:
            print(f"\nSuggestions ({len(suggestions)}):")
            for f in suggestions:
                print(f"  [Turn {f.turn}] {f.description}")
                if f.suggestion:
                    print(f"    → {f.suggestion}")
    else:
        print("\n[OK] No issues found")

    if verbose and report.turns:
        print("\n--- Conversation Summary ---")
        for turn in report.turns[:10]:  # Limit output
            user_preview = turn.user_message[:60].replace("\n", " ")
            assistant_preview = turn.assistant_response[:60].replace("\n", " ")
            tools = ", ".join(tc.name for tc in turn.tool_calls) or "none"
            print(f"\n  Turn {turn.turn_number}:")
            print(f"    User: {user_preview}...")
            print(f"    Tools: {tools}")
            print(f"    Response: {assistant_preview}...")

    print("\n" + "=" * 60)


def find_session(sessions_dir: Path, session_id: str) -> Path | None:
    """Find a session directory by ID (partial match supported)."""
    # Exact match first
    exact = sessions_dir / session_id
    if exact.exists():
        return exact

    # Partial match
    for session_dir in sessions_dir.iterdir():
        if session_id in session_dir.name:
            return session_dir

    return None


def list_recent_sessions(sessions_dir: Path, limit: int = 10) -> list[Path]:
    """List recent sessions."""
    sessions = sorted(
        [d for d in sessions_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return sessions[:limit]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Session debugging analysis")
    parser.add_argument(
        "session_id",
        nargs="?",
        help="Session ID to analyze (partial match supported). Omit to list recent sessions.",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path.home() / ".ash" / "sessions",
        help="Sessions directory",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of text"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    args = parser.parse_args()

    if not args.sessions_dir.exists():
        print(f"Sessions directory not found: {args.sessions_dir}")
        return 1

    # List sessions if no ID provided
    if not args.session_id:
        print("Recent sessions:")
        for session_dir in list_recent_sessions(args.sessions_dir):
            print(f"  {session_dir.name}")
        print(f"\nUsage: {sys.argv[0]} <session_id>")
        return 0

    # Find session
    session_dir = find_session(args.sessions_dir, args.session_id)
    if not session_dir:
        print(f"Session not found: {args.session_id}")
        print("\nRecent sessions:")
        for session_dir in list_recent_sessions(args.sessions_dir, 5):
            print(f"  {session_dir.name}")
        return 1

    report = analyze_session(session_dir)

    if args.json:
        output = {
            "session_id": report.session_id,
            "total_turns": report.total_turns,
            "total_tool_calls": report.total_tool_calls,
            "tool_failures": report.tool_failures,
            "tool_usage": report.tool_usage,
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "turn": f.turn,
                    "description": f.description,
                    "suggestion": f.suggestion,
                }
                for f in report.findings
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Exit with error if errors found
    if any(f.severity == "error" for f in report.findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
