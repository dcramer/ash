"""Rich terminal reporting for eval results."""

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from evals.runner import EvalReport, EvalResult


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_status(result: EvalResult) -> Text:
    """Format pass/fail/error status with color."""
    if result.is_judge_error:
        return Text("ERROR", style="bold yellow")
    if result.passed:
        return Text("PASS", style="bold green")
    return Text("FAIL", style="bold red")


def _format_score(score: float, is_judge_error: bool = False) -> Text:
    """Format score with color based on value."""
    if is_judge_error:
        return Text("N/A", style="dim")
    score_text = f"{score:.2f}"
    if score >= 0.8:
        return Text(score_text, style="green")
    elif score >= 0.5:
        return Text(score_text, style="yellow")
    return Text(score_text, style="red")


def print_report(report: EvalReport, *, console: Console | None = None) -> None:
    """Print a formatted eval report to the terminal.

    Args:
        report: The eval report to print.
        console: Optional Rich console (creates new one if not provided).
    """
    if console is None:
        console = Console()

    # Summary panel
    accuracy_pct = report.accuracy * 100
    if report.accuracy >= 0.8:
        accuracy_style = "bold green"
    elif report.accuracy >= 0.5:
        accuracy_style = "bold yellow"
    else:
        accuracy_style = "bold red"

    summary = Text()
    summary.append(f"Suite: {report.suite_name}\n", style="bold")
    summary.append(f"Total: {report.total} | ")
    summary.append(f"Passed: {report.passed}", style="green")
    summary.append(" | ")
    summary.append(f"Failed: {report.failed}", style="red" if report.failed else "dim")
    if report.judge_errors > 0:
        summary.append(" | ")
        summary.append(f"Judge Errors: {report.judge_errors}", style="yellow")
    summary.append("\n")
    summary.append(f"Accuracy: {accuracy_pct:.1f}%", style=accuracy_style)
    summary.append(f" | Average Score: {report.average_score:.2f}")
    if report.judge_errors > 0:
        summary.append(f" (excluding {report.judge_errors} judge errors)", style="dim")

    console.print(Panel(summary, title="Eval Summary", border_style="blue"))

    # Results table
    table = Table(title="Results", show_header=True, header_style="bold")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Description", max_width=40)
    table.add_column("Score", justify="center")
    table.add_column("Status", justify="center")

    for result in report.results:
        table.add_row(
            result.case.id,
            _truncate(result.case.description, 40),
            _format_score(result.score, result.is_judge_error),
            _format_status(result),
        )

    console.print(table)

    # Judge error details
    judge_errors = report.judge_error_cases()
    if judge_errors:
        console.print("\n[bold yellow]Judge Errors:[/bold yellow]\n")
        console.print(
            "[dim]These cases failed due to judge errors (e.g., API failures, "
            "parse errors) and are excluded from accuracy calculations.[/dim]\n"
        )

        for result in judge_errors:
            _print_judge_error_case(result, console)

    # Failed case details
    failed = report.failed_cases()
    if failed:
        console.print("\n[bold red]Failed Cases:[/bold red]\n")

        for result in failed:
            _print_failed_case(result, console)


def _print_judge_error_case(result: EvalResult, console: Console) -> None:
    """Print details for a case with judge error."""
    console.print(f"[bold cyan]{result.case.id}[/bold cyan]: {result.case.description}")
    console.print(f"  [dim]Prompt:[/dim] {_truncate(result.case.prompt, 80)}")

    error_type = result.judge_result.error_type or "unknown"
    console.print(f"  [yellow]Error Type:[/yellow] {error_type}")
    console.print(f"  [yellow]Details:[/yellow] {result.judge_result.reasoning}")

    if result.response_text:
        console.print(f"  [dim]Response:[/dim] {_truncate(result.response_text, 80)}")

    console.print()


def _print_failed_case(result: EvalResult, console: Console) -> None:
    """Print details for a failed case."""
    console.print(f"[bold cyan]{result.case.id}[/bold cyan]: {result.case.description}")
    console.print(f"  [dim]Prompt:[/dim] {_truncate(result.case.prompt, 80)}")
    console.print(
        f"  [dim]Expected:[/dim] {_truncate(result.case.expected_behavior, 80)}"
    )

    if result.error:
        console.print(f"  [red]Error:[/red] {result.error}")
    else:
        console.print(f"  [dim]Response:[/dim] {_truncate(result.response_text, 80)}")
        console.print(f"  [dim]Reasoning:[/dim] {result.judge_result.reasoning}")

    if result.tool_calls:
        tools = ", ".join(tc["name"] for tc in result.tool_calls[:5])
        if len(result.tool_calls) > 5:
            tools += f" (+{len(result.tool_calls) - 5} more)"
        console.print(f"  [dim]Tools called:[/dim] {tools}")

    # Show criteria scores if available
    if result.judge_result.criteria_scores:
        scores_text = ", ".join(
            f"{k}: {v:.1f}" for k, v in result.judge_result.criteria_scores.items()
        )
        console.print(f"  [dim]Criteria:[/dim] {scores_text}")

    console.print()


# ---------------------------------------------------------------------------
# Live CLI reporter
# ---------------------------------------------------------------------------


class LiveReporter:
    """Real-time output for the standalone eval runner."""

    def __init__(self, console: Console, *, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self._passed = 0
        self._failed = 0

    def suite_header(self, suite_stem: str, case_count: int, agent_type: str) -> None:
        self.console.print(
            f"\n[bold]{suite_stem}[/bold] ({case_count} case{'s' if case_count != 1 else ''}, agent: {agent_type})"
        )
        self.console.print("  Setting up agent...")

    def case_start(self, case_id: str) -> Status:
        return self.console.status(f"  Running {case_id}...")

    def case_result(
        self, suite_stem: str, case_id: str, result: EvalResult, duration: float
    ) -> None:
        score = result.score
        secs = f"{duration:.1f}s"

        if result.is_judge_error:
            tag = "[bold yellow]ERROR[/bold yellow]"
            self._failed += 1
        elif result.passed:
            tag = "[bold green]PASS [/bold green]"
            self._passed += 1
        else:
            tag = "[bold red]FAIL [/bold red]"
            self._failed += 1

        full_id = f"{suite_stem}::{case_id}"
        self.console.print(f"  {tag}  {full_id:<40s} {score:.2f}  {secs}")

        # Show failure reasoning inline
        if not result.passed and result.judge_result.reasoning:
            reason = _truncate(result.judge_result.reasoning, 72)
            self.console.print(f"        [dim]{reason}[/dim]")

    def summary(self, total_duration: float) -> None:
        total = self._passed + self._failed
        pct = (self._passed / total * 100) if total else 0
        style = "bold green" if self._failed == 0 else "bold red"

        self.console.print()
        self.console.print(
            f"[{style}]{self._passed} passed, {self._failed} failed[/{style}] "
            f"| {pct:.1f}% | {total_duration:.1f}s"
        )
