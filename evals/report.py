"""Rich terminal reporting for eval results."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from evals.runner import EvalReport, EvalResult


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_status(passed: bool) -> Text:
    """Format pass/fail status with color."""
    if passed:
        return Text("PASS", style="bold green")
    return Text("FAIL", style="bold red")


def _format_score(score: float) -> Text:
    """Format score with color based on value."""
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
    summary.append("\n")
    summary.append(f"Accuracy: {accuracy_pct:.1f}%", style=accuracy_style)
    summary.append(f" | Average Score: {report.average_score:.2f}")

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
            _format_score(result.score),
            _format_status(result.passed),
        )

    console.print(table)

    # Failed case details
    failed = report.failed_cases()
    if failed:
        console.print("\n[bold red]Failed Cases:[/bold red]\n")

        for result in failed:
            _print_failed_case(result, console)


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

    console.print()
