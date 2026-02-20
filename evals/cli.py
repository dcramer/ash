"""Standalone eval CLI — ``uv run evals``."""

import asyncio
import os
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evals.harness import (
    check_prerequisites,
    configure_eval_logging,
    eval_agent_context,
    resolve_filter,
)
from evals.report import LiveReporter
from evals.runner import discover_eval_suites, load_eval_suite, run_yaml_eval_case
from evals.types import EvalConfig, EvalSuite

app = typer.Typer(name="evals", add_completion=False)
console = Console()


def _load_env() -> None:
    """Load .env / .env.local, matching conftest behaviour."""
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")
    load_dotenv(project_root / ".env.local", override=True)

    env_local = project_root / ".env.local"
    if env_local.exists() and not os.environ.get("OPENAI_API_KEY"):
        content = env_local.read_text().strip()
        if content and "=" not in content:
            os.environ["OPENAI_API_KEY"] = content


def _discover_all() -> list[tuple[str, EvalSuite]]:
    """Discover and load all eval suites, returning (stem, suite) pairs."""
    paths = discover_eval_suites()
    result: list[tuple[str, EvalSuite]] = []
    for p in paths:
        suite = load_eval_suite(p)
        result.append((p.stem, suite))
    return result


@app.command("list")
def list_(
    tag: str | None = typer.Option(None, "--tag", "-t", help="Filter cases by tag"),
) -> None:
    """List available eval cases."""
    suites = _discover_all()
    matched = resolve_filter(suites, filter_str=None, tag=tag)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Suite", style="cyan", no_wrap=True)
    table.add_column("Case ID", style="white")
    table.add_column("Agent", style="magenta")
    table.add_column("Tags", style="dim")
    table.add_column("Description", max_width=50)

    for stem, suite, cases in matched:
        for case in cases:
            agent = case.agent or suite.defaults.agent
            tags = ", ".join(case.tags) if case.tags else ""
            desc = case.description[:50] if case.description else ""
            table.add_row(stem, case.id, agent, tags, desc)

    console.print(table)


@app.command()
def run(
    filter_str: str | None = typer.Argument(None, help="Suite or suite::case filter"),
    tag: str | None = typer.Option(None, "--tag", "-t", help="Filter cases by tag"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs"),
) -> None:
    """Run eval cases."""
    asyncio.run(_run_async(filter_str=filter_str, tag=tag, verbose=verbose))


async def _run_async(
    *,
    filter_str: str | None,
    tag: str | None,
    verbose: bool,
) -> None:
    _load_env()
    configure_eval_logging(verbose)

    err = check_prerequisites()
    if err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        sys.exit(1)

    suites = _discover_all()
    matched = resolve_filter(suites, filter_str=filter_str, tag=tag)

    if not matched:
        console.print("[yellow]No matching eval cases found.[/yellow]")
        sys.exit(0)

    # Judge LLM — one provider reused across all cases
    from ash.llm import OpenAIProvider

    api_key = os.environ["OPENAI_API_KEY"]
    judge_llm = OpenAIProvider(api_key=api_key)

    reporter = LiveReporter(console, verbose=verbose)
    total_start = time.monotonic()
    any_failure = False

    for stem, suite, cases in matched:
        agent_type = suite.defaults.agent
        reporter.suite_header(stem, len(cases), agent_type)

        for case in cases:
            effective_agent = case.agent or agent_type

            case_start = time.monotonic()
            async with eval_agent_context(effective_agent) as components:
                results = await run_yaml_eval_case(
                    components=components,
                    suite=suite,
                    case=case,
                    judge_llm=judge_llm,
                    config=EvalConfig(),
                )
            duration = time.monotonic() - case_start

            # Report each result (multi-turn cases produce multiple)
            for result in results:
                reporter.case_result(stem, result.case.id, result, duration)
                if not result.passed:
                    any_failure = True

    total_duration = time.monotonic() - total_start
    reporter.summary(total_duration)

    if any_failure:
        sys.exit(1)
