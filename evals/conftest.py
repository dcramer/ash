"""Eval-specific pytest fixtures."""

import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env and .env.local files
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")
load_dotenv(_project_root / ".env.local", override=True)

# Handle raw API key in .env.local (no KEY= prefix)
_env_local = _project_root / ".env.local"
if _env_local.exists() and not os.environ.get("OPENAI_API_KEY"):
    content = _env_local.read_text().strip()
    if content and "=" not in content:
        os.environ["OPENAI_API_KEY"] = content

from ash.core.agent import AgentComponents
from ash.llm import LLMProvider, OpenAIProvider
from evals.harness import eval_agent_context


def pytest_configure(config: pytest.Config) -> None:
    """Register eval marker and configure logging for eval visibility."""
    from evals.harness import configure_eval_logging

    config.addinivalue_line("markers", "eval: marks tests as evaluation tests")
    configure_eval_logging(verbose=True)


@pytest.fixture(scope="session", autouse=True)
def _require_docker() -> None:
    """Fail fast if Docker is not available."""
    from evals.harness import check_prerequisites

    err = check_prerequisites()
    if err:
        pytest.skip(err)


@pytest.fixture(autouse=True)
def _isolate_ash_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    """Safety net: ensure ALL evals use a temporary ASH_HOME."""
    home = tmp_path / ".ash"
    home.mkdir()
    monkeypatch.setenv("ASH_HOME", str(home))

    from ash.config.paths import get_ash_home

    get_ash_home.cache_clear()
    yield home
    get_ash_home.cache_clear()


@pytest.fixture
def real_llm() -> LLMProvider:
    """Create a real OpenAI LLM provider.

    Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    return OpenAIProvider(api_key=api_key)


@pytest.fixture
def judge_llm(real_llm: LLMProvider) -> LLMProvider:
    """LLM provider for judging (defaults to real_llm)."""
    return real_llm


@pytest.fixture
async def eval_agent() -> AsyncGenerator[AgentComponents, None]:
    """Create a fully configured default agent for eval testing."""
    async with eval_agent_context("default") as components:
        yield components


@pytest.fixture
async def eval_memory_agent() -> AsyncGenerator[AgentComponents, None]:
    """Create agent with memory (embeddings + extraction) for memory evals."""
    async with eval_agent_context("memory") as components:
        yield components
