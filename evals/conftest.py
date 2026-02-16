"""Eval-specific pytest fixtures."""

import os
import uuid
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
if _env_local.exists() and not os.environ.get("ANTHROPIC_API_KEY"):
    content = _env_local.read_text().strip()
    if content and "=" not in content:
        os.environ["ANTHROPIC_API_KEY"] = content

from ash.config import AshConfig
from ash.config.models import (
    MemoryConfig,
    ModelConfig,
    SandboxConfig,
)
from ash.config.workspace import Workspace, WorkspaceLoader
from ash.core.agent import AgentComponents, create_agent
from ash.llm import AnthropicProvider, LLMProvider


def pytest_configure(config: pytest.Config) -> None:
    """Register eval marker and configure logging for eval visibility."""
    import logging

    config.addinivalue_line("markers", "eval: marks tests as evaluation tests")

    # Add a console handler for key modules so eval runs show what was
    # extracted, recalled, and resolved. This makes flaky eval failures
    # debuggable without re-running.
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    for module in [
        "ash.core.agent",
        "ash.memory.extractor",
        "ash.graph.store",
        "ash.memory.index",
        "evals",
    ]:
        mod_logger = logging.getLogger(module)
        mod_logger.setLevel(logging.DEBUG)
        mod_logger.addHandler(handler)


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
    """Create a real Anthropic LLM provider.

    Requires ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    return AnthropicProvider(api_key=api_key)


@pytest.fixture
def judge_llm(real_llm: LLMProvider) -> LLMProvider:
    """LLM provider for judging (defaults to real_llm).

    Can be overridden to use a different provider/model for judging.
    """
    return real_llm


@pytest.fixture
def eval_workspace_path(tmp_path: Path) -> Path:
    """Create an isolated workspace directory for evals.

    Creates a unique workspace at /tmp/ash-evals-{uuid}/
    """
    workspace_id = uuid.uuid4().hex[:8]
    workspace_path = tmp_path / f"ash-evals-{workspace_id}"
    workspace_path.mkdir(parents=True)

    # Create SOUL.md
    soul_content = """# Eval Assistant

You are a helpful AI assistant being evaluated.

## Behavior

- Be helpful and accurate
- Use tools when appropriate
- Provide clear, concise responses
- When asked to schedule reminders, use the bash tool with the `ash schedule` command
"""
    (workspace_path / "SOUL.md").write_text(soul_content)

    return workspace_path


@pytest.fixture
def eval_workspace(eval_workspace_path: Path) -> Workspace:
    """Load the eval workspace."""
    loader = WorkspaceLoader(eval_workspace_path)
    return loader.load()


@pytest.fixture
def eval_graph_dir(tmp_path: Path) -> Path:
    """Create a temporary graph directory for evals."""
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "embeddings").mkdir()
    return graph_dir


@pytest.fixture
def eval_config(eval_workspace_path: Path, tmp_path: Path) -> AshConfig:
    """Create AshConfig for eval runs with isolated workspace."""
    return AshConfig(
        workspace=eval_workspace_path,
        models={
            "default": ModelConfig(
                provider="anthropic",
                model="claude-sonnet-4-5",
                temperature=0.7,
            ),
            "haiku": ModelConfig(
                provider="anthropic",
                model="claude-haiku-4-5",
                temperature=0,
            ),
        },
        sandbox=SandboxConfig(
            # Use default sandbox settings
            workspace_access="rw",
        ),
        memory=MemoryConfig(
            database_path=tmp_path / "eval.db",
            extraction_enabled=False,  # Disable background extraction for evals
            compaction_enabled=False,  # Disable compaction for evals
        ),
    )


@pytest.fixture
async def eval_agent(
    eval_config: AshConfig,
    eval_workspace: Workspace,
    eval_graph_dir: Path,
) -> AsyncGenerator[AgentComponents, None]:
    """Create a fully configured agent with real tools for eval testing.

    This creates a complete agent with:
    - Real sandbox execution
    - Graph-backed memory store
    - All built-in tools (bash, read_file, write_file, etc.)

    The workspace is isolated to /tmp/ash-evals-{uuid}/ to prevent
    interference with real user data.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    components = await create_agent(
        config=eval_config,
        workspace=eval_workspace,
        graph_dir=eval_graph_dir,
    )

    yield components

    # Cleanup: stop sandbox if running
    if components.sandbox_executor:
        await components.sandbox_executor.cleanup()


@pytest.fixture
def eval_memory_config(eval_workspace_path: Path, tmp_path: Path) -> AshConfig:
    """Create AshConfig with embeddings and extraction enabled for memory evals."""
    from ash.config.models import EmbeddingsConfig

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    return AshConfig(
        workspace=eval_workspace_path,
        models={
            "default": ModelConfig(
                provider="anthropic",
                model="claude-sonnet-4-5",
                temperature=0.7,
            ),
            "haiku": ModelConfig(
                provider="anthropic",
                model="claude-haiku-4-5",
                temperature=0,
            ),
        },
        sandbox=SandboxConfig(
            workspace_access="rw",
        ),
        embeddings=EmbeddingsConfig(),
        memory=MemoryConfig(
            extraction_enabled=True,
            extraction_min_message_length=10,
            extraction_debounce_seconds=0,
            compaction_enabled=False,
        ),
    )


@pytest.fixture
async def eval_memory_agent(
    eval_memory_config: AshConfig,
    eval_workspace: Workspace,
    eval_graph_dir: Path,
) -> AsyncGenerator[AgentComponents, None]:
    """Create agent with memory (embeddings + extraction) for memory evals.

    File stores are isolated via the _isolate_ash_home autouse fixture.
    Requires both ANTHROPIC_API_KEY and OPENAI_API_KEY.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    components = await create_agent(
        config=eval_memory_config,
        workspace=eval_workspace,
        graph_dir=eval_graph_dir,
    )

    yield components

    if components.sandbox_executor:
        await components.sandbox_executor.cleanup()
