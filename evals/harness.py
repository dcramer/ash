"""Reusable eval harness — async context managers for agent setup and teardown.

Extracted from conftest.py so both pytest fixtures and the standalone CLI
can share the same isolation and agent-creation logic.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from ash.config import AshConfig
from ash.config.models import (
    EmbeddingsConfig,
    MemoryConfig,
    ModelConfig,
    SandboxConfig,
)
from ash.config.workspace import WorkspaceLoader
from ash.core.agent import AgentComponents, create_agent
from evals.types import EvalCase, EvalSuite

logger = logging.getLogger(__name__)

SOUL_CONTENT = """\
# Eval Assistant

You are a helpful AI assistant being evaluated.

## Behavior

- Be helpful and accurate
- Use tools when appropriate
- Provide clear, concise responses
- When asked to schedule reminders, use the bash tool with the `ash schedule` command
"""


def _build_config(agent_type: str, workspace_path: Path) -> AshConfig:
    """Build an AshConfig for the given agent type.

    ``"default"`` disables extraction; ``"memory"`` enables embeddings +
    extraction with zero debounce for fast eval turnaround.
    """
    models = {
        "default": ModelConfig(
            provider="openai",
            model="gpt-5.2",
            temperature=0.7,
        ),
        "mini": ModelConfig(
            provider="openai",
            model="gpt-5-mini",
            temperature=0,
        ),
    }
    sandbox = SandboxConfig(workspace_access="rw")

    if agent_type == "memory":
        return AshConfig(
            workspace=workspace_path,
            models=models,
            sandbox=sandbox,
            embeddings=EmbeddingsConfig(),
            memory=MemoryConfig(
                extraction_enabled=True,
                extraction_min_message_length=10,
                extraction_debounce_seconds=0,
                compaction_enabled=False,
            ),
        )

    return AshConfig(
        workspace=workspace_path,
        models=models,
        sandbox=sandbox,
        memory=MemoryConfig(
            extraction_enabled=False,
            compaction_enabled=False,
        ),
    )


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def isolated_ash_home() -> AsyncGenerator[Path, None]:
    """Temporary ASH_HOME with automatic cleanup.

    Sets the ``ASH_HOME`` env var, clears the ``get_ash_home`` cache, and
    restores everything on exit.
    """
    from ash.config.paths import get_ash_home

    original = os.environ.get("ASH_HOME")
    tmp = tempfile.mkdtemp(prefix="ash-eval-")
    tmp_path = Path(tmp)

    os.environ["ASH_HOME"] = tmp
    get_ash_home.cache_clear()

    try:
        yield tmp_path
    finally:
        if original is None:
            os.environ.pop("ASH_HOME", None)
        else:
            os.environ["ASH_HOME"] = original
        get_ash_home.cache_clear()

        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


@asynccontextmanager
async def eval_agent_context(agent_type: str) -> AsyncGenerator[AgentComponents, None]:
    """Full agent lifecycle for a single eval case.

    Creates an isolated ASH_HOME, workspace, graph dir, config, agent, and
    RPC server.  Tears everything down on exit.
    """
    # Eval harness boundary.
    # Spec contract: specs/subsystems.md (Integration Hooks).
    async with isolated_ash_home() as _home:
        # Workspace
        workspace_id = uuid.uuid4().hex[:8]
        workspace_path = _home / f"ash-evals-{workspace_id}"
        workspace_path.mkdir(parents=True)
        (workspace_path / "SOUL.md").write_text(SOUL_CONTENT)

        workspace = WorkspaceLoader(workspace_path).load()

        # Graph dir
        graph_dir = _home / "graph"
        graph_dir.mkdir()
        (graph_dir / "embeddings").mkdir()

        # Config + agent
        config = _build_config(agent_type, workspace_path)
        components = await create_agent(
            config=config,
            workspace=workspace,
            graph_dir=graph_dir,
        )
        from ash.config.paths import (
            get_rpc_socket_path,
            get_schedule_file,
            get_sessions_path,
        )
        from ash.integrations import (
            MemoryIntegration,
            SchedulingIntegration,
            active_integrations,
        )
        from ash.rpc.server import RPCServer

        schedule_integration = SchedulingIntegration(get_schedule_file())
        contributors = [schedule_integration]
        if agent_type == "memory":
            contributors.append(MemoryIntegration())

        async with active_integrations(
            config=config,
            components=components,
            mode="eval",
            sessions_path=get_sessions_path(),
            contributors=contributors,
        ) as (integration_runtime, integration_context):
            rpc_server = RPCServer(get_rpc_socket_path())
            integration_runtime.register_rpc_methods(rpc_server, integration_context)
            await rpc_server.start()

            try:
                yield components
            finally:
                await rpc_server.stop()
                if components.sandbox_executor:
                    await components.sandbox_executor.cleanup()


# ---------------------------------------------------------------------------
# Filter / discovery helpers
# ---------------------------------------------------------------------------


def resolve_filter(
    suites: list[tuple[str, EvalSuite]],
    filter_str: str | None = None,
    tag: str | None = None,
) -> list[tuple[str, EvalSuite, list[EvalCase]]]:
    """Return matched ``(suite_stem, suite, cases)`` triples.

    *filter_str* formats:
    - ``"memory"`` — match suite filename stem
    - ``"memory::recall_cross"`` — match ``suite_stem::case_id``

    *tag* filters cases by their ``tags`` list.
    """
    results: list[tuple[str, EvalSuite, list[EvalCase]]] = []

    for stem, suite in suites:
        if filter_str:
            if "::" in filter_str:
                f_stem, f_case = filter_str.split("::", 1)
                if stem != f_stem:
                    continue
                cases = [c for c in suite.cases if c.id == f_case]
            else:
                if stem != filter_str:
                    continue
                cases = list(suite.cases)
        else:
            cases = list(suite.cases)

        if tag:
            cases = [c for c in cases if tag in c.tags]

        if cases:
            results.append((stem, suite, cases))

    return results


# ---------------------------------------------------------------------------
# Prerequisites & logging
# ---------------------------------------------------------------------------


def check_prerequisites() -> str | None:
    """Check Docker availability and OPENAI_API_KEY. Returns error message or None."""
    import docker

    try:
        client = docker.from_env()
        client.ping()
    except Exception:
        return "Docker is not running — start Docker Desktop and retry"

    if not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY environment variable is not set"

    return None


def configure_eval_logging(verbose: bool = False) -> None:
    """Set up logging for eval runs mirroring conftest's handler."""
    from ash.logging import ComponentFormatter

    level = logging.DEBUG if verbose else logging.WARNING

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        ComponentFormatter("%(levelname)s %(context)s%(component)s | %(message)s")
    )

    for module in [
        "ash",
        "evals",
    ]:
        mod_logger = logging.getLogger(module)
        mod_logger.setLevel(level)
        mod_logger.addHandler(handler)
