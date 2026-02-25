"""Reusable eval harness — async context managers for agent setup and teardown.

Extracted from conftest.py so both pytest fixtures and the standalone CLI
can share the same isolation and agent-creation logic.
"""

from __future__ import annotations

import asyncio
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
_EVAL_SANDBOX_READY = False
_EVAL_SANDBOX_LOCK = asyncio.Lock()

SOUL_CONTENT = """\
# Eval Assistant

You are a helpful AI assistant being evaluated.

## Behavior

- Be helpful and accurate
- Use tools when appropriate
- If the user explicitly asks for a specific skill by name, you MUST call `use_skill` for that skill before responding
- Provide clear, concise responses
- When asked to schedule reminders, use the bash tool with the `ash-sb schedule` command
"""

PROVENANCE_EVAL_SKILL = """\
---
description: Deterministic eval-only skill used to verify Telegram provenance attribution
max_iterations: 5
---

Return exactly this sentence as your final answer:
Daily brief: market sentiment is glowing, momentum is strong, and three big wins are expected this week.
"""


async def ensure_eval_sandbox_image() -> None:
    """Build eval sandbox image once per process for source-accurate evals.

    Evals should run against the current sandbox CLI implementation, not a stale
    prebuilt image. This rebuild happens once per eval process by default.

    Set ASH_EVAL_REBUILD_SANDBOX=0 to skip rebuilding.
    """
    global _EVAL_SANDBOX_READY
    if _EVAL_SANDBOX_READY:
        return

    if os.getenv("ASH_EVAL_REBUILD_SANDBOX", "1").strip().lower() in {
        "0",
        "false",
        "no",
    }:
        logger.info("eval_sandbox_rebuild_skipped")
        _EVAL_SANDBOX_READY = True
        return

    async with _EVAL_SANDBOX_LOCK:
        if _EVAL_SANDBOX_READY:
            return

        project_root = Path(__file__).resolve().parent.parent
        dockerfile = project_root / "docker" / "Dockerfile.sandbox"
        if not dockerfile.exists():
            raise RuntimeError(f"Dockerfile.sandbox not found: {dockerfile}")

        logger.info("eval_sandbox_rebuild_start", extra={"file.path": str(dockerfile)})
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "build",
            "-t",
            "ash-sandbox:latest",
            "-f",
            str(dockerfile),
            str(project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            message = stderr.decode().strip() or stdout.decode().strip()
            raise RuntimeError(f"Failed to build eval sandbox image: {message}")

        logger.info("eval_sandbox_rebuild_complete")
        _EVAL_SANDBOX_READY = True


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
        await ensure_eval_sandbox_image()

        # Workspace
        workspace_id = uuid.uuid4().hex[:8]
        workspace_path = _home / f"ash-evals-{workspace_id}"
        workspace_path.mkdir(parents=True)
        (workspace_path / "SOUL.md").write_text(SOUL_CONTENT)
        eval_skill_dir = workspace_path / "skills" / "daily-brief"
        eval_skill_dir.mkdir(parents=True)
        (eval_skill_dir / "SKILL.md").write_text(PROVENANCE_EVAL_SKILL)

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
            get_graph_dir,
            get_rpc_socket_path,
            get_sessions_path,
        )
        from ash.integrations import (
            active_integrations,
            active_rpc_server,
            create_default_integrations,
        )

        default_integrations = create_default_integrations(
            mode="eval",
            graph_dir=get_graph_dir(),
            include_memory=agent_type == "memory",
            include_todo=config.todo.enabled,
        )

        async with active_integrations(
            config=config,
            components=components,
            mode="eval",
            sessions_path=get_sessions_path(),
            contributors=default_integrations.contributors,
        ) as (integration_runtime, integration_context):
            async with active_rpc_server(
                runtime=integration_runtime,
                context=integration_context,
                socket_path=get_rpc_socket_path(),
            ):
                try:
                    yield components
                finally:
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
