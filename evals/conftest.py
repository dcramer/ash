"""Eval-specific pytest fixtures."""

import os
from collections.abc import AsyncGenerator
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
from ash.config.models import ModelConfig
from ash.config.workspace import Workspace
from ash.core.agent import Agent, AgentConfig
from ash.core.prompt import RuntimeInfo, SystemPromptBuilder
from ash.llm import AnthropicProvider, LLMProvider
from ash.skills import SkillRegistry
from ash.tools import ToolExecutor, ToolRegistry


def pytest_configure(config: pytest.Config) -> None:
    """Register eval marker."""
    config.addinivalue_line("markers", "eval: marks tests as evaluation tests")


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
def eval_config() -> AshConfig:
    """Minimal AshConfig for eval runs."""
    return AshConfig(
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
    )


@pytest.fixture
def eval_workspace(tmp_path: Path) -> Workspace:
    """Create a temporary workspace with basic SOUL.md."""
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    soul_content = """# Eval Assistant

You are a helpful AI assistant being evaluated.

## Behavior

- Be helpful and accurate
- Use tools when appropriate
- Provide clear, concise responses
"""
    (workspace_path / "SOUL.md").write_text(soul_content)

    return Workspace(path=workspace_path, soul=soul_content)


@pytest.fixture
def eval_tool_registry() -> ToolRegistry:
    """Create a minimal tool registry for evals.

    Note: For evals that test tool use, you may want to add specific tools.
    """
    return ToolRegistry()


@pytest.fixture
def eval_skill_registry() -> SkillRegistry:
    """Create an empty skill registry for evals."""
    return SkillRegistry()


@pytest.fixture
def eval_prompt_builder(
    eval_workspace: Workspace,
    eval_tool_registry: ToolRegistry,
    eval_skill_registry: SkillRegistry,
    eval_config: AshConfig,
) -> SystemPromptBuilder:
    """Create a prompt builder for evals."""
    return SystemPromptBuilder(
        workspace=eval_workspace,
        tool_registry=eval_tool_registry,
        skill_registry=eval_skill_registry,
        config=eval_config,
    )


@pytest.fixture
def eval_runtime() -> RuntimeInfo:
    """Create runtime info for evals."""
    return RuntimeInfo.from_environment(
        model="claude-sonnet-4-5",
        provider="anthropic",
        timezone="UTC",
    )


@pytest.fixture
async def eval_agent(
    real_llm: LLMProvider,
    eval_tool_registry: ToolRegistry,
    eval_prompt_builder: SystemPromptBuilder,
    eval_runtime: RuntimeInfo,
) -> AsyncGenerator[Agent, None]:
    """Create a fully configured agent for eval testing.

    This creates a minimal agent without sandbox, memory, or other
    heavyweight components - suitable for basic behavior evals.
    """
    tool_executor = ToolExecutor(eval_tool_registry)

    agent = Agent(
        llm=real_llm,
        tool_executor=tool_executor,
        prompt_builder=eval_prompt_builder,
        runtime=eval_runtime,
        config=AgentConfig(
            model="claude-sonnet-4-5",
            temperature=0.7,
            max_tool_iterations=10,
        ),
    )

    yield agent
