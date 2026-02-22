"""Integration tests for sandbox schedule CLI over real RPC socket."""

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from ash_sandbox_cli.commands.schedule import app
from typer.testing import CliRunner

from ash.config.models import AshConfig, ModelConfig
from ash.integrations import (
    IntegrationContext,
    IntegrationRuntime,
    SchedulingIntegration,
)
from ash.integrations.rpc import active_rpc_server


def _runner(env: dict[str, str]) -> CliRunner:
    return CliRunner(env=env)


@pytest.mark.asyncio
async def test_schedule_cli_end_to_end_via_real_rpc(tmp_path: Path) -> None:
    """Create/list schedule entries through real RPC server + schedule integration."""
    schedule_file = tmp_path / "schedule.jsonl"
    socket_path = tmp_path / "rpc.sock"

    config = AshConfig(
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")}
    )
    components = SimpleNamespace(agent=object())
    context = IntegrationContext(
        config=config,
        components=components,  # type: ignore[arg-type]
        mode="chat",
    )
    runtime = IntegrationRuntime([SchedulingIntegration(schedule_file)])
    await runtime.setup(context)

    env = {
        "ASH_RPC_SOCKET": str(socket_path),
        "ASH_PROVIDER": "telegram",
        "ASH_CHAT_ID": "chat-1",
        "ASH_CHAT_TITLE": "Main Room",
        "ASH_USER_ID": "user-1",
        "ASH_USERNAME": "alice",
        "ASH_TIMEZONE": "UTC",
    }
    runner = _runner(env)

    async with active_rpc_server(
        runtime=runtime,
        context=context,
        socket_path=socket_path,
    ):
        create_result = await asyncio.to_thread(
            runner.invoke,
            app,
            ["create", "take out trash", "--cron", "0 8 * * *"],
        )
        assert create_result.exit_code == 0
        assert "Scheduled recurring task" in create_result.stdout

        match = re.search(r"id=([0-9a-f]{8})", create_result.stdout)
        assert match is not None

        list_result = await asyncio.to_thread(runner.invoke, app, ["list"])
        assert list_result.exit_code == 0
        assert "take out trash" in list_result.stdout
        assert match.group(1) in list_result.stdout
