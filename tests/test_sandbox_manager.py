from pathlib import Path
from typing import Any, cast

from ash.sandbox.manager import SandboxConfig, SandboxManager


class _CreatedContainer:
    id = "test-container-id"


class _FakeContainers:
    def __init__(self) -> None:
        self.last_create_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_create_kwargs = kwargs
        return _CreatedContainer()


class _FakeDockerClient:
    def __init__(self) -> None:
        self.containers = _FakeContainers()


async def test_create_container_mounts_run_dir_for_rpc_socket(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    rpc_socket = run_dir / "rpc.sock"

    manager = SandboxManager(config=SandboxConfig(rpc_socket_path=rpc_socket))
    fake_client = _FakeDockerClient()
    manager._client = cast(Any, fake_client)

    async def _ensure_client():
        return fake_client

    manager._ensure_client = _ensure_client  # type: ignore[method-assign]

    await manager.create_container()

    assert fake_client.containers.last_create_kwargs is not None
    volumes = fake_client.containers.last_create_kwargs["volumes"]
    env = fake_client.containers.last_create_kwargs["environment"]
    assert str(run_dir) in volumes
    assert volumes[str(run_dir)]["bind"] == "/ash/run"
    assert env["ASH_RPC_SOCKET"] == "/ash/run/rpc.sock"


async def test_create_container_skips_rpc_mount_when_run_dir_missing(
    tmp_path: Path,
) -> None:
    rpc_socket = tmp_path / "missing" / "rpc.sock"
    manager = SandboxManager(config=SandboxConfig(rpc_socket_path=rpc_socket))
    fake_client = _FakeDockerClient()
    manager._client = cast(Any, fake_client)

    async def _ensure_client():
        return fake_client

    manager._ensure_client = _ensure_client  # type: ignore[method-assign]

    await manager.create_container()

    assert fake_client.containers.last_create_kwargs is not None
    volumes = fake_client.containers.last_create_kwargs.get("volumes", {})
    env = fake_client.containers.last_create_kwargs.get("environment", {})
    assert str(rpc_socket.parent) not in volumes
    assert "ASH_RPC_SOCKET" not in env
