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


async def test_create_container_preserves_explicit_environment_values(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    rpc_socket = run_dir / "rpc.sock"

    manager = SandboxManager(config=SandboxConfig(rpc_socket_path=rpc_socket))
    fake_client = _FakeDockerClient()
    manager._client = cast(Any, fake_client)

    async def _ensure_client():
        return fake_client

    manager._ensure_client = _ensure_client  # type: ignore[method-assign]

    await manager.create_container(
        environment={
            "ASH_RPC_HOST": "host.docker.internal",
            "ASH_RPC_PORT": "50222",
            "OTHER_VAR": "yes",
        }
    )

    assert fake_client.containers.last_create_kwargs is not None
    env = fake_client.containers.last_create_kwargs["environment"]
    assert env["ASH_RPC_HOST"] == "host.docker.internal"
    assert env["ASH_RPC_PORT"] == "50222"
    assert env["OTHER_VAR"] == "yes"


class _FakeExecAPI:
    def __init__(self) -> None:
        self.last_exec_create_kwargs: dict[str, Any] | None = None

    def exec_create(self, _container_id: str, **kwargs: Any) -> dict[str, str]:
        self.last_exec_create_kwargs = kwargs
        return {"Id": "exec-1"}

    def exec_start(
        self,
        _exec_id: str,
        *,
        demux: bool,
    ) -> tuple[bytes | None, bytes | None]:
        assert demux is True
        return (b"ok", b"")

    def exec_inspect(self, _exec_id: str) -> dict[str, int]:
        return {"ExitCode": 0}

    def inspect_container(self, _container_id: str) -> dict[str, Any]:
        return {
            "Config": {
                "Env": [
                    "ASH_RPC_SOCKET=/ash/run/rpc.sock",
                    "ASH_RPC_HOST=host.docker.internal",
                    "ASH_RPC_PORT=50000",
                ]
            }
        }


class _FakeContainerForExec:
    def __init__(self) -> None:
        self.id = "container-1"
        self.attrs = {
            "Config": {
                "Env": [
                    "ASH_RPC_SOCKET=/ash/run/rpc.sock",
                    "ASH_RPC_HOST=host.docker.internal",
                    "ASH_RPC_PORT=50000",
                ]
            }
        }


class _FakeDockerClientForExec:
    def __init__(self) -> None:
        self.api = _FakeExecAPI()

    class containers:
        @staticmethod
        def get(_container_id: str) -> Any:
            raise AssertionError("container lookup should not be needed")


async def test_exec_command_preserves_container_rpc_env_when_overriding(
    tmp_path: Path,
) -> None:
    manager = SandboxManager(config=SandboxConfig())
    fake_client = _FakeDockerClientForExec()
    manager._client = cast(Any, fake_client)
    manager._containers["container-1"] = cast(Any, _FakeContainerForExec())

    async def _ensure_client():
        return fake_client

    manager._ensure_client = _ensure_client  # type: ignore[method-assign]

    exit_code, stdout, stderr = await manager.exec_command(
        "container-1",
        "env",
        environment={
            "ASH_CONTEXT_TOKEN": "token-value",
            "EXTRA_VAR": "1",
        },
    )

    assert exit_code == 0
    assert stdout == "ok"
    assert stderr == ""
    assert fake_client.api.last_exec_create_kwargs is not None

    env_list = fake_client.api.last_exec_create_kwargs["environment"]
    env_map = dict(item.split("=", 1) for item in env_list)
    assert env_map["ASH_RPC_SOCKET"] == "/ash/run/rpc.sock"
    assert env_map["ASH_RPC_HOST"] == "host.docker.internal"
    assert env_map["ASH_RPC_PORT"] == "50000"
    assert env_map["ASH_CONTEXT_TOKEN"] == "token-value"
    assert env_map["EXTRA_VAR"] == "1"
