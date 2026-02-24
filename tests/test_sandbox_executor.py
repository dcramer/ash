from __future__ import annotations

from ash.sandbox.executor import SandboxExecutor


class _FakeManager:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.removed: list[str] = []
        self.start_should_fail = False
        self.exec_should_fail = False

    async def create_container(self, environment=None) -> str:
        _ = environment
        container_id = f"c{len(self.created) + 1}"
        self.created.append(container_id)
        return container_id

    async def start_container(self, container_id: str) -> None:
        if self.start_should_fail:
            raise RuntimeError("start failed")
        _ = container_id

    async def exec_command(
        self,
        container_id: str,
        command: str,
        command_timeout: int | None = None,
        environment: dict[str, str] | None = None,
        **kwargs,
    ) -> tuple[int, str, str]:
        _ = (
            container_id,
            command,
            command_timeout if command_timeout is not None else kwargs.get("timeout"),
            environment,
            kwargs,
        )
        if self.exec_should_fail:
            raise RuntimeError("exec failed")
        return 0, "ok", ""

    async def remove_container(self, container_id: str, force: bool = True) -> None:
        _ = force
        self.removed.append(container_id)


async def test_execute_ephemeral_container_is_removed_on_success() -> None:
    manager = _FakeManager()
    executor = SandboxExecutor()
    executor._manager = manager
    executor._initialized = True

    result = await executor.execute("echo hi", reuse_container=False)

    assert result.success is True
    assert manager.created == ["c1"]
    assert manager.removed == ["c1"]


async def test_execute_ephemeral_container_is_removed_on_exec_error() -> None:
    manager = _FakeManager()
    manager.exec_should_fail = True
    executor = SandboxExecutor()
    executor._manager = manager
    executor._initialized = True

    result = await executor.execute("echo hi", reuse_container=False)

    assert result.success is False
    assert "exec failed" in result.stderr
    assert manager.created == ["c1"]
    assert manager.removed == ["c1"]


async def test_get_or_create_container_removes_container_when_start_fails() -> None:
    manager = _FakeManager()
    manager.start_should_fail = True
    executor = SandboxExecutor()
    executor._manager = manager
    executor._initialized = True

    result = await executor.execute("echo hi", reuse_container=False)
    assert result.success is False
    assert "start failed" in result.stderr
    assert manager.created == ["c1"]
    assert manager.removed == ["c1"]
