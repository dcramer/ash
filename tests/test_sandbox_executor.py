from __future__ import annotations

from pathlib import Path

from ash.sandbox.executor import SandboxExecutor


class _FakeContainer:
    def __init__(
        self,
        *,
        container_id: str,
        image: str,
        status: str = "running",
    ) -> None:
        self.id = container_id
        self.status = status
        self.attrs = {"Config": {"Image": image}}


class _FakeManager:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.removed: list[str] = []
        self.start_should_fail = False
        self.exec_should_fail = False
        self._containers: dict[str, _FakeContainer] = {}
        self._by_name: dict[str, str] = {}

    async def create_container(self, name=None, environment=None) -> str:
        _ = environment
        if name and name in self._by_name:
            raise RuntimeError("name already in use")
        container_id = f"c{len(self.created) + 1}"
        self.created.append(container_id)
        container = _FakeContainer(
            container_id=container_id, image="ash-sandbox:latest"
        )
        self._containers[container_id] = container
        if name:
            self._by_name[name] = container_id
        return container_id

    async def start_container(self, container_id: str) -> None:
        if self.start_should_fail:
            raise RuntimeError("start failed")
        container = self._containers.get(container_id)
        if container:
            container.status = "running"

    async def get_container(self, container_ref: str):
        container = self._containers.get(container_ref)
        if container is not None:
            return container
        container_id = self._by_name.get(container_ref)
        if container_id:
            return self._containers.get(container_id)
        return None

    async def get_container_status(self, container_ref: str) -> str | None:
        container = await self.get_container(container_ref)
        return container.status if container else None

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
        self._containers.pop(container_id, None)
        for name, cid in list(self._by_name.items()):
            if cid == container_id:
                self._by_name.pop(name, None)


def _patch_runtime_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ash.sandbox.executor.get_run_path", lambda: tmp_path / "run")
    monkeypatch.setattr("ash.sandbox.executor.get_ash_home", lambda: tmp_path / "home")


async def test_execute_ephemeral_container_is_removed_on_success(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_runtime_paths(monkeypatch, tmp_path)
    manager = _FakeManager()
    executor = SandboxExecutor()
    executor._manager = manager
    executor._initialized = True

    result = await executor.execute("echo hi", reuse_container=False)

    assert result.success is True
    assert manager.created == ["c1"]
    assert manager.removed == ["c1"]


async def test_execute_ephemeral_container_is_removed_on_exec_error(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_runtime_paths(monkeypatch, tmp_path)
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


async def test_get_or_create_container_removes_container_when_start_fails(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_runtime_paths(monkeypatch, tmp_path)
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


async def test_reuse_container_reconciles_from_state_file(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_runtime_paths(monkeypatch, tmp_path)
    manager = _FakeManager()
    shared_name = "ash-sandbox-deadbeef"
    manager._containers["preexisting"] = _FakeContainer(
        container_id="preexisting", image="ash-sandbox:latest", status="running"
    )
    manager._by_name[shared_name] = "preexisting"

    executor = SandboxExecutor()
    executor._manager = manager
    executor._initialized = True
    monkeypatch.setattr(executor, "_managed_container_name", lambda: shared_name)

    # First call should attach existing instead of creating.
    result = await executor.execute("echo hi", reuse_container=True)
    assert result.success is True
    assert manager.created == []
    assert executor._container_id == "preexisting"

    # Second call should reuse same container id.
    result2 = await executor.execute("echo hi", reuse_container=True)
    assert result2.success is True
    assert manager.created == []


async def test_reuse_container_prunes_image_mismatch_then_recreates(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_runtime_paths(monkeypatch, tmp_path)
    manager = _FakeManager()
    shared_name = "ash-sandbox-deadbeef"
    manager._containers["stale"] = _FakeContainer(
        container_id="stale", image="old-image-id", status="running"
    )
    manager._by_name[shared_name] = "stale"

    executor = SandboxExecutor()
    executor._manager = manager
    executor._initialized = True
    monkeypatch.setattr(executor, "_managed_container_name", lambda: shared_name)

    result = await executor.execute("echo hi", reuse_container=True)
    assert result.success is True
    assert "stale" in manager.removed
    assert manager.created == ["c1"]
