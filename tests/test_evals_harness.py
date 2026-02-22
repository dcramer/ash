from __future__ import annotations

from pathlib import Path

import pytest

import evals.harness as harness


@pytest.mark.asyncio
async def test_ensure_eval_sandbox_image_builds_once(monkeypatch, tmp_path: Path):
    calls: list[tuple] = []

    class _Proc:
        returncode = 0

        async def communicate(self):
            return (b"ok", b"")

    async def _fake_create_subprocess_exec(*args, **kwargs):
        calls.append(args)
        _ = kwargs
        return _Proc()

    project_root = tmp_path / "proj"
    docker_dir = project_root / "docker"
    docker_dir.mkdir(parents=True)
    (docker_dir / "Dockerfile.sandbox").write_text("FROM alpine:3.20\n")
    fake_file = project_root / "evals" / "harness.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("# fake harness path anchor\n")

    monkeypatch.setenv("ASH_EVAL_REBUILD_SANDBOX", "1")
    monkeypatch.setattr(harness, "_EVAL_SANDBOX_READY", False)
    monkeypatch.setattr(harness, "__file__", str(fake_file))
    monkeypatch.setattr(
        harness.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    await harness.ensure_eval_sandbox_image()
    await harness.ensure_eval_sandbox_image()

    assert len(calls) == 1
    assert calls[0][:4] == ("docker", "build", "-t", "ash-sandbox:latest")


@pytest.mark.asyncio
async def test_ensure_eval_sandbox_image_can_be_skipped(monkeypatch):
    called = False

    async def _fake_create_subprocess_exec(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("should not be called when skip env is set")

    monkeypatch.setenv("ASH_EVAL_REBUILD_SANDBOX", "0")
    monkeypatch.setattr(harness, "_EVAL_SANDBOX_READY", False)
    monkeypatch.setattr(
        harness.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    await harness.ensure_eval_sandbox_image()

    assert called is False
