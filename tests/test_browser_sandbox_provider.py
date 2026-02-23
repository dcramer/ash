from __future__ import annotations

from typing import cast

from ash.browser.providers.sandbox import SandboxBrowserProvider
from ash.sandbox.executor import ExecutionResult, SandboxExecutor


class _FakeExecutor:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self._python_calls = 0

    async def execute(
        self,
        command: str,
        deadline_seconds: int | None = None,
        reuse_container: bool = True,
        environment: dict[str, str] | None = None,
        **kwargs,
    ) -> ExecutionResult:
        _ = (deadline_seconds, reuse_container, environment, kwargs)
        self.commands.append(command)

        if "nohup chromium" in command:
            return ExecutionResult(exit_code=0, stdout="12345\n", stderr="")
        if "/json/version" in command:
            return ExecutionResult(exit_code=0, stdout="ok\n", stderr="")
        if "python -c" in command:
            self._python_calls += 1
        if self._python_calls == 1:
            return ExecutionResult(
                exit_code=0,
                stdout='{"url":"https://example.com","title":"Example","html":"<html></html>"}\n',
                stderr="",
            )
        if self._python_calls == 2:
            return ExecutionResult(exit_code=0, stdout='{"text":"Hello"}\n', stderr="")
        if self._python_calls == 3:
            return ExecutionResult(
                exit_code=0, stdout='{"title":"Example"}\n', stderr=""
            )
        if self._python_calls == 7:
            return ExecutionResult(
                exit_code=0,
                stdout='{"image_b64":"aGVsbG8="}\n',
                stderr="",
            )
        return ExecutionResult(exit_code=0, stdout="{}\n", stderr="")


async def test_sandbox_provider_uses_executor_for_full_flow() -> None:
    executor = _FakeExecutor()
    provider = SandboxBrowserProvider(executor=cast(SandboxExecutor, executor))

    started = await provider.start_session(session_id="s1", profile_name=None)
    assert started.provider_session_id == "s1"

    goto = await provider.goto(
        provider_session_id="s1",
        url="https://example.com",
        timeout_seconds=2.0,
    )
    assert goto.url == "https://example.com"
    assert goto.title == "Example"

    extract_text = await provider.extract(
        provider_session_id="s1",
        html=None,
        mode="text",
        selector=None,
        max_chars=100,
    )
    assert extract_text.data["text"] == "Hello"

    extract_title = await provider.extract(
        provider_session_id="s1",
        html=None,
        mode="title",
        selector=None,
        max_chars=100,
    )
    assert extract_title.data["title"] == "Example"

    await provider.click(provider_session_id="s1", selector="#a")
    await provider.type(
        provider_session_id="s1",
        selector="#q",
        text="hello",
        clear_first=True,
    )
    await provider.wait_for(
        provider_session_id="s1",
        selector="#ready",
        timeout_seconds=1.0,
    )

    shot = await provider.screenshot(provider_session_id="s1")
    assert shot.mime_type == "image/png"
    assert shot.image_bytes == b"hello"

    await provider.close_session(provider_session_id="s1")
    assert any("nohup chromium" in cmd for cmd in executor.commands)
    assert any("/json/version" in cmd for cmd in executor.commands)


async def test_sandbox_provider_requires_executor() -> None:
    provider = SandboxBrowserProvider(executor=None)
    try:
        await provider.start_session(session_id="s1", profile_name=None)
    except ValueError as e:
        assert "sandbox_executor_required" in str(e)
    else:
        raise AssertionError("expected sandbox_executor_required")
