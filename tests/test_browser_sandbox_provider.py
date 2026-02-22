from __future__ import annotations

import pytest

from ash.browser.providers.sandbox import SandboxBrowserProvider, _SandboxSession


class _FakeLocator:
    def __init__(self, text: str = "locator-text") -> None:
        self.first = self
        self._text = text
        self.clicked = False
        self.filled: list[str] = []
        self.typed: list[str] = []

    async def click(self, **kwargs) -> None:
        _ = kwargs
        self.clicked = True

    async def fill(self, value: str) -> None:
        self.filled.append(value)

    async def type(self, value: str, *, delay: int) -> None:
        _ = delay
        self.typed.append(value)

    async def inner_text(self, **kwargs) -> str:
        _ = kwargs
        return self._text


class _FakePage:
    def __init__(self) -> None:
        self.url = "about:blank"
        self._title = "Example"
        self._html = "<html><body>Hello world</body></html>"
        self._text = "Hello world"
        self._locator = _FakeLocator()
        self.waited_selectors: list[str] = []

    def locator(self, selector: str) -> _FakeLocator:
        _ = selector
        return self._locator

    async def goto(self, url: str, **kwargs) -> None:
        _ = kwargs
        self.url = url

    async def title(self) -> str:
        return self._title

    async def content(self) -> str:
        return self._html

    async def evaluate(self, script: str) -> str:
        _ = script
        return self._text

    async def wait_for_selector(self, selector: str, **kwargs) -> None:
        _ = kwargs
        self.waited_selectors.append(selector)

    async def screenshot(self, *, type: str, full_page: bool) -> bytes:
        _ = (type, full_page)
        return b"png-bytes"

    async def close(self) -> None:
        return None


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    async def new_page(self) -> _FakePage:
        return self._page

    async def close(self) -> None:
        return None


class _FakeBrowser:
    def __init__(self, context: _FakeContext) -> None:
        self._context = context

    async def new_context(self, *, viewport: dict[str, int]) -> _FakeContext:
        _ = viewport
        return self._context

    async def close(self) -> None:
        return None


class _FakePlaywright:
    def __init__(self, browser: _FakeBrowser) -> None:
        self.chromium = self
        self._browser = browser

    async def launch(self, **kwargs) -> _FakeBrowser:
        _ = kwargs
        return self._browser

    async def stop(self) -> None:
        return None


@pytest.mark.asyncio
async def test_sandbox_provider_full_action_flow() -> None:
    provider = SandboxBrowserProvider()
    page = _FakePage()
    context = _FakeContext(page)
    browser = _FakeBrowser(context)
    playwright = _FakePlaywright(browser)

    async def _fake_create_session() -> _SandboxSession:
        return _SandboxSession(
            playwright=playwright,
            browser=browser,
            context=context,
            page=page,
        )

    provider._create_session = _fake_create_session  # type: ignore[method-assign]

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
    assert extract_text.data["text"] == "Hello world"

    extract_title = await provider.extract(
        provider_session_id="s1",
        html=None,
        mode="title",
        selector=None,
        max_chars=100,
    )
    assert extract_title.data["title"] == "Example"

    await provider.click(provider_session_id="s1", selector="#a")
    assert page._locator.clicked is True

    await provider.type(
        provider_session_id="s1",
        selector="#q",
        text="hello",
        clear_first=True,
    )
    assert page._locator.filled == [""]
    assert page._locator.typed == ["hello"]

    await provider.wait_for(
        provider_session_id="s1",
        selector="#ready",
        timeout_seconds=1.0,
    )
    assert page.waited_selectors == ["#ready"]

    shot = await provider.screenshot(provider_session_id="s1")
    assert shot.mime_type == "image/png"
    assert shot.image_bytes == b"png-bytes"

    await provider.close_session(provider_session_id="s1")


@pytest.mark.asyncio
async def test_sandbox_provider_missing_session() -> None:
    provider = SandboxBrowserProvider()
    with pytest.raises(ValueError, match="session_not_found"):
        await provider.screenshot(provider_session_id="missing")
