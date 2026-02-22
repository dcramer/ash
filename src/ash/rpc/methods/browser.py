"""Browser RPC method handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.browser import BrowserManager
    from ash.rpc.server import RPCServer


def register_browser_methods(server: RPCServer, manager: BrowserManager) -> None:
    """Register browser RPC methods."""

    async def _execute(action: str, params: dict[str, Any]) -> dict[str, Any]:
        effective_user_id = str(params.get("user_id") or "unknown")
        result = await manager.execute_action(
            action=action,
            effective_user_id=effective_user_id,
            provider_name=(str(params["provider"]) if params.get("provider") else None),
            session_id=(
                str(params["session_id"]) if params.get("session_id") else None
            ),
            session_name=(
                str(params["session_name"]) if params.get("session_name") else None
            ),
            profile_name=(
                str(params["profile_name"]) if params.get("profile_name") else None
            ),
            params=params,
        )
        return result.to_dict()

    async def browser_session_start(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("session.start", params)

    async def browser_session_list(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("session.list", params)

    async def browser_session_show(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("session.show", params)

    async def browser_session_close(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("session.close", params)

    async def browser_session_archive(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("session.archive", params)

    async def browser_page_goto(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("page.goto", params)

    async def browser_page_extract(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("page.extract", params)

    async def browser_page_click(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("page.click", params)

    async def browser_page_type(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("page.type", params)

    async def browser_page_wait_for(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("page.wait_for", params)

    async def browser_page_screenshot(params: dict[str, Any]) -> dict[str, Any]:
        return await _execute("page.screenshot", params)

    server.register("browser.session.start", browser_session_start)
    server.register("browser.session.list", browser_session_list)
    server.register("browser.session.show", browser_session_show)
    server.register("browser.session.close", browser_session_close)
    server.register("browser.session.archive", browser_session_archive)
    server.register("browser.page.goto", browser_page_goto)
    server.register("browser.page.extract", browser_page_extract)
    server.register("browser.page.click", browser_page_click)
    server.register("browser.page.type", browser_page_type)
    server.register("browser.page.wait_for", browser_page_wait_for)
    server.register("browser.page.screenshot", browser_page_screenshot)
