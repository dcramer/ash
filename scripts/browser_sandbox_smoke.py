from __future__ import annotations

import argparse
import asyncio
import json
from urllib.parse import urlparse

from ash.browser import create_browser_manager
from ash.config import load_config
from ash.config.paths import get_config_path
from ash.sandbox.executor import SandboxExecutor
from ash.tools.base import build_sandbox_manager_config


async def _run(url: str, user_id: str, provider: str, session_name: str) -> int:
    config = load_config(get_config_path())
    sandbox_manager_config = build_sandbox_manager_config(
        config.sandbox,
        config.workspace,
    )
    parsed = urlparse(url)
    if (
        parsed.scheme in {"http", "https"}
        and sandbox_manager_config.network_mode == "none"
    ):
        print(
            "config_error",
            json.dumps(
                {
                    "ok": False,
                    "reason": (
                        "sandbox.network_mode is 'none'; external browser navigation "
                        "requires network_mode='bridge'"
                    ),
                    "url": url,
                },
                ensure_ascii=True,
            ),
        )
        return 2

    print(
        "sandbox.config",
        json.dumps(
            {
                "image": sandbox_manager_config.image,
                "network_mode": sandbox_manager_config.network_mode,
                "runtime": sandbox_manager_config.runtime,
            },
            ensure_ascii=True,
        ),
    )
    executor = SandboxExecutor(config=sandbox_manager_config)
    probe = await executor.execute(
        "curl -I -sS --max-time 8 https://example.com >/dev/null && echo ok || echo fail",
        timeout=15,
        reuse_container=True,
    )
    print(
        "sandbox.network_probe",
        json.dumps(
            {
                "exit_code": probe.exit_code,
                "success": probe.success,
                "stdout": probe.stdout.strip(),
                "stderr": probe.stderr.strip(),
            },
            ensure_ascii=True,
        ),
    )

    manager = create_browser_manager(
        config,
        sandbox_executor=executor,
    )

    started = await manager.execute_action(
        action="session.start",
        effective_user_id=user_id,
        provider_name=provider,
        session_name=session_name,
    )
    print("session.start", json.dumps(started.to_dict(), ensure_ascii=True))
    if not started.ok or not started.session_id:
        return 1

    session_id = started.session_id
    goto = await manager.execute_action(
        action="page.goto",
        effective_user_id=user_id,
        provider_name=provider,
        session_id=session_id,
        params={"url": url},
    )
    print("page.goto", json.dumps(goto.to_dict(), ensure_ascii=True))
    if not goto.ok:
        return 1

    extract = await manager.execute_action(
        action="page.extract",
        effective_user_id=user_id,
        provider_name=provider,
        session_id=session_id,
        params={"mode": "title"},
    )
    print("page.extract", json.dumps(extract.to_dict(), ensure_ascii=True))

    shot = await manager.execute_action(
        action="page.screenshot",
        effective_user_id=user_id,
        provider_name=provider,
        session_id=session_id,
    )
    print("page.screenshot", json.dumps(shot.to_dict(), ensure_ascii=True))

    archived = await manager.execute_action(
        action="session.archive",
        effective_user_id=user_id,
        provider_name=provider,
        session_id=session_id,
    )
    print("session.archive", json.dumps(archived.to_dict(), ensure_ascii=True))
    return 0 if archived.ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sandbox browser smoke via BrowserManager+SandboxExecutor."
    )
    parser.add_argument("--url", default="https://example.com")
    parser.add_argument("--user-id", default="local-test")
    parser.add_argument("--provider", default="sandbox")
    parser.add_argument("--session-name", default="local-smoke")
    args = parser.parse_args()
    return asyncio.run(
        _run(
            url=args.url,
            user_id=args.user_id,
            provider=args.provider,
            session_name=args.session_name,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
