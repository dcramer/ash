"""Runtime server orchestration helpers."""

from __future__ import annotations

import asyncio
import logging
import signal as signal_module
import time
from typing import TYPE_CHECKING, Protocol

import uvicorn

if TYPE_CHECKING:
    from fastapi import FastAPI


class TelegramRuntimeProvider(Protocol):
    """Minimal provider contract required by the server runner."""

    async def start(self, handler) -> None: ...

    async def stop(self) -> None: ...


logger = logging.getLogger(__name__)

TELEGRAM_HANDLER_POLL_INTERVAL_SECONDS = 0.1
TELEGRAM_HANDLER_WAIT_TIMEOUT_SECONDS = 60.0


class ServerRunner:
    """Owns uvicorn serving and optional Telegram polling lifecycle."""

    def __init__(
        self,
        app: FastAPI,
        *,
        host: str,
        port: int,
        telegram_provider: TelegramRuntimeProvider | None = None,
    ) -> None:
        self._app = app
        self._host = host
        self._port = port
        self._telegram_provider = telegram_provider

    async def run(self) -> None:
        """Run uvicorn and optional telegram polling with coordinated shutdown."""
        uvicorn_config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="info",
            log_config=None,  # Use shared logging config, not uvicorn's
        )
        server = uvicorn.Server(uvicorn_config)

        # Track tasks for cleanup
        telegram_task: asyncio.Task | None = None

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        shutdown_count = 0

        def handle_signal() -> None:
            nonlocal shutdown_count
            shutdown_count += 1

            if shutdown_count == 1:
                # First signal: graceful shutdown
                logger.info("server_shutting_down")
                server.should_exit = True
                # Stop telegram polling before cancelling task
                provider = self._telegram_provider
                if provider:
                    loop.call_soon(lambda: asyncio.create_task(provider.stop()))
                # Cancel telegram task after stop is scheduled
                if telegram_task and not telegram_task.done():
                    telegram_task.cancel()
            else:
                # Second signal: force immediate exit
                logger.warning("server_force_shutdown")
                import os

                os._exit(1)

        for sig in (signal_module.SIGTERM, signal_module.SIGINT):
            loop.add_signal_handler(sig, handle_signal)

        if self._telegram_provider:
            # Run both uvicorn and telegram polling
            logger.info("telegram_polling_starting")
            server_task = asyncio.create_task(server.serve())

            async def start_telegram() -> None:
                # Wait for server wiring to expose telegram handler.
                # Use a generous timeout to tolerate slow startup paths.
                deadline = time.monotonic() + TELEGRAM_HANDLER_WAIT_TIMEOUT_SECONDS
                while not server_task.done():
                    handler = await self._app.state.server.get_telegram_handler()
                    if handler:
                        try:
                            provider = self._telegram_provider
                            if provider:
                                await provider.start(handler.handle_message)
                        except asyncio.CancelledError:
                            logger.info("telegram_polling_cancelled")
                        return
                    if time.monotonic() >= deadline:
                        logger.error("telegram_handler_timeout")
                        return
                    await asyncio.sleep(TELEGRAM_HANDLER_POLL_INTERVAL_SECONDS)

                logger.error("telegram_handler_unavailable")

            telegram_task = asyncio.create_task(start_telegram())
            # return_exceptions=True ensures we wait for server to finish graceful
            # shutdown after telegram is cancelled, avoiding double Ctrl+C
            await asyncio.gather(server_task, telegram_task, return_exceptions=True)
            return

        await server.serve()
