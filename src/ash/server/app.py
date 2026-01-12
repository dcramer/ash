"""FastAPI application for Ash server."""

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from ash.server.routes import health, webhooks

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ash.core import Agent
    from ash.db import Database
    from ash.providers.telegram import TelegramMessageHandler, TelegramProvider

logger = logging.getLogger(__name__)


class AshServer:
    """Main server application.

    Manages the FastAPI app and provider integrations.
    """

    def __init__(
        self,
        database: "Database",
        agent: "Agent",
        telegram_provider: "TelegramProvider | None" = None,
    ):
        self._database = database
        self._agent = agent
        self._telegram_provider = telegram_provider
        self._telegram_handler: TelegramMessageHandler | None = None

        self._app = self._create_app()

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self._app

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI app."""

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> "AsyncIterator[None]":
            # Startup
            logger.info("Starting Ash server")
            await self._database.connect()

            if self._telegram_provider:
                from ash.providers.telegram import TelegramMessageHandler

                self._telegram_handler = TelegramMessageHandler(
                    provider=self._telegram_provider,
                    agent=self._agent,
                    database=self._database,
                    streaming=False,
                )
                # Start in polling mode if no webhook
                # Webhook mode is handled via the routes

            yield

            # Shutdown
            logger.info("Shutting down Ash server")
            if self._telegram_provider:
                await self._telegram_provider.stop()
            await self._database.disconnect()

        app = FastAPI(
            title="Ash",
            description="Personal Assistant Agent API",
            version="0.1.0",
            lifespan=lifespan,
        )

        # Store references in app state
        app.state.server = self
        app.state.database = self._database
        app.state.agent = self._agent

        # Include routes
        app.include_router(health.router, tags=["health"])

        if self._telegram_provider:
            app.state.telegram_provider = self._telegram_provider
            app.include_router(
                webhooks.router,
                prefix="/webhook",
                tags=["webhooks"],
            )

        return app

    async def get_telegram_handler(self) -> "TelegramMessageHandler | None":
        """Get the Telegram message handler."""
        return self._telegram_handler


def create_app(
    database: "Database",
    agent: "Agent",
    telegram_provider: "TelegramProvider | None" = None,
) -> FastAPI:
    """Create the FastAPI application."""
    server = AshServer(
        database=database,
        agent=agent,
        telegram_provider=telegram_provider,
    )
    return server.app
