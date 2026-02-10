"""FastAPI application for Ash server."""

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from ash.server.routes import health, webhooks

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ash.agents import AgentRegistry
    from ash.config import AshConfig
    from ash.core import Agent
    from ash.db import Database
    from ash.llm import LLMProvider
    from ash.memory import MemoryManager
    from ash.memory.extractor import MemoryExtractor
    from ash.providers.telegram import TelegramMessageHandler, TelegramProvider
    from ash.skills import SkillRegistry
    from ash.tools.registry import ToolRegistry

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
        config: "AshConfig | None" = None,
        agent_registry: "AgentRegistry | None" = None,
        skill_registry: "SkillRegistry | None" = None,
        tool_registry: "ToolRegistry | None" = None,
        llm_provider: "LLMProvider | None" = None,
        memory_manager: "MemoryManager | None" = None,
        memory_extractor: "MemoryExtractor | None" = None,
    ):
        self._database = database
        self._agent = agent
        self._telegram_provider = telegram_provider
        self._config = config
        self._agent_registry = agent_registry
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry
        self._llm_provider = llm_provider
        self._memory_manager = memory_manager
        self._memory_extractor = memory_extractor
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
                    config=self._config,
                    agent_registry=self._agent_registry,
                    skill_registry=self._skill_registry,
                    tool_registry=self._tool_registry,
                    llm_provider=self._llm_provider,
                    memory_manager=self._memory_manager,
                    memory_extractor=self._memory_extractor,
                )
                # Wire up callback handler for checkpoint inline keyboards
                self._telegram_provider.set_callback_handler(
                    self._telegram_handler.handle_callback_query
                )
                # Wire up passive message handler for group listening
                self._telegram_provider.set_passive_handler(
                    self._telegram_handler.handle_passive_message
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
    config: "AshConfig | None" = None,
    agent_registry: "AgentRegistry | None" = None,
    skill_registry: "SkillRegistry | None" = None,
    tool_registry: "ToolRegistry | None" = None,
    llm_provider: "LLMProvider | None" = None,
    memory_manager: "MemoryManager | None" = None,
    memory_extractor: "MemoryExtractor | None" = None,
) -> FastAPI:
    """Create the FastAPI application."""
    server = AshServer(
        database=database,
        agent=agent,
        telegram_provider=telegram_provider,
        config=config,
        agent_registry=agent_registry,
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        llm_provider=llm_provider,
        memory_manager=memory_manager,
        memory_extractor=memory_extractor,
    )
    return server.app
