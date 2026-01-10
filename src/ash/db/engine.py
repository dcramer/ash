"""Async SQLAlchemy database engine."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class Database:
    """Async database connection manager."""

    def __init__(
        self, database_url: str | None = None, database_path: Path | None = None
    ):
        """Initialize database.

        Args:
            database_url: Full database URL (takes precedence).
            database_path: Path to SQLite database file.
        """
        if database_url:
            self._url = database_url
        elif database_path:
            # Ensure parent directory exists
            database_path.parent.mkdir(parents=True, exist_ok=True)
            self._url = f"sqlite+aiosqlite:///{database_path}"
        else:
            raise ValueError("Either database_url or database_path must be provided")

        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine."""
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self._session_factory

    async def connect(self) -> None:
        """Initialize the database connection."""
        self._engine = create_async_engine(
            self._url,
            echo=False,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session.

        Usage:
            async with db.session() as session:
                result = await session.execute(...)
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


# Global database instance
_db: Database | None = None


def get_database() -> Database:
    """Get the global database instance."""
    if _db is None:
        raise RuntimeError("Database not configured. Call init_database() first.")
    return _db


def init_database(
    database_url: str | None = None, database_path: Path | None = None
) -> Database:
    """Initialize the global database instance."""
    global _db
    _db = Database(database_url=database_url, database_path=database_path)
    return _db
