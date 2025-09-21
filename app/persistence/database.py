"""Database configuration and connection management for Tag Sentinel persistence."""

import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool, QueuePool


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class DatabaseConfig:
    """Database configuration and connection management."""

    def __init__(
        self,
        url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
    ):
        """Initialize database configuration.

        Args:
            url: Database URL. If None, reads from POSTGRES_URL env var
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections beyond pool_size
            pool_timeout: Timeout for getting connection from pool
            pool_recycle: Recycle connections after this many seconds
            echo: Enable SQLAlchemy logging
        """
        self.url = url or self._get_database_url()
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo

        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    def _get_database_url(self) -> str:
        """Get database URL from environment variables."""
        # Try different environment variable names
        for env_var in ["POSTGRES_URL", "DATABASE_URL", "DB_URL"]:
            url = os.getenv(env_var)
            if url:
                # Ensure async driver with psycopg3
                if url.startswith("postgresql://"):
                    url = url.replace("postgresql://", "postgresql+psycopg_async://", 1)
                elif url.startswith("postgresql+psycopg://"):
                    url = url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
                return url

        # Default to local development database with async driver
        return "postgresql+psycopg_async://postgres:postgres@localhost:5432/tag_sentinel"

    @property
    def engine(self) -> AsyncEngine:
        """Get or create async database engine."""
        if self._engine is None:
            # Use NullPool for testing, QueuePool for production
            poolclass = NullPool if "test" in self.url else QueuePool

            self._engine = create_async_engine(
                self.url,
                poolclass=poolclass,
                pool_size=self.pool_size if poolclass != NullPool else None,
                max_overflow=self.max_overflow if poolclass != NullPool else None,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                echo=self.echo,
                future=True,
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get or create async session factory."""
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Create async database session context manager."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self) -> None:
        """Close database engine and connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            from sqlalchemy import text
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception:
            return False


# Global database configuration instance
db_config = DatabaseConfig()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection helper for FastAPI routes."""
    async with db_config.session() as session:
        yield session


async def init_database() -> None:
    """Initialize database tables."""
    from .models import Base

    async with db_config.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_database() -> None:
    """Close database connections."""
    await db_config.close()