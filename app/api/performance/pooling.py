"""Connection pooling for Tag Sentinel API performance optimization.

This module provides connection pooling for databases, HTTP clients,
and other resources to improve performance and resource utilization.
"""

import logging
import asyncio
import time
from typing import Any, Optional, Dict, List, Callable, Generic, TypeVar, AsyncContextManager
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PoolConfig:
    """Configuration for connection pools."""
    min_connections: int = 1
    max_connections: int = 10
    max_idle_time: int = 300  # 5 minutes
    connection_timeout: int = 30  # seconds
    acquire_timeout: int = 10  # seconds
    health_check_interval: int = 60  # seconds
    enable_health_checks: bool = True
    retry_attempts: int = 3
    retry_delay: int = 1  # seconds


@dataclass
class PoolStats:
    """Connection pool statistics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    created_connections: int = 0
    closed_connections: int = 0
    failed_connections: int = 0
    pool_hits: int = 0
    pool_misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate pool hit rate."""
        total_requests = self.pool_hits + self.pool_misses
        if total_requests == 0:
            return 0.0
        return self.pool_hits / total_requests


@dataclass
class PooledConnection(Generic[T]):
    """A pooled connection wrapper."""
    connection: T
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    is_healthy: bool = True
    pool_id: Optional[str] = None

    def touch(self) -> None:
        """Update last used time and increment use count."""
        self.last_used = time.time()
        self.use_count += 1

    @property
    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used

    @property
    def age(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at


class ConnectionPool(Generic[T], ABC):
    """Abstract base class for connection pools."""

    def __init__(self, config: PoolConfig):
        """Initialize connection pool.

        Args:
            config: Pool configuration
        """
        self.config = config
        self.stats = PoolStats()
        self._connections: List[PooledConnection[T]] = []
        self._active_connections: weakref.WeakSet = weakref.WeakSet()
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None

        # Start health check task
        if config.enable_health_checks:
            self._start_health_check_task()

    @abstractmethod
    async def create_connection(self) -> T:
        """Create a new connection.

        Returns:
            New connection instance
        """
        pass

    @abstractmethod
    async def close_connection(self, connection: T) -> None:
        """Close a connection.

        Args:
            connection: Connection to close
        """
        pass

    @abstractmethod
    async def validate_connection(self, connection: T) -> bool:
        """Validate if connection is healthy.

        Args:
            connection: Connection to validate

        Returns:
            True if connection is healthy
        """
        pass

    async def acquire(self) -> AsyncContextManager[T]:
        """Acquire a connection from the pool.

        Returns:
            Context manager for connection
        """
        return self._connection_context()

    @asynccontextmanager
    async def _connection_context(self):
        """Context manager for acquiring and releasing connections."""
        connection = None
        try:
            connection = await self._acquire_connection()
            yield connection.connection
        finally:
            if connection:
                await self._release_connection(connection)

    async def _acquire_connection(self) -> PooledConnection[T]:
        """Acquire a connection from the pool."""
        start_time = time.time()

        while True:
            async with self._lock:
                # Try to get an idle connection
                for i, pooled_conn in enumerate(self._connections):
                    if pooled_conn not in self._active_connections:
                        # Check if connection is still healthy
                        if await self._is_connection_healthy(pooled_conn):
                            pooled_conn.touch()
                            self._active_connections.add(pooled_conn)
                            self.stats.active_connections += 1
                            self.stats.idle_connections -= 1
                            self.stats.pool_hits += 1
                            return pooled_conn
                        else:
                            # Remove unhealthy connection
                            self._connections.remove(pooled_conn)
                            await self._close_pooled_connection(pooled_conn)

                # Create new connection if under limit
                if len(self._connections) < self.config.max_connections:
                    new_connection = await self._create_new_connection()
                    if new_connection:
                        new_connection.touch()
                        self._connections.append(new_connection)
                        self._active_connections.add(new_connection)
                        self.stats.total_connections += 1
                        self.stats.active_connections += 1
                        self.stats.pool_misses += 1
                        return new_connection

            # Check timeout
            if time.time() - start_time > self.config.acquire_timeout:
                raise TimeoutError(f"Failed to acquire connection within {self.config.acquire_timeout}s")

            # Wait and retry
            await asyncio.sleep(0.1)

    async def _release_connection(self, pooled_conn: PooledConnection[T]) -> None:
        """Release a connection back to the pool."""
        async with self._lock:
            if pooled_conn in self._active_connections:
                self._active_connections.remove(pooled_conn)
                self.stats.active_connections -= 1
                self.stats.idle_connections += 1

    async def _create_new_connection(self) -> Optional[PooledConnection[T]]:
        """Create a new pooled connection."""
        for attempt in range(self.config.retry_attempts):
            try:
                connection = await asyncio.wait_for(
                    self.create_connection(),
                    timeout=self.config.connection_timeout
                )

                pooled_conn = PooledConnection(connection=connection)
                self.stats.created_connections += 1
                return pooled_conn

            except Exception as e:
                logger.error(f"Failed to create connection (attempt {attempt + 1}): {e}")
                self.stats.failed_connections += 1

                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)

        return None

    async def _close_pooled_connection(self, pooled_conn: PooledConnection[T]) -> None:
        """Close a pooled connection."""
        try:
            await self.close_connection(pooled_conn.connection)
            self.stats.closed_connections += 1
            self.stats.total_connections -= 1
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

    async def _is_connection_healthy(self, pooled_conn: PooledConnection[T]) -> bool:
        """Check if pooled connection is healthy."""
        # Check idle time
        if pooled_conn.idle_time > self.config.max_idle_time:
            return False

        # Check connection health
        try:
            is_healthy = await self.validate_connection(pooled_conn.connection)
            pooled_conn.is_healthy = is_healthy
            return is_healthy
        except Exception as e:
            logger.error(f"Error validating connection: {e}")
            return False

    def _start_health_check_task(self):
        """Start background health check task."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._health_check_task = loop.create_task(self._periodic_health_check())
        except RuntimeError:
            pass

    async def _periodic_health_check(self):
        """Periodically check connection health."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check task: {e}")

    async def _health_check(self):
        """Perform health check on all connections."""
        async with self._lock:
            unhealthy_connections = []

            for pooled_conn in self._connections:
                # Skip active connections
                if pooled_conn in self._active_connections:
                    continue

                if not await self._is_connection_healthy(pooled_conn):
                    unhealthy_connections.append(pooled_conn)

            # Remove unhealthy connections
            for pooled_conn in unhealthy_connections:
                self._connections.remove(pooled_conn)
                await self._close_pooled_connection(pooled_conn)
                self.stats.idle_connections -= 1

            # Ensure minimum connections
            while len(self._connections) < self.config.min_connections:
                new_connection = await self._create_new_connection()
                if new_connection:
                    self._connections.append(new_connection)
                    self.stats.idle_connections += 1
                else:
                    break

    async def get_stats(self) -> PoolStats:
        """Get pool statistics."""
        async with self._lock:
            return PoolStats(
                total_connections=self.stats.total_connections,
                active_connections=self.stats.active_connections,
                idle_connections=self.stats.idle_connections,
                created_connections=self.stats.created_connections,
                closed_connections=self.stats.closed_connections,
                failed_connections=self.stats.failed_connections,
                pool_hits=self.stats.pool_hits,
                pool_misses=self.stats.pool_misses
            )

    async def close(self) -> None:
        """Close all connections and clean up pool."""
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self._lock:
            for pooled_conn in self._connections[:]:
                await self._close_pooled_connection(pooled_conn)
            self._connections.clear()
            self._active_connections.clear()

        logger.info("Connection pool closed")


class DatabaseConnectionPool(ConnectionPool):
    """Connection pool for database connections."""

    def __init__(self, config: PoolConfig, connection_factory: Callable):
        """Initialize database connection pool.

        Args:
            config: Pool configuration
            connection_factory: Function to create new database connections
        """
        super().__init__(config)
        self.connection_factory = connection_factory

    async def create_connection(self):
        """Create a new database connection."""
        return await self.connection_factory()

    async def close_connection(self, connection) -> None:
        """Close a database connection."""
        if hasattr(connection, 'close'):
            if asyncio.iscoroutinefunction(connection.close):
                await connection.close()
            else:
                connection.close()

    async def validate_connection(self, connection) -> bool:
        """Validate database connection health."""
        try:
            # Try a simple query
            if hasattr(connection, 'execute'):
                if asyncio.iscoroutinefunction(connection.execute):
                    await connection.execute("SELECT 1")
                else:
                    connection.execute("SELECT 1")
            return True
        except Exception:
            return False


class HTTPConnectionPool(ConnectionPool):
    """Connection pool for HTTP client connections."""

    def __init__(self, config: PoolConfig, session_factory: Callable):
        """Initialize HTTP connection pool.

        Args:
            config: Pool configuration
            session_factory: Function to create new HTTP sessions
        """
        super().__init__(config)
        self.session_factory = session_factory

    async def create_connection(self):
        """Create a new HTTP client session."""
        return await self.session_factory()

    async def close_connection(self, connection) -> None:
        """Close an HTTP client session."""
        if hasattr(connection, 'close'):
            await connection.close()

    async def validate_connection(self, connection) -> bool:
        """Validate HTTP session health."""
        try:
            # Check if session is closed
            if hasattr(connection, 'closed'):
                return not connection.closed
            return True
        except Exception:
            return False


class ConnectionPoolManager:
    """Manager for multiple connection pools."""

    def __init__(self):
        """Initialize connection pool manager."""
        self.pools: Dict[str, ConnectionPool] = {}

    def add_pool(self, name: str, pool: ConnectionPool) -> None:
        """Add a connection pool.

        Args:
            name: Pool name
            pool: Connection pool instance
        """
        self.pools[name] = pool
        logger.info(f"Added connection pool: {name}")

    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a connection pool by name.

        Args:
            name: Pool name

        Returns:
            Connection pool or None if not found
        """
        return self.pools.get(name)

    async def get_connection(self, pool_name: str):
        """Get a connection from a named pool.

        Args:
            pool_name: Name of the pool

        Returns:
            Connection context manager

        Raises:
            ValueError: If pool doesn't exist
        """
        pool = self.pools.get(pool_name)
        if not pool:
            raise ValueError(f"Pool '{pool_name}' not found")

        return await pool.acquire()

    async def get_all_stats(self) -> Dict[str, PoolStats]:
        """Get statistics for all pools.

        Returns:
            Dictionary mapping pool names to statistics
        """
        stats = {}
        for name, pool in self.pools.items():
            stats[name] = await pool.get_stats()
        return stats

    async def close_all(self) -> None:
        """Close all connection pools."""
        for name, pool in self.pools.items():
            try:
                await pool.close()
                logger.info(f"Closed connection pool: {name}")
            except Exception as e:
                logger.error(f"Error closing pool {name}: {e}")

        self.pools.clear()


# Global connection pool manager
_global_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager() -> ConnectionPoolManager:
    """Get global connection pool manager."""
    global _global_pool_manager
    if _global_pool_manager is None:
        _global_pool_manager = ConnectionPoolManager()
    return _global_pool_manager


def set_pool_manager(manager: ConnectionPoolManager) -> None:
    """Set global connection pool manager."""
    global _global_pool_manager
    _global_pool_manager = manager


# Utility functions for creating common pools

async def create_database_pool(
    name: str,
    connection_factory: Callable,
    config: Optional[PoolConfig] = None
) -> DatabaseConnectionPool:
    """Create a database connection pool.

    Args:
        name: Pool name
        connection_factory: Function to create database connections
        config: Pool configuration

    Returns:
        Database connection pool
    """
    pool_config = config or PoolConfig()
    pool = DatabaseConnectionPool(pool_config, connection_factory)

    # Add to global manager
    manager = get_pool_manager()
    manager.add_pool(name, pool)

    return pool


async def create_http_pool(
    name: str,
    session_factory: Callable,
    config: Optional[PoolConfig] = None
) -> HTTPConnectionPool:
    """Create an HTTP connection pool.

    Args:
        name: Pool name
        session_factory: Function to create HTTP sessions
        config: Pool configuration

    Returns:
        HTTP connection pool
    """
    pool_config = config or PoolConfig()
    pool = HTTPConnectionPool(pool_config, session_factory)

    # Add to global manager
    manager = get_pool_manager()
    manager.add_pool(name, pool)

    return pool