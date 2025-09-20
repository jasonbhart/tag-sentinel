"""Distributed concurrency control for preventing concurrent audit runs.

This module provides distributed locking mechanisms to ensure that only one
audit run executes per site-environment combination across multiple scheduler
instances.
"""

import asyncio
import json
import time
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Any, AsyncGenerator, List
from dataclasses import dataclass
from uuid import uuid4

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    Redis = None  # type: ignore
    REDIS_AVAILABLE = False


logger = logging.getLogger(__name__)


class LockError(Exception):
    """Exception raised when lock operations fail."""
    pass


class LockTimeoutError(LockError):
    """Exception raised when lock acquisition times out."""
    pass


class LockBackendError(LockError):
    """Exception raised when lock backend operations fail."""
    pass


@dataclass
class LockInfo:
    """Information about an acquired lock."""

    key: str
    owner_id: str
    acquired_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any]

    @property
    def is_expired(self) -> bool:
        """Check if the lock has expired."""
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def time_remaining(self) -> timedelta:
        """Get time remaining before lock expires."""
        return max(timedelta(0), self.expires_at - datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'owner_id': self.owner_id,
            'acquired_at': self.acquired_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LockInfo':
        """Create LockInfo from dictionary."""
        return cls(
            key=data['key'],
            owner_id=data['owner_id'],
            acquired_at=datetime.fromisoformat(data['acquired_at']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            metadata=data.get('metadata', {})
        )


class LockBackend(ABC):
    """Abstract base class for distributed lock backends."""

    @abstractmethod
    async def acquire_lock(
        self,
        key: str,
        owner_id: str,
        timeout_seconds: int,
        metadata: Dict[str, Any]
    ) -> bool:
        """Attempt to acquire a distributed lock.

        Args:
            key: Unique lock key
            owner_id: Identifier for lock owner
            timeout_seconds: Lock timeout in seconds
            metadata: Additional lock metadata

        Returns:
            True if lock acquired, False otherwise
        """
        pass

    @abstractmethod
    async def release_lock(self, key: str, owner_id: str) -> bool:
        """Release a distributed lock.

        Args:
            key: Lock key
            owner_id: Lock owner identifier

        Returns:
            True if lock released, False if not owned
        """
        pass

    @abstractmethod
    async def extend_lock(
        self,
        key: str,
        owner_id: str,
        additional_seconds: int
    ) -> bool:
        """Extend a held lock's timeout.

        Args:
            key: Lock key
            owner_id: Lock owner identifier
            additional_seconds: Additional timeout in seconds

        Returns:
            True if extended, False if not owned or expired
        """
        pass

    @abstractmethod
    async def get_lock_info(self, key: str) -> Optional[LockInfo]:
        """Get information about a lock.

        Args:
            key: Lock key

        Returns:
            LockInfo if lock exists, None otherwise
        """
        pass

    @abstractmethod
    async def cleanup_expired_locks(self) -> int:
        """Clean up expired locks.

        Returns:
            Number of locks cleaned up
        """
        pass

    @abstractmethod
    async def list_locks(self, pattern: str = "*") -> List[LockInfo]:
        """List locks matching a pattern.

        Args:
            pattern: Key pattern to match

        Returns:
            List of matching locks
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform backend health check.

        Returns:
            Health status information
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close backend connections."""
        pass


class RedisLockBackend(LockBackend):
    """Redis-based distributed lock backend."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "tag_sentinel:locks:",
        **redis_kwargs
    ):
        """Initialize Redis lock backend.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for lock keys
            **redis_kwargs: Additional Redis connection parameters
        """
        if not REDIS_AVAILABLE:
            raise LockBackendError("Redis not available - install redis package")

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis_kwargs = redis_kwargs
        self._redis = None

    async def _get_redis(self):
        """Get Redis connection, creating if necessary."""
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url, **self.redis_kwargs)
                # Test connection
                await self._redis.ping()
            except Exception as e:
                raise LockBackendError(f"Failed to connect to Redis: {e}")

        return self._redis

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.key_prefix}{key}"

    async def acquire_lock(
        self,
        key: str,
        owner_id: str,
        timeout_seconds: int,
        metadata: Dict[str, Any]
    ) -> bool:
        """Acquire lock using Redis SET with NX and EX options."""
        try:
            redis_client = await self._get_redis()
            redis_key = self._make_key(key)

            lock_data = {
                'owner_id': owner_id,
                'acquired_at': datetime.now(timezone.utc).isoformat(),
                'expires_at': (datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)).isoformat(),
                'metadata': metadata
            }

            # Use SET with NX (only if key doesn't exist) and EX (expire)
            result = await redis_client.set(
                redis_key,
                json.dumps(lock_data),
                nx=True,  # Only set if key doesn't exist
                ex=timeout_seconds  # Expire after timeout_seconds
            )

            return result is not None

        except Exception as e:
            logger.error(f"Failed to acquire lock '{key}': {e}")
            raise LockBackendError(f"Lock acquisition failed: {e}")

    async def release_lock(self, key: str, owner_id: str) -> bool:
        """Release lock using Lua script for atomicity."""
        try:
            redis_client = await self._get_redis()
            redis_key = self._make_key(key)

            # Lua script to ensure we only delete if we own the lock
            lua_script = """
            local current = redis.call('GET', KEYS[1])
            if current then
                local data = cjson.decode(current)
                if data.owner_id == ARGV[1] then
                    return redis.call('DEL', KEYS[1])
                else
                    return 0
                end
            else
                return 0
            end
            """

            result = await redis_client.eval(lua_script, 1, redis_key, owner_id)
            return result == 1

        except Exception as e:
            logger.error(f"Failed to release lock '{key}': {e}")
            raise LockBackendError(f"Lock release failed: {e}")

    async def extend_lock(
        self,
        key: str,
        owner_id: str,
        additional_seconds: int
    ) -> bool:
        """Extend lock timeout using Lua script."""
        try:
            redis_client = await self._get_redis()
            redis_key = self._make_key(key)

            # Compute new expiration time on client side
            new_expires_at = datetime.now(timezone.utc) + timedelta(seconds=additional_seconds)
            new_expires_iso = new_expires_at.isoformat()

            # Lua script to extend lock if we own it (avoid os.time/os.date)
            lua_script = """
            local current = redis.call('GET', KEYS[1])
            if current then
                local data = cjson.decode(current)
                if data.owner_id == ARGV[1] then
                    data.expires_at = ARGV[2]
                    redis.call('SET', KEYS[1], cjson.encode(data), 'EX', tonumber(ARGV[3]))
                    return 1
                else
                    return 0
                end
            else
                return 0
            end
            """

            result = await redis_client.eval(
                lua_script,
                1,
                redis_key,
                owner_id,
                new_expires_iso,
                str(additional_seconds)
            )
            return result == 1

        except Exception as e:
            logger.error(f"Failed to extend lock '{key}': {e}")
            raise LockBackendError(f"Lock extension failed: {e}")

    async def get_lock_info(self, key: str) -> Optional[LockInfo]:
        """Get lock information from Redis."""
        try:
            redis_client = await self._get_redis()
            redis_key = self._make_key(key)

            lock_data_str = await redis_client.get(redis_key)
            if not lock_data_str:
                return None

            lock_data = json.loads(lock_data_str)
            return LockInfo(
                key=key,
                owner_id=lock_data['owner_id'],
                acquired_at=datetime.fromisoformat(lock_data['acquired_at']),
                expires_at=datetime.fromisoformat(lock_data['expires_at']),
                metadata=lock_data.get('metadata', {})
            )

        except Exception as e:
            logger.error(f"Failed to get lock info for '{key}': {e}")
            return None

    async def cleanup_expired_locks(self) -> int:
        """Clean up expired locks by scanning for expired keys."""
        try:
            redis_client = await self._get_redis()
            pattern = f"{self.key_prefix}*"

            count = 0
            async for key in redis_client.scan_iter(match=pattern):
                # Redis automatically expires keys, but we can double-check
                ttl = await redis_client.ttl(key)
                if ttl == -2:  # Key doesn't exist (expired)
                    count += 1

            return count

        except Exception as e:
            logger.error(f"Failed to cleanup expired locks: {e}")
            return 0

    async def list_locks(self, pattern: str = "*") -> List[LockInfo]:
        """List all locks matching pattern."""
        try:
            redis_client = await self._get_redis()
            redis_pattern = f"{self.key_prefix}{pattern}"

            locks = []
            async for key in redis_client.scan_iter(match=redis_pattern):
                lock_key = key.decode('utf-8').replace(self.key_prefix, '')
                lock_info = await self.get_lock_info(lock_key)
                if lock_info:
                    locks.append(lock_info)

            return locks

        except Exception as e:
            logger.error(f"Failed to list locks: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check."""
        try:
            redis_client = await self._get_redis()
            start_time = time.time()
            await redis_client.ping()
            ping_time = time.time() - start_time

            info = await redis_client.info()

            return {
                'backend_type': 'RedisLockBackend',
                'backend': 'redis',
                'status': 'healthy',
                'ping_time_ms': round(ping_time * 1000, 2),
                'redis_version': info.get('redis_version'),
                'connected_clients': info.get('connected_clients'),
                'used_memory_human': info.get('used_memory_human'),
                'url': self.redis_url
            }

        except Exception as e:
            return {
                'backend_type': 'RedisLockBackend',
                'backend': 'redis',
                'status': 'unhealthy',
                'error': str(e),
                'url': self.redis_url
            }

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


class InMemoryLockBackend(LockBackend):
    """In-memory lock backend for single-node deployments."""

    def __init__(self):
        """Initialize in-memory lock backend."""
        self._locks: Dict[str, LockInfo] = {}
        self._lock = asyncio.Lock()

    async def acquire_lock(
        self,
        key: str,
        owner_id: str,
        timeout_seconds: int,
        metadata: Dict[str, Any]
    ) -> bool:
        """Acquire in-memory lock."""
        async with self._lock:
            # Clean up expired lock if exists
            if key in self._locks and self._locks[key].is_expired:
                del self._locks[key]

            # Check if lock already exists and not expired
            if key in self._locks:
                return False

            # Acquire lock
            now = datetime.now(timezone.utc)
            self._locks[key] = LockInfo(
                key=key,
                owner_id=owner_id,
                acquired_at=now,
                expires_at=now + timedelta(seconds=timeout_seconds),
                metadata=metadata
            )

            return True

    async def release_lock(self, key: str, owner_id: str) -> bool:
        """Release in-memory lock."""
        async with self._lock:
            if key not in self._locks:
                return False

            lock = self._locks[key]
            if lock.owner_id != owner_id:
                return False

            del self._locks[key]
            return True

    async def extend_lock(
        self,
        key: str,
        owner_id: str,
        additional_seconds: int
    ) -> bool:
        """Extend in-memory lock."""
        async with self._lock:
            if key not in self._locks:
                return False

            lock = self._locks[key]
            if lock.owner_id != owner_id or lock.is_expired:
                return False

            # Extend lock
            lock.expires_at += timedelta(seconds=additional_seconds)
            return True

    async def get_lock_info(self, key: str) -> Optional[LockInfo]:
        """Get in-memory lock info."""
        async with self._lock:
            if key not in self._locks:
                return None

            lock = self._locks[key]
            if lock.is_expired:
                del self._locks[key]
                return None

            return lock

    async def cleanup_expired_locks(self) -> int:
        """Clean up expired in-memory locks."""
        async with self._lock:
            expired_keys = [
                key for key, lock in self._locks.items()
                if lock.is_expired
            ]

            for key in expired_keys:
                del self._locks[key]

            return len(expired_keys)

    async def list_locks(self, pattern: str = "*") -> List[LockInfo]:
        """List in-memory locks."""
        import fnmatch

        async with self._lock:
            # Clean expired locks first
            await self.cleanup_expired_locks()

            locks = []
            for key, lock in self._locks.items():
                if fnmatch.fnmatch(key, pattern):
                    locks.append(lock)

            return locks

    async def health_check(self) -> Dict[str, Any]:
        """Perform in-memory backend health check."""
        async with self._lock:
            return {
                'backend_type': 'InMemoryLockBackend',
                'backend': 'in_memory',
                'status': 'healthy',
                'active_locks': len(self._locks),
                'expired_locks': sum(1 for lock in self._locks.values() if lock.is_expired)
            }

    async def close(self) -> None:
        """Close in-memory backend (no-op)."""
        async with self._lock:
            self._locks.clear()


class ConcurrencyManager:
    """Manager for distributed concurrency control with pluggable backends."""

    def __init__(
        self,
        backend: LockBackend,
        default_timeout_seconds: int = 3600,  # 1 hour
        cleanup_interval_seconds: int = 300,  # 5 minutes
    ):
        """Initialize concurrency manager.

        Args:
            backend: Lock backend to use
            default_timeout_seconds: Default lock timeout
            cleanup_interval_seconds: Interval for cleanup task
        """
        self.backend = backend
        self.default_timeout_seconds = default_timeout_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.instance_id = str(uuid4())
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self) -> None:
        """Start the concurrency manager."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(f"Started concurrency manager (instance: {self.instance_id})")

    async def stop(self) -> None:
        """Stop the concurrency manager."""
        self._shutdown = True

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        await self.backend.close()
        logger.info(f"Stopped concurrency manager (instance: {self.instance_id})")

    @asynccontextmanager
    async def acquire_lock(
        self,
        site_id: str,
        environment: str,
        timeout_seconds: Optional[int] = None,
        wait_timeout_seconds: int = 60,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[LockInfo, None]:
        """Acquire a lock for site-environment combination.

        Args:
            site_id: Site identifier
            environment: Environment name
            timeout_seconds: Lock timeout (default: manager default)
            wait_timeout_seconds: Time to wait for lock acquisition
            metadata: Additional lock metadata

        Yields:
            LockInfo if lock acquired

        Raises:
            LockTimeoutError: If lock acquisition times out
            LockError: If lock operations fail
        """
        lock_key = self._make_lock_key(site_id, environment)
        owner_id = f"{self.instance_id}:{asyncio.current_task().get_name()}" # type: ignore
        timeout = timeout_seconds or self.default_timeout_seconds
        metadata = metadata or {}

        # Add manager metadata
        metadata.update({
            'site_id': site_id,
            'environment': environment,
            'manager_instance': self.instance_id,
            'acquired_by_task': asyncio.current_task().get_name() if asyncio.current_task() else 'unknown' # type: ignore
        })

        # Try to acquire lock with retries
        start_time = time.time()
        acquired = False

        while not acquired and (time.time() - start_time) < wait_timeout_seconds:
            acquired = await self.backend.acquire_lock(
                lock_key,
                owner_id,
                timeout,
                metadata
            )

            if not acquired:
                await asyncio.sleep(1)  # Wait before retry

        if not acquired:
            raise LockTimeoutError(
                f"Failed to acquire lock for {site_id}:{environment} within {wait_timeout_seconds} seconds"
            )

        # Get lock info
        lock_info = await self.backend.get_lock_info(lock_key)
        if not lock_info:
            raise LockError(f"Lock acquired but info not available for {lock_key}")

        try:
            logger.info(f"Acquired lock: {site_id}:{environment} (owner: {owner_id})")
            yield lock_info
        finally:
            # Release lock
            released = await self.backend.release_lock(lock_key, owner_id)
            if released:
                logger.info(f"Released lock: {site_id}:{environment}")
            else:
                logger.warning(f"Failed to release lock: {site_id}:{environment} (may have expired)")

    async def acquire_lock_manual(
        self,
        site_id: str,
        environment: str,
        timeout_seconds: Optional[int] = None,
        wait_timeout_seconds: int = 60,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[LockInfo]:
        """Manually acquire a lock without automatic release.

        This method returns the LockInfo directly instead of using async context manager.
        The caller is responsible for releasing the lock using release_lock_manual().

        Args:
            site_id: Site identifier
            environment: Environment name
            timeout_seconds: Lock timeout (default: manager default)
            wait_timeout_seconds: Time to wait for lock acquisition
            metadata: Additional lock metadata

        Returns:
            LockInfo if lock acquired successfully, None if acquisition failed
        """
        lock_key = self._make_lock_key(site_id, environment)
        owner_id = f"{self.instance_id}:{asyncio.current_task().get_name()}" # type: ignore
        timeout = timeout_seconds or self.default_timeout_seconds
        metadata = metadata or {}

        # Add manager metadata
        metadata.update({
            'site_id': site_id,
            'environment': environment,
            'manager_instance': self.instance_id,
            'acquired_by_task': asyncio.current_task().get_name() if asyncio.current_task() else 'unknown' # type: ignore
        })

        # Try to acquire lock with retries
        start_time = time.time()
        acquired = False

        while not acquired and (time.time() - start_time) < wait_timeout_seconds:
            acquired = await self.backend.acquire_lock(
                lock_key,
                owner_id,
                timeout,
                metadata
            )

            if not acquired:
                await asyncio.sleep(1)  # Wait before retry

        if not acquired:
            logger.warning(f"Failed to acquire lock for {site_id}:{environment} within {wait_timeout_seconds} seconds")
            return None

        # Get lock info
        lock_info = await self.backend.get_lock_info(lock_key)
        if not lock_info:
            logger.error(f"Lock acquired but info not available for {lock_key}")
            return None

        logger.info(f"Manually acquired lock: {site_id}:{environment} (owner: {owner_id})")
        return lock_info

    async def release_lock_manual(
        self,
        site_id: str,
        environment: str,
        owner_id: str
    ) -> bool:
        """Manually release a lock acquired with acquire_lock_manual().

        Args:
            site_id: Site identifier
            environment: Environment name
            owner_id: Owner ID from LockInfo

        Returns:
            True if lock was successfully released
        """
        lock_key = self._make_lock_key(site_id, environment)
        released = await self.backend.release_lock(lock_key, owner_id)

        if released:
            logger.info(f"Manually released lock: {site_id}:{environment}")
        else:
            logger.warning(f"Failed to manually release lock: {site_id}:{environment} (may have expired)")

        return released

    async def is_locked(self, site_id: str, environment: str) -> bool:
        """Check if site-environment combination is locked.

        Args:
            site_id: Site identifier
            environment: Environment name

        Returns:
            True if locked, False otherwise
        """
        lock_key = self._make_lock_key(site_id, environment)
        lock_info = await self.backend.get_lock_info(lock_key)
        return lock_info is not None and not lock_info.is_expired

    async def get_lock_status(
        self,
        site_id: str,
        environment: str
    ) -> Optional[LockInfo]:
        """Get lock status for site-environment combination.

        Args:
            site_id: Site identifier
            environment: Environment name

        Returns:
            LockInfo if locked, None otherwise
        """
        lock_key = self._make_lock_key(site_id, environment)
        return await self.backend.get_lock_info(lock_key)

    async def list_active_locks(self) -> List[LockInfo]:
        """List all active locks."""
        return await self.backend.list_locks("site:*")

    async def force_release_lock(
        self,
        site_id: str,
        environment: str,
        reason: str = "Manual release"
    ) -> bool:
        """Force release a lock (emergency operation).

        Args:
            site_id: Site identifier
            environment: Environment name
            reason: Reason for forced release

        Returns:
            True if lock was released
        """
        lock_key = self._make_lock_key(site_id, environment)
        lock_info = await self.backend.get_lock_info(lock_key)

        if not lock_info:
            return False

        # Force release by deleting the lock (backend-specific)
        if hasattr(self.backend, '_redis'):
            redis_client = await self.backend._get_redis()  # type: ignore
            result = await redis_client.delete(self.backend._make_key(lock_key))  # type: ignore
            success = result > 0
        elif isinstance(self.backend, InMemoryLockBackend):
            success = await self.backend.release_lock(lock_key, lock_info.owner_id)
        else:
            success = False

        if success:
            logger.warning(
                f"Force released lock: {site_id}:{environment} "
                f"(was owned by {lock_info.owner_id}, reason: {reason})"
            )

        return success

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on concurrency manager."""
        backend_health = await self.backend.health_check()

        active_locks = await self.list_active_locks()

        return {
            'manager_instance': self.instance_id,
            'backend_health': backend_health,
            'active_locks_count': len(active_locks),
            'cleanup_task_running': self._cleanup_task is not None and not self._cleanup_task.done(),
            'default_timeout_seconds': self.default_timeout_seconds,
            'cleanup_interval_seconds': self.cleanup_interval_seconds
        }

    def _make_lock_key(self, site_id: str, environment: str) -> str:
        """Create lock key for site-environment combination."""
        return f"site:{site_id}:env:{environment}"

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up expired locks."""
        logger.info("Started lock cleanup task")

        while not self._shutdown:
            try:
                cleaned_count = await self.backend.cleanup_expired_locks()
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} expired locks")

                await asyncio.sleep(self.cleanup_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in lock cleanup task: {e}")
                await asyncio.sleep(self.cleanup_interval_seconds)

        logger.info("Lock cleanup task stopped")


async def create_concurrency_manager(
    backend_type: str = "redis",
    redis_url: str = "redis://localhost:6379",
    **kwargs
) -> ConcurrencyManager:
    """Create and start a concurrency manager.

    Args:
        backend_type: Backend type ('redis' or 'in_memory')
        redis_url: Redis connection URL (for redis backend)
        **kwargs: Additional manager parameters

    Returns:
        Configured and started ConcurrencyManager

    Raises:
        ValueError: If backend_type is invalid
        LockBackendError: If backend initialization fails
    """
    # Accept alias 'memory' for convenience
    if backend_type == "memory":
        backend_type = "in_memory"

    if backend_type == "redis":
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, falling back to in-memory backend")
            backend = InMemoryLockBackend()
        else:
            backend = RedisLockBackend(redis_url=redis_url)
    elif backend_type == "in_memory":
        backend = InMemoryLockBackend()
    else:
        raise ValueError(f"Invalid backend type: {backend_type}")

    manager = ConcurrencyManager(backend, **kwargs)
    await manager.start()
    return manager
