"""Storage backends for rate limiting data.

This module provides different storage implementations for rate limiting
counters and token bucket state with proper async support.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import json
import aiofiles
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class RateLimitStorage(ABC):
    """Abstract storage interface for rate limiting data."""

    @abstractmethod
    async def get_counter(self, key: str) -> Optional[int]:
        """Get counter value."""
        pass

    @abstractmethod
    async def increment_counter(self, key: str, amount: int = 1, expiry: int = None) -> int:
        """Increment counter and return new value."""
        pass

    @abstractmethod
    async def get_value(self, key: str) -> Any:
        """Get arbitrary value."""
        pass

    @abstractmethod
    async def set_value(self, key: str, value: Any, expiry: int = None) -> None:
        """Set arbitrary value."""
        pass

    @abstractmethod
    async def delete_key(self, key: str) -> bool:
        """Delete key."""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired keys and return count of cleaned keys."""
        pass


class InMemoryRateLimitStorage(RateLimitStorage):
    """In-memory storage for rate limiting (development/testing only)."""

    def __init__(self, cleanup_interval: int = 300):
        """Initialize in-memory storage.

        Args:
            cleanup_interval: Seconds between automatic cleanup runs
        """
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        self._cleanup_interval = cleanup_interval
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """Periodically clean up expired keys."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                cleaned = await self.cleanup_expired()
                if cleaned > 0:
                    logger.debug(f"Cleaned up {cleaned} expired rate limit keys")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limit cleanup task: {e}")

    async def cleanup_expired(self) -> int:
        """Remove expired keys."""
        async with self._lock:
            now = time.time()
            expired_keys = [k for k, exp_time in self._expiry.items() if exp_time <= now]

            for key in expired_keys:
                self._data.pop(key, None)
                self._expiry.pop(key, None)

            return len(expired_keys)

    async def get_counter(self, key: str) -> Optional[int]:
        """Get counter value."""
        async with self._lock:
            # Check if expired
            if key in self._expiry and self._expiry[key] <= time.time():
                self._data.pop(key, None)
                self._expiry.pop(key, None)
                return None

            return self._data.get(key)

    async def increment_counter(self, key: str, amount: int = 1, expiry: int = None) -> int:
        """Increment counter and return new value."""
        async with self._lock:
            current = self._data.get(key, 0)
            new_value = current + amount
            self._data[key] = new_value

            if expiry:
                self._expiry[key] = time.time() + expiry

            return new_value

    async def get_value(self, key: str) -> Any:
        """Get arbitrary value."""
        async with self._lock:
            # Check if expired
            if key in self._expiry and self._expiry[key] <= time.time():
                self._data.pop(key, None)
                self._expiry.pop(key, None)
                return None

            return self._data.get(key)

    async def set_value(self, key: str, value: Any, expiry: int = None) -> None:
        """Set arbitrary value."""
        async with self._lock:
            self._data[key] = value
            if expiry:
                self._expiry[key] = time.time() + expiry

    async def delete_key(self, key: str) -> bool:
        """Delete key."""
        async with self._lock:
            deleted = key in self._data
            self._data.pop(key, None)
            self._expiry.pop(key, None)
            return deleted

    async def close(self):
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    def __del__(self):
        """Destructor to clean up task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


class FileBasedRateLimitStorage(RateLimitStorage):
    """File-based storage for rate limiting data."""

    def __init__(self, storage_dir: str = "./data/rate_limits"):
        """Initialize file-based storage.

        Args:
            storage_dir: Directory to store rate limit data files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key."""
        # Create safe filename from key
        safe_key = key.replace(":", "_").replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_key}.json"

    async def _get_key_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for a specific key."""
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def _read_key_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Read key data from file."""
        file_path = self._get_file_path(key)
        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)

                # Check if expired
                if 'expiry' in data and data['expiry'] <= time.time():
                    await self._delete_key_file(key)
                    return None

                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Error reading rate limit data for key {key}: {e}")
            return None

    async def _write_key_data(self, key: str, data: Dict[str, Any]) -> None:
        """Write key data to file."""
        file_path = self._get_file_path(key)

        try:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data))
        except OSError as e:
            logger.error(f"Error writing rate limit data for key {key}: {e}")

    async def _delete_key_file(self, key: str) -> bool:
        """Delete key file."""
        file_path = self._get_file_path(key)
        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except OSError as e:
            logger.error(f"Error deleting rate limit file for key {key}: {e}")
            return False

    async def get_counter(self, key: str) -> Optional[int]:
        """Get counter value."""
        lock = await self._get_key_lock(key)
        async with lock:
            data = await self._read_key_data(key)
            return data.get('value') if data else None

    async def increment_counter(self, key: str, amount: int = 1, expiry: int = None) -> int:
        """Increment counter and return new value."""
        lock = await self._get_key_lock(key)
        async with lock:
            data = await self._read_key_data(key) or {'value': 0}
            data['value'] = data.get('value', 0) + amount

            if expiry:
                data['expiry'] = time.time() + expiry

            await self._write_key_data(key, data)
            return data['value']

    async def get_value(self, key: str) -> Any:
        """Get arbitrary value."""
        lock = await self._get_key_lock(key)
        async with lock:
            data = await self._read_key_data(key)
            return data.get('value') if data else None

    async def set_value(self, key: str, value: Any, expiry: int = None) -> None:
        """Set arbitrary value."""
        lock = await self._get_key_lock(key)
        async with lock:
            data = {'value': value}

            if expiry:
                data['expiry'] = time.time() + expiry

            await self._write_key_data(key, data)

    async def delete_key(self, key: str) -> bool:
        """Delete key."""
        lock = await self._get_key_lock(key)
        async with lock:
            return await self._delete_key_file(key)

    async def cleanup_expired(self) -> int:
        """Clean up expired keys."""
        cleaned = 0

        # Get all rate limit files
        for file_path in self.storage_dir.glob("*.json"):
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)

                    if 'expiry' in data and data['expiry'] <= time.time():
                        file_path.unlink()
                        cleaned += 1
            except (json.JSONDecodeError, OSError):
                # Remove corrupted files
                try:
                    file_path.unlink()
                    cleaned += 1
                except OSError:
                    pass

        return cleaned


# Redis storage implementation (requires redis package)
try:
    import redis.asyncio as redis

    class RedisRateLimitStorage(RateLimitStorage):
        """Redis storage for rate limiting."""

        def __init__(self, redis_url: str = "redis://localhost:6379", key_prefix: str = "rate_limit:"):
            """Initialize Redis storage.

            Args:
                redis_url: Redis connection URL
                key_prefix: Prefix for all rate limit keys
            """
            self.redis_url = redis_url
            self.key_prefix = key_prefix
            self._redis = None

        async def _get_redis(self):
            """Get Redis connection."""
            if self._redis is None:
                self._redis = redis.from_url(self.redis_url)
            return self._redis

        def _prefixed_key(self, key: str) -> str:
            """Add prefix to key."""
            return f"{self.key_prefix}{key}"

        async def get_counter(self, key: str) -> Optional[int]:
            """Get counter value."""
            r = await self._get_redis()
            value = await r.get(self._prefixed_key(key))
            return int(value) if value else None

        async def increment_counter(self, key: str, amount: int = 1, expiry: int = None) -> int:
            """Increment counter and return new value."""
            r = await self._get_redis()
            prefixed_key = self._prefixed_key(key)

            # Use pipeline for atomic operations
            async with r.pipeline() as pipe:
                await pipe.incrby(prefixed_key, amount)
                if expiry:
                    await pipe.expire(prefixed_key, expiry)
                results = await pipe.execute()
                return results[0]

        async def get_value(self, key: str) -> Any:
            """Get arbitrary value."""
            r = await self._get_redis()
            value = await r.get(self._prefixed_key(key))
            if value:
                import pickle
                return pickle.loads(value)
            return None

        async def set_value(self, key: str, value: Any, expiry: int = None) -> None:
            """Set arbitrary value."""
            r = await self._get_redis()
            import pickle
            serialized = pickle.dumps(value)
            prefixed_key = self._prefixed_key(key)

            if expiry:
                await r.setex(prefixed_key, expiry, serialized)
            else:
                await r.set(prefixed_key, serialized)

        async def delete_key(self, key: str) -> bool:
            """Delete key."""
            r = await self._get_redis()
            result = await r.delete(self._prefixed_key(key))
            return result > 0

        async def cleanup_expired(self) -> int:
            """Redis handles expiry automatically."""
            # Redis automatically removes expired keys
            # Return 0 as we can't know how many were cleaned
            return 0

        async def close(self):
            """Close Redis connection."""
            if self._redis:
                await self._redis.close()

except ImportError:
    class RedisRateLimitStorage:
        """Redis storage placeholder when redis is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Redis package not available. Install with: pip install redis")


def create_storage(storage_type: str = "memory", **kwargs) -> RateLimitStorage:
    """Factory function to create rate limit storage.

    Args:
        storage_type: Type of storage ('memory', 'file', 'redis')
        **kwargs: Additional arguments for storage initialization

    Returns:
        RateLimitStorage instance
    """
    if storage_type == "memory":
        return InMemoryRateLimitStorage(**kwargs)
    elif storage_type == "file":
        return FileBasedRateLimitStorage(**kwargs)
    elif storage_type == "redis":
        return RedisRateLimitStorage(**kwargs)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")