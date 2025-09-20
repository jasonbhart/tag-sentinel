"""Caching system for Tag Sentinel API performance optimization.

This module provides comprehensive caching mechanisms including
in-memory caching, Redis caching, and intelligent cache strategies.
"""

import logging
import time
import asyncio
import hashlib
import pickle
from typing import Any, Optional, Dict, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import functools
import json

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In, First Out


@dataclass
class CacheConfig:
    """Configuration for caching system."""

    # Cache settings
    enable_caching: bool = True
    default_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000
    strategy: CacheStrategy = CacheStrategy.LRU

    # Redis settings
    use_redis: bool = False
    redis_url: str = "redis://localhost:6379"
    redis_key_prefix: str = "tag_sentinel:cache:"

    # Serialization
    serialization_format: str = "pickle"  # pickle, json, msgpack
    compress_data: bool = True
    compression_threshold: int = 1024  # bytes

    # Performance
    cache_stats: bool = True
    background_cleanup: bool = True
    cleanup_interval: int = 300  # 5 minutes


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[int] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    expired: int = 0
    total_size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hits + self.misses
        if total_requests == 0:
            return 0.0
        return self.hits / total_requests

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class InMemoryCacheBackend(CacheBackend):
    """In-memory cache backend with configurable eviction strategies."""

    def __init__(self, config: CacheConfig):
        """Initialize in-memory cache backend."""
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self._lock = asyncio.Lock()

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        if config.background_cleanup:
            self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._cleanup_task = loop.create_task(self._periodic_cleanup())
        except RuntimeError:
            pass

    async def _periodic_cleanup(self):
        """Periodically clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup task: {e}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self.cache.get(key)

            if entry is None:
                self.stats.misses += 1
                return None

            if entry.is_expired:
                del self.cache[key]
                self.stats.expired += 1
                self.stats.misses += 1
                return None

            entry.touch()
            self.stats.hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        async with self._lock:
            # Calculate size
            try:
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
            except Exception:
                size_bytes = 0

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.config.default_ttl,
                size_bytes=size_bytes
            )

            # Check if we need to evict entries
            if len(self.cache) >= self.config.max_cache_size:
                await self._evict_entries()

            self.cache[key] = entry
            self.stats.sets += 1
            self.stats.total_size_bytes += size_bytes

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats.total_size_bytes -= entry.size_bytes
                del self.cache[key]
                self.stats.deletes += 1
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.stats = CacheStats()

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        async with self._lock:
            entry = self.cache.get(key)
            if entry and not entry.is_expired:
                return True
            return False

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        async with self._lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                sets=self.stats.sets,
                deletes=self.stats.deletes,
                evictions=self.stats.evictions,
                expired=self.stats.expired,
                total_size_bytes=self.stats.total_size_bytes
            )

    async def _evict_entries(self) -> None:
        """Evict entries based on configured strategy."""
        if not self.cache:
            return

        evict_count = max(1, len(self.cache) // 10)  # Evict 10% of entries

        if self.config.strategy == CacheStrategy.LRU:
            # Evict least recently used
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_access
            )
        elif self.config.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )
        elif self.config.strategy == CacheStrategy.FIFO:
            # Evict oldest entries
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].timestamp
            )
        else:  # TTL
            # Evict entries with shortest remaining TTL
            current_time = time.time()
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (x[1].timestamp + (x[1].ttl or 0)) - current_time
            )

        # Remove entries
        for key, entry in sorted_entries[:evict_count]:
            self.stats.total_size_bytes -= entry.size_bytes
            del self.cache[key]
            self.stats.evictions += 1

    async def _cleanup_expired(self) -> None:
        """Clean up expired entries."""
        async with self._lock:
            current_time = time.time()
            expired_keys = []

            for key, entry in self.cache.items():
                if entry.is_expired:
                    expired_keys.append(key)

            for key in expired_keys:
                entry = self.cache[key]
                self.stats.total_size_bytes -= entry.size_bytes
                del self.cache[key]
                self.stats.expired += 1

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def close(self) -> None:
        """Close cache backend and clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class RedisCacheBackend(CacheBackend):
    """Redis cache backend for distributed caching."""

    def __init__(self, config: CacheConfig):
        """Initialize Redis cache backend."""
        self.config = config
        self.stats = CacheStats()
        self._redis = None

    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.config.redis_url)
            except ImportError:
                raise ImportError("redis package not available. Install with: pip install redis")
        return self._redis

    def _get_cache_key(self, key: str) -> str:
        """Get full cache key with prefix."""
        return f"{self.config.redis_key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            redis_client = await self._get_redis()
            cache_key = self._get_cache_key(key)

            data = await redis_client.get(cache_key)
            if data is None:
                self.stats.misses += 1
                return None

            # Deserialize value
            value = self._deserialize(data)
            self.stats.hits += 1
            return value

        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            self.stats.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache."""
        try:
            redis_client = await self._get_redis()
            cache_key = self._get_cache_key(key)

            # Serialize value
            data = self._serialize(value)

            # Set in Redis with TTL
            ttl_seconds = ttl or self.config.default_ttl
            await redis_client.setex(cache_key, ttl_seconds, data)

            self.stats.sets += 1

        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            redis_client = await self._get_redis()
            cache_key = self._get_cache_key(key)

            result = await redis_client.delete(cache_key)
            deleted = result > 0

            if deleted:
                self.stats.deletes += 1

            return deleted

        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cache entries with our prefix."""
        try:
            redis_client = await self._get_redis()
            pattern = f"{self.config.redis_key_prefix}*"

            keys = await redis_client.keys(pattern)
            if keys:
                await redis_client.delete(*keys)

        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            redis_client = await self._get_redis()
            cache_key = self._get_cache_key(key)

            result = await redis_client.exists(cache_key)
            return result > 0

        except Exception as e:
            logger.error(f"Error checking Redis cache existence: {e}")
            return False

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(
            hits=self.stats.hits,
            misses=self.stats.misses,
            sets=self.stats.sets,
            deletes=self.stats.deletes,
            evictions=0,  # Redis handles eviction
            expired=0,    # Redis handles expiration
            total_size_bytes=0  # Not easily available in Redis
        )

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.config.serialization_format == "json":
            try:
                data = json.dumps(value).encode('utf-8')
            except (TypeError, ValueError):
                # Fallback to pickle for complex objects
                data = pickle.dumps(value)
        else:
            data = pickle.dumps(value)

        # Compress if enabled and data is large enough
        if (self.config.compress_data and
            len(data) > self.config.compression_threshold):
            try:
                import gzip
                data = gzip.compress(data)
            except ImportError:
                pass

        return data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        # Try to decompress first
        if self.config.compress_data:
            try:
                import gzip
                data = gzip.decompress(data)
            except (ImportError, OSError):
                # Data might not be compressed
                pass

        # Try JSON first if configured
        if self.config.serialization_format == "json":
            try:
                return json.loads(data.decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                # Fallback to pickle
                pass

        return pickle.loads(data)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()


class CacheManager:
    """Main cache manager that handles multiple cache backends."""

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager."""
        self.config = config or CacheConfig()

        # Initialize backend
        if self.config.use_redis:
            self.backend = RedisCacheBackend(self.config)
        else:
            self.backend = InMemoryCacheBackend(self.config)

        logger.info(f"CacheManager initialized with {type(self.backend).__name__}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.config.enable_caching:
            return None

        return await self.backend.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if not self.config.enable_caching:
            return

        await self.backend.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.config.enable_caching:
            return False

        return await self.backend.delete(key)

    async def clear(self) -> None:
        """Clear all cache entries."""
        await self.backend.clear()

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.config.enable_caching:
            return False

        return await self.backend.exists(key)

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return await self.backend.get_stats()

    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a deterministic key from arguments
        key_data = {
            "args": args,
            "kwargs": {k: v for k, v in sorted(kwargs.items())}
        }

        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        return key_hash

    async def close(self) -> None:
        """Close cache manager and clean up resources."""
        await self.backend.close()
        logger.info("CacheManager closed")


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def set_cache_manager(manager: CacheManager) -> None:
    """Set global cache manager instance."""
    global _global_cache_manager
    _global_cache_manager = manager


def cache_decorator(
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
    cache_manager: Optional[CacheManager] = None
):
    """Decorator for caching function results.

    Args:
        ttl: Time to live for cached values
        key_func: Custom function to generate cache keys
        cache_manager: Cache manager to use (uses global if None)

    Example:
        @cache_decorator(ttl=300)
        async def expensive_function(param1, param2):
            # Expensive computation
            return result
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = cache_manager or get_cache_manager()

            if not manager.config.enable_caching:
                return await func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{manager.generate_key(*args, **kwargs)}"

            # Try to get from cache
            cached_result = await manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await manager.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator