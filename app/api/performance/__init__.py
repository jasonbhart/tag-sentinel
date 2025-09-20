"""Performance optimization module for Tag Sentinel API.

This module provides performance optimizations for production deployment
including caching, connection pooling, and response optimization.
"""

from .caching import CacheManager, CacheConfig, cache_decorator
from .pooling import ConnectionPoolManager, PoolConfig
from .optimization import ResponseOptimizer, CompressionMiddleware, StaticFileOptimizer
from .background_tasks import BackgroundTaskManager, TaskConfig

__all__ = [
    "CacheManager",
    "CacheConfig",
    "cache_decorator",
    "ConnectionPoolManager",
    "PoolConfig",
    "ResponseOptimizer",
    "CompressionMiddleware",
    "StaticFileOptimizer",
    "BackgroundTaskManager",
    "TaskConfig"
]