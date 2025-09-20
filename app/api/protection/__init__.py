"""API protection module for Tag Sentinel API.

This module provides rate limiting, abuse detection, and API protection
mechanisms with configurable policies and monitoring.
"""

from .rate_limiter import RateLimiter, RateLimitConfig, RateLimitExceededError, RateLimitResult
from .middleware import RateLimitMiddleware, APIProtectionMiddleware
from .policies import RateLimitPolicy, FixedWindowPolicy, SlidingWindowPolicy, TokenBucketPolicy, PolicyManager
from .storage import RateLimitStorage, InMemoryRateLimitStorage, FileBasedRateLimitStorage, RedisRateLimitStorage

__all__ = [
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitExceededError",
    "RateLimitResult",
    "RateLimitMiddleware",
    "APIProtectionMiddleware",
    "RateLimitPolicy",
    "FixedWindowPolicy",
    "SlidingWindowPolicy",
    "TokenBucketPolicy",
    "PolicyManager",
    "RateLimitStorage",
    "InMemoryRateLimitStorage",
    "FileBasedRateLimitStorage",
    "RedisRateLimitStorage"
]