"""Rate limiting implementation for Tag Sentinel API.

This module provides configurable rate limiting with multiple algorithms
and storage backends for API protection.
"""

import logging
import time
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


class RateLimitScope(str, Enum):
    """Rate limiting scopes."""
    GLOBAL = "global"
    IP = "ip"
    USER = "user"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Basic rate limit settings
    requests_per_window: int = 100
    window_seconds: int = 60
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.FIXED_WINDOW
    scope: RateLimitScope = RateLimitScope.IP

    # Token bucket specific settings
    bucket_size: Optional[int] = None  # Defaults to requests_per_window
    refill_rate: Optional[float] = None  # Tokens per second

    # Sliding window specific settings
    sliding_window_precision: int = 10  # Number of sub-windows

    # Burst handling
    allow_burst: bool = True
    burst_multiplier: float = 1.5

    # Key generation
    key_prefix: str = "rate_limit"
    include_endpoint: bool = True
    include_method: bool = True

    # Response headers
    add_headers: bool = True
    header_prefix: str = "X-RateLimit"

    # Error handling
    error_message: str = "Rate limit exceeded"
    retry_after_header: bool = True

    @classmethod
    def from_string(cls, limit_string: str) -> "RateLimitConfig":
        """Create config from string like '100/minute' or '10/second'.

        Args:
            limit_string: Format like "100/minute", "10/second", "1000/hour"

        Returns:
            RateLimitConfig instance
        """
        try:
            requests, window = limit_string.split("/")
            requests = int(requests)

            window_map = {
                "second": 1,
                "minute": 60,
                "hour": 3600,
                "day": 86400
            }

            window_seconds = window_map.get(window.lower())
            if window_seconds is None:
                raise ValueError(f"Unknown window type: {window}")

            return cls(
                requests_per_window=requests,
                window_seconds=window_seconds
            )
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid rate limit string format '{limit_string}': {e}")


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        remaining: Optional[int] = None,
        reset_time: Optional[int] = None
    ):
        self.message = message
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        self.reset_time = reset_time
        super().__init__(self.message)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    limit: int
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None

    def to_headers(self, prefix: str = "X-RateLimit") -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            f"{prefix}-Limit": str(self.limit),
            f"{prefix}-Remaining": str(self.remaining),
            f"{prefix}-Reset": str(self.reset_time)
        }

        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)

        return headers


class RateLimiter:
    """Main rate limiter class with pluggable algorithms and storage."""

    def __init__(self, config: RateLimitConfig, storage: Optional["RateLimitStorage"] = None):
        """Initialize rate limiter.

        Args:
            config: Rate limiting configuration
            storage: Storage backend (defaults to in-memory)
        """
        self.config = config
        from .storage import InMemoryRateLimitStorage
        self.storage = storage or InMemoryRateLimitStorage()
        self._locks: Dict[str, asyncio.Lock] = {}

        # Validate config
        if config.bucket_size is None:
            config.bucket_size = config.requests_per_window
        if config.refill_rate is None:
            config.refill_rate = config.requests_per_window / config.window_seconds

        logger.info(f"RateLimiter initialized: {config.requests_per_window}/{config.window_seconds}s using {config.algorithm}")

    async def check_rate_limit(
        self,
        key: str,
        cost: int = 1,
        config_override: Optional[RateLimitConfig] = None
    ) -> RateLimitResult:
        """Check if request should be allowed based on rate limit.

        Args:
            key: Unique identifier for the rate limit (IP, user ID, etc.)
            cost: Cost of this request (default 1)
            config_override: Override default config for this check

        Returns:
            RateLimitResult with decision and metadata
        """
        config = config_override or self.config
        full_key = f"{config.key_prefix}:{key}"

        # Use per-key locks to prevent race conditions
        if full_key not in self._locks:
            self._locks[full_key] = asyncio.Lock()

        async with self._locks[full_key]:
            if config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                return await self._check_fixed_window(full_key, cost, config)
            elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return await self._check_sliding_window(full_key, cost, config)
            elif config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                return await self._check_token_bucket(full_key, cost, config)
            else:
                raise ValueError(f"Unknown algorithm: {config.algorithm}")

    async def _check_fixed_window(self, key: str, cost: int, config: RateLimitConfig) -> RateLimitResult:
        """Fixed window rate limiting."""
        now = int(time.time())
        window_start = (now // config.window_seconds) * config.window_seconds
        window_key = f"{key}:{window_start}"

        # Get current count
        current_count = await self.storage.get_counter(window_key) or 0

        # Calculate limits
        limit = config.requests_per_window
        if config.allow_burst:
            limit = int(limit * config.burst_multiplier)

        # Check if request would exceed limit
        if current_count + cost > limit:
            reset_time = window_start + config.window_seconds
            retry_after = reset_time - now

            return RateLimitResult(
                allowed=False,
                limit=limit,
                remaining=max(0, limit - current_count),
                reset_time=reset_time,
                retry_after=retry_after
            )

        # Increment counter
        await self.storage.increment_counter(window_key, cost, config.window_seconds)

        reset_time = window_start + config.window_seconds
        return RateLimitResult(
            allowed=True,
            limit=limit,
            remaining=max(0, limit - current_count - cost),
            reset_time=reset_time
        )

    async def _check_sliding_window(self, key: str, cost: int, config: RateLimitConfig) -> RateLimitResult:
        """Sliding window rate limiting."""
        now = time.time()
        window_size = config.window_seconds / config.sliding_window_precision

        # Get counts from sub-windows
        total_count = 0
        oldest_window = now - config.window_seconds

        for i in range(config.sliding_window_precision):
            window_start = int((now - i * window_size) // window_size) * window_size
            if window_start < oldest_window:
                break

            window_key = f"{key}:sliding:{window_start}"
            count = await self.storage.get_counter(window_key) or 0

            # Weight by how much of the window is within our time range
            window_end = window_start + window_size
            overlap = min(window_end, now) - max(window_start, oldest_window)
            weight = overlap / window_size

            total_count += count * weight

        # Check limit
        limit = config.requests_per_window
        if config.allow_burst:
            limit = int(limit * config.burst_multiplier)

        if total_count + cost > limit:
            reset_time = int(now + config.window_seconds)
            return RateLimitResult(
                allowed=False,
                limit=limit,
                remaining=max(0, int(limit - total_count)),
                reset_time=reset_time,
                retry_after=int(config.window_seconds)
            )

        # Add to current window
        current_window = int(now // window_size) * window_size
        window_key = f"{key}:sliding:{current_window}"
        await self.storage.increment_counter(window_key, cost, int(window_size) + 1)

        reset_time = int(now + config.window_seconds)
        return RateLimitResult(
            allowed=True,
            limit=limit,
            remaining=max(0, int(limit - total_count - cost)),
            reset_time=reset_time
        )

    async def _check_token_bucket(self, key: str, cost: int, config: RateLimitConfig) -> RateLimitResult:
        """Token bucket rate limiting."""
        now = time.time()

        # Get current bucket state
        bucket_data = await self.storage.get_value(key)
        if bucket_data:
            tokens, last_refill = bucket_data
        else:
            tokens = float(config.bucket_size)
            last_refill = now

        # Refill tokens based on time elapsed
        time_elapsed = now - last_refill
        tokens_to_add = time_elapsed * config.refill_rate
        tokens = min(config.bucket_size, tokens + tokens_to_add)

        # Check if we have enough tokens
        if tokens < cost:
            # Calculate when enough tokens will be available
            tokens_needed = cost - tokens
            retry_after = int(tokens_needed / config.refill_rate)

            return RateLimitResult(
                allowed=False,
                limit=config.bucket_size,
                remaining=int(tokens),
                reset_time=int(now + retry_after),
                retry_after=retry_after
            )

        # Consume tokens
        tokens -= cost

        # Store updated bucket state
        await self.storage.set_value(key, (tokens, now), config.window_seconds * 2)

        return RateLimitResult(
            allowed=True,
            limit=config.bucket_size,
            remaining=int(tokens),
            reset_time=int(now + (config.bucket_size - tokens) / config.refill_rate)
        )

    def generate_key(
        self,
        request_info: Dict[str, Any],
        config_override: Optional[RateLimitConfig] = None
    ) -> str:
        """Generate rate limit key based on request information.

        Args:
            request_info: Dictionary containing request details
            config_override: Override default config

        Returns:
            Generated rate limit key
        """
        config = config_override or self.config
        key_parts = []

        if config.scope == RateLimitScope.GLOBAL:
            key_parts.append("global")
        elif config.scope == RateLimitScope.IP:
            key_parts.append(f"ip:{request_info.get('client_ip', 'unknown')}")
        elif config.scope == RateLimitScope.USER:
            key_parts.append(f"user:{request_info.get('user_id', 'anonymous')}")
        elif config.scope == RateLimitScope.API_KEY:
            key_parts.append(f"key:{request_info.get('api_key', 'none')}")
        elif config.scope == RateLimitScope.ENDPOINT:
            endpoint = request_info.get('endpoint', 'unknown')
            if config.include_method:
                method = request_info.get('method', 'GET')
                endpoint = f"{method}:{endpoint}"
            key_parts.append(f"endpoint:{endpoint}")

        if config.include_endpoint and config.scope != RateLimitScope.ENDPOINT:
            endpoint = request_info.get('endpoint', 'unknown')
            if config.include_method:
                method = request_info.get('method', 'GET')
                endpoint = f"{method}:{endpoint}"
            key_parts.append(endpoint)

        return ":".join(key_parts)


