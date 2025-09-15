"""Per-host rate limiting with token bucket algorithm and exponential backoff.

This module implements politeness controls for web crawling, including
per-host rate limiting, exponential backoff, and respect for server responses.
"""

import asyncio
import time
import random
from typing import Dict, Optional, Any
from urllib.parse import urlparse
import logging
from dataclasses import dataclass
from enum import Enum
import threading

from ..utils.url_normalizer import normalize, URLNormalizationError


logger = logging.getLogger(__name__)


class BackoffReason(Enum):
    """Reasons for applying backoff delays."""
    RATE_LIMIT = "rate_limit"          # 429 Too Many Requests
    SERVER_ERROR = "server_error"       # 5xx server errors
    TIMEOUT = "timeout"                 # Request timeout
    CONNECTION_ERROR = "connection"     # Network connection error
    RETRY_AFTER = "retry_after"        # Explicit Retry-After header


@dataclass
class BackoffState:
    """State tracking for exponential backoff per host."""
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    current_delay: float = 1.0
    reason: Optional[BackoffReason] = None
    retry_after_until: float = 0.0
    
    def calculate_delay(self, base_delay: float = 1.0, max_delay: float = 300.0, jitter: bool = True) -> float:
        """Calculate the next backoff delay."""
        if self.retry_after_until > time.time():
            return self.retry_after_until - time.time()
        
        if self.consecutive_failures == 0:
            return 0.0
        
        # Exponential backoff: base_delay * 2^(failures-1)
        delay = base_delay * (2 ** (self.consecutive_failures - 1))
        delay = min(delay, max_delay)
        
        if jitter:
            # Add ±10% jitter to prevent thundering herd
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.0, delay)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: float
    tokens: float
    last_refill: float
    refill_rate: float  # tokens per second
    
    def can_consume(self, tokens: int = 1) -> bool:
        """Check if tokens can be consumed without refilling."""
        return self.tokens >= tokens
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.last_refill = now
        
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
    
    def time_until_tokens_available(self, tokens: int = 1) -> float:
        """Calculate time until requested tokens will be available."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class HostRateLimiter:
    """Rate limiter for a specific host with concurrent request tracking."""
    
    def __init__(
        self,
        host: str,
        requests_per_second: float = 2.0,
        max_concurrent: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 300.0
    ):
        """Initialize host rate limiter.
        
        Args:
            host: Hostname this limiter applies to
            requests_per_second: Maximum requests per second
            max_concurrent: Maximum concurrent requests
            base_delay: Base delay for exponential backoff
            max_delay: Maximum backoff delay
        """
        self.host = host
        self.requests_per_second = requests_per_second
        self.max_concurrent = max_concurrent
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Token bucket for rate limiting
        capacity = max(requests_per_second * 2, 1.0)  # Allow some burst capacity
        self._bucket = TokenBucket(
            capacity=capacity,
            tokens=capacity,
            last_refill=time.time(),
            refill_rate=requests_per_second
        )
        
        # Concurrent request tracking
        self._concurrent_requests = 0
        self._concurrent_lock = asyncio.Lock()
        self._concurrent_condition = asyncio.Condition()
        
        # Backoff state
        self._backoff = BackoffState()
        self._backoff_lock = asyncio.Lock()

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_failure_threshold = 10  # Open circuit after 10 consecutive failures
        self._circuit_timeout = 300.0  # Keep circuit open for 5 minutes

        # Statistics
        self._stats = {
            "total_requests": 0,
            "rate_limited": 0,
            "concurrent_limited": 0,
            "backoff_events": 0,
            "retry_after_events": 0,
            "circuit_breaker_events": 0
        }
    
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open.

        Returns:
            True if circuit is open (requests should be rejected)
        """
        current_time = time.time()
        if current_time >= self._circuit_open_until:
            # Circuit timeout has expired, reset if it was open
            if self._circuit_open_until > 0:
                self._circuit_open_until = 0.0
                self._consecutive_failures = 0
                logger.info(f"Circuit breaker for {self.host} reset after timeout")
            return False
        return True

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission to make a request.

        Args:
            timeout: Maximum time to wait for permission

        Returns:
            True if permission granted, False if timeout or circuit open
        """
        # Check circuit breaker first
        if self.is_circuit_open():
            self._stats["circuit_breaker_events"] += 1
            logger.debug(f"Circuit breaker open for {self.host}, rejecting request")
            return False

        start_time = time.time()

        while True:
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return False

            # Check backoff delay
            async with self._backoff_lock:
                backoff_delay = self._backoff.calculate_delay(self.base_delay, self.max_delay)
                if backoff_delay > 0:
                    logger.debug(f"Host {self.host} in backoff for {backoff_delay:.2f}s")
                    await asyncio.sleep(min(backoff_delay, 1.0))  # Check frequently
                    continue
            
            # Check concurrent limit
            async with self._concurrent_condition:
                while self._concurrent_requests >= self.max_concurrent:
                    self._stats["concurrent_limited"] += 1
                    logger.debug(f"Host {self.host} at concurrent limit ({self._concurrent_requests})")
                    
                    try:
                        remaining_timeout = None
                        if timeout:
                            remaining_timeout = timeout - (time.time() - start_time)
                            if remaining_timeout <= 0:
                                return False
                        
                        await asyncio.wait_for(
                            self._concurrent_condition.wait(), 
                            timeout=remaining_timeout
                        )
                    except asyncio.TimeoutError:
                        return False
                
                # Check rate limit
                if not self._bucket.can_consume():
                    wait_time = self._bucket.time_until_tokens_available()
                    self._stats["rate_limited"] += 1
                    logger.debug(f"Host {self.host} rate limited, waiting {wait_time:.2f}s")
                    
                    # Wait for tokens or timeout
                    try:
                        remaining_timeout = None
                        if timeout:
                            remaining_timeout = timeout - (time.time() - start_time)
                            if remaining_timeout <= 0:
                                return False
                            wait_time = min(wait_time, remaining_timeout)
                        
                        await asyncio.sleep(wait_time)
                        continue
                    except asyncio.CancelledError:
                        return False
                
                # All checks passed, consume token and increment concurrent counter
                if self._bucket.consume():
                    self._concurrent_requests += 1
                    self._stats["total_requests"] += 1
                    logger.debug(f"Host {self.host} request acquired ({self._concurrent_requests}/{self.max_concurrent})")
                    return True
    
    async def release(self):
        """Release a concurrent request slot."""
        async with self._concurrent_condition:
            if self._concurrent_requests > 0:
                self._concurrent_requests -= 1
                self._concurrent_condition.notify_all()
                logger.debug(f"Host {self.host} request released ({self._concurrent_requests}/{self.max_concurrent})")
    
    async def record_success(self):
        """Record a successful request (resets backoff and circuit breaker)."""
        async with self._backoff_lock:
            if self._backoff.consecutive_failures > 0:
                logger.debug(f"Host {self.host} backoff reset after success")
                self._backoff.consecutive_failures = 0
                self._backoff.current_delay = self.base_delay
                self._backoff.reason = None

            # Reset circuit breaker state on success
            if self._consecutive_failures > 0:
                logger.debug(f"Host {self.host} circuit breaker reset after success")
                self._consecutive_failures = 0
                self._circuit_open_until = 0.0
    
    async def record_failure(
        self, 
        reason: BackoffReason, 
        retry_after: Optional[float] = None
    ):
        """Record a failed request (triggers backoff).
        
        Args:
            reason: Reason for the failure
            retry_after: Explicit retry-after delay in seconds
        """
        async with self._backoff_lock:
            self._backoff.consecutive_failures += 1
            self._backoff.last_failure_time = time.time()
            self._backoff.reason = reason

            # Update circuit breaker state
            self._consecutive_failures += 1

            # Check if we should open the circuit breaker
            if self._consecutive_failures >= self._circuit_failure_threshold:
                self._circuit_open_until = time.time() + self._circuit_timeout
                logger.warning(
                    f"Circuit breaker opened for {self.host} after {self._consecutive_failures} "
                    f"consecutive failures. Blocked for {self._circuit_timeout}s"
                )
                self._stats["circuit_breaker_events"] += 1

            if retry_after:
                self._backoff.retry_after_until = time.time() + retry_after
                self._stats["retry_after_events"] += 1
                logger.info(f"Host {self.host} explicit retry-after: {retry_after}s")

            self._stats["backoff_events"] += 1
            delay = self._backoff.calculate_delay(self.base_delay, self.max_delay)
            logger.info(f"Host {self.host} backoff triggered: {reason.value}, delay: {delay:.2f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this host limiter."""
        return {
            "host": self.host,
            "requests_per_second": self.requests_per_second,
            "max_concurrent": self.max_concurrent,
            "current_concurrent": self._concurrent_requests,
            "backoff_failures": self._backoff.consecutive_failures,
            "backoff_reason": self._backoff.reason.value if self._backoff.reason else None,
            "bucket_tokens": self._bucket.tokens,
            "bucket_capacity": self._bucket.capacity,
            **self._stats
        }


class PerHostRateLimiter:
    """Per-host rate limiting with automatic limiter creation and management."""
    
    def __init__(
        self,
        default_requests_per_second: float = 2.0,
        default_max_concurrent: int = 2,
        default_base_delay: float = 1.0,
        default_max_delay: float = 300.0,
        cleanup_interval: float = 600.0  # 10 minutes
    ):
        """Initialize per-host rate limiter.
        
        Args:
            default_requests_per_second: Default RPS limit for new hosts
            default_max_concurrent: Default concurrent limit for new hosts
            default_base_delay: Default base backoff delay
            default_max_delay: Default maximum backoff delay
            cleanup_interval: Interval to cleanup unused limiters
        """
        self._default_rps = default_requests_per_second
        self._default_concurrent = default_max_concurrent
        self._default_base_delay = default_base_delay
        self._default_max_delay = default_max_delay
        
        self._limiters: Dict[str, HostRateLimiter] = {}
        self._limiter_access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
        # Start cleanup task
        self._cleanup_task = None
        if cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop(cleanup_interval))
    
    async def get_limiter(self, url: str) -> HostRateLimiter:
        """Get or create rate limiter for URL's host.
        
        Args:
            url: URL to get limiter for
            
        Returns:
            HostRateLimiter for the URL's host
        """
        try:
            normalized_url = normalize(url)
            parsed = urlparse(normalized_url)
            host = parsed.netloc.lower()
        except (URLNormalizationError, Exception):
            # Fallback to simple parsing for invalid URLs
            try:
                parsed = urlparse(url)
                host = parsed.netloc.lower()
            except Exception:
                host = "unknown"
        
        async with self._lock:
            if host not in self._limiters:
                self._limiters[host] = HostRateLimiter(
                    host=host,
                    requests_per_second=self._default_rps,
                    max_concurrent=self._default_concurrent,
                    base_delay=self._default_base_delay,
                    max_delay=self._default_max_delay
                )
                logger.debug(f"Created rate limiter for host: {host}")
            
            self._limiter_access_times[host] = time.time()
            return self._limiters[host]
    
    async def acquire(self, url: str, timeout: Optional[float] = None) -> Optional[HostRateLimiter]:
        """Acquire permission to make request to URL.
        
        Args:
            url: URL to make request to
            timeout: Maximum time to wait
            
        Returns:
            HostRateLimiter if permission granted, None if timeout
        """
        limiter = await self.get_limiter(url)
        if await limiter.acquire(timeout=timeout):
            return limiter
        return None
    
    async def record_response(
        self, 
        url: str, 
        status_code: int, 
        retry_after: Optional[float] = None
    ):
        """Record response for rate limiting decisions.
        
        Args:
            url: URL that was requested
            status_code: HTTP status code received
            retry_after: Retry-After header value in seconds
        """
        limiter = await self.get_limiter(url)
        
        if 200 <= status_code < 300:
            await limiter.record_success()
        elif status_code == 429:
            await limiter.record_failure(BackoffReason.RATE_LIMIT, retry_after)
        elif 500 <= status_code < 600:
            await limiter.record_failure(BackoffReason.SERVER_ERROR, retry_after)
    
    async def record_error(self, url: str, error_type: str):
        """Record error for rate limiting decisions.
        
        Args:
            url: URL that failed
            error_type: Type of error (timeout, connection, etc.)
        """
        limiter = await self.get_limiter(url)
        
        if error_type == "timeout":
            await limiter.record_failure(BackoffReason.TIMEOUT)
        elif error_type in ("connection", "network"):
            await limiter.record_failure(BackoffReason.CONNECTION_ERROR)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all host limiters."""
        return {host: limiter.get_stats() for host, limiter in self._limiters.items()}
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all hosts."""
        total_hosts = len(self._limiters)
        total_requests = sum(limiter._stats["total_requests"] for limiter in self._limiters.values())
        total_rate_limited = sum(limiter._stats["rate_limited"] for limiter in self._limiters.values())
        total_backoff_events = sum(limiter._stats["backoff_events"] for limiter in self._limiters.values())
        
        hosts_in_backoff = sum(
            1 for limiter in self._limiters.values() 
            if limiter._backoff.consecutive_failures > 0
        )
        
        return {
            "total_hosts": total_hosts,
            "hosts_in_backoff": hosts_in_backoff,
            "total_requests": total_requests,
            "total_rate_limited": total_rate_limited,
            "total_backoff_events": total_backoff_events,
            "default_requests_per_second": self._default_rps,
            "default_max_concurrent": self._default_concurrent
        }
    
    async def close(self):
        """Close the rate limiter and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Per-host rate limiter closed")
    
    async def _cleanup_loop(self, interval: float):
        """Cleanup unused limiters periodically."""
        while True:
            try:
                await asyncio.sleep(interval)
                await self._cleanup_unused_limiters(interval * 2)  # Cleanup if unused for 2x interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {e}")
    
    async def _cleanup_unused_limiters(self, max_age: float):
        """Remove limiters that haven't been used recently."""
        now = time.time()
        hosts_to_remove = []
        
        async with self._lock:
            for host, access_time in self._limiter_access_times.items():
                if (now - access_time) > max_age:
                    limiter = self._limiters[host]
                    # Only cleanup if no concurrent requests
                    if limiter._concurrent_requests == 0:
                        hosts_to_remove.append(host)
            
            for host in hosts_to_remove:
                del self._limiters[host]
                del self._limiter_access_times[host]
                logger.debug(f"Cleaned up unused rate limiter for host: {host}")
        
        if hosts_to_remove:
            logger.info(f"Cleaned up {len(hosts_to_remove)} unused rate limiters")