"""Rate limiting policies for Tag Sentinel API.

This module provides predefined rate limiting policies for different
types of API endpoints and users with configurable parameters.
"""

import logging
from typing import Dict, List, Optional, Pattern
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re

from .rate_limiter import RateLimitConfig, RateLimitAlgorithm, RateLimitScope

logger = logging.getLogger(__name__)


class RateLimitPolicy(ABC):
    """Abstract base class for rate limiting policies."""

    @abstractmethod
    def get_config(self, endpoint: str, user_type: str = "anonymous") -> RateLimitConfig:
        """Get rate limit configuration for endpoint and user type."""
        pass

    @abstractmethod
    def matches(self, endpoint: str) -> bool:
        """Check if this policy applies to the given endpoint."""
        pass


@dataclass
class EndpointPattern:
    """Pattern matching for endpoints."""
    pattern: str
    regex: Pattern = None

    def __post_init__(self):
        if self.regex is None:
            self.regex = re.compile(self.pattern)

    def matches(self, endpoint: str) -> bool:
        """Check if endpoint matches this pattern."""
        return bool(self.regex.match(endpoint))


class FixedWindowPolicy(RateLimitPolicy):
    """Fixed window rate limiting policy."""

    def __init__(
        self,
        patterns: List[str],
        requests_per_window: int = 100,
        window_seconds: int = 60,
        scope: RateLimitScope = RateLimitScope.IP,
        authenticated_multiplier: float = 2.0
    ):
        """Initialize fixed window policy.

        Args:
            patterns: List of endpoint patterns this policy applies to
            requests_per_window: Number of requests allowed per window
            window_seconds: Window duration in seconds
            scope: Rate limiting scope
            authenticated_multiplier: Multiplier for authenticated users
        """
        self.patterns = [EndpointPattern(p) for p in patterns]
        self.base_config = RateLimitConfig(
            requests_per_window=requests_per_window,
            window_seconds=window_seconds,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=scope
        )
        self.authenticated_multiplier = authenticated_multiplier

    def matches(self, endpoint: str) -> bool:
        """Check if this policy applies to the endpoint."""
        return any(pattern.matches(endpoint) for pattern in self.patterns)

    def get_config(self, endpoint: str, user_type: str = "anonymous") -> RateLimitConfig:
        """Get rate limit configuration."""
        config = RateLimitConfig(
            requests_per_window=self.base_config.requests_per_window,
            window_seconds=self.base_config.window_seconds,
            algorithm=self.base_config.algorithm,
            scope=self.base_config.scope
        )

        # Apply multiplier for authenticated users
        if user_type == "authenticated":
            config.requests_per_window = int(config.requests_per_window * self.authenticated_multiplier)

        return config


class SlidingWindowPolicy(RateLimitPolicy):
    """Sliding window rate limiting policy."""

    def __init__(
        self,
        patterns: List[str],
        requests_per_window: int = 100,
        window_seconds: int = 60,
        precision: int = 10,
        scope: RateLimitScope = RateLimitScope.IP
    ):
        """Initialize sliding window policy."""
        self.patterns = [EndpointPattern(p) for p in patterns]
        self.base_config = RateLimitConfig(
            requests_per_window=requests_per_window,
            window_seconds=window_seconds,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            scope=scope,
            sliding_window_precision=precision
        )

    def matches(self, endpoint: str) -> bool:
        """Check if this policy applies to the endpoint."""
        return any(pattern.matches(endpoint) for pattern in self.patterns)

    def get_config(self, endpoint: str, user_type: str = "anonymous") -> RateLimitConfig:
        """Get rate limit configuration."""
        return self.base_config


class TokenBucketPolicy(RateLimitPolicy):
    """Token bucket rate limiting policy."""

    def __init__(
        self,
        patterns: List[str],
        bucket_size: int = 100,
        refill_rate: float = 1.0,
        scope: RateLimitScope = RateLimitScope.IP
    ):
        """Initialize token bucket policy."""
        self.patterns = [EndpointPattern(p) for p in patterns]
        self.base_config = RateLimitConfig(
            requests_per_window=bucket_size,
            window_seconds=60,  # Not used for token bucket
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=scope,
            bucket_size=bucket_size,
            refill_rate=refill_rate
        )

    def matches(self, endpoint: str) -> bool:
        """Check if this policy applies to the endpoint."""
        return any(pattern.matches(endpoint) for pattern in self.patterns)

    def get_config(self, endpoint: str, user_type: str = "anonymous") -> RateLimitConfig:
        """Get rate limit configuration."""
        return self.base_config


class PolicyManager:
    """Manages multiple rate limiting policies."""

    def __init__(self):
        """Initialize policy manager."""
        self.policies: List[RateLimitPolicy] = []
        self.default_policy = FixedWindowPolicy(
            patterns=[".*"],  # Matches all endpoints
            requests_per_window=100,
            window_seconds=60
        )

    def add_policy(self, policy: RateLimitPolicy) -> None:
        """Add a rate limiting policy."""
        self.policies.append(policy)

    def get_config(self, endpoint: str, user_type: str = "anonymous") -> RateLimitConfig:
        """Get rate limit configuration for endpoint."""
        # Find first matching policy
        for policy in self.policies:
            if policy.matches(endpoint):
                return policy.get_config(endpoint, user_type)

        # Fall back to default policy
        return self.default_policy.get_config(endpoint, user_type)

    def set_default_policy(self, policy: RateLimitPolicy) -> None:
        """Set default rate limiting policy."""
        self.default_policy = policy


# Predefined policies for common API patterns

def create_standard_policies() -> PolicyManager:
    """Create standard rate limiting policies for Tag Sentinel API."""
    manager = PolicyManager()

    # High-frequency endpoints (health checks, etc.)
    health_policy = FixedWindowPolicy(
        patterns=[r"/health", r"/ping", r"/status"],
        requests_per_window=1000,
        window_seconds=60,
        scope=RateLimitScope.IP
    )
    manager.add_policy(health_policy)

    # Authentication endpoints (stricter limits)
    auth_policy = SlidingWindowPolicy(
        patterns=[r"/auth/.*", r"/login", r"/logout"],
        requests_per_window=10,
        window_seconds=60,
        precision=6,
        scope=RateLimitScope.IP
    )
    manager.add_policy(auth_policy)

    # Export endpoints (resource intensive)
    export_policy = TokenBucketPolicy(
        patterns=[r"/api/v1/audits/.*/exports/.*"],
        bucket_size=5,
        refill_rate=0.1,  # 1 token per 10 seconds
        scope=RateLimitScope.USER
    )
    manager.add_policy(export_policy)

    # Audit creation (moderate limits)
    audit_create_policy = FixedWindowPolicy(
        patterns=[r"/api/v1/audits$"],
        requests_per_window=20,
        window_seconds=60,
        scope=RateLimitScope.USER,
        authenticated_multiplier=2.0
    )
    manager.add_policy(audit_create_policy)

    # General API endpoints
    api_policy = FixedWindowPolicy(
        patterns=[r"/api/v1/.*"],
        requests_per_window=200,
        window_seconds=60,
        scope=RateLimitScope.IP,
        authenticated_multiplier=3.0
    )
    manager.add_policy(api_policy)

    # Documentation and static files (relaxed limits)
    docs_policy = FixedWindowPolicy(
        patterns=[r"/docs.*", r"/redoc.*", r"/openapi\.json", r"/static/.*"],
        requests_per_window=500,
        window_seconds=60,
        scope=RateLimitScope.IP
    )
    manager.add_policy(docs_policy)

    return manager


def create_development_policies() -> PolicyManager:
    """Create relaxed policies for development environment."""
    manager = PolicyManager()

    # Very permissive policy for development
    dev_policy = FixedWindowPolicy(
        patterns=[".*"],
        requests_per_window=10000,
        window_seconds=60,
        scope=RateLimitScope.IP
    )
    manager.set_default_policy(dev_policy)

    return manager


def create_production_policies() -> PolicyManager:
    """Create strict policies for production environment."""
    manager = PolicyManager()

    # Stricter limits for production

    # Authentication - very strict
    auth_policy = FixedWindowPolicy(
        patterns=[r"/auth/.*"],
        requests_per_window=5,
        window_seconds=60,
        scope=RateLimitScope.IP
    )
    manager.add_policy(auth_policy)

    # Exports - extremely limited
    export_policy = TokenBucketPolicy(
        patterns=[r"/api/v1/audits/.*/exports/.*"],
        bucket_size=2,
        refill_rate=0.05,  # 1 token per 20 seconds
        scope=RateLimitScope.USER
    )
    manager.add_policy(export_policy)

    # API endpoints - moderate but strict
    api_policy = SlidingWindowPolicy(
        patterns=[r"/api/v1/.*"],
        requests_per_window=100,
        window_seconds=60,
        precision=10,
        scope=RateLimitScope.IP
    )
    manager.add_policy(api_policy)

    # Default strict policy
    default_policy = FixedWindowPolicy(
        patterns=[".*"],
        requests_per_window=50,
        window_seconds=60,
        scope=RateLimitScope.IP
    )
    manager.set_default_policy(default_policy)

    return manager