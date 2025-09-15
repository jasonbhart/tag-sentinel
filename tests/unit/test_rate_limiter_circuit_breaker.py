"""Unit tests for rate limiter circuit breaker functionality."""

import pytest
import asyncio
import time
from unittest.mock import patch

from app.audit.queue.rate_limiter import HostRateLimiter, BackoffReason


class TestRateLimiterCircuitBreaker:
    """Test cases for circuit breaker functionality in rate limiter."""

    @pytest.fixture
    def limiter(self):
        """Create a rate limiter for testing."""
        return HostRateLimiter(
            host="test.example.com",
            requests_per_second=1.0,
            max_concurrent=2,
            base_delay=0.1,
            max_delay=1.0
        )

    def test_circuit_breaker_initialization(self, limiter):
        """Test circuit breaker is properly initialized."""
        assert limiter._consecutive_failures == 0
        assert limiter._circuit_open_until == 0.0
        assert limiter._circuit_failure_threshold == 10
        assert limiter._circuit_timeout == 300.0
        assert not limiter.is_circuit_open()

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, limiter):
        """Test circuit breaker opens after consecutive failures."""
        # Simulate 10 consecutive failures (threshold)
        for i in range(10):
            await limiter.record_failure(BackoffReason.CONNECTION_ERROR)
            assert limiter._consecutive_failures == i + 1

            # Circuit should not be open until we hit threshold
            if i < 9:
                assert not limiter.is_circuit_open()

        # After 10 failures, circuit should be open
        assert limiter.is_circuit_open()
        assert limiter._circuit_open_until > time.time()

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_requests_when_open(self, limiter):
        """Test circuit breaker rejects requests when open."""
        # Force circuit to open by simulating failures
        for _ in range(10):
            await limiter.record_failure(BackoffReason.TIMEOUT)

        assert limiter.is_circuit_open()

        # Acquire should return False immediately when circuit is open
        result = await limiter.acquire(timeout=1.0)
        assert result is False

        # Stats should reflect circuit breaker events
        stats = limiter.get_stats()
        assert stats["circuit_breaker_events"] > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_after_timeout(self, limiter):
        """Test circuit breaker resets after timeout period."""
        # Force circuit open
        for _ in range(10):
            await limiter.record_failure(BackoffReason.CONNECTION_ERROR)

        assert limiter.is_circuit_open()

        # Mock time to simulate timeout passing
        original_time = time.time()
        with patch('time.time', return_value=original_time + 301):  # 301 seconds later
            assert not limiter.is_circuit_open()
            assert limiter._consecutive_failures == 0
            assert limiter._circuit_open_until == 0.0

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self, limiter):
        """Test circuit breaker resets on successful request."""
        # Simulate some failures (but not enough to open circuit)
        for _ in range(5):
            await limiter.record_failure(BackoffReason.TIMEOUT)

        assert limiter._consecutive_failures == 5
        assert not limiter.is_circuit_open()

        # Record success - should reset failure count
        await limiter.record_success()
        assert limiter._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_backoff_interaction(self, limiter):
        """Test circuit breaker works alongside backoff mechanism."""
        # Simulate failures that trigger both backoff and circuit breaker
        for i in range(10):
            await limiter.record_failure(BackoffReason.CONNECTION_ERROR)

        # Circuit should be open
        assert limiter.is_circuit_open()

        # Backoff should also be active
        assert limiter._backoff.consecutive_failures == 10

        # Both mechanisms should prevent request acquisition
        result = await limiter.acquire(timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_stats_tracking(self, limiter):
        """Test circuit breaker events are properly tracked in stats."""
        initial_stats = limiter.get_stats()
        assert initial_stats["circuit_breaker_events"] == 0

        # Force circuit open
        for _ in range(10):
            await limiter.record_failure(BackoffReason.TIMEOUT)

        # Should have triggered circuit breaker event
        stats = limiter.get_stats()
        assert stats["circuit_breaker_events"] == 1

        # Multiple acquire attempts while circuit is open
        for _ in range(3):
            await limiter.acquire(timeout=0.1)

        # Should track additional circuit breaker events
        final_stats = limiter.get_stats()
        assert final_stats["circuit_breaker_events"] > stats["circuit_breaker_events"]

    @pytest.mark.asyncio
    async def test_different_error_types_contribute_to_circuit_breaker(self, limiter):
        """Test that different error types all contribute to circuit breaker."""
        error_types = [
            BackoffReason.TIMEOUT,
            BackoffReason.CONNECTION_ERROR,
            BackoffReason.TIMEOUT,
            BackoffReason.CONNECTION_ERROR,
            BackoffReason.TIMEOUT
        ]

        # Mix different error types
        for i, error_type in enumerate(error_types * 2):  # 10 total errors
            await limiter.record_failure(error_type)

        # Circuit should be open regardless of error type mix
        assert limiter.is_circuit_open()
        assert limiter._consecutive_failures == 10

    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout_configuration(self):
        """Test circuit breaker with different timeout configuration."""
        # Create limiter with shorter timeout for testing
        limiter = HostRateLimiter(
            host="test.example.com",
            requests_per_second=1.0,
            max_concurrent=2
        )

        # Override timeout for testing
        limiter._circuit_timeout = 1.0  # 1 second timeout

        # Force circuit open
        for _ in range(10):
            await limiter.record_failure(BackoffReason.TIMEOUT)

        circuit_open_time = limiter._circuit_open_until
        assert limiter.is_circuit_open()

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Circuit should reset automatically
        assert not limiter.is_circuit_open()
        assert limiter._circuit_open_until == 0.0
        assert limiter._consecutive_failures == 0

    def test_circuit_breaker_logging(self, limiter, caplog):
        """Test circuit breaker generates appropriate log messages."""
        import logging
        caplog.set_level(logging.WARNING)

        async def run_test():
            # Force circuit open
            for _ in range(10):
                await limiter.record_failure(BackoffReason.CONNECTION_ERROR)

        asyncio.run(run_test())

        # Should have logged circuit breaker opening
        assert any("Circuit breaker opened" in record.message for record in caplog.records)
        assert any("consecutive failures" in record.message for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__])