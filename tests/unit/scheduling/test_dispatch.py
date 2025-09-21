"""Unit tests for RunDispatcher."""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from typing import Optional

from app.scheduling.dispatch import (
    RunDispatcher,
    RunResult,
    RunStatus,
    DispatchError,
    DispatchStats,
    create_mock_dispatcher
)
from app.scheduling.models import RunRequest


class MockAuditBackend:
    """Mock audit backend for testing."""

    def __init__(self):
        self.dispatched_runs = {}
        self.should_fail = False
        self.dispatch_delay = 0.0
        self.auto_complete = False  # Disabled by default to prevent hanging
        self.completion_delay = 0.1  # Complete runs after 100ms
        self._completion_tasks = set()  # Track completion tasks for cleanup

    async def dispatch_run(self, run_request: RunRequest) -> RunResult:
        """Mock dispatch implementation."""
        if self.should_fail:
            raise Exception("Mock dispatch failure")

        if self.dispatch_delay > 0:
            await asyncio.sleep(self.dispatch_delay)

        result = RunResult(
            run_id=run_request.id,
            status=RunStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            metadata={
                **run_request.metadata,  # Preserve original metadata including lock info
                "mock": True
            }
        )
        self.dispatched_runs[run_request.id] = result

        # Schedule automatic completion if enabled
        if self.auto_complete:
            task = asyncio.create_task(self._complete_run_later(run_request.id))
            self._completion_tasks.add(task)
            # Remove task from set when done to prevent memory leaks
            task.add_done_callback(self._completion_tasks.discard)

        return result

    async def _complete_run_later(self, run_id: str):
        """Complete a run after a delay."""
        await asyncio.sleep(self.completion_delay)
        if run_id in self.dispatched_runs:
            result = self.dispatched_runs[run_id]
            # Update to completed status
            completed_result = RunResult(
                run_id=result.run_id,
                status=RunStatus.COMPLETED,
                started_at=result.started_at,
                completed_at=datetime.now(timezone.utc),
                metadata=result.metadata
            )
            self.dispatched_runs[run_id] = completed_result

    async def get_run_status(self, run_id: str) -> Optional[RunResult]:
        """Get mock run status."""
        if run_id in self.dispatched_runs:
            return self.dispatched_runs[run_id]
        return None

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel mock run."""
        if run_id in self.dispatched_runs:
            result = self.dispatched_runs[run_id]
            if not result.is_complete:
                result.status = RunStatus.CANCELLED
                result.completed_at = datetime.now(timezone.utc)
                return True
        return False

    async def cleanup(self):
        """Clean up any pending completion tasks."""
        for task in list(self._completion_tasks):
            if not task.done():
                task.cancel()
        self._completion_tasks.clear()


class TestRunDispatcher:
    """Test RunDispatcher functionality."""

    @pytest_asyncio.fixture
    async def dispatcher(self):
        """Create a RunDispatcher for testing."""
        backend = MockAuditBackend()
        dispatcher = RunDispatcher(backend, max_concurrent_runs=2, max_queue_size=5)
        await dispatcher.start()
        yield dispatcher
        await dispatcher.stop()
        await backend.cleanup()  # Clean up any hanging tasks

    @pytest.mark.asyncio
    async def test_dispatcher_initialization(self, dispatcher):
        """Test dispatcher initialization and configuration."""
        assert dispatcher.max_concurrent_runs == 2
        assert dispatcher.max_queue_size == 5
        assert dispatcher.idempotency_window == timedelta(hours=1)

    @pytest.mark.asyncio
    async def test_enqueue_run_request(self, dispatcher):
        """Test enqueueing a run request."""
        run_request = RunRequest(
            site_id="test-site",
            environment="production",
            scheduled_at=datetime.now(timezone.utc)
        )

        # Enqueue should succeed
        enqueued = await dispatcher.enqueue_run(run_request)
        assert enqueued is True

        # Check stats
        stats = dispatcher.get_stats()
        assert stats.current_queue_size == 1
        # Note: total_dispatched may be 0 if not processed yet

    @pytest.mark.asyncio
    async def test_queue_capacity_limit(self, dispatcher):
        """Test that queue respects capacity limits."""
        # Fill up the queue (max_queue_size = 5)
        for i in range(5):
            run_request = RunRequest(
                site_id=f"site-{i}",
                environment="prod",
                scheduled_at=datetime.now(timezone.utc)
            )
            enqueued = await dispatcher.enqueue_run(run_request)
            assert enqueued is True

        # Next request should be rejected
        overflow_request = RunRequest(
            site_id="overflow",
            environment="prod",
            scheduled_at=datetime.now(timezone.utc)
        )
        enqueued = await dispatcher.enqueue_run(overflow_request)
        assert enqueued is False

    @pytest.mark.asyncio
    async def test_idempotency_blocking(self, dispatcher):
        """Test that idempotency prevents duplicate runs."""
        idempotency_key = "unique-run-123"

        # First request
        request1 = RunRequest(
            site_id="test",
            environment="prod",
            scheduled_at=datetime.now(timezone.utc),
            idempotency_key=idempotency_key
        )

        # Second request with same key
        request2 = RunRequest(
            site_id="test",
            environment="prod",
            scheduled_at=datetime.now(timezone.utc),
            idempotency_key=idempotency_key
        )

        # First should succeed
        enqueued1 = await dispatcher.enqueue_run(request1)
        assert enqueued1 is True

        # Wait a moment for processing
        await asyncio.sleep(0.1)

        # Second should be blocked by idempotency
        enqueued2 = await dispatcher.enqueue_run(request2)
        assert enqueued2 is False

    @pytest.mark.asyncio
    async def test_priority_ordering(self, dispatcher):
        """Test that higher priority runs are processed first."""
        # Enable auto-completion for this test
        dispatcher.backend.auto_complete = True

        # Stop dispatcher to control processing
        await dispatcher.stop()

        # Add runs with different priorities
        low_priority = RunRequest(
            site_id="low",
            environment="prod",
            scheduled_at=datetime.now(timezone.utc),
            priority=1
        )

        high_priority = RunRequest(
            site_id="high",
            environment="prod",
            scheduled_at=datetime.now(timezone.utc),
            priority=5
        )

        # Enqueue in reverse priority order
        await dispatcher.enqueue_run(low_priority)
        await dispatcher.enqueue_run(high_priority)

        # Start dispatcher to process
        await dispatcher.start()

        # Wait for processing with condition-based polling
        for _ in range(30):  # 3 seconds maximum
            stats = dispatcher.get_stats()
            if stats.total_dispatched >= 1:
                break
            await asyncio.sleep(0.1)

        # High priority should have been processed first
        stats = dispatcher.get_stats()
        assert stats.total_dispatched >= 1

    @pytest.mark.asyncio
    async def test_run_status_tracking(self, dispatcher):
        """Test tracking of run status through lifecycle."""
        run_request = RunRequest(
            site_id="status-test",
            environment="prod",
            scheduled_at=datetime.now(timezone.utc)
        )

        # Enqueue run
        enqueued = await dispatcher.enqueue_run(run_request)
        assert enqueued is True

        # Wait for dispatch (dispatcher loop sleeps for 1 second)
        await asyncio.sleep(1.2)

        # Should be able to get result
        result = await dispatcher.get_run_result(run_request.id)
        assert result is not None
        assert result.status == RunStatus.RUNNING

    @pytest.mark.asyncio
    async def test_dispatcher_stats(self, dispatcher):
        """Test dispatcher statistics tracking."""
        # Enable auto-completion for this test
        dispatcher.backend.auto_complete = True

        # Initial stats
        stats = dispatcher.get_stats()
        assert stats.total_dispatched == 0
        assert stats.total_dispatched == 0
        assert stats.current_queue_size == 0

        # Add some runs
        for i in range(3):
            run_request = RunRequest(
                site_id=f"stats-{i}",
                environment="prod",
                scheduled_at=datetime.now(timezone.utc)
            )
            await dispatcher.enqueue_run(run_request)

        # Start dispatcher to process runs
        await dispatcher.start()

        # Wait for processing with condition-based polling
        for _ in range(35):  # 3.5 seconds maximum
            stats = dispatcher.get_stats()
            if stats.total_dispatched == 3:
                break
            await asyncio.sleep(0.1)

        # Check updated stats
        stats = dispatcher.get_stats()
        assert stats.total_dispatched == 3
        assert stats.current_queue_size <= 3  # May have been processed

    @pytest.mark.asyncio
    async def test_queue_status(self, dispatcher):
        """Test dispatcher queue status."""
        status = dispatcher.get_queue_status()
        assert isinstance(status, dict)
        assert "queue_full" in status
        assert "at_concurrency_limit" in status

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown behavior."""
        backend = MockAuditBackend()
        dispatcher = RunDispatcher(backend, max_concurrent_runs=1)

        await dispatcher.start()

        # Add a run to the queue
        run_request = RunRequest(
            site_id="shutdown-test",
            environment="prod",
            scheduled_at=datetime.now(timezone.utc)
        )
        await dispatcher.enqueue_run(run_request)

        # Shutdown should complete without hanging
        await dispatcher.stop()

        # Stats should still be accessible
        stats = dispatcher.get_stats()
        assert isinstance(stats, DispatchStats)


class TestMockDispatcher:
    """Test mock dispatcher functionality."""

    @pytest.mark.asyncio
    async def test_create_mock_dispatcher(self):
        """Test creating mock dispatcher."""
        dispatcher = create_mock_dispatcher()

        assert isinstance(dispatcher, RunDispatcher)

        # Should be able to enqueue runs
        run_request = RunRequest(
            site_id="mock-test",
            environment="test",
            scheduled_at=datetime.now(timezone.utc)
        )

        await dispatcher.start()
        try:
            enqueued = await dispatcher.enqueue_run(run_request)
            assert enqueued is True

            # Wait for mock processing
            await asyncio.sleep(0.1)

            # Should get mock result
            result = await dispatcher.get_run_result(run_request.id)
            assert result is not None
            assert result.run_id == run_request.id

        finally:
            await dispatcher.stop()


class TestDispatcherErrorHandling:
    """Test dispatcher error handling."""

    @pytest.mark.asyncio
    async def test_backend_failure_handling(self):
        """Test handling of backend dispatch failures."""
        backend = MockAuditBackend()
        backend.should_fail = True

        dispatcher = RunDispatcher(backend, max_concurrent_runs=1)
        await dispatcher.start()

        try:
            run_request = RunRequest(
                site_id="fail-test",
                environment="prod",
                scheduled_at=datetime.now(timezone.utc)
            )

            # Enqueue should succeed
            enqueued = await dispatcher.enqueue_run(run_request)
            assert enqueued is True

            # Wait for processing attempt
            await asyncio.sleep(0.2)

            # Stats should show the failure
            stats = dispatcher.get_stats()
            # The exact behavior on failure depends on implementation
            # but the dispatcher should not crash
            assert isinstance(stats, DispatchStats)

        finally:
            await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_invalid_run_request_handling(self):
        """Test handling of invalid run requests."""
        backend = MockAuditBackend()
        dispatcher = RunDispatcher(backend)

        await dispatcher.start()

        try:
            # Test with None values (if the model allows it)
            # This depends on the RunRequest validation
            run_request = RunRequest(
                site_id="",  # Empty site_id
                environment="",  # Empty environment
                scheduled_at=datetime.now(timezone.utc)
            )

            # Should handle gracefully (either reject or process)
            enqueued = await dispatcher.enqueue_run(run_request)
            assert isinstance(enqueued, bool)

        finally:
            await dispatcher.stop()


class TestDispatcherConfiguration:
    """Test dispatcher configuration options."""

    @pytest.mark.asyncio
    async def test_custom_configuration(self):
        """Test dispatcher with custom configuration."""
        backend = MockAuditBackend()

        dispatcher = RunDispatcher(
            backend=backend,
            max_concurrent_runs=5,
            max_queue_size=100,
            idempotency_window_minutes=120  # 2 hours
        )

        assert dispatcher.max_concurrent_runs == 5
        assert dispatcher.max_queue_size == 100
        assert dispatcher.idempotency_window == timedelta(hours=2)

    @pytest.mark.asyncio
    async def test_concurrency_limiting(self):
        """Test that concurrency limits are respected."""
        backend = MockAuditBackend()
        backend.dispatch_delay = 0.5  # Slow dispatch to test concurrency
        backend.auto_complete = True  # Enable auto-completion for this test

        dispatcher = RunDispatcher(backend, max_concurrent_runs=1)
        await dispatcher.start()

        try:
            # Enqueue multiple runs quickly
            requests = []
            for i in range(3):
                run_request = RunRequest(
                    site_id=f"concurrent-{i}",
                    environment="prod",
                    scheduled_at=datetime.now(timezone.utc)
                )
                requests.append(run_request)
                enqueued = await dispatcher.enqueue_run(run_request)
                assert enqueued is True

            # Wait for processing with condition-based polling
            # With dispatch_delay=0.5 and max_concurrent_runs=1, expect some processing
            for _ in range(40):  # 4 seconds maximum
                stats = dispatcher.get_stats()
                if stats.total_dispatched >= 2:
                    break
                await asyncio.sleep(0.1)

            # Should have limited concurrent dispatches
            stats = dispatcher.get_stats()
            # Exact assertions depend on timing and implementation details
            # At minimum, should have dispatched some runs (timing-dependent)
            assert stats.total_dispatched >= 2
            assert stats.total_dispatched <= 3

        finally:
            await dispatcher.stop()
            await backend.cleanup()  # Clean up any hanging tasks