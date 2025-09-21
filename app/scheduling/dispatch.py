"""Run request dispatching and queue management system.

This module provides robust run request queuing and dispatching with support for:
- Synchronous and asynchronous run dispatching
- Idempotency handling to prevent duplicate runs
- Priority-based run scheduling
- Queue monitoring and health checking
- Error handling and retry logic
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
import heapq
from uuid import uuid4

from .models import RunRequest, ScheduleState


logger = logging.getLogger(__name__)


class DispatchError(Exception):
    """Exception raised when dispatch operations fail."""
    pass


class RunStatus(str, Enum):
    """Status of a run request."""
    PENDING = "pending"           # Queued but not started
    DISPATCHED = "dispatched"     # Sent to audit runner
    RUNNING = "running"           # Currently executing
    COMPLETED = "completed"       # Finished successfully
    FAILED = "failed"            # Failed with error
    CANCELLED = "cancelled"      # Cancelled before execution
    TIMEOUT = "timeout"          # Timed out during execution


@dataclass
class RunResult:
    """Result of a run dispatch operation."""

    run_id: str
    status: RunStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    audit_results: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> Optional[timedelta]:
        """Get run duration if both start and end times are available."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def is_complete(self) -> bool:
        """Check if run is in a completed state."""
        return self.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.TIMEOUT)


@dataclass
class DispatchStats:
    """Statistics for run dispatcher operations."""

    total_dispatched: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    cancelled_runs: int = 0
    timeout_runs: int = 0
    current_queue_size: int = 0
    average_queue_time_seconds: float = 0.0
    average_run_time_seconds: float = 0.0
    idempotency_blocks: int = 0
    last_dispatch_time: Optional[datetime] = None

    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_dispatched == 0:
            return 0.0
        return (self.successful_runs / self.total_dispatched) * 100.0


# Type alias for audit runner function
AuditRunnerFunc = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class RunDispatcherBackend(ABC):
    """Abstract backend for run dispatching."""

    @abstractmethod
    async def dispatch_run(self, run_request: RunRequest) -> RunResult:
        """Dispatch a run request to the audit runner.

        Args:
            run_request: Run request to dispatch

        Returns:
            RunResult with dispatch outcome

        Raises:
            DispatchError: If dispatch fails
        """
        pass

    @abstractmethod
    async def get_run_status(self, run_id: str) -> Optional[RunResult]:
        """Get status of a dispatched run.

        Args:
            run_id: Run ID to check

        Returns:
            RunResult if found, None otherwise
        """
        pass

    @abstractmethod
    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a pending or running audit.

        Args:
            run_id: Run ID to cancel

        Returns:
            True if cancelled, False if not found or already complete
        """
        pass


class MockAuditRunnerBackend(RunDispatcherBackend):
    """Mock backend for testing run dispatching."""

    def __init__(self, simulate_delay: bool = True, failure_rate: float = 0.0):
        """Initialize mock backend.

        Args:
            simulate_delay: Whether to simulate audit run delays
            failure_rate: Rate of simulated failures (0.0 = never, 1.0 = always)
        """
        self.simulate_delay = simulate_delay
        self.failure_rate = failure_rate
        self._running_tasks: Dict[str, asyncio.Task] = {}

    async def dispatch_run(self, run_request: RunRequest) -> RunResult:
        """Dispatch run to mock audit runner."""
        run_result = RunResult(
            run_id=run_request.id,
            status=RunStatus.DISPATCHED,
            started_at=datetime.now(timezone.utc),
            metadata={
                **run_request.metadata,  # Preserve original metadata including lock info
                "site_id": run_request.site_id,
                "environment": run_request.environment,
                "mock_backend": True
            }
        )

        # Start async task to simulate audit run
        task = asyncio.create_task(self._simulate_audit_run(run_request, run_result))
        self._running_tasks[run_request.id] = task

        return run_result

    async def get_run_status(self, run_id: str) -> Optional[RunResult]:
        """Get status of mock run."""
        if run_id not in self._running_tasks:
            return None

        task = self._running_tasks[run_id]
        if task.done():
            try:
                return await task
            except Exception:
                # Task failed - create failed result
                return RunResult(
                    run_id=run_id,
                    status=RunStatus.FAILED,
                    error="Mock audit run failed",
                    completed_at=datetime.now(timezone.utc)
                )
        else:
            # Still running
            return RunResult(
                run_id=run_id,
                status=RunStatus.RUNNING,
                started_at=datetime.now(timezone.utc) - timedelta(seconds=1)
            )

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel mock run."""
        if run_id not in self._running_tasks:
            return False

        task = self._running_tasks[run_id]
        if not task.done():
            task.cancel()
            return True

        return False

    async def _simulate_audit_run(self, run_request: RunRequest, run_result: RunResult) -> RunResult:
        """Simulate audit run execution."""
        try:
            run_result.status = RunStatus.RUNNING

            if self.simulate_delay:
                # Simulate audit run time
                await asyncio.sleep(2.0)

            # Simulate failure based on failure rate
            import random
            if random.random() < self.failure_rate:
                run_result.status = RunStatus.FAILED
                run_result.error = "Simulated audit failure"
            else:
                run_result.status = RunStatus.COMPLETED
                run_result.audit_results = {
                    "pages_crawled": 10,
                    "tags_found": 5,
                    "issues_detected": 1
                }

            run_result.completed_at = datetime.now(timezone.utc)
            return run_result

        except asyncio.CancelledError:
            run_result.status = RunStatus.CANCELLED
            run_result.completed_at = datetime.now(timezone.utc)
            return run_result
        finally:
            # Clean up task reference
            if run_request.id in self._running_tasks:
                del self._running_tasks[run_request.id]


class RunDispatcher:
    """Main run dispatcher with queue management and idempotency handling."""

    def __init__(
        self,
        backend: RunDispatcherBackend,
        max_queue_size: int = 1000,
        max_concurrent_runs: int = 10,
        idempotency_window_minutes: int = 60,
        cleanup_interval_seconds: int = 300,
        completion_callback: Optional[Callable[[RunResult], Awaitable[None]]] = None
    ):
        """Initialize run dispatcher.

        Args:
            backend: Backend for dispatching runs
            max_queue_size: Maximum size of pending run queue
            max_concurrent_runs: Maximum concurrent runs
            idempotency_window_minutes: Window for idempotency checking
            cleanup_interval_seconds: Interval for cleanup tasks
            completion_callback: Optional callback called when runs complete
        """
        self.backend = backend
        self.max_queue_size = max_queue_size
        self.max_concurrent_runs = max_concurrent_runs
        self.idempotency_window = timedelta(minutes=idempotency_window_minutes)
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.completion_callback = completion_callback

        # Queue and tracking
        self._pending_queue: List[RunRequest] = []
        self._running_runs: Dict[str, RunResult] = {}
        self._completed_runs: Dict[str, RunResult] = {}
        self._idempotency_cache: Dict[str, str] = {}  # key -> run_id

        # Statistics
        self._stats = DispatchStats()

        # Control
        self._shutdown = False
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the dispatcher."""
        if self._dispatcher_task is None:
            self._shutdown = False  # Reset shutdown flag
            self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started run dispatcher")

    async def stop(self) -> None:
        """Stop the dispatcher gracefully."""
        self._shutdown = True

        # Cancel tasks
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Reset task references to allow restart
        self._dispatcher_task = None
        self._cleanup_task = None

        logger.info("Stopped run dispatcher")

    async def enqueue_run(self, run_request: RunRequest) -> bool:
        """Enqueue a run request for execution.

        Args:
            run_request: Run request to enqueue

        Returns:
            True if enqueued, False if blocked by idempotency or queue full

        Raises:
            DispatchError: If enqueuing fails
        """
        # Check idempotency
        if run_request.idempotency_key:
            if await self._check_idempotency(run_request):
                self._stats.idempotency_blocks += 1
                logger.info(f"Run blocked by idempotency: {run_request.idempotency_key}")
                return False

        # Check queue capacity
        if len(self._pending_queue) >= self.max_queue_size:
            logger.warning(f"Run queue full ({self.max_queue_size}), rejecting request")
            return False

        # Add to queue
        heapq.heappush(self._pending_queue, run_request)
        self._stats.current_queue_size = len(self._pending_queue)

        # Update idempotency cache
        if run_request.idempotency_key:
            self._idempotency_cache[run_request.idempotency_key] = run_request.id

        logger.info(f"Enqueued run: {run_request.site_id}:{run_request.environment} (id: {run_request.id})")
        return True

    async def get_run_result(self, run_id: str) -> Optional[RunResult]:
        """Get result of a run by ID.

        Args:
            run_id: Run ID to look up

        Returns:
            RunResult if found, None otherwise
        """
        # Check running runs
        if run_id in self._running_runs:
            # Get updated status from backend
            updated_result = await self.backend.get_run_status(run_id)
            if updated_result and updated_result.is_complete:
                # Move to completed
                self._completed_runs[run_id] = updated_result
                del self._running_runs[run_id]

                # Call completion callback if provided
                if self.completion_callback:
                    try:
                        await self.completion_callback(updated_result)
                    except Exception as e:
                        logger.warning(f"Completion callback failed for run {run_id}: {e}")

                return updated_result
            return self._running_runs[run_id]

        # Check completed runs
        if run_id in self._completed_runs:
            return self._completed_runs[run_id]

        # Check pending queue
        for request in self._pending_queue:
            if request.id == run_id:
                return RunResult(
                    run_id=run_id,
                    status=RunStatus.PENDING,
                    metadata={
                        "site_id": request.site_id,
                        "environment": request.environment
                    }
                )

        return None

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a run by ID.

        Args:
            run_id: Run ID to cancel

        Returns:
            True if cancelled, False if not found or already complete
        """
        # Check pending queue
        for i, request in enumerate(self._pending_queue):
            if request.id == run_id:
                cancelled_request = self._pending_queue[i]
                del self._pending_queue[i]
                heapq.heapify(self._pending_queue)
                self._stats.current_queue_size = len(self._pending_queue)
                self._stats.cancelled_runs += 1
                logger.info(f"Cancelled pending run: {run_id}")

                # Create cancelled run result and invoke completion callback to release locks
                cancelled_result = RunResult(
                    run_id=run_id,
                    status=RunStatus.CANCELLED,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    metadata={
                        **cancelled_request.metadata,  # Preserve original metadata including lock info
                        "site_id": cancelled_request.site_id,
                        "environment": cancelled_request.environment
                    }
                )

                # Invoke completion callback to ensure locks are released
                if self.completion_callback:
                    try:
                        await self.completion_callback(cancelled_result)
                    except Exception as e:
                        logger.error(f"Error in completion callback for cancelled run {run_id}: {e}")

                return True

        # Check running runs
        if run_id in self._running_runs:
            if await self.backend.cancel_run(run_id):
                result = self._running_runs[run_id]
                result.status = RunStatus.CANCELLED
                result.completed_at = datetime.now(timezone.utc)
                self._completed_runs[run_id] = result
                del self._running_runs[run_id]
                self._stats.cancelled_runs += 1

                # Call completion callback for cancelled run if provided
                if self.completion_callback:
                    try:
                        await self.completion_callback(result)
                    except Exception as callback_e:
                        logger.warning(f"Completion callback failed for cancelled run {run_id}: {callback_e}")

                logger.info(f"Cancelled running run: {run_id}")
                return True

        return False

    def get_stats(self) -> DispatchStats:
        """Get current dispatcher statistics."""
        self._stats.current_queue_size = len(self._pending_queue)
        return self._stats

    def get_queue_status(self) -> Dict[str, Any]:
        """Get detailed queue status."""
        pending_by_priority = {}
        for request in self._pending_queue:
            priority = request.priority
            if priority not in pending_by_priority:
                pending_by_priority[priority] = 0
            pending_by_priority[priority] += 1

        running_by_site = {}
        for result in self._running_runs.values():
            site_id = result.metadata.get("site_id", "unknown")
            if site_id not in running_by_site:
                running_by_site[site_id] = 0
            running_by_site[site_id] += 1

        return {
            "pending_count": len(self._pending_queue),
            "running_count": len(self._running_runs),
            "completed_count": len(self._completed_runs),
            "pending_by_priority": pending_by_priority,
            "running_by_site": running_by_site,
            "max_queue_size": self.max_queue_size,
            "max_concurrent_runs": self.max_concurrent_runs,
            "queue_full": len(self._pending_queue) >= self.max_queue_size,
            "at_concurrency_limit": len(self._running_runs) >= self.max_concurrent_runs
        }

    async def _check_idempotency(self, run_request: RunRequest) -> bool:
        """Check if run is blocked by idempotency."""
        if not run_request.idempotency_key:
            return False

        # Check cache
        existing_run_id = self._idempotency_cache.get(run_request.idempotency_key)
        if existing_run_id:
            # Check if existing run is still active or recent
            existing_result = await self.get_run_result(existing_run_id)
            if existing_result:
                # Block if run is still pending or running
                if existing_result.status in (RunStatus.PENDING, RunStatus.DISPATCHED, RunStatus.RUNNING):
                    return True  # Block duplicate

                # For completed/failed runs, check if within idempotency window
                if existing_result.started_at:
                    age = datetime.now(timezone.utc) - existing_result.started_at
                    if age <= self.idempotency_window:
                        return True  # Block duplicate

        return False  # Allow run

    async def _dispatcher_loop(self) -> None:
        """Main dispatcher loop."""
        logger.info("Started dispatcher loop")

        while not self._shutdown:
            try:
                # Check if we can dispatch more runs
                if (len(self._running_runs) < self.max_concurrent_runs and
                    self._pending_queue):

                    # Get highest priority request
                    run_request = heapq.heappop(self._pending_queue)
                    self._stats.current_queue_size = len(self._pending_queue)

                    # Dispatch run
                    try:
                        result = await self.backend.dispatch_run(run_request)
                        self._running_runs[run_request.id] = result
                        self._stats.total_dispatched += 1
                        self._stats.last_dispatch_time = datetime.now(timezone.utc)

                        logger.info(f"Dispatched run: {run_request.id}")

                    except Exception as e:
                        # Failed to dispatch - create failed result
                        failed_result = RunResult(
                            run_id=run_request.id,
                            status=RunStatus.FAILED,
                            error=f"Dispatch failed: {e}",
                            started_at=datetime.now(timezone.utc),
                            completed_at=datetime.now(timezone.utc),
                            metadata={
                                **run_request.metadata,
                                "site_id": run_request.site_id,
                                "environment": run_request.environment
                            }
                        )
                        self._completed_runs[run_request.id] = failed_result
                        self._stats.failed_runs += 1

                        # Call completion callback for failed run if provided
                        if self.completion_callback:
                            try:
                                await self.completion_callback(failed_result)
                            except Exception as callback_e:
                                logger.warning(f"Completion callback failed for failed run {run_request.id}: {callback_e}")

                        logger.error(f"Failed to dispatch run {run_request.id}: {e}")

                # Check for completed runs
                completed_run_ids = []
                for run_id, result in self._running_runs.items():
                    updated_result = await self.backend.get_run_status(run_id)
                    if updated_result and updated_result.is_complete:
                        completed_run_ids.append(run_id)
                        self._completed_runs[run_id] = updated_result

                        # Call completion callback if provided
                        if self.completion_callback:
                            try:
                                await self.completion_callback(updated_result)
                            except Exception as callback_e:
                                logger.warning(f"Completion callback failed for run {run_id}: {callback_e}")

                        # Update statistics
                        if updated_result.status == RunStatus.COMPLETED:
                            self._stats.successful_runs += 1
                        elif updated_result.status == RunStatus.FAILED:
                            self._stats.failed_runs += 1
                        elif updated_result.status == RunStatus.CANCELLED:
                            self._stats.cancelled_runs += 1
                        elif updated_result.status == RunStatus.TIMEOUT:
                            self._stats.timeout_runs += 1

                # Remove completed runs from running
                for run_id in completed_run_ids:
                    del self._running_runs[run_id]

                # Brief sleep to prevent busy waiting
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dispatcher loop: {e}")
                await asyncio.sleep(5.0)  # Back off on error

        logger.info("Dispatcher loop stopped")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        logger.info("Started cleanup loop")

        while not self._shutdown:
            try:
                # Clean up old completed runs
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                old_runs = [
                    run_id for run_id, result in self._completed_runs.items()
                    if result.completed_at and result.completed_at < cutoff_time
                ]

                for run_id in old_runs:
                    del self._completed_runs[run_id]

                if old_runs:
                    logger.info(f"Cleaned up {len(old_runs)} old completed runs")

                # Clean up old idempotency cache entries
                cutoff_time = datetime.now(timezone.utc) - self.idempotency_window
                old_keys = []

                for key, run_id in self._idempotency_cache.items():
                    result = self._completed_runs.get(run_id)
                    if result and result.completed_at and result.completed_at < cutoff_time:
                        old_keys.append(key)

                for key in old_keys:
                    del self._idempotency_cache[key]

                if old_keys:
                    logger.info(f"Cleaned up {len(old_keys)} old idempotency entries")

                await asyncio.sleep(self.cleanup_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.cleanup_interval_seconds)

        logger.info("Cleanup loop stopped")


# Factory function for creating dispatcher with mock backend
def create_mock_dispatcher(
    max_queue_size: int = 100,
    max_concurrent_runs: int = 5,
    simulate_delay: bool = True,
    failure_rate: float = 0.1
) -> RunDispatcher:
    """Create a run dispatcher with mock backend for testing.

    Args:
        max_queue_size: Maximum queue size
        max_concurrent_runs: Maximum concurrent runs
        simulate_delay: Whether to simulate audit delays
        failure_rate: Rate of simulated failures

    Returns:
        Configured RunDispatcher with mock backend
    """
    backend = MockAuditRunnerBackend(
        simulate_delay=simulate_delay,
        failure_rate=failure_rate
    )

    return RunDispatcher(
        backend=backend,
        max_queue_size=max_queue_size,
        max_concurrent_runs=max_concurrent_runs
    )