"""Background task management for Tag Sentinel API.

This module provides efficient background task processing for
long-running operations and scheduled tasks.
"""

import logging
import asyncio
import time
from typing import Any, Optional, Dict, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Background task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class TaskConfig:
    """Configuration for background task management."""
    max_workers: int = 4
    max_queue_size: int = 1000
    task_timeout: int = 3600  # 1 hour
    retry_attempts: int = 3
    retry_delay: int = 60  # seconds
    cleanup_interval: int = 300  # 5 minutes
    keep_completed_tasks: int = 100
    enable_metrics: bool = True


@dataclass
class TaskResult:
    """Result of a background task."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration: Optional[float] = None
    retry_count: int = 0

    @property
    def is_completed(self) -> bool:
        """Check if task is in a final state."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]


@dataclass
class BackgroundTask:
    """A background task definition."""
    id: str
    name: str
    func: Callable[..., Awaitable[Any]]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[int] = None
    retry_attempts: int = 3
    retry_delay: int = 60
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None

    def __lt__(self, other):
        """Compare tasks for priority queue ordering."""
        if not isinstance(other, BackgroundTask):
            return NotImplemented
        # Higher priority tasks come first
        return self.priority.value > other.priority.value


class TaskQueue:
    """Priority queue for background tasks."""

    def __init__(self, max_size: int = 1000):
        """Initialize task queue.

        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queue = asyncio.PriorityQueue(maxsize=max_size)
        self._scheduled_tasks: List[BackgroundTask] = []
        self._lock = asyncio.Lock()

    async def put(self, task: BackgroundTask) -> None:
        """Add task to queue.

        Args:
            task: Task to add

        Raises:
            QueueFull: If queue is at capacity
        """
        if task.scheduled_at:
            async with self._lock:
                self._scheduled_tasks.append(task)
                self._scheduled_tasks.sort(key=lambda t: t.scheduled_at)
        else:
            await self._queue.put(task)

    async def get(self) -> BackgroundTask:
        """Get next task from queue.

        Returns:
            Next task to execute
        """
        # Check for scheduled tasks that are ready
        async with self._lock:
            current_time = time.time()
            ready_tasks = []

            for task in self._scheduled_tasks[:]:
                if task.scheduled_at <= current_time:
                    ready_tasks.append(task)
                    self._scheduled_tasks.remove(task)

            # Add ready scheduled tasks to main queue
            for task in ready_tasks:
                try:
                    self._queue.put_nowait(task)
                except asyncio.QueueFull:
                    logger.warning(f"Queue full, dropping scheduled task: {task.name}")

        # Get task from main queue
        return await self._queue.get()

    async def size(self) -> int:
        """Get current queue size."""
        async with self._lock:
            return self._queue.qsize() + len(self._scheduled_tasks)

    def task_done(self) -> None:
        """Mark task as done."""
        self._queue.task_done()

    async def join(self) -> None:
        """Wait for all tasks to complete."""
        await self._queue.join()


class BackgroundTaskManager:
    """Manager for background task execution."""

    def __init__(self, config: Optional[TaskConfig] = None):
        """Initialize background task manager.

        Args:
            config: Task configuration
        """
        self.config = config or TaskConfig()
        self.task_queue = TaskQueue(self.config.max_queue_size)
        self.task_results: Dict[str, TaskResult] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}

        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Worker tasks
        self.workers: List[asyncio.Task] = []
        self.running = False

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(f"BackgroundTaskManager initialized with {self.config.max_workers} workers")

    async def start(self) -> None:
        """Start the background task manager."""
        if self.running:
            return

        self.running = True

        # Start worker tasks
        for i in range(self.config.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

        logger.info(f"Started {len(self.workers)} background workers")

    async def stop(self) -> None:
        """Stop the background task manager."""
        if not self.running:
            return

        self.running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel active tasks
        for task in self.active_tasks.values():
            task.cancel()

        # Close thread pool
        self.thread_pool.shutdown(wait=True)

        logger.info("Background task manager stopped")

    async def submit_task(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[int] = None,
        retry_attempts: Optional[int] = None,
        retry_delay: Optional[int] = None,
        delay: Optional[int] = None,
        **kwargs
    ) -> str:
        """Submit a background task.

        Args:
            func: Async function to execute
            *args: Function arguments
            name: Task name
            priority: Task priority
            timeout: Task timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds
            delay: Delay before execution in seconds
            **kwargs: Function keyword arguments

        Returns:
            Task ID
        """
        if not self.running:
            raise RuntimeError("Task manager is not running")

        task_id = str(uuid.uuid4())
        task_name = name or func.__name__

        # Calculate scheduled time if delay is specified
        scheduled_at = None
        if delay:
            scheduled_at = time.time() + delay

        task = BackgroundTask(
            id=task_id,
            name=task_name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout or self.config.task_timeout,
            retry_attempts=retry_attempts or self.config.retry_attempts,
            retry_delay=retry_delay or self.config.retry_delay,
            scheduled_at=scheduled_at
        )

        # Create initial result
        self.task_results[task_id] = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING
        )

        # Add to queue
        await self.task_queue.put(task)

        logger.info(f"Submitted task: {task_name} ({task_id})")
        return task_id

    async def submit_cpu_task(
        self,
        func: Callable[..., Any],
        *args,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> str:
        """Submit a CPU-bound task to thread pool.

        Args:
            func: Function to execute (not async)
            *args: Function arguments
            name: Task name
            priority: Task priority
            **kwargs: Function keyword arguments

        Returns:
            Task ID
        """
        async def cpu_task_wrapper():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_pool, func, *args)

        return await self.submit_task(
            cpu_task_wrapper,
            name=name or f"cpu_{func.__name__}",
            priority=priority,
            **kwargs
        )

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result by ID.

        Args:
            task_id: Task ID

        Returns:
            Task result or None if not found
        """
        return self.task_results.get(task_id)

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Wait for task to complete.

        Args:
            task_id: Task ID
            timeout: Wait timeout in seconds

        Returns:
            Task result

        Raises:
            TimeoutError: If task doesn't complete within timeout
            ValueError: If task not found
        """
        if task_id not in self.task_results:
            raise ValueError(f"Task {task_id} not found")

        start_time = time.time()

        while True:
            result = self.task_results[task_id]
            if result.is_completed:
                return result

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

            await asyncio.sleep(0.1)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: Task ID

        Returns:
            True if task was cancelled
        """
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
            self.task_results[task_id].status = TaskStatus.CANCELLED
            return True

        # Mark pending task as cancelled
        if task_id in self.task_results:
            result = self.task_results[task_id]
            if result.status == TaskStatus.PENDING:
                result.status = TaskStatus.CANCELLED
                return True

        return False

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status information.

        Returns:
            Queue status dictionary
        """
        queue_size = await self.task_queue.size()

        status_counts = {}
        for result in self.task_results.values():
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "queue_size": queue_size,
            "active_tasks": len(self.active_tasks),
            "total_tasks": len(self.task_results),
            "status_counts": status_counts,
            "workers": len(self.workers),
            "running": self.running
        }

    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine that processes tasks."""
        logger.info(f"Worker {worker_name} started")

        while self.running:
            try:
                # Get next task
                task = await self.task_queue.get()

                # Execute task
                await self._execute_task(task)

                # Mark task as done
                self.task_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {e}")

        logger.info(f"Worker {worker_name} stopped")

    async def _execute_task(self, task: BackgroundTask) -> None:
        """Execute a single task."""
        result = self.task_results[task.id]
        result.status = TaskStatus.RUNNING
        result.started_at = time.time()

        logger.debug(f"Executing task: {task.name} ({task.id})")

        for attempt in range(task.retry_attempts + 1):
            try:
                # Create and run task
                async_task = asyncio.create_task(task.func(*task.args, **task.kwargs))
                self.active_tasks[task.id] = async_task

                # Execute with timeout
                task_result = await asyncio.wait_for(async_task, timeout=task.timeout)

                # Task completed successfully
                result.status = TaskStatus.COMPLETED
                result.result = task_result
                result.completed_at = time.time()
                result.duration = result.completed_at - result.started_at

                logger.info(f"Task completed: {task.name} ({task.id})")
                break

            except asyncio.TimeoutError:
                logger.error(f"Task timeout: {task.name} ({task.id})")
                result.status = TaskStatus.FAILED
                result.error = f"Task timed out after {task.timeout}s"

            except asyncio.CancelledError:
                logger.info(f"Task cancelled: {task.name} ({task.id})")
                result.status = TaskStatus.CANCELLED
                break

            except Exception as e:
                logger.error(f"Task failed: {task.name} ({task.id}) - {e}")
                result.retry_count = attempt + 1

                if attempt < task.retry_attempts:
                    logger.info(f"Retrying task {task.name} in {task.retry_delay}s (attempt {attempt + 2})")
                    await asyncio.sleep(task.retry_delay)
                else:
                    result.status = TaskStatus.FAILED
                    result.error = str(e)

            finally:
                # Clean up
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]

        # Set completion time if not already set
        if not result.completed_at:
            result.completed_at = time.time()
            result.duration = result.completed_at - result.started_at

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up old task results."""
        while self.running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_old_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def _cleanup_old_tasks(self) -> None:
        """Clean up old completed task results."""
        if not self.config.keep_completed_tasks:
            return

        # Sort tasks by completion time
        completed_tasks = [
            (task_id, result) for task_id, result in self.task_results.items()
            if result.is_completed and result.completed_at
        ]

        completed_tasks.sort(key=lambda x: x[1].completed_at, reverse=True)

        # Keep only the most recent completed tasks
        if len(completed_tasks) > self.config.keep_completed_tasks:
            tasks_to_remove = completed_tasks[self.config.keep_completed_tasks:]

            for task_id, _ in tasks_to_remove:
                del self.task_results[task_id]

            logger.debug(f"Cleaned up {len(tasks_to_remove)} old task results")


# Global background task manager
_global_task_manager: Optional[BackgroundTaskManager] = None


def get_task_manager() -> BackgroundTaskManager:
    """Get global background task manager."""
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = BackgroundTaskManager()
    return _global_task_manager


def set_task_manager(manager: BackgroundTaskManager) -> None:
    """Set global background task manager."""
    global _global_task_manager
    _global_task_manager = manager


# Convenience functions

async def submit_background_task(
    func: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> str:
    """Submit a background task using global manager."""
    manager = get_task_manager()
    return await manager.submit_task(func, *args, **kwargs)


async def submit_cpu_task(
    func: Callable[..., Any],
    *args,
    **kwargs
) -> str:
    """Submit a CPU-bound task using global manager."""
    manager = get_task_manager()
    return await manager.submit_cpu_task(func, *args, **kwargs)


def background_task(
    name: Optional[str] = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: Optional[int] = None,
    retry_attempts: Optional[int] = None,
    retry_delay: Optional[int] = None
):
    """Decorator to make a function a background task.

    Args:
        name: Task name
        priority: Task priority
        timeout: Task timeout
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries

    Example:
        @background_task(name="process_data", priority=TaskPriority.HIGH)
        async def process_data(data):
            # Process data
            return result
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await submit_background_task(
                func,
                *args,
                name=name or func.__name__,
                priority=priority,
                timeout=timeout,
                retry_attempts=retry_attempts,
                retry_delay=retry_delay,
                **kwargs
            )
        return wrapper
    return decorator