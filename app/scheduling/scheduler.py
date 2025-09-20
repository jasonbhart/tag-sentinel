"""Core scheduling engine that orchestrates cron evaluation, blackout checks, locks, and dispatch."""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from uuid import uuid4

from .models import Schedule, RunRequest, ScheduleStatus, ScheduleState as ScheduleStateEnum
from .cron import CronEvaluator
from .blackout import BlackoutManager
from .locks import ConcurrencyManager
from .dispatch import RunDispatcher
from .environments import EnvironmentConfigManager
from .metrics import MetricsCollector, SchedulingMetrics

logger = logging.getLogger(__name__)


@dataclass
class ScheduleRuntimeState:
    """Runtime state for a schedule."""
    schedule: Schedule
    status: ScheduleStateEnum = ScheduleStateEnum.ACTIVE
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineStats:
    """Statistics for the schedule engine."""
    total_schedules: int = 0
    active_schedules: int = 0
    paused_schedules: int = 0
    disabled_schedules: int = 0
    total_runs_scheduled: int = 0
    total_runs_completed: int = 0
    total_runs_failed: int = 0
    blackout_blocks: int = 0
    lock_conflicts: int = 0
    catch_up_runs: int = 0
    last_tick_time: Optional[datetime] = None
    average_tick_duration_ms: float = 0.0


class ScheduleEngine:
    """Core scheduling engine that orchestrates all scheduling components."""

    def __init__(
        self,
        cron_evaluator: CronEvaluator,
        blackout_manager: BlackoutManager,
        concurrency_manager: ConcurrencyManager,
        run_dispatcher: RunDispatcher,
        environment_manager: EnvironmentConfigManager,
        metrics_collector: Optional[MetricsCollector] = None,
        tick_interval_seconds: int = 60,
        max_catch_up_hours: int = 24,
        max_consecutive_failures: int = 5
    ):
        """Initialize the schedule engine.

        Args:
            cron_evaluator: Cron expression evaluator
            blackout_manager: Blackout window manager
            concurrency_manager: Distributed concurrency control
            run_dispatcher: Run dispatcher
            environment_manager: Environment configuration manager
            metrics_collector: Optional metrics collector for monitoring
            tick_interval_seconds: How often to check schedules
            max_catch_up_hours: Maximum hours to look back for catch-up
            max_consecutive_failures: Max failures before disabling schedule
        """
        self.cron_evaluator = cron_evaluator
        self.blackout_manager = blackout_manager
        self.concurrency_manager = concurrency_manager
        self.run_dispatcher = run_dispatcher
        self.environment_manager = environment_manager

        # Set up completion callback for lock release
        if hasattr(run_dispatcher, 'completion_callback') and run_dispatcher.completion_callback is None:
            run_dispatcher.completion_callback = self._on_run_completion

        # Metrics
        self.metrics_collector = metrics_collector
        self.metrics = SchedulingMetrics(metrics_collector) if metrics_collector else None

        self.tick_interval_seconds = tick_interval_seconds
        self.max_catch_up_hours = max_catch_up_hours
        self.max_consecutive_failures = max_consecutive_failures

        # Schedule state tracking
        self._schedules: Dict[str, ScheduleRuntimeState] = {}

        # Engine control
        self._running = False
        self._engine_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._stats = EngineStats()

        # Timing tracking for performance metrics
        self._tick_times: List[float] = []
        self._max_tick_samples = 100

    async def start(self) -> None:
        """Start the schedule engine."""
        if self._running:
            logger.warning("Schedule engine is already running")
            return

        logger.info("Starting schedule engine")
        self._running = True
        self._shutdown_event.clear()

        # Start dependent services
        await self.concurrency_manager.start()
        await self.run_dispatcher.start()

        # Start metrics collector
        if self.metrics_collector:
            await self.metrics_collector.start()

        # Start engine loop
        self._engine_task = asyncio.create_task(self._engine_loop())

        logger.info(f"Schedule engine started (tick interval: {self.tick_interval_seconds}s)")

    async def stop(self) -> None:
        """Stop the schedule engine."""
        if not self._running:
            return

        logger.info("Stopping schedule engine")
        self._running = False
        self._shutdown_event.set()

        # Cancel engine task
        if self._engine_task:
            self._engine_task.cancel()
            try:
                await self._engine_task
            except asyncio.CancelledError:
                pass
            self._engine_task = None

        # Stop dependent services
        await self.run_dispatcher.stop()
        await self.concurrency_manager.stop()

        # Stop metrics collector
        if self.metrics_collector:
            await self.metrics_collector.stop()

        logger.info("Schedule engine stopped")

    def add_schedule(self, schedule: Schedule) -> None:
        """Add a schedule to the engine."""
        if schedule.id in self._schedules:
            logger.warning(f"Schedule {schedule.id} already exists, updating")

        # Calculate next run time
        next_run = None
        if schedule.enabled:
            try:
                next_run = self.cron_evaluator.get_next_run_time(
                    schedule.cron,
                    timezone_str=schedule.timezone
                )
            except Exception as e:
                logger.error(f"Failed to calculate next run for schedule {schedule.id}: {e}")

        schedule_state = ScheduleRuntimeState(
            schedule=schedule,
            status=ScheduleStateEnum.ACTIVE if schedule.enabled else ScheduleStateEnum.DISABLED,
            next_run=next_run
        )

        self._schedules[schedule.id] = schedule_state
        self._update_schedule_stats()

        logger.info(f"Added schedule {schedule.id} (next run: {next_run})")

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule from the engine."""
        if schedule_id not in self._schedules:
            return False

        del self._schedules[schedule_id]
        self._update_schedule_stats()

        logger.info(f"Removed schedule {schedule_id}")
        return True

    def get_schedule(self, schedule_id: str) -> Optional[ScheduleRuntimeState]:
        """Get schedule state by ID."""
        return self._schedules.get(schedule_id)

    def list_schedules(self) -> List[ScheduleRuntimeState]:
        """List all schedules."""
        return list(self._schedules.values())

    def pause_schedule(self, schedule_id: str) -> bool:
        """Pause a schedule."""
        if schedule_id not in self._schedules:
            return False

        schedule_state = self._schedules[schedule_id]
        schedule_state.status = ScheduleStateEnum.PAUSED
        schedule_state.next_run = None

        self._update_schedule_stats()
        logger.info(f"Paused schedule {schedule_id}")
        return True

    def resume_schedule(self, schedule_id: str) -> bool:
        """Resume a paused schedule."""
        if schedule_id not in self._schedules:
            return False

        schedule_state = self._schedules[schedule_id]
        if schedule_state.status != ScheduleStateEnum.PAUSED:
            return False

        schedule_state.status = ScheduleStateEnum.ACTIVE

        # Recalculate next run
        try:
            schedule_state.next_run = self.cron_evaluator.get_next_run_time(
                schedule_state.schedule.cron,
                timezone_str=schedule_state.schedule.timezone
            )
        except Exception as e:
            logger.error(f"Failed to resume schedule {schedule_id}: {e}")
            return False

        self._update_schedule_stats()
        logger.info(f"Resumed schedule {schedule_id} (next run: {schedule_state.next_run})")
        return True

    def get_stats(self) -> EngineStats:
        """Get engine statistics."""
        return self._stats

    async def trigger_schedule(self, schedule_id: str, force: bool = False) -> Optional[str]:
        """Manually trigger a schedule run.

        Args:
            schedule_id: Schedule to trigger
            force: Whether to bypass blackout and concurrency checks

        Returns:
            Run ID if triggered, None otherwise
        """
        if schedule_id not in self._schedules:
            logger.error(f"Schedule {schedule_id} not found")
            return None

        schedule_state = self._schedules[schedule_id]
        schedule = schedule_state.schedule

        if not force:
            # Check blackout windows
            status = self.blackout_manager.is_blackout_active(
                schedule.blackout_windows
            )
            if status.is_blackout:
                logger.info(f"Skipping triggered run for {schedule_id} due to blackout")
                return None

        # Create run request
        run_request = RunRequest(
            id=str(uuid4()),
            schedule_id=schedule_id,
            site_id=schedule.site_id,
            environment=schedule.environment,
            priority=5 if force else 0,  # Higher priority for forced runs
            scheduled_at=datetime.now(timezone.utc),
            params={
                "triggered": True,
                "force": force,
                **schedule.params,
                "schedule_metadata": schedule.metadata
            }
        )

        # Dispatch run
        return await self._dispatch_run(run_request, schedule_state, force_concurrency=force)

    async def _engine_loop(self) -> None:
        """Main engine loop that processes schedules."""
        logger.info("Starting schedule engine loop")

        while self._running:
            try:
                tick_start = datetime.now(timezone.utc)

                await self._process_tick()

                # Update performance metrics
                tick_duration = (datetime.now(timezone.utc) - tick_start).total_seconds() * 1000
                self._tick_times.append(tick_duration)
                if len(self._tick_times) > self._max_tick_samples:
                    self._tick_times.pop(0)

                self._stats.last_tick_time = tick_start
                self._stats.average_tick_duration_ms = sum(self._tick_times) / len(self._tick_times)

                # Record metrics
                if self.metrics:
                    self.metrics.record_tick_duration(tick_duration)

                # Wait for next tick or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.tick_interval_seconds
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal tick timeout, continue

            except Exception as e:
                logger.error(f"Error in engine loop: {e}", exc_info=True)
                # Continue running despite errors
                await asyncio.sleep(1)

    async def _process_tick(self) -> None:
        """Process a single tick - check all schedules for due runs."""
        now = datetime.now(timezone.utc)
        due_schedules = []

        # Find schedules that are due
        for schedule_state in self._schedules.values():
            if (schedule_state.status == ScheduleStateEnum.ACTIVE and
                schedule_state.next_run and
                schedule_state.next_run <= now):
                due_schedules.append(schedule_state)

        if not due_schedules:
            return

        logger.debug(f"Processing {len(due_schedules)} due schedules")

        # Record metrics for schedules processed
        if self.metrics:
            self.metrics.increment_schedules_processed(len(due_schedules))

        # Process due schedules
        for schedule_state in due_schedules:
            try:
                await self._process_schedule(schedule_state, now)
            except Exception as e:
                logger.error(f"Error processing schedule {schedule_state.schedule.id}: {e}", exc_info=True)
                self._handle_schedule_error(schedule_state, str(e))

    async def _process_schedule(self, schedule_state: ScheduleRuntimeState, current_time: datetime) -> None:
        """Process a single schedule that is due."""
        schedule = schedule_state.schedule

        # Check blackout windows
        status = self.blackout_manager.is_blackout_active(
            schedule.blackout_windows
        )
        if status.is_blackout:
            logger.info(f"Skipping run for {schedule.id} due to blackout")
            self._stats.blackout_blocks += 1

            # Record metrics
            if self.metrics:
                self.metrics.increment_blackout_blocks()

            # Calculate next run after blackout
            await self._calculate_next_run(schedule_state)
            return

        # Resolve environment config and merge with schedule params first
        try:
            resolved_config = self.environment_manager.resolve_environment_config(
                site_id=schedule.site_id,
                environment=schedule.environment,
                overrides=schedule.params  # Schedule params as overrides
            )

            # Get audit params from resolved config and merge with schedule params
            merged_params = resolved_config.get_audit_params()
            merged_params.update(schedule.params)  # Schedule params take precedence

        except Exception as e:
            logger.error(f"Failed to resolve environment config for {schedule.site_id}:{schedule.environment}: {e}")
            # Environment validation is required - do not proceed with invalid config
            logger.warning(f"Skipping scheduled run for {schedule.site_id}:{schedule.environment} due to environment resolution failure")

            # Use existing error handling system to track failure and manage schedule state
            self._handle_schedule_error(schedule_state, f"Environment resolution failed: {e}")

            # Only recalculate next run if schedule wasn't disabled by error handling
            if schedule_state.status != ScheduleStateEnum.DISABLED:
                await self._calculate_next_run(schedule_state)

            return  # Exit early - do not create any run requests

        # Check for catch-up runs if enabled
        catch_up_runs = []
        if schedule.catch_up_policy and getattr(schedule.catch_up_policy, 'enabled', True):
            catch_up_runs = await self._calculate_catch_up_runs(schedule_state, current_time, merged_params)

        # Create run requests (current + catch-up)
        run_requests = []

        # Current scheduled run
        current_run = RunRequest(
            id=str(uuid4()),
            schedule_id=schedule.id,
            site_id=schedule.site_id,
            environment=schedule.environment,
            priority=0,  # Normal priority for scheduled runs
            scheduled_at=schedule_state.next_run or current_time,
            params=merged_params,
            metadata=schedule.metadata.copy()
        )
        run_requests.append(current_run)

        # Add catch-up runs
        run_requests.extend(catch_up_runs)

        # Dispatch runs
        dispatched_count = 0
        for run_request in run_requests:
            run_id = await self._dispatch_run(run_request, schedule_state)
            if run_id:
                dispatched_count += 1

        if dispatched_count > 0:
            self._stats.total_runs_scheduled += dispatched_count
            if len(catch_up_runs) > 0:
                self._stats.catch_up_runs += len(catch_up_runs)

            # Record metrics
            if self.metrics:
                self.metrics.increment_runs_dispatched()
                if len(catch_up_runs) > 0:
                    self.metrics.increment_catch_up_runs()

            schedule_state.last_run = current_time
            schedule_state.consecutive_failures = 0  # Reset failure count on successful dispatch

        # Calculate next run time
        await self._calculate_next_run(schedule_state)

    async def _dispatch_run(
        self,
        run_request: RunRequest,
        schedule_state: ScheduleRuntimeState,
        force_concurrency: bool = False
    ) -> Optional[str]:
        """Dispatch a single run request."""
        schedule = schedule_state.schedule

        if not force_concurrency:
            # Check concurrency limits
            try:
                # Acquire lock manually (don't use async with to avoid early release)
                lock_info = await self.concurrency_manager.acquire_lock_manual(
                    schedule.site_id,
                    schedule.environment,
                    timeout_seconds=30,  # Quick timeout for scheduling
                    wait_timeout_seconds=1
                )

                if lock_info is None:
                    logger.info(f"Skipping run for {schedule.id} due to concurrency limit")
                    self._stats.lock_conflicts += 1

                    # Record metrics
                    if self.metrics:
                        self.metrics.increment_lock_conflicts()

                    return None

                # Lock acquired, store lock info in run request metadata for later release
                run_request.metadata["_lock_owner_id"] = lock_info.owner_id
                run_request.metadata["_lock_key"] = lock_info.key

                # Dispatch the run (lock will be released when run completes)
                try:
                    result = await self._enqueue_run(run_request)
                    if result is None:
                        # Enqueuing failed - release the lock immediately
                        logger.warning(f"Enqueuing failed for {schedule.id}, releasing acquired lock")
                        await self.concurrency_manager.release_lock_manual(
                            site_id=schedule.site_id,
                            environment=schedule.environment,
                            owner_id=lock_info.owner_id
                        )
                        # Remove lock metadata since we released it
                        run_request.metadata.pop("_lock_owner_id", None)
                        run_request.metadata.pop("_lock_key", None)
                    return result
                except Exception as enqueue_error:
                    # Enqueuing raised an exception - release the lock immediately
                    logger.error(f"Enqueuing raised exception for {schedule.id}, releasing acquired lock: {enqueue_error}")
                    await self.concurrency_manager.release_lock_manual(
                        site_id=schedule.site_id,
                        environment=schedule.environment,
                        owner_id=lock_info.owner_id
                    )
                    # Remove lock metadata since we released it
                    run_request.metadata.pop("_lock_owner_id", None)
                    run_request.metadata.pop("_lock_key", None)
                    return None

            except Exception as e:
                logger.warning(f"Failed to acquire lock for {schedule.id}: {e}")
                self._stats.lock_conflicts += 1

                # Record metrics
                if self.metrics:
                    self.metrics.increment_lock_conflicts()

                return None
        else:
            # Force dispatch without concurrency check
            return await self._enqueue_run(run_request)

    async def _enqueue_run(self, run_request: RunRequest) -> Optional[str]:
        """Enqueue a run request for dispatch."""
        try:
            success = await self.run_dispatcher.enqueue_run(run_request)
            if success:
                logger.info(f"Enqueued run {run_request.id} for schedule {run_request.schedule_id}")
                return run_request.id
            else:
                logger.warning(f"Failed to enqueue run {run_request.id}: blocked by idempotency or queue full")
                return None
        except Exception as e:
            logger.error(f"Failed to enqueue run {run_request.id}: {e}")
            return None

    async def _on_run_completion(self, run_result) -> None:
        """Callback invoked when a run completes - releases any held locks.

        Args:
            run_result: The completed run result
        """
        try:
            # Check if this run had a lock that needs to be released
            if (hasattr(run_result, 'metadata') and
                run_result.metadata and
                '_lock_owner_id' in run_result.metadata and
                '_lock_key' in run_result.metadata):

                # Extract site_id and environment from metadata
                site_id = run_result.metadata.get('site_id')
                environment = run_result.metadata.get('environment')
                owner_id = run_result.metadata.get('_lock_owner_id')

                if site_id and environment and owner_id:
                    # Release the lock
                    released = await self.concurrency_manager.release_lock_manual(
                        site_id=site_id,
                        environment=environment,
                        owner_id=owner_id
                    )

                    if released:
                        logger.info(f"Released lock for completed run {run_result.run_id} ({site_id}:{environment})")
                    else:
                        logger.warning(f"Failed to release lock for run {run_result.run_id} ({site_id}:{environment}) - may have expired")
                else:
                    logger.warning(f"Run {run_result.run_id} has lock metadata but missing site_id/environment/owner_id")

        except Exception as e:
            logger.error(f"Error releasing lock for completed run {run_result.run_id}: {e}")

    async def _calculate_next_run(self, schedule_state: ScheduleRuntimeState) -> None:
        """Calculate the next run time for a schedule."""
        try:
            schedule_state.next_run = self.cron_evaluator.get_next_run_time(
                schedule_state.schedule.cron,
                timezone_str=schedule_state.schedule.timezone
            )
        except Exception as e:
            logger.error(f"Failed to calculate next run for schedule {schedule_state.schedule.id}: {e}")
            schedule_state.next_run = None
            self._handle_schedule_error(schedule_state, f"Cron calculation error: {e}")

    async def _calculate_catch_up_runs(
        self,
        schedule_state: ScheduleRuntimeState,
        current_time: datetime,
        merged_params: Dict[str, Any]
    ) -> List[RunRequest]:
        """Calculate catch-up runs based on policy."""
        schedule = schedule_state.schedule
        catch_up_policy = schedule.catch_up_policy

        if not catch_up_policy or not catch_up_policy.enabled:
            return []

        # Calculate missed run times
        start_time = max(
            schedule_state.last_run or (current_time - timedelta(hours=self.max_catch_up_hours)),
            current_time - timedelta(hours=self.max_catch_up_hours)
        )

        try:
            missed_times = self.cron_evaluator.get_run_times_in_range(
                schedule.cron,
                start_time,
                schedule_state.next_run or current_time,
                schedule.timezone
            )
        except Exception as e:
            logger.error(f"Failed to calculate catch-up runs for {schedule.id}: {e}")
            return []

        if not missed_times:
            return []

        logger.info(f"Found {len(missed_times)} missed runs for schedule {schedule.id}")

        # Apply catch-up policy
        catch_up_runs = []

        if catch_up_policy.strategy == "skip":
            # Don't create any catch-up runs
            pass
        elif catch_up_policy.strategy == "run_immediately":
            # Create one run with highest priority for immediate execution
            run_request = RunRequest(
                id=str(uuid4()),
                schedule_id=schedule.id,
                site_id=schedule.site_id,
                environment=schedule.environment,
                priority=5,  # High priority for immediate catch-up
                scheduled_at=current_time,
                params={
                    **merged_params,
                    "catch_up": True,
                    "missed_runs": len(missed_times),
                    "catch_up_strategy": "run_immediately"
                },
                metadata=schedule.metadata.copy()
            )
            catch_up_runs.append(run_request)
        elif catch_up_policy.strategy == "schedule_next":
            # Create one run for the next available slot
            run_request = RunRequest(
                id=str(uuid4()),
                schedule_id=schedule.id,
                site_id=schedule.site_id,
                environment=schedule.environment,
                priority=0,  # Normal priority for scheduled catch-up
                scheduled_at=current_time,
                params={
                    **merged_params,
                    "catch_up": True,
                    "missed_runs": len(missed_times),
                    "catch_up_strategy": "schedule_next"
                },
                metadata=schedule.metadata.copy()
            )
            catch_up_runs.append(run_request)
        elif catch_up_policy.strategy == "gradual_catchup":
            # Create runs for each missed time, limited by max_catch_up_runs
            max_runs = min(len(missed_times), catch_up_policy.max_catch_up_runs or 10)

            for i, missed_time in enumerate(missed_times[:max_runs]):
                run_request = RunRequest(
                    id=str(uuid4()),
                    schedule_id=schedule.id,
                    site_id=schedule.site_id,
                    environment=schedule.environment,
                    priority=1,  # Lower priority for gradual catch-up runs
                    scheduled_at=missed_time,
                    params={
                        **merged_params,
                        "catch_up": True,
                        "missed_time": missed_time.isoformat(),
                        "catch_up_index": i + 1,
                        "catch_up_strategy": "gradual_catchup"
                    },
                    metadata=schedule.metadata.copy()
                )
                catch_up_runs.append(run_request)

        return catch_up_runs

    def _handle_schedule_error(self, schedule_state: ScheduleRuntimeState, error: str) -> None:
        """Handle an error in schedule processing."""
        schedule_state.consecutive_failures += 1
        schedule_state.last_error = error

        logger.error(f"Schedule {schedule_state.schedule.id} error #{schedule_state.consecutive_failures}: {error}")

        # Disable schedule if too many consecutive failures
        if schedule_state.consecutive_failures >= self.max_consecutive_failures:
            schedule_state.status = ScheduleStateEnum.DISABLED
            schedule_state.next_run = None
            logger.error(f"Disabled schedule {schedule_state.schedule.id} after {schedule_state.consecutive_failures} failures")
            self._update_schedule_stats()

    def _update_schedule_stats(self) -> None:
        """Update schedule statistics."""
        self._stats.total_schedules = len(self._schedules)
        self._stats.active_schedules = sum(
            1 for s in self._schedules.values()
            if s.status == ScheduleStateEnum.ACTIVE
        )
        self._stats.paused_schedules = sum(
            1 for s in self._schedules.values()
            if s.status == ScheduleStateEnum.PAUSED
        )
        self._stats.disabled_schedules = sum(
            1 for s in self._schedules.values()
            if s.status == ScheduleStateEnum.DISABLED
        )
