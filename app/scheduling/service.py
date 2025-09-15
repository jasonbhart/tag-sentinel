"""Schedule service lifecycle management and orchestration."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .scheduler import ScheduleEngine, EngineStats
from .models import Schedule
from .cron import CronEvaluator
from .blackout import BlackoutManager
from .locks import ConcurrencyManager, create_concurrency_manager
from .dispatch import RunDispatcher, create_mock_dispatcher
from .environments import EnvironmentConfigManager
from .audit_integration import create_audit_runner_backend
from .metrics import create_metrics_collector, create_scheduling_metrics

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service lifecycle status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ServiceConfig:
    """Configuration for the scheduling service."""

    # Engine configuration
    tick_interval_seconds: int = 60
    max_catch_up_hours: int = 24
    max_consecutive_failures: int = 5

    # Concurrency configuration
    concurrency_backend_type: str = "memory"  # "memory" or "redis"
    redis_url: Optional[str] = None
    default_lock_timeout_seconds: int = 3600
    lock_cleanup_interval_seconds: int = 300

    # Dispatcher configuration
    max_queue_size: int = 1000
    max_concurrent_runs: int = 10
    idempotency_window_minutes: int = 60
    dispatch_cleanup_interval_seconds: int = 300
    use_real_audit_runner: bool = False  # Use real audit runner vs mock
    audit_timeout_minutes: int = 60

    # Blackout configuration
    emergency_blackout_enabled: bool = True

    # Environment configuration
    environment_config_path: Optional[str] = None

    # Schedule loading
    schedule_config_paths: List[str] = None
    auto_reload_configs: bool = False
    config_reload_interval_seconds: int = 300


@dataclass
class ServiceHealth:
    """Health status of the scheduling service."""
    status: ServiceStatus
    uptime_seconds: float
    schedules_active: int
    schedules_total: int
    runs_queued: int
    runs_running: int
    last_error: Optional[str] = None
    component_status: Dict[str, str] = None


class SchedulingService:
    """High-level scheduling service that manages the complete lifecycle."""

    def __init__(self, config: ServiceConfig):
        """Initialize the scheduling service.

        Args:
            config: Service configuration
        """
        self.config = config
        self._status = ServiceStatus.STOPPED
        self._start_time: Optional[float] = None
        self._last_error: Optional[str] = None

        # Core components
        self.cron_evaluator = CronEvaluator()
        self.blackout_manager: Optional[BlackoutManager] = None
        self.concurrency_manager: Optional[ConcurrencyManager] = None
        self.run_dispatcher: Optional[RunDispatcher] = None
        self.environment_manager: Optional[EnvironmentConfigManager] = None
        self.schedule_engine: Optional[ScheduleEngine] = None

        # Config reloading
        self._config_reload_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the scheduling service."""
        if self._status != ServiceStatus.STOPPED:
            raise RuntimeError(f"Cannot start service in status {self._status}")

        logger.info("Starting scheduling service")
        self._status = ServiceStatus.STARTING

        try:
            # Initialize components
            await self._initialize_components()

            # Load schedules
            await self._load_schedules()

            # Start the engine
            await self.schedule_engine.start()

            # Start config reloading if enabled
            if self.config.auto_reload_configs:
                self._config_reload_task = asyncio.create_task(self._config_reload_loop())

            self._status = ServiceStatus.RUNNING
            self._start_time = asyncio.get_event_loop().time()

            logger.info("Scheduling service started successfully")

        except Exception as e:
            self._last_error = str(e)
            self._status = ServiceStatus.ERROR
            logger.error(f"Failed to start scheduling service: {e}", exc_info=True)

            # Cleanup on failure
            await self._cleanup_components()
            raise

    async def stop(self) -> None:
        """Stop the scheduling service."""
        if self._status == ServiceStatus.STOPPED:
            return

        logger.info("Stopping scheduling service")
        self._status = ServiceStatus.STOPPING

        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Stop config reloading
            if self._config_reload_task:
                self._config_reload_task.cancel()
                try:
                    await self._config_reload_task
                except asyncio.CancelledError:
                    pass
                self._config_reload_task = None

            # Stop engine and components
            await self._cleanup_components()

            self._status = ServiceStatus.STOPPED
            logger.info("Scheduling service stopped")

        except Exception as e:
            self._last_error = str(e)
            self._status = ServiceStatus.ERROR
            logger.error(f"Error stopping scheduling service: {e}", exc_info=True)
            raise

    async def pause(self) -> None:
        """Pause the scheduling service (stop scheduling new runs)."""
        if self._status != ServiceStatus.RUNNING:
            raise RuntimeError(f"Cannot pause service in status {self._status}")

        logger.info("Pausing scheduling service")

        # Pause all active schedules
        if self.schedule_engine:
            for schedule_state in self.schedule_engine.list_schedules():
                if schedule_state.status.value == "active":
                    self.schedule_engine.pause_schedule(schedule_state.schedule.id)

        self._status = ServiceStatus.PAUSED
        logger.info("Scheduling service paused")

    async def resume(self) -> None:
        """Resume the scheduling service."""
        if self._status != ServiceStatus.PAUSED:
            raise RuntimeError(f"Cannot resume service in status {self._status}")

        logger.info("Resuming scheduling service")

        # Resume all paused schedules
        if self.schedule_engine:
            for schedule_state in self.schedule_engine.list_schedules():
                if schedule_state.status.value == "paused":
                    self.schedule_engine.resume_schedule(schedule_state.schedule.id)

        self._status = ServiceStatus.RUNNING
        logger.info("Scheduling service resumed")

    async def reload_config(self) -> None:
        """Reload configuration and schedules."""
        if self._status not in [ServiceStatus.RUNNING, ServiceStatus.PAUSED]:
            raise RuntimeError(f"Cannot reload config in status {self._status}")

        logger.info("Reloading service configuration")

        try:
            # Reload environment configuration
            if self.environment_manager:
                self.environment_manager.load_config()

            # Reload schedules
            await self._load_schedules()

            logger.info("Configuration reloaded successfully")

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to reload configuration: {e}", exc_info=True)
            raise

    def get_status(self) -> ServiceStatus:
        """Get current service status."""
        return self._status

    def get_health(self) -> ServiceHealth:
        """Get service health information."""
        uptime = 0.0
        if self._start_time:
            uptime = asyncio.get_event_loop().time() - self._start_time

        schedules_active = 0
        schedules_total = 0
        runs_queued = 0
        runs_running = 0

        component_status = {}

        if self.schedule_engine:
            stats = self.schedule_engine.get_stats()
            schedules_active = stats.active_schedules
            schedules_total = stats.total_schedules
            component_status["engine"] = "healthy"
        else:
            component_status["engine"] = "not_initialized"

        if self.run_dispatcher:
            try:
                dispatcher_stats = self.run_dispatcher.get_stats()
                queue_status = self.run_dispatcher.get_queue_status()
                runs_queued = dispatcher_stats.current_queue_size
                runs_running = queue_status["running_count"]
                component_status["dispatcher"] = "healthy"
            except Exception:
                component_status["dispatcher"] = "error"
        else:
            component_status["dispatcher"] = "not_initialized"

        if self.concurrency_manager:
            component_status["concurrency"] = "healthy"
        else:
            component_status["concurrency"] = "not_initialized"

        if self.blackout_manager:
            component_status["blackout"] = "healthy"
        else:
            component_status["blackout"] = "not_initialized"

        return ServiceHealth(
            status=self._status,
            uptime_seconds=uptime,
            schedules_active=schedules_active,
            schedules_total=schedules_total,
            runs_queued=runs_queued,
            runs_running=runs_running,
            last_error=self._last_error,
            component_status=component_status
        )

    def get_engine_stats(self) -> Optional[EngineStats]:
        """Get schedule engine statistics."""
        if self.schedule_engine:
            return self.schedule_engine.get_stats()
        return None

    async def add_schedule(self, schedule: Schedule) -> None:
        """Add a schedule to the service."""
        if not self.schedule_engine:
            raise RuntimeError("Service not initialized")

        self.schedule_engine.add_schedule(schedule)
        logger.info(f"Added schedule {schedule.id}")

    async def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule from the service."""
        if not self.schedule_engine:
            raise RuntimeError("Service not initialized")

        result = self.schedule_engine.remove_schedule(schedule_id)
        if result:
            logger.info(f"Removed schedule {schedule_id}")
        return result

    async def trigger_schedule(self, schedule_id: str, force: bool = False) -> Optional[str]:
        """Manually trigger a schedule."""
        if not self.schedule_engine:
            raise RuntimeError("Service not initialized")

        return await self.schedule_engine.trigger_schedule(schedule_id, force)

    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all schedules with their current state."""
        if not self.schedule_engine:
            return []

        schedules = []
        for schedule_state in self.schedule_engine.list_schedules():
            schedules.append({
                "id": schedule_state.schedule.id,
                "name": schedule_state.schedule.name,
                "site_id": schedule_state.schedule.site_id,
                "environment": schedule_state.schedule.environment,
                "cron_expression": schedule_state.schedule.cron,
                "enabled": schedule_state.schedule.enabled,
                "status": schedule_state.status.value,
                "next_run": schedule_state.next_run.isoformat() if schedule_state.next_run else None,
                "last_run": schedule_state.last_run.isoformat() if schedule_state.last_run else None,
                "consecutive_failures": schedule_state.consecutive_failures,
                "last_error": schedule_state.last_error
            })

        return schedules

    async def _initialize_components(self) -> None:
        """Initialize all service components."""
        logger.info("Initializing service components")

        # Initialize environment manager
        if self.config.environment_config_path:
            self.environment_manager = EnvironmentConfigManager(self.config.environment_config_path)
            self.environment_manager.load_config()

        # Initialize blackout manager
        # Note: BlackoutManager currently accepts only a timezone string
        self.blackout_manager = BlackoutManager()

        # Initialize concurrency manager
        self.concurrency_manager = await create_concurrency_manager(
            backend_type=self.config.concurrency_backend_type,
            redis_url=self.config.redis_url,
            default_timeout_seconds=self.config.default_lock_timeout_seconds,
            cleanup_interval_seconds=self.config.lock_cleanup_interval_seconds
        )

        # Initialize run dispatcher
        if self.config.use_real_audit_runner:
            # Use real audit runner backend
            audit_backend = create_audit_runner_backend(
                max_concurrent_audits=self.config.max_concurrent_runs,
                default_timeout_minutes=self.config.audit_timeout_minutes
            )
            self.run_dispatcher = RunDispatcher(
                backend=audit_backend,
                max_queue_size=self.config.max_queue_size,
                max_concurrent_runs=self.config.max_concurrent_runs,
                idempotency_window_minutes=self.config.idempotency_window_minutes,
                cleanup_interval_seconds=self.config.dispatch_cleanup_interval_seconds
            )
        else:
            # Use mock dispatcher for testing/development
            self.run_dispatcher = create_mock_dispatcher(
                max_queue_size=self.config.max_queue_size,
                max_concurrent_runs=self.config.max_concurrent_runs
            )

        # Initialize metrics collector
        metrics_collector = create_metrics_collector(retention_hours=24)

        # Initialize schedule engine
        self.schedule_engine = ScheduleEngine(
            cron_evaluator=self.cron_evaluator,
            blackout_manager=self.blackout_manager,
            concurrency_manager=self.concurrency_manager,
            run_dispatcher=self.run_dispatcher,
            environment_manager=self.environment_manager or EnvironmentConfigManager(),
            metrics_collector=metrics_collector,
            tick_interval_seconds=self.config.tick_interval_seconds,
            max_catch_up_hours=self.config.max_catch_up_hours,
            max_consecutive_failures=self.config.max_consecutive_failures
        )

        logger.info("Service components initialized")

    async def _cleanup_components(self) -> None:
        """Clean up all service components."""
        logger.info("Cleaning up service components")

        if self.schedule_engine:
            await self.schedule_engine.stop()
            self.schedule_engine = None

        if self.run_dispatcher:
            await self.run_dispatcher.stop()
            self.run_dispatcher = None

        if self.concurrency_manager:
            await self.concurrency_manager.stop()
            self.concurrency_manager = None

        self.blackout_manager = None
        self.environment_manager = None

        logger.info("Service components cleaned up")

    async def _load_schedules(self) -> None:
        """Load schedules from configuration files."""
        if not self.config.schedule_config_paths or not self.schedule_engine:
            logger.info("No schedule configuration paths specified")
            return

        logger.info(f"Loading schedules from {len(self.config.schedule_config_paths)} configuration files")

        for config_path in self.config.schedule_config_paths:
            try:
                await self._load_schedules_from_file(config_path)
            except Exception as e:
                logger.error(f"Failed to load schedules from {config_path}: {e}")

    async def _load_schedules_from_file(self, config_path: str) -> None:
        """Load schedules from a single configuration file."""
        # This is a placeholder - in a real implementation, you'd load from YAML/JSON
        # For now, just log that we would load schedules
        logger.info(f"Would load schedules from {config_path}")

    async def _config_reload_loop(self) -> None:
        """Background task for automatic config reloading."""
        logger.info("Starting config reload loop")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.config_reload_interval_seconds
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                # Time to reload config
                try:
                    await self.reload_config()
                except Exception as e:
                    logger.error(f"Error in config reload: {e}", exc_info=True)


# Factory functions for common service configurations

def create_development_service(
    environment_config_path: Optional[str] = None,
    schedule_config_paths: Optional[List[str]] = None
) -> SchedulingService:
    """Create a scheduling service configured for development."""
    config = ServiceConfig(
        tick_interval_seconds=30,  # More frequent ticks for development
        concurrency_backend_type="memory",
        max_concurrent_runs=3,
        max_queue_size=50,
        auto_reload_configs=True,
        config_reload_interval_seconds=60,  # Reload every minute in dev
        environment_config_path=environment_config_path,
        schedule_config_paths=schedule_config_paths or []
    )

    return SchedulingService(config)


def create_production_service(
    redis_url: str,
    environment_config_path: str,
    schedule_config_paths: List[str]
) -> SchedulingService:
    """Create a scheduling service configured for production."""
    config = ServiceConfig(
        tick_interval_seconds=60,
        concurrency_backend_type="redis",
        redis_url=redis_url,
        max_concurrent_runs=20,
        max_queue_size=2000,
        auto_reload_configs=False,  # Manual reloading in production
        environment_config_path=environment_config_path,
        schedule_config_paths=schedule_config_paths,
        default_lock_timeout_seconds=7200,  # 2 hours
        max_catch_up_hours=48,  # Larger catch-up window
        use_real_audit_runner=True,  # Use real audit runner in production
        audit_timeout_minutes=90  # Longer timeout for production
    )

    return SchedulingService(config)
