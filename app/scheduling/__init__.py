"""Scheduling system for automated audit runs.

This package provides comprehensive scheduling functionality including:
- Cron-based schedule evaluation with timezone support
- Blackout window management for maintenance periods
- Environment-specific configuration handling
- Distributed concurrency control
- Run request dispatching and queue management
- Catch-up policies for missed runs
"""

from .models import (
    Schedule,
    ScheduleState,
    ScheduleStatus,
    ScheduleCollection,
    RunRequest,
    BlackoutWindow,
    CatchUpPolicy,
)

from .cron import (
    CronEvaluator,
    CronValidationError,
    CronEvaluationError,
    validate_cron_expression,
    get_next_run_time,
)

from .blackout import (
    BlackoutManager,
    BlackoutStatus,
    BlackoutEvaluationError,
    create_maintenance_window,
    create_absolute_blackout,
    create_emergency_blackout,
)

from .environments import (
    EnvironmentConfig,
    EnvironmentConfigManager,
    ConfigValidationError,
    ResolvedEnvironmentConfig,
)

from .locks import (
    ConcurrencyManager,
    LockInfo,
    LockError,
    LockTimeoutError,
    create_concurrency_manager,
)

from .dispatch import (
    RunDispatcher,
    RunResult,
    RunStatus,
    DispatchError,
    DispatchStats,
    create_mock_dispatcher,
)

__all__ = [
    # Models
    "Schedule",
    "ScheduleState",
    "ScheduleStatus",
    "ScheduleCollection",
    "RunRequest",
    "BlackoutWindow",
    "CatchUpPolicy",

    # Cron evaluation
    "CronEvaluator",
    "CronValidationError",
    "CronEvaluationError",
    "validate_cron_expression",
    "get_next_run_time",

    # Blackout management
    "BlackoutManager",
    "BlackoutStatus",
    "BlackoutEvaluationError",
    "create_maintenance_window",
    "create_absolute_blackout",
    "create_emergency_blackout",

    # Environment configuration
    "EnvironmentConfig",
    "EnvironmentConfigManager",
    "ConfigValidationError",
    "ResolvedEnvironmentConfig",

    # Concurrency control
    "ConcurrencyManager",
    "LockInfo",
    "LockError",
    "LockTimeoutError",
    "create_concurrency_manager",

    # Run dispatching
    "RunDispatcher",
    "RunResult",
    "RunStatus",
    "DispatchError",
    "DispatchStats",
    "create_mock_dispatcher",
]