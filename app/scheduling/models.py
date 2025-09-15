"""Comprehensive data models for scheduling system.

This module defines all data models needed for the scheduling system including
schedules, run requests, blackout windows, and state tracking.
"""

import re
from datetime import datetime, date, time, timezone
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
import zoneinfo


class CatchUpPolicy(BaseModel):
    """Policy for handling missed schedules after downtime."""

    enabled: bool = Field(
        default=True,
        description="Whether catch-up processing is enabled"
    )

    strategy: Literal['skip', 'run_immediately', 'schedule_next', 'gradual_catchup'] = Field(
        default='skip',
        description="Strategy for handling missed schedules"
    )

    max_catchup_window_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # 1 week max
        description="Maximum time window to consider for catch-up runs"
    )

    max_catchup_runs: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of missed runs to catch up (None = unlimited)"
    )

    gradual_spacing_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,  # 24 hours max
        description="Minutes between gradual catch-up runs"
    )

    @model_validator(mode='after')
    def validate_gradual_settings(self):
        """Validate gradual catch-up specific settings."""
        if self.strategy == 'gradual_catchup':
            if self.max_catchup_runs is None:
                raise ValueError("max_catchup_runs required for gradual_catchup strategy")
        return self


class BlackoutWindow(BaseModel):
    """Defines a time window when schedules should not run."""

    name: str = Field(description="Human-readable name for the blackout window")

    # Recurring patterns
    days_of_week: List[str] = Field(
        default_factory=list,
        description="Days of week: Mon, Tue, Wed, Thu, Fri, Sat, Sun"
    )

    start_time: str = Field(
        description="Start time in HH:MM format (24-hour)"
    )

    end_time: str = Field(
        description="End time in HH:MM format (24-hour)"
    )

    # Absolute date ranges (override recurring if set)
    start_date: Optional[date] = Field(
        default=None,
        description="Absolute start date (overrides recurring pattern)"
    )

    end_date: Optional[date] = Field(
        default=None,
        description="Absolute end date (overrides recurring pattern)"
    )

    timezone: str = Field(
        default='UTC',
        description="IANA timezone identifier for the blackout window"
    )

    priority: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Priority level for overlapping blackouts (higher wins)"
    )

    emergency: bool = Field(
        default=False,
        description="Emergency blackout that activates immediately"
    )

    @field_validator('days_of_week')
    @classmethod
    def validate_days_of_week(cls, v):
        """Validate day names."""
        valid_days = {'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'}
        for day in v:
            if day not in valid_days:
                raise ValueError(f"Invalid day '{day}'. Must be one of: {valid_days}")
        return v

    @field_validator('start_time', 'end_time')
    @classmethod
    def validate_time_format(cls, v):
        """Validate HH:MM format."""
        pattern = r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid time format '{v}'. Must be HH:MM (24-hour)")
        return v

    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v):
        """Validate IANA timezone identifier."""
        try:
            zoneinfo.ZoneInfo(v)
        except zoneinfo.ZoneInfoNotFoundError:
            raise ValueError(f"Invalid timezone '{v}'. Must be valid IANA identifier")
        return v

    @model_validator(mode='after')
    def validate_date_range(self):
        """Validate absolute date ranges."""
        if self.start_date is not None or self.end_date is not None:
            # If either date is set, both must be set for absolute windows
            if self.start_date is None:
                raise ValueError("start_date required when end_date is set for absolute blackout window")
            if self.end_date is None:
                raise ValueError("end_date required when start_date is set for absolute blackout window")

            # Validate date order
            if self.start_date > self.end_date:
                raise ValueError("start_date must be before or equal to end_date")

        return self

    def is_absolute_window(self) -> bool:
        """Check if this is an absolute date range window."""
        return self.start_date is not None and self.end_date is not None


class RunRequest(BaseModel):
    """Request to execute an audit run."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique run request ID")

    site_id: str = Field(description="Site identifier for the audit")

    environment: str = Field(description="Environment name (production, staging, etc.)")

    scheduled_at: datetime = Field(description="When the run was scheduled to execute")

    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the run request was created"
    )

    schedule_id: Optional[str] = Field(
        default=None,
        description="Schedule that generated this request (None for manual runs)"
    )

    priority: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Run priority (higher values run first)"
    )

    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Audit run parameters (URLs, configuration, etc.)"
    )

    idempotency_key: Optional[str] = Field(
        default=None,
        description="Key to prevent duplicate runs"
    )

    timeout_minutes: int = Field(
        default=60,
        ge=1,
        le=480,  # 8 hours max
        description="Maximum runtime before timeout"
    )

    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retry attempts"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the run"
    )

    def __lt__(self, other):
        """Support priority queue ordering (higher priority = lower value for heapq)."""
        if isinstance(other, RunRequest):
            return self.priority > other.priority
        return NotImplemented


class ScheduleState(str, Enum):
    """Current state of a schedule."""
    ACTIVE = "active"           # Normal operation
    PAUSED = "paused"          # Temporarily disabled
    DISABLED = "disabled"       # Permanently disabled
    ERROR = "error"            # Configuration or runtime error
    BLACKOUT = "blackout"      # Currently in blackout window


class ScheduleStatus(BaseModel):
    """Runtime status information for a schedule."""

    state: ScheduleState = Field(description="Current schedule state")

    last_run_at: Optional[datetime] = Field(
        default=None,
        description="When the schedule last executed successfully"
    )

    last_attempt_at: Optional[datetime] = Field(
        default=None,
        description="When the schedule last attempted to run"
    )

    next_run_at: Optional[datetime] = Field(
        default=None,
        description="When the schedule will next run"
    )

    consecutive_failures: int = Field(
        default=0,
        ge=0,
        description="Count of consecutive failures"
    )

    total_runs: int = Field(
        default=0,
        ge=0,
        description="Total number of runs executed"
    )

    successful_runs: int = Field(
        default=0,
        ge=0,
        description="Number of successful runs"
    )

    last_error: Optional[str] = Field(
        default=None,
        description="Last error message if any"
    )

    last_error_at: Optional[datetime] = Field(
        default=None,
        description="When the last error occurred"
    )

    current_blackout: Optional[str] = Field(
        default=None,
        description="Name of current active blackout window"
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the status was last updated"
    )


class Schedule(BaseModel):
    """Complete schedule definition for automated audit runs."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique schedule ID")

    name: str = Field(description="Human-readable schedule name")

    site_id: str = Field(description="Site identifier to audit")

    environment: Union[Literal['production', 'staging'], str] = Field(
        description="Environment to audit (production, staging, or custom)"
    )

    cron: str = Field(description="Cron expression for schedule timing")

    timezone: str = Field(
        default='UTC',
        description="IANA timezone identifier for schedule evaluation"
    )

    blackout_windows: List[BlackoutWindow] = Field(
        default_factory=list,
        description="Blackout windows when schedule should not run"
    )

    catch_up_policy: CatchUpPolicy = Field(
        default_factory=CatchUpPolicy,
        description="Policy for handling missed schedules"
    )

    max_concurrent_runs: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of concurrent runs for this schedule"
    )

    run_timeout_minutes: int = Field(
        default=60,
        ge=5,
        le=480,  # 8 hours
        description="Timeout for individual runs"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed runs"
    )

    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Audit run parameters (serialized RunParams)"
    )

    enabled: bool = Field(
        default=True,
        description="Whether the schedule is enabled"
    )

    # Metadata and tracking
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the schedule was created"
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the schedule was last updated"
    )

    created_by: Optional[str] = Field(
        default=None,
        description="User who created the schedule"
    )

    tags: List[str] = Field(
        default_factory=list,
        description="Tags for organizing and filtering schedules"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    # Runtime status (not persisted with schedule definition)
    status: Optional[ScheduleStatus] = Field(
        default=None,
        exclude=True,
        description="Current runtime status (populated at runtime)"
    )

    @field_validator('cron')
    @classmethod
    def validate_cron_expression(cls, v):
        """Validate cron expression format."""
        # Check for named expressions first
        named_expressions = ['@yearly', '@annually', '@monthly', '@weekly', '@daily', '@hourly']
        if v.strip() in named_expressions:
            return v

        # Basic validation for regular cron expressions
        parts = v.split()
        if len(parts) < 5 or len(parts) > 6:
            raise ValueError("Cron expression must have 5 or 6 fields")

        # Check for valid cron field patterns
        for part in parts:
            if not re.match(r'^[\d\*\-\,\/\?]+$', part):
                raise ValueError(f"Invalid cron field: {part}")

        return v

    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v):
        """Validate IANA timezone identifier."""
        try:
            zoneinfo.ZoneInfo(v)
        except zoneinfo.ZoneInfoNotFoundError:
            raise ValueError(f"Invalid timezone '{v}'. Must be valid IANA identifier")
        return v

    @field_validator('site_id')
    @classmethod
    def validate_site_id(cls, v):
        """Validate site ID format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("site_id must contain only alphanumeric characters, underscores, and hyphens")
        return v

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment name."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("environment must contain only alphanumeric characters, underscores, and hyphens")
        return v

    @model_validator(mode='after')
    def validate_blackout_priorities(self):
        """Validate blackout window priorities are unique or properly ordered."""
        priorities = [bw.priority for bw in self.blackout_windows]
        if len(priorities) != len(set(priorities)) and len(priorities) > 1:
            # Allow duplicate priorities but warn in logs (handled elsewhere)
            pass
        return self

    def get_site_env_key(self) -> str:
        """Get unique key for site-environment combination."""
        return f"{self.site_id}:{self.environment}"

    def is_enabled_for_execution(self) -> bool:
        """Check if schedule is enabled and ready for execution."""
        return self.enabled and (self.status is None or self.status.state == ScheduleState.ACTIVE)


class ScheduleCollection(BaseModel):
    """Collection of schedules with metadata."""

    schedules: List[Schedule] = Field(default_factory=list, description="List of schedules")

    version: str = Field(default="1.0", description="Configuration version")

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the collection was last updated"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Collection metadata"
    )

    def get_by_site_env(self, site_id: str, environment: str) -> List[Schedule]:
        """Get all schedules for a site-environment combination."""
        return [s for s in self.schedules if s.site_id == site_id and s.environment == environment]

    def get_enabled_schedules(self) -> List[Schedule]:
        """Get all enabled schedules."""
        return [s for s in self.schedules if s.is_enabled_for_execution()]

    def get_by_tag(self, tag: str) -> List[Schedule]:
        """Get schedules with a specific tag."""
        return [s for s in self.schedules if tag in s.tags]
