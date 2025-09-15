"""Blackout window management for preventing scheduled runs during maintenance.

This module provides sophisticated blackout window evaluation with support for:
- Recurring daily/weekly patterns
- Absolute date/time ranges
- Timezone-aware calculations
- Emergency blackout activation
- Blackout window testing and validation
"""

from datetime import datetime, date, time, timedelta, timezone
from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
import logging
import zoneinfo

from .models import BlackoutWindow


logger = logging.getLogger(__name__)


class BlackoutEvaluationError(Exception):
    """Exception raised when blackout evaluation fails."""
    pass


@dataclass
class BlackoutStatus:
    """Result of blackout window evaluation."""

    is_blackout: bool
    active_windows: List[BlackoutWindow] = field(default_factory=list)
    next_blackout_start: Optional[datetime] = None
    next_blackout_end: Optional[datetime] = None
    time_until_next_blackout: Optional[timedelta] = None
    time_until_blackout_end: Optional[timedelta] = None

    def get_primary_blackout(self) -> Optional[BlackoutWindow]:
        """Get the highest priority active blackout window."""
        if not self.active_windows:
            return None
        return max(self.active_windows, key=lambda bw: bw.priority)

    def get_blackout_reason(self) -> Optional[str]:
        """Get reason for blackout from primary window."""
        primary = self.get_primary_blackout()
        return primary.name if primary else None


@dataclass
class BlackoutEvaluation:
    """Detailed evaluation of blackout windows for a specific time."""

    check_time: datetime
    status: BlackoutStatus
    evaluated_windows: List[Tuple[BlackoutWindow, bool]] = field(default_factory=list)
    evaluation_timezone: str = 'UTC'

    def get_evaluation_summary(self) -> str:
        """Get human-readable evaluation summary."""
        if self.status.is_blackout:
            reason = self.status.get_blackout_reason()
            return f"BLACKOUT ACTIVE: {reason}"
        else:
            next_time = self.status.time_until_next_blackout
            if next_time:
                return f"No blackout (next in {next_time})"
            else:
                return "No blackout scheduled"


class BlackoutManager:
    """Manager for evaluating and handling blackout windows."""

    def __init__(self, timezone_str: str = 'UTC'):
        """Initialize blackout manager.

        Args:
            timezone_str: Default timezone for evaluations
        """
        self.default_timezone = timezone_str
        self.emergency_blackouts: Set[str] = set()  # Active emergency blackout IDs

    def is_blackout_active(
        self,
        windows: List[BlackoutWindow],
        check_time: Optional[datetime] = None,
        timezone_str: Optional[str] = None
    ) -> BlackoutStatus:
        """Check if any blackout windows are active at the given time.

        Args:
            windows: List of blackout windows to evaluate
            check_time: Time to check (default: current time)
            timezone_str: Timezone for evaluation (default: manager default)

        Returns:
            BlackoutStatus with evaluation results

        Raises:
            BlackoutEvaluationError: If evaluation fails
        """
        tz_str = timezone_str or self.default_timezone
        try:
            tz = zoneinfo.ZoneInfo(tz_str)
        except zoneinfo.ZoneInfoNotFoundError:
            raise BlackoutEvaluationError(f"Invalid timezone: {tz_str}")

        if check_time is None:
            check_time = datetime.now(tz)
        elif check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=tz)
        else:
            check_time = check_time.astimezone(tz)

        active_windows = []

        for window in windows:
            # Check emergency blackouts first
            if window.emergency and window.name in self.emergency_blackouts:
                active_windows.append(window)
                continue

            if self._is_window_active(window, check_time):
                active_windows.append(window)

        # Calculate next blackout timing
        next_start, next_end = self._get_next_blackout_times(windows, check_time)

        # Calculate time deltas
        time_until_next_blackout = None
        time_until_blackout_end = None

        if next_start:
            time_until_next_blackout = next_start - check_time

        if active_windows and next_end:
            time_until_blackout_end = next_end - check_time

        return BlackoutStatus(
            is_blackout=len(active_windows) > 0,
            active_windows=active_windows,
            next_blackout_start=next_start,
            next_blackout_end=next_end,
            time_until_next_blackout=time_until_next_blackout,
            time_until_blackout_end=time_until_blackout_end
        )

    def evaluate_detailed(
        self,
        windows: List[BlackoutWindow],
        check_time: Optional[datetime] = None,
        timezone_str: Optional[str] = None
    ) -> BlackoutEvaluation:
        """Perform detailed evaluation of blackout windows.

        Args:
            windows: List of blackout windows to evaluate
            check_time: Time to check (default: current time)
            timezone_str: Timezone for evaluation (default: manager default)

        Returns:
            Detailed BlackoutEvaluation with per-window results
        """
        tz_str = timezone_str or self.default_timezone
        try:
            tz = zoneinfo.ZoneInfo(tz_str)
        except zoneinfo.ZoneInfoNotFoundError:
            raise BlackoutEvaluationError(f"Invalid timezone: {tz_str}")

        if check_time is None:
            check_time = datetime.now(tz)
        elif check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=tz)
        else:
            check_time = check_time.astimezone(tz)

        evaluated_windows = []
        for window in windows:
            is_active = (
                (window.emergency and window.name in self.emergency_blackouts) or
                self._is_window_active(window, check_time)
            )
            evaluated_windows.append((window, is_active))

        status = self.is_blackout_active(windows, check_time, tz_str)

        return BlackoutEvaluation(
            check_time=check_time,
            status=status,
            evaluated_windows=evaluated_windows,
            evaluation_timezone=tz_str
        )

    def _is_window_active(self, window: BlackoutWindow, check_time: datetime) -> bool:
        """Check if a specific blackout window is active."""
        try:
            # Convert check_time to window's timezone
            window_tz = zoneinfo.ZoneInfo(window.timezone)
            window_time = check_time.astimezone(window_tz)

            # Handle absolute date ranges first
            if window.is_absolute_window():
                return self._is_absolute_window_active(window, window_time)
            else:
                return self._is_recurring_window_active(window, window_time)

        except Exception as e:
            logger.error(f"Error evaluating blackout window '{window.name}': {e}")
            return False

    def _is_absolute_window_active(self, window: BlackoutWindow, window_time: datetime) -> bool:
        """Check if absolute date range blackout window is active."""
        current_date = window_time.date()
        current_time = window_time.time()

        # Parse window times
        start_time = time.fromisoformat(window.start_time)
        end_time = time.fromisoformat(window.end_time)

        # Check date range
        if window.start_date and current_date < window.start_date:
            return False

        if window.end_date and current_date > window.end_date:
            return False

        # For single-day absolute windows, check time range
        if window.start_date and window.end_date and window.start_date == window.end_date:
            if start_time <= end_time:
                return start_time <= current_time <= end_time
            else:
                # Spans midnight
                return current_time >= start_time or current_time <= end_time

        # For multi-day absolute windows
        if window.start_date and window.end_date:
            if current_date == window.start_date:
                # First day - check from start_time onwards
                return current_time >= start_time
            elif current_date == window.end_date:
                # Last day - check until end_time
                return current_time <= end_time
            else:
                # Middle days - always active
                return window.start_date < current_date < window.end_date

        return False

    def _is_recurring_window_active(self, window: BlackoutWindow, window_time: datetime) -> bool:
        """Check if recurring blackout window is active."""
        current_day = window_time.strftime('%a')  # Mon, Tue, etc.
        current_time = window_time.time()

        # Check if current day matches
        if window.days_of_week and current_day not in window.days_of_week:
            return False

        # Parse window times
        start_time = time.fromisoformat(window.start_time)
        end_time = time.fromisoformat(window.end_time)

        # Check time range
        if start_time <= end_time:
            # Same day window
            return start_time <= current_time <= end_time
        else:
            # Window spans midnight
            return current_time >= start_time or current_time <= end_time

    def _get_next_blackout_times(
        self,
        windows: List[BlackoutWindow],
        from_time: datetime
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get next blackout start and end times."""
        next_starts = []
        next_ends = []

        for window in windows:
            if window.emergency and window.name in self.emergency_blackouts:
                continue  # Emergency blackouts are always active

            try:
                start, end = self._get_next_window_times(window, from_time)
                if start:
                    next_starts.append(start)
                if end:
                    next_ends.append(end)
            except Exception as e:
                logger.error(f"Error calculating next times for window '{window.name}': {e}")

        next_start = min(next_starts) if next_starts else None
        next_end = min(next_ends) if next_ends else None

        return next_start, next_end

    def _get_next_window_times(
        self,
        window: BlackoutWindow,
        from_time: datetime
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get next start and end times for a specific window."""
        try:
            window_tz = zoneinfo.ZoneInfo(window.timezone)
            window_time = from_time.astimezone(window_tz)

            if window.is_absolute_window():
                return self._get_next_absolute_window_times(window, window_time)
            else:
                return self._get_next_recurring_window_times(window, window_time)
        except Exception as e:
            logger.error(f"Error calculating next times for window '{window.name}': {e}")
            return None, None

    def _get_next_absolute_window_times(
        self,
        window: BlackoutWindow,
        from_time: datetime
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get next start/end times for absolute window."""
        if not window.start_date:
            return None, None

        start_time_obj = time.fromisoformat(window.start_time)
        end_time_obj = time.fromisoformat(window.end_time)

        # Get timezone object
        window_tz = zoneinfo.ZoneInfo(window.timezone)

        # Calculate absolute start time
        start_dt = datetime.combine(window.start_date, start_time_obj, window_tz)

        if window.end_date:
            end_dt = datetime.combine(window.end_date, end_time_obj, window_tz)
        else:
            # Single day window
            end_dt = datetime.combine(window.start_date, end_time_obj, window_tz)
            if end_time_obj < start_time_obj:
                # Spans midnight
                end_dt += timedelta(days=1)

        # Convert to same timezone as from_time
        start_dt = start_dt.astimezone(from_time.tzinfo)
        end_dt = end_dt.astimezone(from_time.tzinfo)

        # Return future times only
        next_start = start_dt if start_dt > from_time else None
        next_end = end_dt if end_dt > from_time else None

        return next_start, next_end

    def _get_next_recurring_window_times(
        self,
        window: BlackoutWindow,
        from_time: datetime
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get next start/end times for recurring window."""
        if not window.days_of_week:
            # Daily window
            days_to_check = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        else:
            days_to_check = window.days_of_week

        start_time_obj = time.fromisoformat(window.start_time)
        end_time_obj = time.fromisoformat(window.end_time)

        # Map day names to numbers (Monday = 0)
        day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}

        current_weekday = from_time.weekday()
        next_starts = []
        next_ends = []

        for day_name in days_to_check:
            target_weekday = day_map[day_name]

            # Calculate days until target weekday
            days_ahead = target_weekday - current_weekday
            if days_ahead < 0:  # Target day already passed this week
                days_ahead += 7
            elif days_ahead == 0:  # Today - check if time has passed
                today_start = datetime.combine(from_time.date(), start_time_obj, from_time.tzinfo)
                if today_start <= from_time:
                    days_ahead = 7  # Next occurrence is next week

            # Calculate next occurrence
            target_date = from_time.date() + timedelta(days=days_ahead)
            next_start = datetime.combine(target_date, start_time_obj, from_time.tzinfo)

            # Calculate end time
            if end_time_obj >= start_time_obj:
                # Same day
                next_end = datetime.combine(target_date, end_time_obj, from_time.tzinfo)
            else:
                # Next day
                next_end = datetime.combine(target_date + timedelta(days=1), end_time_obj, from_time.tzinfo)

            next_starts.append(next_start)
            next_ends.append(next_end)

        # Return the earliest times
        next_start = min(next_starts) if next_starts else None
        next_end = min(next_ends) if next_ends else None

        return next_start, next_end

    def activate_emergency_blackout(self, blackout_name: str) -> None:
        """Activate an emergency blackout.

        Args:
            blackout_name: Name of the blackout window to activate
        """
        self.emergency_blackouts.add(blackout_name)
        logger.warning(f"Emergency blackout activated: {blackout_name}")

    def deactivate_emergency_blackout(self, blackout_name: str) -> None:
        """Deactivate an emergency blackout.

        Args:
            blackout_name: Name of the blackout window to deactivate
        """
        self.emergency_blackouts.discard(blackout_name)
        logger.info(f"Emergency blackout deactivated: {blackout_name}")

    def clear_all_emergency_blackouts(self) -> None:
        """Clear all active emergency blackouts."""
        count = len(self.emergency_blackouts)
        self.emergency_blackouts.clear()
        logger.info(f"Cleared {count} emergency blackouts")

    def list_active_emergency_blackouts(self) -> List[str]:
        """List all active emergency blackouts."""
        return list(self.emergency_blackouts)

    def test_blackout_window(
        self,
        window: BlackoutWindow,
        test_times: Optional[List[datetime]] = None,
        timezone_str: Optional[str] = None
    ) -> List[Tuple[datetime, bool]]:
        """Test a blackout window against multiple times.

        Args:
            window: Blackout window to test
            test_times: Times to test against (default: generate test times)
            timezone_str: Timezone for test (default: manager default)

        Returns:
            List of (datetime, is_active) tuples
        """
        tz_str = timezone_str or self.default_timezone
        try:
            tz = zoneinfo.ZoneInfo(tz_str)
        except zoneinfo.ZoneInfoNotFoundError:
            raise BlackoutEvaluationError(f"Invalid timezone: {tz_str}")

        if test_times is None:
            # Generate test times for the next week
            now = datetime.now(tz)
            test_times = [now + timedelta(hours=h) for h in range(0, 168, 2)]  # Every 2 hours for a week

        results = []
        for test_time in test_times:
            if test_time.tzinfo is None:
                test_time = test_time.replace(tzinfo=tz)

            is_active = self._is_window_active(window, test_time)
            results.append((test_time, is_active))

        return results

    def validate_blackout_windows(self, windows: List[BlackoutWindow]) -> List[str]:
        """Validate a list of blackout windows.

        Args:
            windows: List of blackout windows to validate

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []

        for i, window in enumerate(windows):
            try:
                # Test timezone
                zoneinfo.ZoneInfo(window.timezone)

                # Test time parsing
                time.fromisoformat(window.start_time)
                time.fromisoformat(window.end_time)

                # Test date range consistency
                if window.start_date and window.end_date:
                    if window.start_date > window.end_date:
                        errors.append(f"Window {i} '{window.name}': start_date after end_date")

                # Test with current time to catch any evaluation errors
                self._is_window_active(window, datetime.now(timezone.utc))

            except Exception as e:
                errors.append(f"Window {i} '{window.name}': {str(e)}")

        return errors


def create_maintenance_window(
    name: str,
    days: List[str],
    start_time: str,
    end_time: str,
    timezone_str: str = 'UTC',
    priority: int = 0
) -> BlackoutWindow:
    """Create a recurring maintenance blackout window.

    Args:
        name: Name of the maintenance window
        days: List of day names (Mon, Tue, etc.)
        start_time: Start time in HH:MM format
        end_time: End time in HH:MM format
        timezone_str: IANA timezone identifier
        priority: Window priority level

    Returns:
        Configured BlackoutWindow
    """
    return BlackoutWindow(
        name=name,
        days_of_week=days,
        start_time=start_time,
        end_time=end_time,
        timezone=timezone_str,
        priority=priority
    )


def create_absolute_blackout(
    name: str,
    start_date: date,
    end_date: date,
    start_time: str,
    end_time: str,
    timezone_str: str = 'UTC',
    priority: int = 5
) -> BlackoutWindow:
    """Create an absolute date range blackout window.

    Args:
        name: Name of the blackout period
        start_date: Start date
        end_date: End date
        start_time: Start time in HH:MM format
        end_time: End time in HH:MM format
        timezone_str: IANA timezone identifier
        priority: Window priority level

    Returns:
        Configured BlackoutWindow
    """
    return BlackoutWindow(
        name=name,
        start_time=start_time,
        end_time=end_time,
        start_date=start_date,
        end_date=end_date,
        timezone=timezone_str,
        priority=priority
    )


def create_emergency_blackout(name: str, priority: int = 10) -> BlackoutWindow:
    """Create an emergency blackout window.

    Args:
        name: Name of the emergency blackout
        priority: Window priority level (default: highest)

    Returns:
        Configured BlackoutWindow (must be activated via BlackoutManager)
    """
    return BlackoutWindow(
        name=name,
        start_time="00:00",
        end_time="23:59",
        emergency=True,
        priority=priority
    )