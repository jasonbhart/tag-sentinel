"""Timezone-aware cron expression evaluation with DST handling.

This module provides robust cron expression evaluation with comprehensive timezone
support, DST transitions, and various cron formats.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import zoneinfo

from croniter import croniter
from croniter.croniter import CroniterError


logger = logging.getLogger(__name__)


class CronValidationError(Exception):
    """Exception raised when cron expression validation fails."""
    pass


class CronEvaluationError(Exception):
    """Exception raised when cron evaluation fails."""
    pass


@dataclass
class CronField:
    """Represents a single field in a cron expression."""

    value: str
    min_val: int
    max_val: int
    name: str

    def __post_init__(self):
        """Validate field after initialization."""
        if self.value == '*':
            self.parsed_values = list(range(self.min_val, self.max_val + 1))
        elif self.value == '?':
            # ? is equivalent to * for day fields
            self.parsed_values = list(range(self.min_val, self.max_val + 1))
        else:
            self.parsed_values = self._parse_field_value()

    def _parse_field_value(self) -> List[int]:
        """Parse cron field value into list of integers."""
        values = []

        for part in self.value.split(','):
            if '/' in part:
                # Step values (e.g., */5, 1-10/2)
                range_part, step = part.split('/', 1)
                try:
                    step_val = int(step)
                except ValueError:
                    raise CronValidationError(f"Invalid step value '{step}' in {self.name} field")

                if range_part == '*':
                    range_values = list(range(self.min_val, self.max_val + 1, step_val))
                elif '-' in range_part:
                    start, end = range_part.split('-', 1)
                    try:
                        start_val, end_val = int(start), int(end)
                    except ValueError:
                        raise CronValidationError(f"Invalid range '{range_part}' in {self.name} field")
                    range_values = list(range(start_val, end_val + 1, step_val))
                else:
                    try:
                        start_val = int(range_part)
                    except ValueError:
                        raise CronValidationError(f"Invalid start value '{range_part}' in {self.name} field")
                    range_values = list(range(start_val, self.max_val + 1, step_val))

                values.extend(range_values)

            elif '-' in part:
                # Range values (e.g., 1-5)
                try:
                    start, end = part.split('-', 1)
                    start_val, end_val = int(start), int(end)
                except ValueError:
                    raise CronValidationError(f"Invalid range '{part}' in {self.name} field")

                if start_val > end_val:
                    raise CronValidationError(f"Invalid range '{part}' in {self.name} field: start > end")

                values.extend(range(start_val, end_val + 1))

            else:
                # Single value
                try:
                    val = int(part)
                except ValueError:
                    raise CronValidationError(f"Invalid value '{part}' in {self.name} field")
                values.append(val)

        # Validate all values are within bounds
        for val in values:
            if val < self.min_val or val > self.max_val:
                raise CronValidationError(
                    f"Value {val} out of range [{self.min_val}-{self.max_val}] in {self.name} field"
                )

        return sorted(list(set(values)))  # Remove duplicates and sort

    def matches(self, value: int) -> bool:
        """Check if the field matches the given value."""
        return value in self.parsed_values


@dataclass
class ParsedCron:
    """Parsed and validated cron expression."""

    original: str
    second: Optional[CronField] = None
    minute: CronField = None
    hour: CronField = None
    day: CronField = None
    month: CronField = None
    weekday: CronField = None

    def is_six_field(self) -> bool:
        """Check if this is a 6-field cron with seconds."""
        return self.second is not None


class CronEvaluator:
    """Timezone-aware cron expression evaluator with DST handling."""

    # Named cron expressions
    NAMED_EXPRESSIONS = {
        '@yearly': '0 0 1 1 *',
        '@annually': '0 0 1 1 *',
        '@monthly': '0 0 1 * *',
        '@weekly': '0 0 * * 0',
        '@daily': '0 0 * * *',
        '@hourly': '0 * * * *',
    }

    def __init__(self):
        """Initialize the cron evaluator."""
        self._cache = {}  # Cache for parsed cron expressions

    def validate_cron_expression(self, cron_expr: str) -> None:
        """Validate a cron expression format.

        Args:
            cron_expr: Cron expression to validate

        Raises:
            CronValidationError: If expression is invalid
        """
        try:
            self._parse_cron_expression(cron_expr)
        except CronValidationError:
            raise
        except Exception as e:
            raise CronValidationError(f"Invalid cron expression '{cron_expr}': {e}")

    def _parse_cron_expression(self, cron_expr: str) -> ParsedCron:
        """Parse a cron expression into validated fields.

        Args:
            cron_expr: Cron expression to parse

        Returns:
            ParsedCron object with validated fields

        Raises:
            CronValidationError: If expression is invalid
        """
        # Check cache first
        if cron_expr in self._cache:
            return self._cache[cron_expr]

        # Handle named expressions
        expr = self.NAMED_EXPRESSIONS.get(cron_expr, cron_expr)

        # Split into fields
        fields = expr.strip().split()

        if len(fields) == 5:
            # Standard 5-field cron
            minute_str, hour_str, day_str, month_str, weekday_str = fields
            parsed = ParsedCron(
                original=cron_expr,
                minute=CronField(minute_str, 0, 59, 'minute'),
                hour=CronField(hour_str, 0, 23, 'hour'),
                day=CronField(day_str, 1, 31, 'day'),
                month=CronField(month_str, 1, 12, 'month'),
                weekday=CronField(weekday_str, 0, 6, 'weekday')  # Sunday=0
            )
        elif len(fields) == 6:
            # 6-field cron with seconds
            second_str, minute_str, hour_str, day_str, month_str, weekday_str = fields
            parsed = ParsedCron(
                original=cron_expr,
                second=CronField(second_str, 0, 59, 'second'),
                minute=CronField(minute_str, 0, 59, 'minute'),
                hour=CronField(hour_str, 0, 23, 'hour'),
                day=CronField(day_str, 1, 31, 'day'),
                month=CronField(month_str, 1, 12, 'month'),
                weekday=CronField(weekday_str, 0, 6, 'weekday')
            )
        else:
            raise CronValidationError(f"Cron expression must have 5 or 6 fields, got {len(fields)}")

        # Cache the parsed result
        self._cache[cron_expr] = parsed
        return parsed

    def get_next_run_time(
        self,
        cron_expr: str,
        from_time: Optional[datetime] = None,
        timezone_str: str = 'UTC',
        max_iterations: int = 1000
    ) -> datetime:
        """Calculate the next run time for a cron expression.

        Args:
            cron_expr: Cron expression
            from_time: Calculate from this time (default: now in timezone)
            timezone_str: IANA timezone identifier
            max_iterations: Maximum iterations to prevent infinite loops

        Returns:
            Next run time in the specified timezone

        Raises:
            CronEvaluationError: If evaluation fails
        """
        try:
            tz = zoneinfo.ZoneInfo(timezone_str)
        except zoneinfo.ZoneInfoNotFoundError:
            raise CronEvaluationError(f"Invalid timezone: {timezone_str}")

        if from_time is None:
            from_time = datetime.now(tz)
        elif from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=tz)
        else:
            from_time = from_time.astimezone(tz)

        # Use croniter for robust cron evaluation
        try:
            # Convert named expressions
            expr = self.NAMED_EXPRESSIONS.get(cron_expr, cron_expr)

            # croniter expects local time, so we work in the target timezone
            cron = croniter(expr, from_time)
            next_run = cron.get_next(datetime)

            # Ensure the result has the correct timezone
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=tz)
            else:
                next_run = next_run.astimezone(tz)

            return next_run

        except CroniterError as e:
            raise CronEvaluationError(f"Failed to evaluate cron expression '{cron_expr}': {e}")
        except Exception as e:
            raise CronEvaluationError(f"Unexpected error evaluating cron expression: {e}")

    def get_previous_run_time(
        self,
        cron_expr: str,
        from_time: Optional[datetime] = None,
        timezone_str: str = 'UTC'
    ) -> datetime:
        """Calculate the previous run time for a cron expression.

        Args:
            cron_expr: Cron expression
            from_time: Calculate from this time (default: now in timezone)
            timezone_str: IANA timezone identifier

        Returns:
            Previous run time in the specified timezone

        Raises:
            CronEvaluationError: If evaluation fails
        """
        try:
            tz = zoneinfo.ZoneInfo(timezone_str)
        except zoneinfo.ZoneInfoNotFoundError:
            raise CronEvaluationError(f"Invalid timezone: {timezone_str}")

        if from_time is None:
            from_time = datetime.now(tz)
        elif from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=tz)
        else:
            from_time = from_time.astimezone(tz)

        try:
            expr = self.NAMED_EXPRESSIONS.get(cron_expr, cron_expr)
            cron = croniter(expr, from_time)
            prev_run = cron.get_prev(datetime)

            if prev_run.tzinfo is None:
                prev_run = prev_run.replace(tzinfo=tz)
            else:
                prev_run = prev_run.astimezone(tz)

            return prev_run

        except CroniterError as e:
            raise CronEvaluationError(f"Failed to evaluate cron expression '{cron_expr}': {e}")

    def get_run_times_in_range(
        self,
        cron_expr: str,
        start_time: datetime,
        end_time: datetime,
        timezone_str: str = 'UTC',
        max_results: int = 1000
    ) -> List[datetime]:
        """Get all run times in a date range.

        Args:
            cron_expr: Cron expression
            start_time: Start of range
            end_time: End of range
            timezone_str: IANA timezone identifier
            max_results: Maximum number of results to prevent runaway

        Returns:
            List of run times in the specified timezone

        Raises:
            CronEvaluationError: If evaluation fails
        """
        try:
            tz = zoneinfo.ZoneInfo(timezone_str)
        except zoneinfo.ZoneInfoNotFoundError:
            raise CronEvaluationError(f"Invalid timezone: {timezone_str}")

        # Ensure times are in the target timezone
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=tz)
        else:
            start_time = start_time.astimezone(tz)

        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=tz)
        else:
            end_time = end_time.astimezone(tz)

        if start_time >= end_time:
            return []

        try:
            expr = self.NAMED_EXPRESSIONS.get(cron_expr, cron_expr)
            cron = croniter(expr, start_time)

            run_times = []
            count = 0

            while count < max_results:
                next_run = cron.get_next(datetime)

                if next_run.tzinfo is None:
                    next_run = next_run.replace(tzinfo=tz)
                else:
                    next_run = next_run.astimezone(tz)

                if next_run >= end_time:
                    break

                run_times.append(next_run)
                count += 1

            return run_times

        except CroniterError as e:
            raise CronEvaluationError(f"Failed to evaluate cron expression '{cron_expr}': {e}")

    def is_dst_transition_time(
        self,
        dt: datetime,
        timezone_str: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if a datetime falls during a DST transition.

        Args:
            dt: Datetime to check
            timezone_str: IANA timezone identifier

        Returns:
            Tuple of (is_transition, transition_type)
            transition_type: 'spring_forward', 'fall_back', or None

        Raises:
            CronEvaluationError: If timezone is invalid
        """
        try:
            tz = zoneinfo.ZoneInfo(timezone_str)
        except zoneinfo.ZoneInfoNotFoundError:
            raise CronEvaluationError(f"Invalid timezone: {timezone_str}")

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        else:
            dt = dt.astimezone(tz)

        # Check if DST is active before and after this time
        before = dt - timedelta(hours=1)
        after = dt + timedelta(hours=1)

        before_dst = before.dst() != timedelta(0)
        after_dst = after.dst() != timedelta(0)
        current_dst = dt.dst() != timedelta(0)

        if not before_dst and current_dst:
            # Spring forward - entered DST
            return True, 'spring_forward'
        elif before_dst and not current_dst:
            # Fall back - exited DST
            return True, 'fall_back'
        else:
            return False, None

    def handle_dst_transition(
        self,
        cron_expr: str,
        scheduled_time: datetime,
        timezone_str: str
    ) -> List[datetime]:
        """Handle DST transitions for scheduled times.

        Args:
            cron_expr: Cron expression
            scheduled_time: Originally scheduled time
            timezone_str: IANA timezone identifier

        Returns:
            List of actual run times (may be empty, one, or two times)
        """
        is_transition, transition_type = self.is_dst_transition_time(scheduled_time, timezone_str)

        if not is_transition:
            return [scheduled_time]

        if transition_type == 'spring_forward':
            # Time may not exist - check if we can create it
            try:
                tz = zoneinfo.ZoneInfo(timezone_str)
                # Try to create the time - if it raises an exception, it doesn't exist
                localized = scheduled_time.astimezone(tz)
                return [localized]
            except (ValueError, OSError):
                # Time doesn't exist - skip this occurrence
                logger.info(f"Skipping non-existent time during spring forward: {scheduled_time}")
                return []

        elif transition_type == 'fall_back':
            # Time exists twice - we return the first occurrence
            # croniter handles this by default
            return [scheduled_time]

        return [scheduled_time]

    def clear_cache(self) -> None:
        """Clear the cron expression parsing cache."""
        self._cache.clear()
        logger.debug("Cleared cron expression cache")


def validate_cron_expression(cron_expr: str) -> None:
    """Convenience function to validate a cron expression.

    Args:
        cron_expr: Cron expression to validate

    Raises:
        CronValidationError: If expression is invalid
    """
    evaluator = CronEvaluator()
    evaluator.validate_cron_expression(cron_expr)


def get_next_run_time(
    cron_expr: str,
    from_time: Optional[datetime] = None,
    timezone_str: str = 'UTC'
) -> datetime:
    """Convenience function to get next run time.

    Args:
        cron_expr: Cron expression
        from_time: Calculate from this time (default: now)
        timezone_str: IANA timezone identifier

    Returns:
        Next run time in the specified timezone
    """
    evaluator = CronEvaluator()
    return evaluator.get_next_run_time(cron_expr, from_time, timezone_str)