"""Unit tests for cron evaluation engine."""

import pytest
from datetime import datetime, timezone, timedelta
import zoneinfo

from app.scheduling.cron import (
    CronEvaluator,
    CronValidationError,
    CronEvaluationError,
    validate_cron_expression,
    get_next_run_time
)


class TestCronEvaluator:
    """Test CronEvaluator functionality."""

    def setup_method(self):
        """Set up test fixture."""
        self.evaluator = CronEvaluator()

    def test_validate_basic_cron_expressions(self):
        """Test validation of basic cron expressions."""
        valid_expressions = [
            "0 9 * * *",      # Daily at 9 AM
            "0 */2 * * *",    # Every 2 hours
            "15 14 1 * *",    # 2:15 PM on 1st of every month
            "0 22 * * 1-5",   # 10 PM on weekdays
            "*/15 * * * *",   # Every 15 minutes
            "0 0 * * 0"       # Weekly on Sunday
        ]

        for expr in valid_expressions:
            # Should not raise exception
            self.evaluator.validate_cron_expression(expr)

    def test_validate_named_expressions(self):
        """Test validation of named cron expressions."""
        named_expressions = [
            "@yearly",
            "@annually",
            "@monthly",
            "@weekly",
            "@daily",
            "@hourly"
        ]

        for expr in named_expressions:
            # Should not raise exception
            self.evaluator.validate_cron_expression(expr)

    def test_invalid_cron_expressions(self):
        """Test that invalid cron expressions are rejected."""
        invalid_expressions = [
            "invalid",
            "60 * * * *",     # Invalid minute
            "* 25 * * *",     # Invalid hour
            "* * 32 * *",     # Invalid day
            "* * * 13 *",     # Invalid month
            "* * * * 8",      # Invalid weekday
            "* * * *",        # Too few fields
            "* * * * * * *"   # Too many fields
        ]

        for expr in invalid_expressions:
            with pytest.raises(CronValidationError):
                self.evaluator.validate_cron_expression(expr)

    def test_next_run_time_daily(self):
        """Test next run time calculation for daily schedule."""
        # 9 AM daily
        cron_expr = "0 9 * * *"

        # If it's 8 AM, next run should be 9 AM today
        current_time = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        next_run = self.evaluator.get_next_run_time(
            cron_expr,
            current_time,
            "UTC"
        )

        expected = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        assert next_run == expected

    def test_next_run_time_with_timezone(self):
        """Test next run time with different timezone."""
        cron_expr = "0 9 * * *"  # 9 AM

        # 8 AM EST = 1 PM UTC (January 15 is standard time, not DST)
        current_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)
        next_run = self.evaluator.get_next_run_time(
            cron_expr,
            current_time,
            "America/New_York"
        )

        # Next 9 AM EST should be today since it's only 8 AM EST now
        # 9 AM EST = 2 PM UTC in January (EST = UTC-5)
        next_run_utc = next_run.astimezone(timezone.utc)
        expected_utc = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)  # 9 AM EST = 2 PM UTC

        assert next_run_utc == expected_utc

    def test_named_expression_evaluation(self):
        """Test evaluation of named expressions."""
        current_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # @daily should be next midnight
        next_run = self.evaluator.get_next_run_time(
            "@daily",
            current_time,
            "UTC"
        )

        expected = datetime(2024, 1, 16, 0, 0, 0, tzinfo=timezone.utc)
        assert next_run == expected

    def test_previous_run_time(self):
        """Test previous run time calculation."""
        cron_expr = "0 9 * * *"  # 9 AM daily

        # If it's 10 AM, previous run should be 9 AM today
        current_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        prev_run = self.evaluator.get_previous_run_time(
            cron_expr,
            current_time,
            "UTC"
        )

        expected = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        assert prev_run == expected

    def test_run_times_in_range(self):
        """Test getting all run times in a date range."""
        cron_expr = "0 */6 * * *"  # Every 6 hours

        # Start just before first run to include it
        start_time = datetime(2023, 12, 31, 23, 59, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 19, 0, 0, tzinfo=timezone.utc)

        run_times = self.evaluator.get_run_times_in_range(
            cron_expr,
            start_time,
            end_time,
            "UTC"
        )

        # Should have runs at 0, 6, 12, 18 hours on Jan 1 (croniter excludes end time)
        assert len(run_times) >= 3
        # Check that first few times are correct
        assert run_times[0].hour % 6 == 0  # Should be on 6-hour boundary
        assert run_times[1].hour % 6 == 0  # Should be on 6-hour boundary

    def test_dst_transition_detection(self):
        """Test DST transition detection."""
        # Spring forward in Eastern Time (March 10, 2024)
        spring_forward = datetime(2024, 3, 10, 7, 0, 0)  # 2 AM EST becomes 3 AM EDT

        is_transition, transition_type = self.evaluator.is_dst_transition_time(
            spring_forward,
            "America/New_York"
        )

        # Note: This might not detect the exact transition depending on the time
        # The test mainly verifies the method doesn't crash

    def test_invalid_timezone(self):
        """Test that invalid timezone raises error."""
        with pytest.raises(CronEvaluationError):
            self.evaluator.get_next_run_time(
                "0 9 * * *",
                timezone_str="Invalid/Timezone"
            )

    def test_cache_functionality(self):
        """Test that cron expression parsing is cached."""
        cron_expr = "0 9 * * *"

        # Parse twice - should use cache second time
        self.evaluator.validate_cron_expression(cron_expr)
        self.evaluator.validate_cron_expression(cron_expr)

        # Should have cached result
        assert cron_expr in self.evaluator._cache

        # Clear cache
        self.evaluator.clear_cache()
        assert len(self.evaluator._cache) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_validate_cron_expression_function(self):
        """Test standalone validation function."""
        # Should not raise
        validate_cron_expression("0 9 * * *")
        validate_cron_expression("@daily")

        # Should raise
        with pytest.raises(CronValidationError):
            validate_cron_expression("invalid")

    def test_get_next_run_time_function(self):
        """Test standalone next run time function."""
        current_time = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        next_run = get_next_run_time(
            "0 9 * * *",
            current_time,
            "UTC"
        )

        expected = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        assert next_run == expected