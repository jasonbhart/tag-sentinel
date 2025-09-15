"""Unit tests for scheduling models."""

import pytest
from datetime import datetime, date, timezone, timedelta
from pydantic import ValidationError

from app.scheduling.models import (
    Schedule,
    ScheduleState,
    BlackoutWindow,
    RunRequest,
    CatchUpPolicy,
    ScheduleCollection
)


class TestSchedule:
    """Test Schedule model validation and functionality."""

    def test_valid_schedule_creation(self):
        """Test creating a valid schedule."""
        schedule = Schedule(
            name="Daily Production Audit",
            site_id="ecommerce",
            environment="production",
            cron="0 9 * * *",
            timezone="America/New_York"
        )

        assert schedule.name == "Daily Production Audit"
        assert schedule.site_id == "ecommerce"
        assert schedule.environment == "production"
        assert schedule.cron == "0 9 * * *"
        assert schedule.timezone == "America/New_York"
        assert schedule.enabled is True
        assert schedule.max_concurrent_runs == 1

    def test_named_cron_expressions(self):
        """Test that named cron expressions are accepted."""
        named_exprs = ["@yearly", "@annually", "@monthly", "@weekly", "@daily", "@hourly"]

        for expr in named_exprs:
            schedule = Schedule(
                name=f"Test {expr}",
                site_id="test",
                environment="prod",
                cron=expr,
                timezone="UTC"
            )
            assert schedule.cron == expr

    def test_invalid_cron_expression(self):
        """Test that invalid cron expressions are rejected."""
        with pytest.raises(ValidationError):
            Schedule(
                name="Invalid Cron",
                site_id="test",
                environment="prod",
                cron="invalid",
                timezone="UTC"
            )

    def test_invalid_timezone(self):
        """Test that invalid timezones are rejected."""
        with pytest.raises(ValidationError):
            Schedule(
                name="Invalid TZ",
                site_id="test",
                environment="prod",
                cron="0 9 * * *",
                timezone="Invalid/Timezone"
            )

    def test_site_env_key(self):
        """Test site-environment key generation."""
        schedule = Schedule(
            name="Test",
            site_id="blog",
            environment="staging",
            cron="@daily",
            timezone="UTC"
        )
        assert schedule.get_site_env_key() == "blog:staging"

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        schedule = Schedule(
            name="Serialize Test",
            site_id="test",
            environment="dev",
            cron="0 */2 * * *",
            timezone="Europe/London"
        )

        # Test serialization
        json_data = schedule.model_dump_json()
        assert isinstance(json_data, str)
        assert "Serialize Test" in json_data

        # Test deserialization
        restored = Schedule.model_validate_json(json_data)
        assert restored.name == schedule.name
        assert restored.site_id == schedule.site_id
        assert restored.cron == schedule.cron


class TestBlackoutWindow:
    """Test BlackoutWindow model validation and functionality."""

    def test_recurring_blackout_window(self):
        """Test creating a recurring blackout window."""
        window = BlackoutWindow(
            name="Weekly Maintenance",
            days_of_week=["Sat", "Sun"],
            start_time="01:00",
            end_time="05:00",
            timezone="UTC"
        )

        assert window.name == "Weekly Maintenance"
        assert window.days_of_week == ["Sat", "Sun"]
        assert window.start_time == "01:00"
        assert window.end_time == "05:00"
        assert window.is_absolute_window() is False
        assert window.emergency is False

    def test_absolute_blackout_window(self):
        """Test creating an absolute blackout window."""
        window = BlackoutWindow(
            name="Holiday Blackout",
            start_time="00:00",
            end_time="23:59",
            start_date=date(2024, 12, 24),
            end_date=date(2024, 12, 26),
            timezone="UTC"
        )

        assert window.name == "Holiday Blackout"
        assert window.start_date == date(2024, 12, 24)
        assert window.end_date == date(2024, 12, 26)
        assert window.is_absolute_window() is True

    def test_emergency_blackout_window(self):
        """Test creating an emergency blackout window."""
        window = BlackoutWindow(
            name="System Emergency",
            start_time="00:00",
            end_time="23:59",
            emergency=True,
            priority=10
        )

        assert window.emergency is True
        assert window.priority == 10

    def test_invalid_day_names(self):
        """Test that invalid day names are rejected."""
        with pytest.raises(ValidationError):
            BlackoutWindow(
                name="Invalid Days",
                days_of_week=["Monday", "InvalidDay"],  # Should be Mon, not Monday
                start_time="01:00",
                end_time="05:00",
                timezone="UTC"
            )

    def test_invalid_time_format(self):
        """Test that invalid time formats are rejected."""
        with pytest.raises(ValidationError):
            BlackoutWindow(
                name="Invalid Time",
                days_of_week=["Mon"],
                start_time="25:00",  # Invalid hour
                end_time="05:00",
                timezone="UTC"
            )

    def test_absolute_window_requires_both_dates(self):
        """Test that absolute windows require both start and end dates."""
        # Only start_date should fail
        with pytest.raises(ValidationError):
            BlackoutWindow(
                name="Invalid Absolute",
                start_time="00:00",
                end_time="23:59",
                start_date=date(2024, 12, 24),
                # Missing end_date
                timezone="UTC"
            )

        # Only end_date should fail
        with pytest.raises(ValidationError):
            BlackoutWindow(
                name="Invalid Absolute",
                start_time="00:00",
                end_time="23:59",
                # Missing start_date
                end_date=date(2024, 12, 26),
                timezone="UTC"
            )

    def test_date_order_validation(self):
        """Test that start_date must be before end_date."""
        with pytest.raises(ValidationError):
            BlackoutWindow(
                name="Invalid Date Order",
                start_time="00:00",
                end_time="23:59",
                start_date=date(2024, 12, 26),
                end_date=date(2024, 12, 24),  # Before start_date
                timezone="UTC"
            )


class TestRunRequest:
    """Test RunRequest model functionality."""

    def test_run_request_creation(self):
        """Test creating a run request."""
        now = datetime.now(timezone.utc)
        request = RunRequest(
            site_id="ecommerce",
            environment="production",
            scheduled_at=now
        )

        assert request.site_id == "ecommerce"
        assert request.environment == "production"
        assert request.scheduled_at == now
        assert request.priority == 0
        assert request.retry_count == 0
        assert isinstance(request.id, str)

    def test_run_request_priority_ordering(self):
        """Test that run requests order by priority correctly."""
        low_priority = RunRequest(
            site_id="test",
            environment="prod",
            scheduled_at=datetime.now(timezone.utc),
            priority=1
        )

        high_priority = RunRequest(
            site_id="test",
            environment="prod",
            scheduled_at=datetime.now(timezone.utc),
            priority=5
        )

        # Higher priority should be "less than" for heapq
        assert high_priority < low_priority


class TestCatchUpPolicy:
    """Test CatchUpPolicy model validation."""

    def test_default_catch_up_policy(self):
        """Test default catch-up policy."""
        policy = CatchUpPolicy()

        assert policy.strategy == "skip"
        assert policy.max_catchup_window_hours == 24
        assert policy.gradual_spacing_minutes == 30
        assert policy.max_catchup_runs is None

    def test_gradual_catchup_validation(self):
        """Test that gradual catch-up requires max_catchup_runs."""
        # Should fail without max_catchup_runs
        with pytest.raises(ValidationError):
            CatchUpPolicy(
                strategy="gradual_catchup",
                # Missing max_catchup_runs
            )

        # Should succeed with max_catchup_runs
        policy = CatchUpPolicy(
            strategy="gradual_catchup",
            max_catchup_runs=5
        )
        assert policy.strategy == "gradual_catchup"
        assert policy.max_catchup_runs == 5


class TestScheduleCollection:
    """Test ScheduleCollection functionality."""

    def test_schedule_collection_operations(self):
        """Test schedule collection filtering and queries."""
        schedules = [
            Schedule(
                name="Ecommerce Prod",
                site_id="ecommerce",
                environment="production",
                cron="@daily",
                timezone="UTC",
                tags=["critical"]
            ),
            Schedule(
                name="Ecommerce Staging",
                site_id="ecommerce",
                environment="staging",
                cron="@daily",
                timezone="UTC",
                tags=["testing"]
            ),
            Schedule(
                name="Blog Prod",
                site_id="blog",
                environment="production",
                cron="@weekly",
                timezone="UTC",
                tags=["content"]
            )
        ]

        collection = ScheduleCollection(schedules=schedules)

        # Test site-environment filtering
        ecommerce_schedules = collection.get_by_site_env("ecommerce", "production")
        assert len(ecommerce_schedules) == 1
        assert ecommerce_schedules[0].name == "Ecommerce Prod"

        # Test enabled schedules
        enabled = collection.get_enabled_schedules()
        assert len(enabled) == 3  # All enabled by default

        # Test tag filtering
        critical_schedules = collection.get_by_tag("critical")
        assert len(critical_schedules) == 1
        assert critical_schedules[0].name == "Ecommerce Prod"