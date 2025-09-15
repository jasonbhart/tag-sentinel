"""Unit tests for blackout window management."""

import pytest
from datetime import datetime, date, time, timezone, timedelta

from app.scheduling.blackout import (
    BlackoutManager,
    BlackoutStatus,
    BlackoutEvaluationError,
    create_maintenance_window,
    create_absolute_blackout,
    create_emergency_blackout
)
from app.scheduling.models import BlackoutWindow


class TestBlackoutStatus:
    """Test BlackoutStatus functionality."""

    def test_primary_blackout_selection(self):
        """Test primary blackout selection by priority."""
        high_priority = BlackoutWindow(
            name="High Priority",
            start_time="01:00",
            end_time="02:00",
            priority=10,
            emergency=True
        )

        low_priority = BlackoutWindow(
            name="Low Priority",
            start_time="01:00",
            end_time="02:00",
            priority=1
        )

        status = BlackoutStatus(
            is_blackout=True,
            active_windows=[low_priority, high_priority]
        )

        primary = status.get_primary_blackout()
        assert primary == high_priority
        assert status.get_blackout_reason() == "High Priority"

    def test_no_active_blackouts(self):
        """Test status with no active blackouts."""
        status = BlackoutStatus(is_blackout=False)

        assert status.get_primary_blackout() is None
        assert status.get_blackout_reason() is None


class TestBlackoutManager:
    """Test BlackoutManager functionality."""

    def setup_method(self):
        """Set up test fixture."""
        self.manager = BlackoutManager("UTC")

    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.default_timezone == "UTC"
        assert len(self.manager.emergency_blackouts) == 0

    def test_recurring_blackout_active(self):
        """Test recurring blackout window detection."""
        # Weekend maintenance window
        window = BlackoutWindow(
            name="Weekend Maintenance",
            days_of_week=["Sat", "Sun"],
            start_time="01:00",
            end_time="05:00",
            timezone="UTC"
        )

        # Test during blackout (Saturday 3 AM)
        check_time = datetime(2024, 1, 6, 3, 0, 0, tzinfo=timezone.utc)  # Saturday
        status = self.manager.is_blackout_active([window], check_time)

        assert status.is_blackout is True
        assert len(status.active_windows) == 1
        assert status.active_windows[0] == window

    def test_recurring_blackout_inactive(self):
        """Test recurring blackout window when inactive."""
        # Weekend maintenance window
        window = BlackoutWindow(
            name="Weekend Maintenance",
            days_of_week=["Sat", "Sun"],
            start_time="01:00",
            end_time="05:00",
            timezone="UTC"
        )

        # Test outside blackout (Monday 3 AM)
        check_time = datetime(2024, 1, 8, 3, 0, 0, tzinfo=timezone.utc)  # Monday
        status = self.manager.is_blackout_active([window], check_time)

        assert status.is_blackout is False
        assert len(status.active_windows) == 0

    def test_absolute_blackout_active(self):
        """Test absolute blackout window detection."""
        # Holiday blackout
        window = BlackoutWindow(
            name="Holiday Blackout",
            start_time="00:00",
            end_time="23:59",
            start_date=date(2024, 12, 24),
            end_date=date(2024, 12, 26),
            timezone="UTC"
        )

        # Test during blackout (Christmas Day)
        check_time = datetime(2024, 12, 25, 12, 0, 0, tzinfo=timezone.utc)
        status = self.manager.is_blackout_active([window], check_time)

        assert status.is_blackout is True
        assert len(status.active_windows) == 1

    def test_absolute_blackout_inactive(self):
        """Test absolute blackout window when inactive."""
        # Holiday blackout
        window = BlackoutWindow(
            name="Holiday Blackout",
            start_time="00:00",
            end_time="23:59",
            start_date=date(2024, 12, 24),
            end_date=date(2024, 12, 26),
            timezone="UTC"
        )

        # Test outside blackout (December 27)
        check_time = datetime(2024, 12, 27, 12, 0, 0, tzinfo=timezone.utc)
        status = self.manager.is_blackout_active([window], check_time)

        assert status.is_blackout is False
        assert len(status.active_windows) == 0

    def test_multiple_windows_priority_ordering(self):
        """Test multiple blackout windows with different priorities."""
        low_priority = BlackoutWindow(
            name="Regular Maintenance",
            days_of_week=["Sun"],
            start_time="01:00",
            end_time="05:00",
            priority=1,
            timezone="UTC"
        )

        high_priority = BlackoutWindow(
            name="Emergency Maintenance",
            days_of_week=["Sun"],
            start_time="02:00",
            end_time="04:00",
            priority=10,
            emergency=True,
            timezone="UTC"
        )

        # Test during both blackouts (Sunday 3 AM)
        check_time = datetime(2024, 1, 7, 3, 0, 0, tzinfo=timezone.utc)  # Sunday
        status = self.manager.is_blackout_active([low_priority, high_priority], check_time)

        assert status.is_blackout is True
        assert len(status.active_windows) == 2
        assert status.get_primary_blackout() == high_priority

    def test_timezone_handling(self):
        """Test timezone-aware blackout evaluation."""
        # Window in Eastern time
        window = BlackoutWindow(
            name="Eastern Maintenance",
            days_of_week=["Mon"],
            start_time="09:00",
            end_time="17:00",
            timezone="America/New_York"
        )

        # Test at 2 PM UTC = 9 AM EST (during blackout)
        check_time = datetime(2024, 1, 8, 14, 0, 0, tzinfo=timezone.utc)  # Monday
        status = self.manager.is_blackout_active([window], check_time, timezone_str="America/New_York")

        assert status.is_blackout is True

    def test_emergency_blackout_management(self):
        """Test emergency blackout activation and deactivation."""
        # Initially no emergency blackouts
        assert len(self.manager.list_active_emergency_blackouts()) == 0

        # Activate emergency blackout
        self.manager.activate_emergency_blackout("system-failure")
        emergencies = self.manager.list_active_emergency_blackouts()
        assert "system-failure" in emergencies
        assert len(emergencies) == 1

        # Activate another emergency
        self.manager.activate_emergency_blackout("security-incident")
        emergencies = self.manager.list_active_emergency_blackouts()
        assert len(emergencies) == 2
        assert "system-failure" in emergencies
        assert "security-incident" in emergencies

        # Deactivate one emergency
        self.manager.deactivate_emergency_blackout("system-failure")
        emergencies = self.manager.list_active_emergency_blackouts()
        assert len(emergencies) == 1
        assert "security-incident" in emergencies

        # Clear all emergencies
        self.manager.clear_all_emergency_blackouts()
        assert len(self.manager.list_active_emergency_blackouts()) == 0

    def test_detailed_evaluation(self):
        """Test detailed blackout evaluation."""
        window = BlackoutWindow(
            name="Test Window",
            days_of_week=["Mon"],
            start_time="09:00",
            end_time="17:00",
            timezone="UTC"
        )

        # Test during active period
        check_time = datetime(2024, 1, 8, 12, 0, 0, tzinfo=timezone.utc)  # Monday noon
        details = self.manager.evaluate_detailed([window], check_time)

        assert len(details.evaluated_windows) == 1
        window_eval, is_active = details.evaluated_windows[0]
        assert window_eval == window
        assert is_active is True

    def test_blackout_window_validation(self):
        """Test blackout window validation."""
        valid_window = BlackoutWindow(
            name="Valid Window",
            days_of_week=["Mon"],
            start_time="09:00",
            end_time="17:00",
            timezone="UTC"
        )

        # Should have no validation errors
        errors = self.manager.validate_blackout_windows([valid_window])
        assert len(errors) == 0

    def test_time_boundaries(self):
        """Test time boundary conditions."""
        # Simple non-crossing window first
        window = BlackoutWindow(
            name="Simple Window",
            days_of_week=["Sun"],
            start_time="14:00",
            end_time="16:00",
            timezone="UTC"
        )

        # Test at Sunday 15:00 (should be active)
        check_time = datetime(2024, 1, 7, 15, 0, 0, tzinfo=timezone.utc)  # Sunday
        status = self.manager.is_blackout_active([window], check_time)
        assert status.is_blackout is True

        # Test at Sunday 13:00 (should be inactive)
        check_time = datetime(2024, 1, 7, 13, 0, 0, tzinfo=timezone.utc)  # Sunday
        status = self.manager.is_blackout_active([window], check_time)
        assert status.is_blackout is False


class TestConvenienceFunctions:
    """Test convenience functions for creating blackout windows."""

    def test_create_maintenance_window(self):
        """Test maintenance window creation."""
        window = create_maintenance_window(
            name="Weekly Maintenance",
            days=["Sat", "Sun"],
            start_time="01:00",
            end_time="05:00",
            timezone_str="UTC"
        )

        assert window.name == "Weekly Maintenance"
        assert window.days_of_week == ["Sat", "Sun"]
        assert window.start_time == "01:00"
        assert window.end_time == "05:00"
        assert window.timezone == "UTC"
        assert window.emergency is False

    def test_create_absolute_blackout(self):
        """Test absolute blackout creation."""
        window = create_absolute_blackout(
            name="Holiday Blackout",
            start_date=date(2024, 12, 24),
            end_date=date(2024, 12, 26),
            start_time="00:00",
            end_time="23:59",
            timezone_str="UTC"
        )

        assert window.name == "Holiday Blackout"
        assert window.start_date == date(2024, 12, 24)
        assert window.end_date == date(2024, 12, 26)
        assert window.timezone == "UTC"
        assert window.is_absolute_window() is True

    def test_create_emergency_blackout(self):
        """Test emergency blackout creation."""
        window = create_emergency_blackout(
            name="System Emergency",
            priority=10
        )

        assert window.name == "System Emergency"
        assert window.emergency is True
        assert window.priority == 10
        assert window.start_time == "00:00"
        assert window.end_time == "23:59"


class TestErrorConditions:
    """Test error conditions and edge cases."""

    def test_malformed_time_handling(self):
        """Test handling of edge cases in time parsing."""
        manager = BlackoutManager("UTC")

        # Test same start and end time (zero duration)
        window = BlackoutWindow(
            name="Zero Duration",
            days_of_week=["Mon"],
            start_time="12:00",
            end_time="12:00",
            timezone="UTC"
        )

        # Should not crash, but likely inactive
        check_time = datetime(2024, 1, 8, 12, 0, 0, tzinfo=timezone.utc)
        status = manager.is_blackout_active([window], check_time)
        # The exact behavior for zero duration may depend on implementation
        assert isinstance(status, BlackoutStatus)

    def test_empty_window_list(self):
        """Test handling empty blackout window list."""
        manager = BlackoutManager("UTC")

        status = manager.is_blackout_active([])
        assert status.is_blackout is False
        assert len(status.active_windows) == 0