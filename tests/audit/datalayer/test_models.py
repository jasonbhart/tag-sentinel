"""Unit tests for DataLayer models."""

import pytest
from datetime import datetime
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.audit.datalayer.models import (
    DataLayerSnapshot,
    ValidationIssue,
    DLContext,
    DLResult,
    DLAggregate,
    ValidationSeverity,
    RedactionMethod
)


class TestDataLayerSnapshot:
    """Test cases for DataLayerSnapshot model."""
    
    def test_basic_snapshot_creation(self):
        """Test basic snapshot creation."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"page": "home", "user_id": "123"},
            events=[{"event": "page_view"}]
        )
        
        assert str(snapshot.page_url) == "https://example.com/"
        assert snapshot.exists is True
        assert snapshot.latest == {"page": "home", "user_id": "123"}
        assert len(snapshot.events) == 1
        assert isinstance(snapshot.capture_time, datetime)
    
    def test_empty_datalayer_snapshot(self):
        """Test snapshot with empty dataLayer."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=False,
            latest=None
        )

        assert snapshot.latest is None
        assert snapshot.variable_count == 0
        assert not snapshot.exists
    
    def test_snapshot_with_events(self):
        """Test snapshot with separated events."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"user_id": "123"},
            events=[
                {"event": "page_view", "page": "home"},
                {"event": "click", "element": "button"}
            ]
        )

        assert len(snapshot.events) == 2
        assert snapshot.events[0]["event"] == "page_view"
        assert snapshot.event_count == 2
    
    def test_snapshot_metrics(self):
        """Test snapshot size and metrics calculations."""
        large_data = {f"var_{i}": f"value_{i}" for i in range(100)}

        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest=large_data
        )

        assert snapshot.variable_count == 100
        assert snapshot.exists
    
    def test_snapshot_json_serialization(self):
        """Test JSON serialization."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"test": "value"}
        )

        json_data = snapshot.model_dump()
        assert str(json_data["page_url"]) == "https://example.com/"
        assert "capture_time" in json_data
        assert json_data["exists"] is True


class TestValidationIssue:
    """Test cases for ValidationIssue model."""
    
    def test_basic_issue_creation(self):
        """Test basic validation issue creation."""
        issue = ValidationIssue(
            page_url="https://example.com",
            path="/user/id",
            message="Invalid user ID format",
            severity=ValidationSeverity.CRITICAL,
            schema_rule="/properties/user/properties/id"
        )

        assert issue.path == "/user/id"
        assert issue.message == "Invalid user ID format"
        assert issue.severity == ValidationSeverity.CRITICAL
        assert issue.is_critical
    
    def test_issue_severities(self):
        """Test different issue severities."""
        error_issue = ValidationIssue(
            page_url="https://example.com",
            path="/test",
            message="Test error",
            severity=ValidationSeverity.CRITICAL
        )

        warning_issue = ValidationIssue(
            page_url="https://example.com",
            path="/test",
            message="Test warning",
            severity=ValidationSeverity.WARNING
        )

        info_issue = ValidationIssue(
            page_url="https://example.com",
            path="/test",
            message="Test info",
            severity=ValidationSeverity.INFO
        )
        
        assert error_issue.is_critical
        assert error_issue.severity == ValidationSeverity.CRITICAL

        assert not warning_issue.is_critical
        assert warning_issue.severity == ValidationSeverity.WARNING

        assert not info_issue.is_critical
        assert info_issue.severity == ValidationSeverity.INFO
    
    def test_issue_with_context(self):
        """Test issue with additional context."""
        issue = ValidationIssue(
            page_url="https://example.com",
            path="/ecommerce/purchase",
            message="Missing required field",
            severity=ValidationSeverity.CRITICAL,
            expected="object",
            observed="null"
        )

        assert issue.expected == "object"
        assert issue.observed == "null"


class TestDLContext:
    """Test cases for DLContext model."""
    
    def test_basic_context_creation(self):
        """Test basic context creation."""
        context = DLContext(
            env="test"
        )

        assert context.env == "test"
        assert context.data_layer_object == "dataLayer"  # default value
        assert context.max_depth == 6  # default value
    
    def test_context_with_metadata(self):
        """Test context with site-specific configuration."""
        site_config = {"campaign": "summer", "ab_test": "version_a"}

        context = DLContext(
            env="staging",
            site_config=site_config
        )

        assert context.site_config == site_config
        assert context.site_config["campaign"] == "summer"
    
    def test_context_with_custom_settings(self):
        """Test context with custom capture settings."""
        context = DLContext(
            env="production",
            data_layer_object="customDataLayer",
            max_depth=10,
            max_entries=1000
        )

        assert context.env == "production"
        assert context.data_layer_object == "customDataLayer"
        assert context.max_depth == 10
        assert context.max_entries == 1000


class TestDLResult:
    """Test cases for DLResult model."""
    
    def test_basic_result_creation(self):
        """Test basic result creation."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"test": "value"}
        )

        result = DLResult(snapshot=snapshot)

        assert result.snapshot == snapshot
        assert result.issues == []
        assert len(result.aggregate_delta) == 0
    
    def test_result_with_validation_issues(self):
        """Test result with validation issues."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"test": "value"}
        )

        issues = [
            ValidationIssue(
                page_url="https://example.com",
                path="/test",
                message="Invalid format",
                severity=ValidationSeverity.CRITICAL
            ),
            ValidationIssue(
                page_url="https://example.com",
                path="/other",
                message="Missing field",
                severity=ValidationSeverity.WARNING
            )
        ]
        
        result = DLResult(
            snapshot=snapshot,
            issues=issues
        )
        
        assert len(result.issues) == 2
    
    def test_result_with_processing_metadata(self):
        """Test result with processing metadata."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"user_id": "123"}
        )
        
        result = DLResult(
            snapshot=snapshot,
            processing_time_ms=150.5
        )
        
        assert result.processing_time_ms == 150.5
        assert str(result.snapshot.page_url) == "https://example.com/"
    
    def test_result_summary_statistics(self):
        """Test result summary statistics."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"test": "value"}
        )

        issues = [
            ValidationIssue(page_url="https://example.com", path="/test1", message="Error 1", severity=ValidationSeverity.CRITICAL),
            ValidationIssue(page_url="https://example.com", path="/test2", message="Error 2", severity=ValidationSeverity.CRITICAL),
            ValidationIssue(page_url="https://example.com", path="/test3", message="Warning 1", severity=ValidationSeverity.WARNING),
        ]
        
        result = DLResult(
            snapshot=snapshot,
            issues=issues
        )
        
        assert len(result.issues) == 3


class TestDLAggregate:
    """Test cases for DLAggregate model."""
    
    def test_basic_aggregate_creation(self):
        """Test basic aggregate creation."""
        aggregate = DLAggregate(
            run_id="test-run-123",
            total_pages=10,
            pages_successful=8,
            pages_failed=2
        )

        assert aggregate.run_id == "test-run-123"
        assert aggregate.total_pages == 10
        assert aggregate.success_rate == 80.0
        assert isinstance(aggregate.start_time, datetime)
    
    def test_aggregate_with_variable_stats(self):
        """Test aggregate with variable statistics."""
        from app.audit.datalayer.models import VariablePresence

        variables = {
            "page_title": VariablePresence(
                name="page_title",
                path="/page_title",
                pages_with_variable=9,
                total_pages=10,
                example_values=["Home", "About"]
            ),
            "user_id": VariablePresence(
                name="user_id",
                path="/user_id",
                pages_with_variable=7,
                total_pages=10,
                example_values=["123", "456"]
            )
        }

        aggregate = DLAggregate(
            run_id="test-run-123",
            total_pages=10,
            pages_successful=10,
            pages_failed=0,
            variables=variables
        )

        assert len(aggregate.variables) == 2
        assert aggregate.variables["page_title"].presence_percentage == 90.0
    
    def test_aggregate_with_validation_summary(self):
        """Test aggregate with validation statistics."""
        aggregate = DLAggregate(
            run_id="test-run-123",
            total_pages=10,
            pages_successful=10,
            pages_failed=0,
            total_validation_issues=15,
            critical_issues=8,
            warning_issues=7
        )
        
        assert aggregate.total_validation_issues == 15
        assert aggregate.critical_issues == 8
        assert aggregate.warning_issues == 7
    
    def test_aggregate_success_rate_calculation(self):
        """Test success rate calculation edge cases."""
        # Zero pages
        empty_aggregate = DLAggregate(
            run_id="empty-run",
            total_pages=0,
            pages_successful=0,
            pages_failed=0
        )
        assert empty_aggregate.success_rate == 0.0

        # Perfect success
        perfect_aggregate = DLAggregate(
            run_id="perfect-run",
            total_pages=5,
            pages_successful=5,
            pages_failed=0
        )
        assert perfect_aggregate.success_rate == 100.0
    
    def test_aggregate_json_serialization(self):
        """Test JSON serialization of aggregate."""
        aggregate = DLAggregate(
            run_id="test-run",
            total_pages=5,
            pages_successful=4,
            pages_failed=1
        )

        json_data = aggregate.model_dump()
        assert json_data["run_id"] == "test-run"
        assert json_data["total_pages"] == 5
        assert json_data["pages_successful"] == 4
        assert "start_time" in json_data
        # success_rate is a computed property, test it separately
        assert aggregate.success_rate == 80.0



if __name__ == "__main__":
    pytest.main([__file__, "-v"])