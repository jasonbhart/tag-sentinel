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
            url="https://example.com",
            raw_data=None,
            processed_data=None
        )
        
        assert snapshot.raw_data is None
        assert snapshot.processed_data is None
        assert snapshot.size == 0
        assert snapshot.variable_count == 0
        assert not snapshot.has_data
    
    def test_snapshot_with_events(self):
        """Test snapshot with separated events."""
        raw_data = [
            {"event": "page_view", "page": "home"},
            {"user_id": "123"},
            {"event": "click", "element": "button"}
        ]
        
        snapshot = DataLayerSnapshot(
            url="https://example.com",
            raw_data=raw_data,
            processed_data={"user_id": "123"},
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
            url="https://example.com",
            raw_data=large_data,
            processed_data=large_data
        )
        
        assert snapshot.variable_count == 100
        assert snapshot.size > 1000  # Should be substantial
        assert snapshot.has_data
    
    def test_snapshot_json_serialization(self):
        """Test JSON serialization."""
        snapshot = DataLayerSnapshot(
            url="https://example.com",
            raw_data={"test": "value"},
            processed_data={"test": "value"}
        )
        
        json_data = snapshot.dict()
        assert json_data["url"] == "https://example.com"
        assert "captured_at" in json_data
        assert json_data["size"] > 0


class TestValidationIssue:
    """Test cases for ValidationIssue model."""
    
    def test_basic_issue_creation(self):
        """Test basic validation issue creation."""
        issue = ValidationIssue(
            path="/user/id",
            message="Invalid user ID format",
            severity=ValidationSeverity.ERROR,
            schema_path="/properties/user/properties/id"
        )
        
        assert issue.path == "/user/id"
        assert issue.message == "Invalid user ID format"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.is_error
        assert not issue.is_warning
    
    def test_issue_severities(self):
        """Test different issue severities."""
        error_issue = ValidationIssue(
            path="/test",
            message="Test error",
            severity=ValidationSeverity.ERROR
        )
        
        warning_issue = ValidationIssue(
            path="/test",
            message="Test warning",
            severity=ValidationSeverity.WARNING
        )
        
        info_issue = ValidationIssue(
            path="/test",
            message="Test info",
            severity=ValidationSeverity.INFO
        )
        
        assert error_issue.is_error
        assert not error_issue.is_warning
        assert not error_issue.is_info
        
        assert not warning_issue.is_error
        assert warning_issue.is_warning
        assert not warning_issue.is_info
        
        assert not info_issue.is_error
        assert not info_issue.is_warning
        assert info_issue.is_info
    
    def test_issue_with_context(self):
        """Test issue with additional context."""
        issue = ValidationIssue(
            path="/ecommerce/purchase",
            message="Missing required field",
            severity=ValidationSeverity.ERROR,
            context={"expected_type": "object", "actual_type": "null"}
        )
        
        assert issue.context["expected_type"] == "object"
        assert issue.context["actual_type"] == "null"


class TestDLContext:
    """Test cases for DLContext model."""
    
    def test_basic_context_creation(self):
        """Test basic context creation."""
        context = DLContext(
            url="https://example.com",
            page_title="Example Page"
        )
        
        assert context.url == "https://example.com"
        assert context.page_title == "Example Page"
        assert isinstance(context.timestamp, datetime)
    
    def test_context_with_metadata(self):
        """Test context with metadata."""
        metadata = {"campaign": "summer", "ab_test": "version_a"}
        
        context = DLContext(
            url="https://example.com/landing",
            page_title="Landing Page",
            metadata=metadata
        )
        
        assert context.metadata == metadata
        assert context.metadata["campaign"] == "summer"
    
    def test_context_with_user_agent(self):
        """Test context with user agent."""
        user_agent = "Mozilla/5.0 (compatible; TestBot/1.0)"
        
        context = DLContext(
            url="https://example.com",
            user_agent=user_agent
        )
        
        assert context.user_agent == user_agent


class TestDLResult:
    """Test cases for DLResult model."""
    
    def test_basic_result_creation(self):
        """Test basic result creation."""
        snapshot = DataLayerSnapshot(
            url="https://example.com",
            raw_data={"test": "value"},
            processed_data={"test": "value"}
        )
        
        result = DLResult(snapshot=snapshot)
        
        assert result.snapshot == snapshot
        assert result.issues == []
        assert len(result.aggregate_delta) == 0
    
    def test_result_with_validation_issues(self):
        """Test result with validation issues."""
        context = DLContext(url="https://example.com")
        snapshot = DataLayerSnapshot(
            url="https://example.com",
            raw_data={"test": "value"},
            processed_data={"test": "value"}
        )
        
        issues = [
            ValidationIssue(
                path="/test",
                message="Invalid format",
                severity=ValidationSeverity.ERROR
            ),
            ValidationIssue(
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
        context = DLContext(url="https://example.com")
        snapshot = DataLayerSnapshot(
            url="https://example.com",
            raw_data={"user_id": "123"},
            processed_data={"user_id": "123"}
        )
        
        result = DLResult(
            snapshot=snapshot,
            processing_time_ms=150.5
        )
        
        assert result.processing_time_ms == 150.5
        assert result.snapshot.url == "https://example.com"
    
    def test_result_summary_statistics(self):
        """Test result summary statistics."""
        context = DLContext(url="https://example.com")
        snapshot = DataLayerSnapshot(
            url="https://example.com",
            raw_data={"test": "value"},
            processed_data={"test": "value"}
        )
        
        issues = [
            ValidationIssue(path="/test1", message="Error 1", severity=ValidationSeverity.ERROR),
            ValidationIssue(path="/test2", message="Error 2", severity=ValidationSeverity.ERROR),
            ValidationIssue(path="/test3", message="Warning 1", severity=ValidationSeverity.WARNING),
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
            successful_captures=8,
            failed_captures=2
        )
        
        assert aggregate.run_id == "test-run-123"
        assert aggregate.total_pages == 10
        assert aggregate.success_rate == 80.0
        assert isinstance(aggregate.created_at, datetime)
    
    def test_aggregate_with_variable_stats(self):
        """Test aggregate with variable statistics."""
        variable_stats = {
            "page_title": {"presence": 0.9, "example_values": ["Home", "About"]},
            "user_id": {"presence": 0.7, "example_values": ["123", "456"]}
        }
        
        aggregate = DLAggregate(
            run_id="test-run-123",
            total_pages=10,
            successful_captures=10,
            failed_captures=0,
            variable_stats=variable_stats
        )
        
        assert len(aggregate.variable_stats) == 2
        assert aggregate.variable_stats["page_title"]["presence"] == 0.9
    
    def test_aggregate_with_validation_summary(self):
        """Test aggregate with validation summary."""
        validation_summary = {
            "total_issues": 15,
            "error_count": 8,
            "warning_count": 7,
            "most_common_issues": [
                {"message": "Missing required field", "count": 5},
                {"message": "Invalid format", "count": 3}
            ]
        }
        
        aggregate = DLAggregate(
            run_id="test-run-123",
            total_pages=10,
            successful_captures=10,
            failed_captures=0,
            validation_summary=validation_summary
        )
        
        assert aggregate.validation_summary["total_issues"] == 15
        assert len(aggregate.validation_summary["most_common_issues"]) == 2
    
    def test_aggregate_success_rate_calculation(self):
        """Test success rate calculation edge cases."""
        # Zero pages
        empty_aggregate = DLAggregate(
            run_id="empty-run",
            total_pages=0,
            successful_captures=0,
            failed_captures=0
        )
        assert empty_aggregate.success_rate == 0.0
        
        # Perfect success
        perfect_aggregate = DLAggregate(
            run_id="perfect-run",
            total_pages=5,
            successful_captures=5,
            failed_captures=0
        )
        assert perfect_aggregate.success_rate == 100.0
    
    def test_aggregate_json_serialization(self):
        """Test JSON serialization of aggregate."""
        aggregate = DLAggregate(
            run_id="test-run",
            total_pages=5,
            successful_captures=4,
            failed_captures=1,
            variable_stats={"test": {"presence": 1.0}}
        )
        
        json_data = aggregate.dict()
        assert json_data["run_id"] == "test-run"
        assert json_data["success_rate"] == 80.0
        assert "created_at" in json_data



if __name__ == "__main__":
    pytest.main([__file__, "-v"])