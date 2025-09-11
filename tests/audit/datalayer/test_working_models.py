"""Working unit tests for DataLayer models that match actual implementation."""

import pytest
from datetime import datetime
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.audit.datalayer.models import (
    DataLayerSnapshot,
    ValidationIssue,
    DLResult,
    DLAggregate,
    RedactionMethod,
    ValidationSeverity
)


class TestDataLayerSnapshot:
    """Test cases for DataLayerSnapshot model with correct fields."""
    
    def test_basic_snapshot_creation(self):
        """Test creating a basic DataLayerSnapshot."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"test": "value"},
            events=[{"event": "page_view"}]
        )
        
        assert str(snapshot.page_url) == "https://example.com/"
        assert snapshot.exists is True
        assert snapshot.latest == {"test": "value"}
        assert len(snapshot.events) == 1
        assert snapshot.events[0] == {"event": "page_view"}
        
    def test_snapshot_with_missing_datalayer(self):
        """Test creating snapshot for missing dataLayer."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=False,
            latest=None,
            events=[]
        )
        
        assert snapshot.exists is False
        assert snapshot.latest is None
        assert len(snapshot.events) == 0
        
    def test_snapshot_with_redaction_info(self):
        """Test snapshot with redaction metadata."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"user": "***REDACTED***"},
            redacted_paths=["user.email", "user.phone"],
            redaction_method_used=RedactionMethod.MASK
        )
        
        assert len(snapshot.redacted_paths) == 2
        assert snapshot.redaction_method_used == RedactionMethod.MASK
        
    def test_snapshot_properties(self):
        """Test snapshot computed properties."""
        snapshot = DataLayerSnapshot(
            page_url="https://staging.example.com/page",
            exists=True,
            latest={"var1": "value1", "var2": "value2", "nested": {"key": "val"}}
        )
        
        assert snapshot.host == "staging.example.com"
        assert snapshot.variable_count == 3  # var1, var2, nested


class TestValidationIssue:
    """Test cases for ValidationIssue model."""
    
    def test_basic_validation_issue(self):
        """Test creating a basic validation issue."""
        issue = ValidationIssue(
            page_url="https://example.com",
            path="/user/email",
            message="Invalid email format",
            severity=ValidationSeverity.WARNING
        )
        
        assert issue.path == "/user/email"
        assert issue.message == "Invalid email format"
        assert issue.severity == ValidationSeverity.WARNING
        
    def test_critical_validation_issue(self):
        """Test creating a critical validation issue."""
        issue = ValidationIssue(
            page_url="https://example.com",
            path="required_field",
            message="Required field is missing",
            severity=ValidationSeverity.CRITICAL,
            expected="string",
            observed=None
        )
        
        assert issue.severity == ValidationSeverity.CRITICAL
        assert issue.expected == "string"
        assert issue.observed is None


class TestDLResult:
    """Test cases for DLResult model."""
    
    def test_basic_result_creation(self):
        """Test creating a basic DLResult."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"test": "value"}
        )
        
        result = DLResult(
            snapshot=snapshot,
            issues=[],
            processing_time_ms=100.0
        )
        
        assert str(result.snapshot.page_url) == "https://example.com/"
        assert result.processing_time_ms == 100.0
        assert len(result.issues) == 0
        
    def test_result_with_validation_issues(self):
        """Test result with validation issues."""
        snapshot = DataLayerSnapshot(
            page_url="https://example.com",
            exists=True,
            latest={"invalid": "data"}
        )
        
        issue = ValidationIssue(
            page_url="https://example.com",
            path="invalid",
            message="Invalid value",
            severity=ValidationSeverity.WARNING
        )
        
        result = DLResult(
            snapshot=snapshot,
            issues=[issue],
            processing_time_ms=200.0
        )
        
        assert len(result.issues) == 1
        assert result.issues[0].message == "Invalid value"


class TestDLAggregate:
    """Test cases for DLAggregate model."""
    
    def test_basic_aggregate_creation(self):
        """Test creating a basic DLAggregate."""
        aggregate = DLAggregate(
            run_id="test-run-123",
            total_pages=10,
            pages_successful=9,
            pages_failed=1
        )
        
        assert aggregate.run_id == "test-run-123"
        assert aggregate.total_pages == 10
        assert aggregate.pages_successful == 9
        assert aggregate.pages_failed == 1
        
    def test_aggregate_success_rate(self):
        """Test aggregate success rate calculation."""
        aggregate = DLAggregate(
            run_id="test-run-123",
            total_pages=20,
            pages_successful=18,
            pages_failed=2
        )
        
        # Should calculate success rate as 90%
        expected_rate = 90.0  # Property likely returns percentage
        assert abs(aggregate.success_rate - expected_rate) < 0.1


class TestEnums:
    """Test cases for enum definitions."""
    
    def test_redaction_methods(self):
        """Test RedactionMethod enum values."""
        assert RedactionMethod.REMOVE == "remove"
        assert RedactionMethod.HASH == "hash"
        assert RedactionMethod.MASK == "mask"
        assert RedactionMethod.TRUNCATE == "truncate"
        
    def test_validation_severities(self):
        """Test ValidationSeverity enum values."""
        assert ValidationSeverity.INFO == "info"
        assert ValidationSeverity.WARNING == "warning"
        assert ValidationSeverity.CRITICAL == "critical"