"""Shared fixtures for DataLayer tests."""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.audit.datalayer.models import (
    DataLayerSnapshot,
    ValidationIssue,
    ValidationSeverity,
    DLContext,
    DLResult
)
from app.audit.datalayer.config import DataLayerConfig


@pytest.fixture
def sample_datalayer_config():
    """Sample DataLayer configuration for testing."""
    return DataLayerConfig(
        enabled=True,
        capture_timeout=5.0,
        max_depth=10,
        max_size=10000
    )


@pytest.fixture
def sample_dl_context():
    """Sample DL context for testing."""
    return DLContext(
        url="https://example.com/test",
        page_title="Test Page"
    )


@pytest.fixture
def sample_snapshot():
    """Sample DataLayer snapshot for testing."""
    return DataLayerSnapshot(
        url="https://example.com/test",
        raw_data={"page": "test", "user_id": "123"},
        processed_data={"page": "test", "user_id": "123"}
    )


@pytest.fixture
def sample_snapshot_with_events():
    """Sample DataLayer snapshot with events."""
    return DataLayerSnapshot(
        url="https://example.com/test",
        raw_data=[{"event": "page_view"}, {"user_id": "123"}],
        processed_data={"user_id": "123"},
        events=[{"event": "page_view"}]
    )


@pytest.fixture
def sample_validation_issues():
    """Sample validation issues for testing."""
    return [
        ValidationIssue(
            path="/user/email",
            message="Invalid email format",
            severity=ValidationSeverity.CRITICAL
        ),
        ValidationIssue(
            path="/user/name",
            message="Missing required field",
            severity=ValidationSeverity.WARNING
        )
    ]


@pytest.fixture
def sample_dl_result(sample_dl_context, sample_snapshot, sample_validation_issues):
    """Sample DL result for testing."""
    return DLResult(
        context=sample_dl_context,
        snapshot=sample_snapshot,
        issues=sample_validation_issues
    )


@pytest.fixture
def sample_schema():
    """Sample JSON schema for testing."""
    return {
        "type": "object",
        "properties": {
            "page": {"type": "string"},
            "user_id": {"type": "string"},
            "event": {"type": "string"}
        },
        "required": ["page"]
    }


@pytest.fixture
def temp_schema_file(sample_schema):
    """Temporary schema file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_schema, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def gtm_datalayer_sample():
    """Sample GTM-style dataLayer for testing."""
    return [
        {"gtm.start": 1234567890},
        {"event": "page_view", "page": "home", "user_id": "123"},
        {"category": "products"},
        {"event": "click", "element": "button", "user_id": "456"}
    ]


@pytest.fixture
def large_datalayer_sample():
    """Large dataLayer sample for performance testing."""
    return {
        f"variable_{i}": f"value_{i}" 
        for i in range(1000)
    }


@pytest.fixture
def sensitive_data_sample():
    """Sample data with sensitive information."""
    return {
        "user_email": "user@example.com",
        "phone_number": "555-123-4567",
        "ssn": "123-45-6789",
        "credit_card": "4111-1111-1111-1111",
        "safe_data": "This is safe",
        "ip_address": "192.168.1.1"
    }


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "datalayer: marks tests specific to dataLayer functionality"
    )