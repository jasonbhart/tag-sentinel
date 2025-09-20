"""Unit tests for API schemas.

Tests validation, serialization, and edge cases for all Pydantic models
used in the API.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.api.schemas import (
    CreateAuditRequest,
    ListAuditsRequest,
    ExportRequest,
    AuditDetail,
    AuditRef,
    AuditList,
    ErrorResponse,
    HealthResponse,
    RequestLogExport,
    CookieExport,
    TagExport,
)


class TestCreateAuditRequest:
    """Test CreateAuditRequest schema validation."""

    def test_valid_basic_request(self):
        """Test basic valid audit request."""
        data = {
            "site_id": "test_site",
            "env": "staging"
        }

        request = CreateAuditRequest(**data)

        assert request.site_id == "test_site"
        assert request.env == "staging"
        assert request.params is None
        assert request.priority == 0

    def test_valid_full_request(self):
        """Test valid audit request with all fields."""
        data = {
            "site_id": "ecommerce_site",
            "env": "production",
            "params": {
                "max_pages": 100,
                "discovery_mode": "hybrid"
            },
            "rules_path": "/path/to/rules.yaml",
            "idempotency_key": "test_key_001",
            "priority": 10,
            "metadata": {
                "triggered_by": "ci_pipeline"
            }
        }

        request = CreateAuditRequest(**data)

        assert request.site_id == "ecommerce_site"
        assert request.env == "production"
        assert request.params["max_pages"] == 100
        assert request.rules_path == "/path/to/rules.yaml"
        assert request.idempotency_key == "test_key_001"
        assert request.priority == 10
        assert request.metadata["triggered_by"] == "ci_pipeline"

    def test_invalid_site_id(self):
        """Test invalid site_id format."""
        data = {
            "site_id": "invalid@site",  # Contains invalid character
            "env": "staging"
        }

        with pytest.raises(ValidationError) as exc_info:
            CreateAuditRequest(**data)

        errors = exc_info.value.errors()
        assert any("pattern" in str(error) for error in errors)

    def test_empty_site_id(self):
        """Test empty site_id."""
        data = {
            "site_id": "",
            "env": "staging"
        }

        with pytest.raises(ValidationError) as exc_info:
            CreateAuditRequest(**data)

        errors = exc_info.value.errors()
        assert any("min_length" in str(error) for error in errors)

    def test_invalid_env(self):
        """Test invalid environment format."""
        data = {
            "site_id": "test_site",
            "env": "invalid@env"  # Contains invalid character
        }

        with pytest.raises(ValidationError) as exc_info:
            CreateAuditRequest(**data)

        errors = exc_info.value.errors()
        assert any("pattern" in str(error) for error in errors)

    def test_priority_range(self):
        """Test priority value validation."""
        # Valid priority
        data = {
            "site_id": "test_site",
            "env": "staging",
            "priority": 50
        }
        request = CreateAuditRequest(**data)
        assert request.priority == 50

        # Invalid priority (too high)
        data["priority"] = 150
        with pytest.raises(ValidationError):
            CreateAuditRequest(**data)

        # Invalid priority (too low)
        data["priority"] = -150
        with pytest.raises(ValidationError):
            CreateAuditRequest(**data)

    def test_params_validation(self):
        """Test params field validation."""
        # Valid params
        data = {
            "site_id": "test_site",
            "env": "staging",
            "params": {
                "max_pages": 100,
                "include_patterns": [".*\\.html$"]
            }
        }

        request = CreateAuditRequest(**data)
        assert request.params["max_pages"] == 100

        # Invalid regex pattern
        data["params"]["include_patterns"] = ["[invalid_regex"]
        with pytest.raises(ValidationError) as exc_info:
            CreateAuditRequest(**data)

        assert "Invalid regex pattern" in str(exc_info.value)


class TestListAuditsRequest:
    """Test ListAuditsRequest schema validation."""

    def test_valid_basic_request(self):
        """Test basic valid list request."""
        request = ListAuditsRequest()

        assert request.site_id is None
        assert request.env is None
        assert request.sort_by == "created_at"
        assert request.sort_order == "desc"
        assert request.limit == 20

    def test_valid_full_request(self):
        """Test valid list request with all fields."""
        data = {
            "site_id": "test_site",
            "env": "staging",
            "status": ["completed", "failed"],
            "search": "checkout",
            "sort_by": "updated_at",
            "sort_order": "asc",
            "cursor": "cursor_123",
            "limit": 50,
            "include_stats": True
        }

        request = ListAuditsRequest(**data)

        assert request.site_id == "test_site"
        assert request.env == "staging"
        assert request.status == ["completed", "failed"]
        assert request.search == "checkout"
        assert request.sort_by == "updated_at"
        assert request.sort_order == "asc"
        assert request.limit == 50
        assert request.include_stats is True

    def test_limit_validation(self):
        """Test limit field validation."""
        # Valid limit
        data = {"limit": 50}
        request = ListAuditsRequest(**data)
        assert request.limit == 50

        # Invalid limit (too high)
        data["limit"] = 150
        with pytest.raises(ValidationError):
            ListAuditsRequest(**data)

        # Invalid limit (too low)
        data["limit"] = 0
        with pytest.raises(ValidationError):
            ListAuditsRequest(**data)


class TestExportRequest:
    """Test ExportRequest schema validation."""

    def test_valid_basic_request(self):
        """Test basic valid export request."""
        request = ExportRequest()

        assert request.format == "json"
        assert request.fields is None
        assert request.compress is False

    def test_valid_full_request(self):
        """Test valid export request with all fields."""
        data = {
            "format": "csv",
            "fields": ["url", "status", "response_time"],
            "filters": {
                "status": ["success"],
                "response_time_gt": 1000
            },
            "compress": True,
            "include_metadata": False
        }

        request = ExportRequest(**data)

        assert request.format == "csv"
        assert request.fields == ["url", "status", "response_time"]
        assert request.filters["status"] == ["success"]
        assert request.compress is True
        assert request.include_metadata is False


class TestResponseSchemas:
    """Test response schema models."""

    def test_audit_ref(self):
        """Test AuditRef schema."""
        data = {
            "id": "audit_123",
            "site_id": "test_site",
            "env": "staging",
            "status": "completed",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        audit_ref = AuditRef(**data)

        assert audit_ref.id == "audit_123"
        assert audit_ref.site_id == "test_site"
        assert audit_ref.status == "completed"

    def test_audit_detail(self):
        """Test AuditDetail schema."""
        now = datetime.utcnow()
        data = {
            "id": "audit_123",
            "site_id": "test_site",
            "env": "staging",
            "status": "completed",
            "created_at": now,
            "updated_at": now,
            "links": {
                "self": "http://test.com/api/audits/audit_123",
                "requests_export": "http://test.com/api/exports/audit_123/requests.json",
                "cookies_export": "http://test.com/api/exports/audit_123/cookies.csv",
                "tags_export": "http://test.com/api/exports/audit_123/tags.json"
            }
        }

        audit_detail = AuditDetail(**data)

        assert audit_detail.id == "audit_123"
        assert audit_detail.status == "completed"
        assert str(audit_detail.links.self) == "http://test.com/api/audits/audit_123"

    def test_audit_list(self):
        """Test AuditList schema."""
        now = datetime.utcnow()
        audit_ref_data = {
            "id": "audit_123",
            "site_id": "test_site",
            "env": "staging",
            "status": "completed",
            "created_at": now,
            "updated_at": now
        }

        data = {
            "audits": [audit_ref_data],
            "total_count": 1,
            "has_more": False,
            "next_cursor": None
        }

        audit_list = AuditList(**data)

        assert len(audit_list.audits) == 1
        assert audit_list.total_count == 1
        assert audit_list.has_more is False

    def test_error_response(self):
        """Test ErrorResponse schema."""
        data = {
            "error": "validation_error",
            "message": "Invalid input",
            "details": {"field": "site_id"},
            "request_id": "req_123"
        }

        error_response = ErrorResponse(**data)

        assert error_response.error == "validation_error"
        assert error_response.message == "Invalid input"
        assert error_response.details["field"] == "site_id"
        assert error_response.request_id == "req_123"

    def test_health_response(self):
        """Test HealthResponse schema."""
        data = {
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "database": "healthy",
                "cache": "healthy"
            },
            "uptime_seconds": 3600.0
        }

        health_response = HealthResponse(**data)

        assert health_response.status == "healthy"
        assert health_response.version == "1.0.0"
        assert health_response.services["database"] == "healthy"
        assert health_response.uptime_seconds == 3600.0


class TestExportSchemas:
    """Test export data schemas."""

    def test_request_log_export(self):
        """Test RequestLogExport schema."""
        data = {
            "id": "req_001",
            "audit_id": "audit_123",
            "page_url": "https://example.com/page",
            "url": "https://example.com/script.js",
            "method": "GET",
            "resource_type": "script",
            "status": "success",
            "timestamp": datetime.utcnow(),
            "is_analytics": True
        }

        export = RequestLogExport(**data)

        assert export.id == "req_001"
        assert export.resource_type == "script"
        assert export.is_analytics is True

    def test_cookie_export(self):
        """Test CookieExport schema."""
        data = {
            "audit_id": "audit_123",
            "page_url": "https://example.com/page",
            "name": "_ga",
            "domain": ".example.com",
            "discovered_at": datetime.utcnow(),
            "source": "javascript",
            "secure": True,
            "http_only": False
        }

        export = CookieExport(**data)

        assert export.name == "_ga"
        assert export.domain == ".example.com"
        assert export.secure is True

    def test_tag_export(self):
        """Test TagExport schema."""
        data = {
            "audit_id": "audit_123",
            "page_url": "https://example.com/page",
            "tag_id": "ga4_tag_001",
            "vendor": "Google Analytics 4",
            "tag_type": "pageview",
            "implementation_method": "gtm",
            "detected_at": datetime.utcnow(),
            "confidence_score": 0.95,
            "detection_method": "network_analysis",
            "uses_data_layer": True,
            "consent_required": True
        }

        export = TagExport(**data)

        assert export.vendor == "Google Analytics 4"
        assert export.tag_type == "pageview"
        assert export.confidence_score == 0.95
        assert export.uses_data_layer is True


class TestSchemaEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_handling(self):
        """Test Unicode string handling within pattern constraints."""
        data = {
            "site_id": "test_site_123",  # Valid pattern
            "env": "staging"
        }

        # Should handle valid characters
        request = CreateAuditRequest(**data)
        assert request.site_id == "test_site_123"

        # Test that invalid Unicode is rejected
        invalid_data = {
            "site_id": "test_site_ñ",  # Contains non-ASCII character
            "env": "staging"
        }

        with pytest.raises(ValidationError) as exc_info:
            CreateAuditRequest(**invalid_data)

        errors = exc_info.value.errors()
        assert any("pattern" in str(error) for error in errors)

    def test_none_values(self):
        """Test handling of None values."""
        data = {
            "site_id": "test_site",
            "env": "staging",
            "params": None,
            "metadata": None
        }

        request = CreateAuditRequest(**data)
        assert request.params is None
        assert request.metadata is None

    def test_empty_collections(self):
        """Test handling of empty collections."""
        data = {
            "format": "json",
            "fields": [],
            "filters": {}
        }

        request = ExportRequest(**data)
        assert request.fields == []
        assert request.filters == {}

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        original_data = {
            "site_id": "test_site",
            "env": "staging",
            "params": {
                "max_pages": 100
            },
            "priority": 5
        }

        # Create object from data
        request = CreateAuditRequest(**original_data)

        # Serialize to dict
        serialized = request.model_dump()

        # Create new object from serialized data
        new_request = CreateAuditRequest(**serialized)

        # Should be equivalent
        assert new_request.site_id == request.site_id
        assert new_request.env == request.env
        assert new_request.params == request.params
        assert new_request.priority == request.priority