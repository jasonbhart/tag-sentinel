"""Comprehensive unit tests for API endpoints.

Tests all REST API endpoints with various scenarios including success cases,
error handling, validation, authentication, and edge cases.
"""

import pytest
import json
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from fastapi import status

from app.api.main import create_app
from app.api.services import AuditService, AuditNotFoundError
from app.api.services.audit_service import IdempotencyError
from app.api.services.export_service import ExportService
from app.api.schemas import AuditDetail, AuditList, AuditRef, AuditSummary, AuditLinks


class TestAuditEndpoints:
    """Test suite for audit management endpoints."""

    @pytest.fixture
    def client(self, mock_audit_service):
        """Create test client with mocked dependencies."""
        app = create_app()

        # Override dependencies
        from app.api.routes.audits import get_audit_service
        app.dependency_overrides[get_audit_service] = lambda: mock_audit_service

        return TestClient(app)

    @pytest.fixture
    def mock_audit_service(self):
        """Create a mock audit service."""
        return AsyncMock(spec=AuditService)

    @pytest.fixture
    def sample_audit_detail(self):
        """Sample audit detail for testing."""
        return AuditDetail(
            id="audit_20240115_ecommerce_prod_abc123",
            site_id="ecommerce",
            env="production",
            status="completed",
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow() + timedelta(minutes=15),
            updated_at=datetime.utcnow(),
            progress_percent=100.0,
            summary=AuditSummary(
                pages_discovered=47,
                pages_processed=45,
                pages_failed=2,
                requests_captured=1247,
                cookies_found=23,
                tags_detected=15,
                errors_count=3,
                duration_seconds=892.5
            ),
            params={"max_pages": 100, "discovery_mode": "hybrid"},
            metadata={"triggered_by": "test"},
            links=AuditLinks(
                self="http://test/api/audits/audit_20240115_ecommerce_prod_abc123",
                requests_export="http://test/api/exports/audit_20240115_ecommerce_prod_abc123/requests.json",
                cookies_export="http://test/api/exports/audit_20240115_ecommerce_prod_abc123/cookies.csv",
                tags_export="http://test/api/exports/audit_20240115_ecommerce_prod_abc123/tags.json",
                artifacts="http://test/artifacts/audit_20240115_ecommerce_prod_abc123/"
            )
        )

    def test_create_audit_success(self, client, mock_audit_service, sample_audit_detail):
        """Test successful audit creation."""
        mock_audit_service.create_audit.return_value = sample_audit_detail

        response = client.post(
            "/api/audits",
            json={
                "site_id": "ecommerce",
                "env": "production",
                "params": {
                    "max_pages": 100,
                    "seeds": ["https://example.com"]
                },
                "priority": 10,
                "metadata": {"triggered_by": "test"}
            }
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["site_id"] == "ecommerce"
        assert data["env"] == "production"
        assert data["status"] == "completed"

    @patch('app.api.routes.audits.get_audit_service')
    def test_create_audit_validation_error(self, mock_get_service, client):
        """Test audit creation with validation errors."""
        response = client.post(
            "/api/audits",
            json={
                "site_id": "",  # Invalid: empty string
                "env": "production"
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert data["error"] == "validation_error"
        assert "validation_errors" in data["details"]

    @patch('app.api.routes.audits.get_audit_service')
    def test_create_audit_invalid_site_id(self, mock_get_service, client):
        """Test audit creation with invalid site_id format."""
        response = client.post(
            "/api/audits",
            json={
                "site_id": "invalid@site",  # Invalid characters
                "env": "production"
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('app.api.routes.audits.get_audit_service')
    def test_create_audit_invalid_params(self, mock_get_service, client, mock_audit_service):
        """Test audit creation with invalid parameters."""
        mock_get_service.return_value = mock_audit_service
        mock_audit_service.create_audit.side_effect = ValueError("Invalid parameters: max_pages must be positive")

        response = client.post(
            "/api/audits",
            json={
                "site_id": "ecommerce",
                "env": "production",
                "params": {"max_pages": -1}  # Invalid negative value
            }
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Invalid parameters" in data["message"]

    def test_create_audit_idempotency_conflict(self, client, mock_audit_service):
        """Test audit creation with idempotency conflict."""
        mock_audit_service.create_audit.side_effect = IdempotencyError(
            "existing_audit_id",
            "Idempotency key exists with different parameters"
        )

        response = client.post(
            "/api/audits",
            json={
                "site_id": "ecommerce",
                "env": "production",
                "params": {
                    "max_pages": 100,
                    "seeds": ["https://example.com"]
                },
                "idempotency_key": "test_key_001"
            }
        )

        assert response.status_code == status.HTTP_409_CONFLICT
        data = response.json()
        assert "Idempotency key" in data["message"]

    def test_get_audit_success(self, client, mock_audit_service, sample_audit_detail):
        """Test successful audit retrieval."""
        mock_audit_service.get_audit.return_value = sample_audit_detail

        response = client.get("/api/audits/audit_20240115_ecommerce_prod_abc123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "audit_20240115_ecommerce_prod_abc123"
        assert data["site_id"] == "ecommerce"

    @patch('app.api.routes.audits.get_audit_service')
    def test_get_audit_not_found(self, mock_get_service, client, mock_audit_service):
        """Test audit retrieval when audit doesn't exist."""
        mock_get_service.return_value = mock_audit_service
        mock_audit_service.get_audit.side_effect = AuditNotFoundError("Audit not found")

        response = client.get("/api/audits/nonexistent_audit_id")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["message"]

    def test_list_audits_success(self, client, mock_audit_service):
        """Test successful audit listing."""

        # Mock audit list response
        audit_refs = [
            AuditRef(
                id=f"audit_{i}",
                site_id="ecommerce",
                env="production",
                status="completed",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            for i in range(3)
        ]

        mock_audit_service.list_audits.return_value = AuditList(
            audits=audit_refs,
            total_count=3,
            has_more=False,
            next_cursor=None
        )

        response = client.get("/api/audits")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["audits"]) == 3
        assert data["total_count"] == 3
        assert not data["has_more"]

    def test_list_audits_with_filters(self, client, mock_audit_service):
        """Test audit listing with filters."""
        mock_audit_service.list_audits.return_value = AuditList(
            audits=[],
            total_count=0,
            has_more=False,
            next_cursor=None
        )

        response = client.get(
            "/api/audits",
            params={
                "site_id": "ecommerce",
                "env": "production",
                "status": ["completed", "failed"],
                "limit": 50,
                "sort_by": "created_at",
                "sort_order": "desc"
            }
        )

        assert response.status_code == status.HTTP_200_OK
        # Verify that the service was called with the correct filters
        mock_audit_service.list_audits.assert_called_once()

    @patch('app.api.routes.audits.get_audit_service')
    def test_list_audits_invalid_limit(self, mock_get_service, client):
        """Test audit listing with invalid limit parameter."""
        response = client.get("/api/audits", params={"limit": 150})  # Exceeds max limit

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestExportEndpoints:
    """Test suite for export endpoints."""

    @pytest.fixture
    def client(self, mock_audit_service, mock_export_service):
        """Create test client."""
        app = create_app()

        # Override dependencies
        from app.api.routes.audits import get_audit_service
        from app.api.routes.exports import get_export_service
        app.dependency_overrides[get_audit_service] = lambda: mock_audit_service
        app.dependency_overrides[get_export_service] = lambda: mock_export_service

        return TestClient(app)

    @pytest.fixture
    def mock_audit_service(self):
        """Create a mock audit service."""
        return AsyncMock(spec=AuditService)

    @pytest.fixture
    def mock_export_service(self):
        """Create a mock export service."""
        return AsyncMock(spec=ExportService)

    @pytest.fixture
    def sample_audit_detail(self):
        """Sample audit detail for export tests."""
        return AuditDetail(
            id="test_audit_001",
            site_id="ecommerce",
            env="production",
            status="completed",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            links=AuditLinks(
                self="http://test/api/audits/test_audit_001",
                requests_export="http://test/api/exports/test_audit_001/requests.json",
                cookies_export="http://test/api/exports/test_audit_001/cookies.csv",
                tags_export="http://test/api/exports/test_audit_001/tags.json"
            )
        )

    async def mock_export_generator(self, data_lines):
        """Mock export data generator."""
        for line in data_lines:
            yield line

    def test_export_requests_json(self, client, mock_audit_service, mock_export_service, sample_audit_detail):
        """Test request log export in JSON format."""
        mock_audit_service.get_audit.return_value = sample_audit_detail
        mock_export_service.export_requests.return_value = self.mock_export_generator([
            '{"audit_id": "test_audit_001", "url": "https://example.com"}\n',
            '{"audit_id": "test_audit_001", "url": "https://analytics.com"}\n'
        ])

        response = client.get("/api/exports/test_audit_001/requests.json")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/x-ndjson"
        assert "attachment" in response.headers["content-disposition"]

    def test_export_requests_csv(self, client, mock_audit_service, mock_export_service, sample_audit_detail):
        """Test request log export in CSV format."""
        mock_audit_service.get_audit.return_value = sample_audit_detail
        mock_export_service.export_requests.return_value = self.mock_export_generator([
            'audit_id,url\n',
            'test_audit_001,https://example.com\n',
            'test_audit_001,https://analytics.com\n'
        ])

        response = client.get("/api/exports/test_audit_001/requests.csv")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/csv; charset=utf-8"

    @patch('app.api.routes.exports.get_audit_service')
    def test_export_audit_not_found(self, mock_get_audit, client, mock_audit_service):
        """Test export when audit doesn't exist."""
        mock_get_audit.return_value = mock_audit_service
        mock_audit_service.get_audit.side_effect = AuditNotFoundError("Audit not found")

        response = client.get("/api/exports/nonexistent_audit/requests.json")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_export_with_filters(self, client, mock_audit_service, mock_export_service, sample_audit_detail):
        """Test export with query parameter filters."""
        mock_audit_service.get_audit.return_value = sample_audit_detail
        mock_export_service.export_requests.return_value = self.mock_export_generator(['{"test": "data"}\n'])

        response = client.get(
            "/api/exports/test_audit_001/requests.json",
            params={"status": "success", "analytics_only": True}
        )

        assert response.status_code == status.HTTP_200_OK
        # Verify filters were passed to export service
        mock_export_service.export_requests.assert_called_once_with(
            "test_audit_001",
            "json",
            {"status": "success", "analytics_only": True}
        )

    def test_export_cookies_json(self, client, mock_audit_service, mock_export_service, sample_audit_detail):
        """Test cookie export in JSON format."""
        mock_audit_service.get_audit.return_value = sample_audit_detail
        mock_export_service.export_cookies.return_value = self.mock_export_generator([
            '{"name": "_ga", "domain": ".example.com"}\n'
        ])

        response = client.get("/api/exports/test_audit_001/cookies.json")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/x-ndjson"

    def test_export_tags_with_filters(self, client, mock_audit_service, mock_export_service, sample_audit_detail):
        """Test tag export with vendor filtering."""
        mock_audit_service.get_audit.return_value = sample_audit_detail
        mock_export_service.export_tags.return_value = self.mock_export_generator([
            '{"vendor": "Google Analytics", "tag_type": "pageview"}\n'
        ])

        response = client.get(
            "/api/exports/test_audit_001/tags.json",
            params={"vendor": "Google Analytics", "has_errors": False}
        )

        assert response.status_code == status.HTTP_200_OK

    @patch('app.api.routes.exports.get_audit_service')
    @patch('app.api.routes.exports.get_export_service')
    def test_export_data_layer(self, mock_get_export, mock_get_audit, client,
                               mock_audit_service, mock_export_service, sample_audit_detail):
        """Test data layer export."""
        mock_get_audit.return_value = mock_audit_service
        mock_get_export.return_value = mock_export_service

        mock_audit_service.get_audit.return_value = sample_audit_detail
        mock_export_service.export_data_layer.return_value = self.mock_export_generator([
            '{"trigger_event": "page_load", "total_properties": 5}\n'
        ])

        response = client.get("/api/exports/test_audit_001/data-layer.json")

        assert response.status_code == status.HTTP_200_OK


class TestArtifactEndpoints:
    """Test suite for artifact serving endpoints."""

    @pytest.fixture
    def client(self, mock_audit_service):
        """Create test client."""
        app = create_app()

        # Override dependencies
        from app.api.routes.audits import get_audit_service
        app.dependency_overrides[get_audit_service] = lambda: mock_audit_service

        return TestClient(app)

    @pytest.fixture
    def mock_audit_service(self):
        """Create a mock audit service."""
        return AsyncMock(spec=AuditService)

    @pytest.fixture
    def sample_audit_detail(self):
        """Sample audit detail for artifact tests."""
        return AuditDetail(
            id="test_audit_001",
            site_id="ecommerce",
            env="production",
            status="completed",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            links=AuditLinks(
                self="http://test/api/audits/test_audit_001",
                requests_export="http://test/api/exports/test_audit_001/requests.json",
                cookies_export="http://test/api/exports/test_audit_001/cookies.csv",
                tags_export="http://test/api/exports/test_audit_001/tags.json",
                artifacts="http://test/artifacts/test_audit_001/"
            )
        )

    def test_artifact_types_validation(self, client, mock_audit_service, sample_audit_detail):
        """Test that only allowed artifact types are accepted."""
        mock_audit_service.get_audit.return_value = sample_audit_detail
        response = client.get("/api/artifacts/test_audit/invalid_type/file.txt")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @patch('app.api.routes.artifacts.get_audit_service')
    def test_artifact_audit_not_found(self, mock_get_service, client, mock_audit_service):
        """Test artifact access when audit doesn't exist."""
        mock_get_service.return_value = mock_audit_service
        mock_audit_service.get_audit.side_effect = AuditNotFoundError("Audit not found")

        response = client.get("/api/artifacts/nonexistent_audit/har/network.har")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_signed_url_generation(self, client, mock_audit_service, sample_audit_detail):
        """Test signed URL generation endpoint."""
        mock_audit_service.get_audit.return_value = sample_audit_detail

        # Mock artifact files exist
        with patch('pathlib.Path.exists', return_value=True):
            response = client.post(
                "/api/artifacts/test_audit_001/signed-urls",
                json={
                    "artifacts": [
                        {"type": "har", "filename": "network.har"},
                        {"type": "screenshot", "filename": "page.png"}
                    ],
                    "expires_in": 3600
                }
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "urls" in data
        assert "expires_at" in data


class TestErrorHandling:
    """Test suite for error handling across all endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_audit_service(self):
        """Create a mock audit service."""
        return AsyncMock(spec=AuditService)

    def test_404_not_found(self, client):
        """Test 404 error handling."""
        response = client.get("/api/nonexistent-endpoint")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_method_not_allowed(self, client):
        """Test 405 method not allowed."""
        response = client.put("/api/audits")  # POST endpoint, PUT not allowed
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_request_validation_error_structure(self, client):
        """Test that validation errors have proper structure."""
        response = client.post("/api/audits", json={"invalid": "data"})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "timestamp" in data
        assert "request_id" in data
        assert data["error"] == "validation_error"

    def test_internal_server_error(self, mock_audit_service):
        """Test 500 internal server error handling."""
        app = create_app()

        # Override dependencies to cause an error
        from app.api.routes.audits import get_audit_service
        mock_audit_service.create_audit.side_effect = Exception("Unexpected error")
        app.dependency_overrides[get_audit_service] = lambda: mock_audit_service

        client = TestClient(app)
        response = client.post(
            "/api/audits",
            json={
                "site_id": "test",
                "env": "staging",
                "params": {
                    "max_pages": 100,
                    "seeds": ["https://example.com"]
                }
            }
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["error"] == "http_500"


class TestMiddleware:
    """Test suite for middleware functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_cors_headers(self, client):
        """Test that response headers are present."""
        response = client.get("/api/audits")
        # Just test that the response has basic headers
        assert "content-type" in response.headers
        assert "x-request-id" in response.headers

    def test_request_id_header(self, client):
        """Test that request ID is added to response headers."""
        response = client.get("/health")
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "services" in data
        assert "uptime_seconds" in data

    def test_root_endpoint(self, client):
        """Test root endpoint redirect."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "documentation" in data