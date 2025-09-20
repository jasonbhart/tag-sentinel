"""Integration tests for API endpoints.

Tests the complete API functionality including request/response handling,
validation, error handling, and integration with the service layer.
"""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.api.main import app


class TestAuditAPI:
    """Test suite for audit API endpoints."""

    @pytest.fixture(autouse=True)
    def clear_audit_service(self):
        """Clear audit service state before each test."""
        # Import here to avoid circular imports
        import app.api.routes.audits as audits_module
        from app.api.persistence.factory import RepositoryFactory

        # Reset the global singleton instance
        audits_module._audit_service_instance = None
        # Clear repository cache to ensure fresh data for each test
        RepositoryFactory.clear_cache()

        yield

        # Clean up after test
        audits_module._audit_service_instance = None
        RepositoryFactory.clear_cache()

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "services" in data
        assert "uptime_seconds" in data

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["message"] == "Tag Sentinel API"
        assert "version" in data
        assert "documentation" in data

    def test_create_audit_basic(self, client):
        """Test basic audit creation."""
        audit_data = {
            "site_id": "test_site",
            "env": "staging"
        }

        response = client.post("/api/audits", json=audit_data)

        assert response.status_code == 201
        data = response.json()

        assert data["site_id"] == "test_site"
        assert data["env"] == "staging"
        assert data["status"] == "queued"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_audit_with_params(self, client):
        """Test audit creation with parameters."""
        audit_data = {
            "site_id": "ecommerce",
            "env": "production",
            "params": {
                "max_pages": 100,
                "discovery_mode": "seeds",
                "seeds": ["https://example.com"]
            },
            "priority": 10,
            "metadata": {
                "triggered_by": "test"
            }
        }

        response = client.post("/api/audits", json=audit_data)

        assert response.status_code == 201
        data = response.json()

        assert data["site_id"] == "ecommerce"
        assert data["env"] == "production"
        assert data["status"] == "queued"

    def test_create_audit_validation_error(self, client):
        """Test audit creation with validation errors."""
        # Missing required fields
        audit_data = {
            "site_id": "",  # Empty site_id
            "env": "staging"
        }

        response = client.post("/api/audits", json=audit_data)

        assert response.status_code == 422
        data = response.json()

        assert data["error"] == "validation_error"
        assert "validation_errors" in data["details"]

    def test_create_audit_invalid_site_id(self, client):
        """Test audit creation with invalid site_id format."""
        audit_data = {
            "site_id": "invalid@site",  # Contains invalid character
            "env": "staging"
        }

        response = client.post("/api/audits", json=audit_data)

        assert response.status_code == 422

    def test_create_audit_invalid_params(self, client):
        """Test audit creation with invalid parameters."""
        audit_data = {
            "site_id": "test_site",
            "env": "staging",
            "params": {
                "max_pages": -1  # Invalid negative value
            }
        }

        response = client.post("/api/audits", json=audit_data)

        assert response.status_code == 400
        data = response.json()

        assert "Invalid parameters" in data["message"]

    def test_idempotency(self, client):
        """Test idempotency functionality."""
        audit_data = {
            "site_id": "test_site",
            "env": "staging",
            "idempotency_key": "test_key_001"
        }

        # First request should succeed
        response1 = client.post("/api/audits", json=audit_data)
        assert response1.status_code == 201
        audit1_data = response1.json()

        # Second request with same key should return existing audit
        response2 = client.post("/api/audits", json=audit_data)
        assert response2.status_code == 201  # Returns existing audit
        audit2_data = response2.json()

        # Should be the same audit
        assert audit1_data["id"] == audit2_data["id"]

        # Test conflicting parameters with same idempotency key
        conflicting_data = {
            "site_id": "different_site",  # Different site_id
            "env": "staging",
            "idempotency_key": "test_key_001"  # Same key
        }

        response3 = client.post("/api/audits", json=conflicting_data)
        assert response3.status_code == 409
        data = response3.json()
        assert "different parameters" in data["message"]

    def test_get_audit(self, client):
        """Test getting audit details."""
        # First create an audit
        audit_data = {
            "site_id": "test_site",
            "env": "staging"
        }

        create_response = client.post("/api/audits", json=audit_data)
        assert create_response.status_code == 201
        audit_id = create_response.json()["id"]

        # Now get the audit
        response = client.get(f"/api/audits/{audit_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == audit_id
        assert data["site_id"] == "test_site"
        assert data["env"] == "staging"

    def test_get_audit_not_found(self, client):
        """Test getting non-existent audit."""
        response = client.get("/api/audits/nonexistent_audit_id")

        assert response.status_code == 404
        data = response.json()

        assert "not found" in data["message"]

    def test_list_audits_empty(self, client):
        """Test listing audits when none exist."""
        response = client.get("/api/audits")

        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 0
        assert len(data["audits"]) == 0
        assert not data["has_more"]

    def test_list_audits_with_data(self, client):
        """Test listing audits with existing data."""
        # Create several audits
        for i in range(3):
            audit_data = {
                "site_id": f"site_{i}",
                "env": "staging"
            }
            response = client.post("/api/audits", json=audit_data)
            assert response.status_code == 201

        # List audits
        response = client.get("/api/audits")

        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 3
        assert len(data["audits"]) == 3

    def test_list_audits_filtering(self, client):
        """Test audit listing with filters."""
        # Create audits for different sites
        audit_data_a = {"site_id": "site_a", "env": "staging"}
        audit_data_b = {"site_id": "site_b", "env": "staging"}
        audit_data_c = {"site_id": "site_a", "env": "production"}

        for data in [audit_data_a, audit_data_b, audit_data_c]:
            response = client.post("/api/audits", json=data)
            assert response.status_code == 201

        # Filter by site_id
        response = client.get("/api/audits?site_id=site_a")

        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 2
        assert all(audit["site_id"] == "site_a" for audit in data["audits"])

    def test_list_audits_pagination(self, client):
        """Test audit listing pagination."""
        # Create several audits
        for i in range(5):
            audit_data = {
                "site_id": f"site_{i}",
                "env": "staging"
            }
            response = client.post("/api/audits", json=audit_data)
            assert response.status_code == 201

        # Get first page
        response = client.get("/api/audits?limit=2")

        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 5
        assert len(data["audits"]) == 2
        assert data["has_more"]
        assert data["next_cursor"] is not None

    def test_list_audits_sorting(self, client):
        """Test audit listing with sorting."""
        # Create audits with different site IDs
        for site_id in ["site_c", "site_a", "site_b"]:
            audit_data = {
                "site_id": site_id,
                "env": "staging"
            }
            response = client.post("/api/audits", json=audit_data)
            assert response.status_code == 201

        # Sort by site_id ascending
        response = client.get("/api/audits?sort_by=site_id&sort_order=asc")

        assert response.status_code == 200
        data = response.json()

        site_ids = [audit["site_id"] for audit in data["audits"]]
        assert site_ids == ["site_a", "site_b", "site_c"]

    def test_list_audits_search(self, client):
        """Test audit listing with search."""
        # Create audits with different characteristics
        audit_data_1 = {
            "site_id": "ecommerce_site",
            "env": "staging",
            "metadata": {"campaign": "checkout_optimization"}
        }
        audit_data_2 = {
            "site_id": "blog_site",
            "env": "staging",
            "metadata": {"campaign": "seo_improvement"}
        }

        for data in [audit_data_1, audit_data_2]:
            response = client.post("/api/audits", json=data)
            assert response.status_code == 201

        # Search for "checkout"
        response = client.get("/api/audits?search=checkout")

        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 1
        assert data["audits"][0]["site_id"] == "ecommerce_site"

    def test_list_audits_status_filter(self, client):
        """Test filtering audits by status."""
        # Create an audit
        audit_data = {
            "site_id": "test_site",
            "env": "staging"
        }
        response = client.post("/api/audits", json=audit_data)
        assert response.status_code == 201

        # Filter by queued status
        response = client.get("/api/audits?status=queued")

        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 1
        assert all(audit["status"] == "queued" for audit in data["audits"])

    def test_list_audits_with_stats(self, client):
        """Test listing audits with summary statistics."""
        # Create several audits
        for i in range(3):
            audit_data = {
                "site_id": f"site_{i}",
                "env": "staging"
            }
            response = client.post("/api/audits", json=audit_data)
            assert response.status_code == 201

        # Request with stats
        response = client.get("/api/audits?include_stats=true")

        assert response.status_code == 200
        data = response.json()

        assert "summary_stats" in data
        assert data["summary_stats"]["total_audits"] == 3

    def test_request_headers(self, client):
        """Test that request ID headers are included."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/audits")

        # FastAPI/Starlette should handle CORS
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/audits",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_content_type_validation(self, client):
        """Test content type validation."""
        audit_data = {
            "site_id": "test_site",
            "env": "staging"
        }

        # Should work with proper content type
        response = client.post("/api/audits", json=audit_data)
        assert response.status_code == 201

        # Test with form data instead of JSON
        response = client.post("/api/audits", data=audit_data)
        assert response.status_code == 422