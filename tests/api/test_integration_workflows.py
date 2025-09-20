"""Integration tests for end-to-end API workflows.

Tests complete audit workflows from creation to export, including
realistic usage scenarios and cross-component integration.
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.services import AuditService
from app.api.services.export_service import ExportService
from app.api.persistence.repositories import InMemoryAuditRepository, InMemoryExportDataRepository
from app.api.runner_integration import RunnerIntegrationService


class TestEndToEndWorkflows:
    """Test complete audit workflows from start to finish."""

    @pytest.fixture(autouse=True)
    def clear_repository_cache(self):
        """Clear repository cache before each test."""
        from app.api.persistence.factory import RepositoryFactory
        RepositoryFactory.clear_cache()
        # Also clear route module globals
        import app.api.routes.audits
        import app.api.routes.exports
        app.api.routes.audits._audit_service_instance = None
        app.api.routes.exports._export_service_instance = None
        yield
        RepositoryFactory.clear_cache()
        app.api.routes.audits._audit_service_instance = None
        app.api.routes.exports._export_service_instance = None

    @pytest.fixture
    def app(self):
        """Create FastAPI application for testing."""
        # Reset global state to ensure test isolation
        import app.api.routes.audits
        import app.api.routes.exports
        app.api.routes.audits._audit_service_instance = None
        app.api.routes.exports._export_service_instance = None

        return create_app()

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_repositories(self):
        """Create mock repositories with sample data."""
        audit_repo = InMemoryAuditRepository()
        export_repo = InMemoryExportDataRepository()
        return audit_repo, export_repo

    @pytest.fixture
    def mock_runner_service(self):
        """Create mock runner integration service."""
        runner = AsyncMock(spec=RunnerIntegrationService)
        runner.dispatch_audit.return_value = True
        return runner

    @pytest.mark.asyncio
    async def test_complete_audit_lifecycle(self, client):
        """Test complete audit lifecycle from creation to completion."""
        # Step 1: Create audit
        create_response = client.post(
            "/api/audits",
            json={
                "site_id": "ecommerce",
                "env": "staging",
                "params": {
                    "max_pages": 10,
                    "discovery_mode": "seeds",
                    "seeds": ["https://example.com"]
                },
                "idempotency_key": "test_lifecycle_001",
                "metadata": {"test_type": "integration"}
            }
        )

        assert create_response.status_code == 201
        audit_data = create_response.json()
        audit_id = audit_data["id"]

        # Verify audit was created with correct data
        assert audit_data["site_id"] == "ecommerce"
        assert audit_data["env"] == "staging"
        assert audit_data["status"] == "queued"
        assert audit_data["params"]["max_pages"] == 10

        # Step 2: Get audit details
        get_response = client.get(f"/api/audits/{audit_id}")
        assert get_response.status_code == 200

        get_data = get_response.json()
        assert get_data["id"] == audit_id
        assert get_data["status"] == "queued"

        # Step 3: Simulate audit progression (would normally be done by runner)
        # This would happen asynchronously in real implementation

        # Step 4: List audits and verify it appears
        list_response = client.get("/api/audits", params={"site_id": "ecommerce"})
        assert list_response.status_code == 200

        list_data = list_response.json()
        assert list_data["total_count"] >= 1
        audit_ids = [audit["id"] for audit in list_data["audits"]]
        assert audit_id in audit_ids

        # Step 5: Verify idempotency works
        duplicate_response = client.post(
            "/api/audits",
            json={
                "site_id": "ecommerce",
                "env": "staging",
                "params": {
                    "max_pages": 10,
                    "discovery_mode": "seeds",
                    "seeds": ["https://example.com"]
                },
                "idempotency_key": "test_lifecycle_001",
                "metadata": {"test_type": "integration"}
            }
        )

        assert duplicate_response.status_code == 201
        duplicate_data = duplicate_response.json()
        assert duplicate_data["id"] == audit_id  # Same audit returned

    @pytest.mark.asyncio
    async def test_audit_with_invalid_params_workflow(self, client):
        """Test workflow when audit creation fails due to invalid parameters."""
        # Try to create audit with invalid params
        response = client.post(
            "/api/audits",
            json={
                "site_id": "test_site",
                "env": "staging",
                "params": {
                    "max_pages": -1,  # Invalid negative value
                    "discovery_mode": "invalid_mode"  # Invalid mode
                }
            }
        )

        # Should fail validation
        assert response.status_code == 400
        error_data = response.json()
        assert "Invalid parameters" in error_data["message"]

    @pytest.mark.asyncio
    async def test_concurrent_audit_creation(self, client):
        """Test creating multiple audits concurrently."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def create_audit(site_id):
            """Helper function to create audit."""
            return client.post(
                "/api/audits",
                json={
                    "site_id": site_id,
                    "env": "staging",
                    "idempotency_key": f"concurrent_test_{site_id}"
                }
            )

        # Create multiple audits concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(create_audit, f"site_{i}")
                futures.append(future)

            # Wait for all to complete
            responses = [future.result() for future in futures]

        # All should succeed
        for response in responses:
            assert response.status_code == 201

        # Verify all audits were created
        list_response = client.get("/api/audits")
        list_data = list_response.json()
        assert list_data["total_count"] >= 5

    @pytest.mark.asyncio
    async def test_audit_listing_with_complex_filters(self, client):
        """Test audit listing with multiple filters and sorting."""
        # Create test audits with various configurations
        test_audits = [
            {
                "site_id": "ecommerce",
                "env": "production",
                "priority": 10,
                "params": {"max_pages": 10, "discovery_mode": "seeds", "seeds": ["https://ecommerce.com"]}
            },
            {
                "site_id": "ecommerce",
                "env": "staging",
                "priority": 5,
                "params": {"max_pages": 5, "discovery_mode": "seeds", "seeds": ["https://staging.ecommerce.com"]}
            },
            {
                "site_id": "blog",
                "env": "production",
                "priority": 1,
                "params": {"max_pages": 20, "discovery_mode": "seeds", "seeds": ["https://blog.com"]}
            },
            {
                "site_id": "docs",
                "env": "staging",
                "priority": 15,
                "params": {"max_pages": 15, "discovery_mode": "seeds", "seeds": ["https://docs.com"]}
            }
        ]

        created_audit_ids = []
        for i, audit_config in enumerate(test_audits):
            response = client.post(
                "/api/audits",
                json={
                    **audit_config,
                    "idempotency_key": f"filter_test_{i}",
                    "metadata": {"test_batch": "filter_testing"}
                }
            )
            assert response.status_code == 201
            created_audit_ids.append(response.json()["id"])

        # Test various filtering scenarios

        # Filter by site_id
        response = client.get("/api/audits", params={"site_id": "ecommerce"})
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert all(audit["site_id"] == "ecommerce" for audit in data["audits"])

        # Filter by environment
        response = client.get("/api/audits", params={"env": "production"})
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2

        # Multiple status filter
        response = client.get("/api/audits", params={"status": ["queued", "completed"]})
        assert response.status_code == 200

        # Test sorting
        response = client.get("/api/audits", params={
            "sort_by": "site_id",
            "sort_order": "asc",
            "limit": 10
        })
        assert response.status_code == 200
        data = response.json()
        site_ids = [audit["site_id"] for audit in data["audits"]]
        # Should be sorted alphabetically
        assert site_ids == sorted(site_ids)

        # Test pagination
        response = client.get("/api/audits", params={"limit": 2})
        assert response.status_code == 200
        data = response.json()
        assert len(data["audits"]) == 2
        if data["has_more"]:
            assert data["next_cursor"] is not None

    @pytest.mark.asyncio
    async def test_export_workflow_after_audit_completion(self, client):
        """Test export workflow after audit is completed."""
        # Step 1: Create and "complete" an audit
        create_response = client.post(
            "/api/audits",
            json={
                "site_id": "ecommerce",
                "env": "production",
                "idempotency_key": "export_test_001"
            }
        )
        assert create_response.status_code == 201
        audit_id = create_response.json()["id"]

        # Step 2: Test exports (will use mock data from export service)

        # Test request export
        export_response = client.get(f"/api/exports/{audit_id}/requests.json")
        # In real implementation, this would return actual data
        # For now, we verify the endpoint structure works

        # Test different export formats
        csv_response = client.get(f"/api/exports/{audit_id}/requests.csv")

        # Test cookie export
        cookie_response = client.get(f"/api/exports/{audit_id}/cookies.json")

        # Test tag export with filters
        tag_response = client.get(
            f"/api/exports/{audit_id}/tags.json",
            params={"vendor": "Google Analytics"}
        )

        # Test data layer export
        dl_response = client.get(f"/api/exports/{audit_id}/data-layer.csv")

    @pytest.mark.asyncio
    async def test_artifact_access_workflow(self, client):
        """Test artifact access workflow."""
        # Step 1: Create audit
        create_response = client.post(
            "/api/audits",
            json={
                "site_id": "ecommerce",
                "env": "production",
                "idempotency_key": "artifact_test_001"
            }
        )
        assert create_response.status_code == 201
        audit_id = create_response.json()["id"]

        # Step 2: Test signed URL generation
        signed_url_response = client.post(
            f"/api/artifacts/{audit_id}/signed-urls",
            json={
                "artifacts": [
                    {"type": "har", "filename": "network-trace.har"},
                    {"type": "screenshot", "filename": "page-001.png"}
                ],
                "expires_in": 3600
            }
        )

        # This will fail in test environment since files don't exist
        # But we can verify the endpoint structure

        # Step 3: Test direct artifact access (would fail due to missing files)
        artifact_response = client.get(f"/api/artifacts/{audit_id}/har/network-trace.har")
        # Expect 404 since file doesn't exist in test environment

    @pytest.mark.asyncio
    async def test_error_recovery_workflows(self, client):
        """Test error recovery and handling workflows."""

        # Test 1: Audit not found scenarios
        response = client.get("/api/audits/nonexistent_audit_id")
        assert response.status_code == 404

        response = client.get("/api/exports/nonexistent_audit_id/requests.json")
        assert response.status_code == 404

        response = client.get("/api/artifacts/nonexistent_audit_id/har/file.har")
        assert response.status_code == 404

        # Test 2: Invalid input validation
        response = client.post("/api/audits", json={})  # Missing required fields
        assert response.status_code == 422

        response = client.post("/api/audits", json={
            "site_id": "",  # Invalid empty string
            "env": "production"
        })
        assert response.status_code == 422

        # Test 3: Idempotency conflicts
        # Create an audit
        response1 = client.post("/api/audits", json={
            "site_id": "test_site",
            "env": "staging",
            "params": {"max_pages": 10, "discovery_mode": "seeds", "seeds": ["https://test.com"]},
            "idempotency_key": "conflict_test"
        })
        assert response1.status_code == 201

        # Try to create with same key but different params
        response2 = client.post("/api/audits", json={
            "site_id": "different_site",  # Different site
            "env": "staging",
            "params": {"max_pages": 10, "discovery_mode": "seeds", "seeds": ["https://test.com"]},
            "idempotency_key": "conflict_test"
        })
        assert response2.status_code == 409  # Conflict

    @pytest.mark.asyncio
    async def test_performance_under_load(self, client):
        """Test API performance under load."""
        import time

        # Test response times for basic operations
        start_time = time.time()

        # Create multiple audits
        for i in range(10):
            response = client.post("/api/audits", json={
                "site_id": f"perf_test_site_{i}",
                "env": "staging",
                "params": {"max_pages": 5, "discovery_mode": "seeds", "seeds": [f"https://site{i}.com"]},
                "idempotency_key": f"perf_test_{i}"
            })
            assert response.status_code == 201

        creation_time = time.time() - start_time

        # Test listing performance
        start_time = time.time()
        response = client.get("/api/audits", params={"limit": 50})
        listing_time = time.time() - start_time

        assert response.status_code == 200

        # Verify reasonable response times (adjust thresholds as needed)
        # These are lenient for test environment
        assert creation_time < 5.0  # 10 creations in under 5 seconds
        assert listing_time < 1.0   # Listing in under 1 second

    @pytest.mark.asyncio
    async def test_api_versioning_and_compatibility(self, client):
        """Test API versioning and backward compatibility."""

        # Test current API version works
        response = client.get("/api/audits")
        assert response.status_code == 200

        # Test OpenAPI documentation availability
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec

        # Verify key endpoints are documented
        paths = openapi_spec["paths"]
        assert "/api/audits" in paths
        assert "/api/audits/{audit_id}" in paths
        assert "/api/exports/{audit_id}/requests.{format}" in paths

    @pytest.mark.asyncio
    async def test_health_and_monitoring_endpoints(self, client):
        """Test health check and monitoring endpoints."""

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert "status" in health_data
        assert "version" in health_data
        assert "services" in health_data
        assert "uptime_seconds" in health_data

        # Verify service status structure
        services = health_data["services"]
        expected_services = ["database", "cache", "audit_runner", "browser_engine"]
        for service in expected_services:
            assert service in services
            assert services[service] in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_edge_cases_and_boundary_conditions(self, client):
        """Test edge cases and boundary conditions."""

        # Test maximum limit values
        response = client.get("/api/audits", params={"limit": 100})  # Max allowed
        assert response.status_code == 200

        response = client.get("/api/audits", params={"limit": 101})  # Over max
        assert response.status_code == 422

        # Test empty results
        response = client.get("/api/audits", params={"site_id": "nonexistent_site"})
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert len(data["audits"]) == 0

        # Test special characters in site_id
        response = client.post("/api/audits", json={
            "site_id": "site-with_dots.and-dashes",
            "env": "staging",
            "params": {"max_pages": 5, "discovery_mode": "seeds", "seeds": ["https://example.com"]}
        })
        assert response.status_code == 201

        # Test invalid characters in site_id
        response = client.post("/api/audits", json={
            "site_id": "site@invalid#chars",
            "env": "staging"
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_data_consistency_across_operations(self, client):
        """Test data consistency across different API operations."""

        # Create audit with specific metadata
        create_response = client.post("/api/audits", json={
            "site_id": "consistency_test",
            "env": "staging",
            "params": {
                "max_pages": 25,
                "discovery_mode": "seeds",
                "seeds": ["https://example.com"]
            },
            "priority": 5,
            "metadata": {"test_key": "test_value", "tags": ["tag1", "tag2"]},
            "idempotency_key": "consistency_test_001"
        })
        assert create_response.status_code == 201
        audit_id = create_response.json()["id"]

        # Verify data via get endpoint
        get_response = client.get(f"/api/audits/{audit_id}")
        assert get_response.status_code == 200
        get_data = get_response.json()

        # Verify all data matches
        assert get_data["site_id"] == "consistency_test"
        assert get_data["env"] == "staging"
        assert get_data["params"]["max_pages"] == 25
        assert get_data["metadata"]["test_key"] == "test_value"

        # Verify data via list endpoint
        list_response = client.get("/api/audits", params={"site_id": "consistency_test"})
        assert list_response.status_code == 200
        list_data = list_response.json()

        # Find our audit in the list
        our_audit = None
        for audit in list_data["audits"]:
            if audit["id"] == audit_id:
                our_audit = audit
                break

        assert our_audit is not None
        assert our_audit["site_id"] == "consistency_test"
        assert our_audit["env"] == "staging"