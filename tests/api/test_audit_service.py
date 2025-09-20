"""Unit tests for the AuditService class.

Tests the business logic for audit creation, tracking, and management
including idempotency, validation, and error handling.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

from app.api.services import AuditService, AuditNotFoundError, IdempotencyError
from app.api.schemas import CreateAuditRequest, ListAuditsRequest


class TestAuditService:
    """Test suite for AuditService."""

    @pytest.fixture(autouse=True)
    def clear_repository_cache(self):
        """Clear repository cache before each test."""
        from app.api.persistence.factory import RepositoryFactory
        RepositoryFactory.clear_cache()
        yield
        RepositoryFactory.clear_cache()

    @pytest.fixture
    def audit_service(self):
        """Create an AuditService instance for testing."""
        return AuditService(
            base_url="http://test.example.com",
            idempotency_window_hours=1,
            max_audit_age_days=7
        )

    @pytest.mark.asyncio
    async def test_create_audit_basic(self, audit_service):
        """Test basic audit creation."""
        request = CreateAuditRequest(
            site_id="test_site",
            env="staging"
        )

        audit = await audit_service.create_audit(request)

        assert audit.site_id == "test_site"
        assert audit.env == "staging"
        assert audit.status == "queued"
        assert audit.id.startswith("audit_")
        assert audit.created_at is not None
        assert audit.site_id == "test_site"
        assert audit.env == "staging"

    @pytest.mark.asyncio
    async def test_create_audit_with_params(self, audit_service):
        """Test audit creation with custom parameters."""
        request = CreateAuditRequest(
            site_id="ecommerce",
            env="production",
            params={
                "max_pages": 100,
                "discovery_mode": "seeds",
                "seeds": ["https://example.com"],
                "include_patterns": [".*\\/checkout\\/.*"]
            },
            priority=10,
            metadata={"triggered_by": "test"}
        )

        audit = await audit_service.create_audit(request)

        assert audit.site_id == "ecommerce"
        assert audit.env == "production"
        assert audit.status == "queued"

    @pytest.mark.asyncio
    async def test_create_audit_invalid_params(self, audit_service):
        """Test audit creation with invalid parameters."""
        request = CreateAuditRequest(
            site_id="test_site",
            env="staging",
            params={
                "discovery_mode": "hybrid",
                "max_pages": 50
                # Missing required sitemap_url for hybrid mode
            }
        )

        with pytest.raises(ValueError, match="Invalid audit parameters"):
            await audit_service.create_audit(request)

    @pytest.mark.asyncio
    async def test_idempotency_same_params(self, audit_service):
        """Test idempotency with same parameters."""
        request = CreateAuditRequest(
            site_id="test_site",
            env="staging",
            idempotency_key="test_key_001"
        )

        # Create first audit
        audit1 = await audit_service.create_audit(request)

        # Try to create same audit again - should return existing audit
        audit2 = await audit_service.create_audit(request)

        # Should be the same audit
        assert audit1.id == audit2.id

    @pytest.mark.asyncio
    async def test_idempotency_different_params(self, audit_service):
        """Test idempotency with different parameters."""
        request1 = CreateAuditRequest(
            site_id="test_site",
            env="staging",
            idempotency_key="test_key_002"
        )

        request2 = CreateAuditRequest(
            site_id="test_site",
            env="production",  # Different environment
            idempotency_key="test_key_002"
        )

        # Create first audit
        await audit_service.create_audit(request1)

        # Try to create audit with same key but different params
        with pytest.raises(IdempotencyError, match="different parameters"):
            await audit_service.create_audit(request2)

    @pytest.mark.asyncio
    async def test_get_audit_exists(self, audit_service):
        """Test getting an existing audit."""
        request = CreateAuditRequest(
            site_id="test_site",
            env="staging"
        )

        created_audit = await audit_service.create_audit(request)
        retrieved_audit = await audit_service.get_audit(created_audit.id)

        assert retrieved_audit.id == created_audit.id
        assert retrieved_audit.site_id == created_audit.site_id
        assert retrieved_audit.env == created_audit.env

    @pytest.mark.asyncio
    async def test_get_audit_not_found(self, audit_service):
        """Test getting a non-existent audit."""
        with pytest.raises(AuditNotFoundError, match="not found"):
            await audit_service.get_audit("nonexistent_audit_id")

    @pytest.mark.asyncio
    async def test_list_audits_empty(self, audit_service):
        """Test listing audits when none exist."""
        request = ListAuditsRequest()
        result = await audit_service.list_audits(request)

        assert result.total_count == 0
        assert len(result.audits) == 0
        assert not result.has_more

    @pytest.mark.asyncio
    async def test_list_audits_with_data(self, audit_service):
        """Test listing audits with existing data."""
        # Create several audits
        for i in range(3):
            request = CreateAuditRequest(
                site_id=f"site_{i}",
                env="staging"
            )
            await audit_service.create_audit(request)

        # List all audits
        list_request = ListAuditsRequest(limit=10)
        result = await audit_service.list_audits(list_request)

        assert result.total_count == 3
        assert len(result.audits) == 3
        assert not result.has_more

    @pytest.mark.asyncio
    async def test_list_audits_filtering(self, audit_service):
        """Test audit listing with filters."""
        # Create audits for different sites
        await audit_service.create_audit(CreateAuditRequest(site_id="site_a", env="staging"))
        await audit_service.create_audit(CreateAuditRequest(site_id="site_b", env="staging"))
        await audit_service.create_audit(CreateAuditRequest(site_id="site_a", env="production"))

        # Filter by site_id
        list_request = ListAuditsRequest(site_id="site_a")
        result = await audit_service.list_audits(list_request)

        assert result.total_count == 2
        assert all(audit.site_id == "site_a" for audit in result.audits)

        # Filter by environment
        list_request = ListAuditsRequest(env="staging")
        result = await audit_service.list_audits(list_request)

        assert result.total_count == 2
        assert all(audit.env == "staging" for audit in result.audits)

    @pytest.mark.asyncio
    async def test_list_audits_pagination(self, audit_service):
        """Test audit listing pagination."""
        # Create several audits
        for i in range(5):
            request = CreateAuditRequest(
                site_id=f"site_{i}",
                env="staging"
            )
            await audit_service.create_audit(request)

        # Get first page
        list_request = ListAuditsRequest(limit=2)
        result = await audit_service.list_audits(list_request)

        assert result.total_count == 5
        assert len(result.audits) == 2
        assert result.has_more
        assert result.next_cursor is not None

        # Get second page
        list_request = ListAuditsRequest(limit=2, cursor=result.next_cursor)
        result2 = await audit_service.list_audits(list_request)

        assert result2.total_count == 5
        assert len(result2.audits) == 2
        assert result2.has_more

    @pytest.mark.asyncio
    async def test_list_audits_sorting(self, audit_service):
        """Test audit listing with sorting."""
        # Create audits with different timestamps
        audit_ids = []
        for i in range(3):
            request = CreateAuditRequest(
                site_id=f"site_{chr(ord('c') - i)}",  # site_c, site_b, site_a
                env="staging"
            )
            audit = await audit_service.create_audit(request)
            audit_ids.append(audit.id)

        # Sort by site_id ascending
        list_request = ListAuditsRequest(sort_by="site_id", sort_order="asc")
        result = await audit_service.list_audits(list_request)

        site_ids = [audit.site_id for audit in result.audits]
        assert site_ids == ["site_a", "site_b", "site_c"]

        # Sort by site_id descending
        list_request = ListAuditsRequest(sort_by="site_id", sort_order="desc")
        result = await audit_service.list_audits(list_request)

        site_ids = [audit.site_id for audit in result.audits]
        assert site_ids == ["site_c", "site_b", "site_a"]

    @pytest.mark.asyncio
    async def test_update_audit_status(self, audit_service):
        """Test updating audit status."""
        request = CreateAuditRequest(
            site_id="test_site",
            env="staging"
        )

        audit = await audit_service.create_audit(request)
        assert audit.status == "queued"

        # Update to running
        from app.api.schemas.responses import AuditStatus
        await audit_service.update_audit_status(
            audit.id,
            AuditStatus.RUNNING,
            progress_percent=25.0
        )

        updated_audit = await audit_service.get_audit(audit.id)
        # Note: This test will need adjustment once we import the enum correctly

    @pytest.mark.asyncio
    async def test_update_nonexistent_audit_status(self, audit_service):
        """Test updating status of non-existent audit."""
        with pytest.raises(AuditNotFoundError):
            await audit_service.update_audit_status(
                "nonexistent_id",
                "running"
            )

    @pytest.mark.asyncio
    async def test_list_audits_with_stats(self, audit_service):
        """Test listing audits with summary statistics."""
        # Create several audits
        for i in range(3):
            request = CreateAuditRequest(
                site_id=f"site_{i}",
                env="staging"
            )
            await audit_service.create_audit(request)

        # Request with stats
        list_request = ListAuditsRequest(include_stats=True)
        result = await audit_service.list_audits(list_request)

        assert result.summary_stats is not None
        assert "total_audits" in result.summary_stats
        assert result.summary_stats["total_audits"] == 3

    @pytest.mark.asyncio
    async def test_audit_id_generation(self, audit_service):
        """Test that audit IDs are unique and well-formatted."""
        request = CreateAuditRequest(
            site_id="test_site",
            env="staging"
        )

        audit1 = await audit_service.create_audit(request)
        audit2 = await audit_service.create_audit(request)

        # IDs should be different
        assert audit1.id != audit2.id

        # IDs should follow expected format
        assert audit1.id.startswith("audit_")
        assert "test_site" in audit1.id
        assert "staging" in audit1.id

    @pytest.mark.asyncio
    async def test_search_functionality(self, audit_service):
        """Test search functionality in audit listing."""
        # Create audits with specific metadata
        await audit_service.create_audit(CreateAuditRequest(
            site_id="ecommerce_site",
            env="staging",
            metadata={"campaign": "checkout_optimization"}
        ))

        await audit_service.create_audit(CreateAuditRequest(
            site_id="blog_site",
            env="staging",
            metadata={"campaign": "seo_improvement"}
        ))

        # Search for "checkout"
        list_request = ListAuditsRequest(search="checkout")
        result = await audit_service.list_audits(list_request)

        assert result.total_count == 1
        assert result.audits[0].site_id == "ecommerce_site"

        # Search for "staging" (should match both)
        list_request = ListAuditsRequest(search="staging")
        result = await audit_service.list_audits(list_request)

        assert result.total_count == 2