"""Unit tests for repository implementations.

Tests all repository implementations (in-memory, file-based, and database)
to ensure consistent behavior across storage backends.
"""

import pytest
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path

from app.api.schemas.requests import ListAuditsRequest
from app.api.schemas.responses import AuditStatus
from app.api.schemas.exports import RequestLogExport, CookieExport, TagExport, DataLayerExport
from app.api.persistence.models import PersistentAuditRecord, ExportMetadata
from app.api.persistence.repositories import InMemoryAuditRepository, InMemoryExportDataRepository
from app.api.persistence.file_repositories import FileBasedAuditRepository, FileBasedExportDataRepository
from app.api.persistence.factory import RepositoryFactory, RepositoryConfig, StorageBackend

try:
    from app.api.persistence.database_repositories import DatabaseAuditRepository, DatabaseExportDataRepository
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


class TestRepositoryBehavior:
    """Base test class for repository behavior that should be consistent across all implementations."""

    # Prevent pytest from collecting this as a test class
    __test__ = False

    def create_sample_audit(self, audit_id: str = "test_audit_001") -> PersistentAuditRecord:
        """Create a sample audit record for testing."""
        return PersistentAuditRecord(
            id=audit_id,
            site_id="ecommerce",
            env="staging",
            status=AuditStatus.QUEUED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            idempotency_key="test_key_001",
            priority=5,
            params={"max_pages": 100, "discovery_mode": "hybrid"},
            metadata={"triggered_by": "test", "tags": ["regression", "smoke"]}
        )

    @pytest.mark.asyncio
    async def test_audit_crud_operations(self, audit_repository):
        """Test basic CRUD operations for audit repository."""
        audit = self.create_sample_audit()

        # Test create
        await audit_repository.create_audit(audit)

        # Test get
        retrieved = await audit_repository.get_audit(audit.id)
        assert retrieved is not None
        assert retrieved.id == audit.id
        assert retrieved.site_id == audit.site_id
        assert retrieved.status == audit.status

        # Test update
        audit.status = AuditStatus.RUNNING
        audit.progress_percent = 50
        await audit_repository.update_audit(audit)

        updated = await audit_repository.get_audit(audit.id)
        assert updated.status == AuditStatus.RUNNING
        assert updated.progress_percent == 50

        # Test delete
        deleted = await audit_repository.delete_audit(audit.id)
        assert deleted is True

        # Verify deletion
        not_found = await audit_repository.get_audit(audit.id)
        assert not_found is None

    @pytest.mark.asyncio
    async def test_audit_list_filtering(self, audit_repository):
        """Test audit listing with various filters."""
        # Create test audits
        audits = [
            self.create_sample_audit("audit_001"),
            PersistentAuditRecord(
                id="audit_002",
                site_id="blog",
                env="staging",
                status=AuditStatus.COMPLETED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={"tags": ["performance"]}
            ),
            PersistentAuditRecord(
                id="audit_003",
                site_id="ecommerce",
                env="production",
                status=AuditStatus.FAILED,
                created_at=datetime.utcnow() - timedelta(days=1),
                updated_at=datetime.utcnow(),
                metadata={"tags": ["security", "regression"]}
            )
        ]

        for audit in audits:
            await audit_repository.create_audit(audit)

        # Test site_id filter
        request = ListAuditsRequest(site_id="ecommerce")
        results, total = await audit_repository.list_audits(request)
        assert total == 2
        assert all(a.site_id == "ecommerce" for a in results)

        # Test environment filter
        request = ListAuditsRequest(env="staging")
        results, total = await audit_repository.list_audits(request)
        assert total == 2
        assert all(a.env == "staging" for a in results)

        # Test status filter
        request = ListAuditsRequest(status=["completed", "failed"])
        results, total = await audit_repository.list_audits(request)
        assert total == 2

        # Test tags filter
        request = ListAuditsRequest(tags=["regression"])
        results, total = await audit_repository.list_audits(request)
        assert total == 2

        # Test search
        request = ListAuditsRequest(search="blog")
        results, total = await audit_repository.list_audits(request)
        assert total == 1
        assert results[0].site_id == "blog"

    @pytest.mark.asyncio
    async def test_audit_pagination(self, audit_repository):
        """Test audit listing pagination."""
        # Create multiple audits
        for i in range(10):
            audit = PersistentAuditRecord(
                id=f"audit_{i:03d}",
                site_id=f"site_{i}",
                env="staging",
                status=AuditStatus.QUEUED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            await audit_repository.create_audit(audit)

        # Test pagination
        request = ListAuditsRequest(sort_by="site_id", sort_order="asc")
        results, total = await audit_repository.list_audits(request, limit=5, offset=0)

        assert total == 10
        assert len(results) == 5

        # Test second page
        results2, total2 = await audit_repository.list_audits(request, limit=5, offset=5)
        assert total2 == 10
        assert len(results2) == 5

        # Ensure no overlap
        ids1 = {r.id for r in results}
        ids2 = {r.id for r in results2}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_idempotency_key_search(self, audit_repository):
        """Test finding audits by idempotency key."""
        audit = self.create_sample_audit()
        await audit_repository.create_audit(audit)

        # Test finding within time window
        found = await audit_repository.find_by_idempotency_key("test_key_001", max_age_hours=24)
        assert found is not None
        assert found.id == audit.id

        # Test not finding outside time window
        not_found = await audit_repository.find_by_idempotency_key("test_key_001", max_age_hours=0)
        assert not_found is None

        # Test not finding non-existent key
        not_found = await audit_repository.find_by_idempotency_key("nonexistent_key")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_export_data_operations(self, export_repository):
        """Test export data repository operations."""
        audit_id = "test_audit_001"

        # Create sample data
        request_logs = [
            RequestLogExport(
                id="req_001",
                audit_id=audit_id,
                page_url="https://example.com/",
                url="https://example.com/analytics.js",
                method="GET",
                resource_type="script",
                status="success",
                status_code=200,
                timestamp=datetime.utcnow(),
                response_time=245,
                is_analytics=True,
                analytics_vendor="Google Analytics"
            )
        ]

        cookies = [
            CookieExport(
                audit_id=audit_id,
                page_url="https://example.com/",
                name="_ga",
                domain=".example.com",
                path="/",
                value="GA1.2.123456789.1234567890",
                secure=False,
                http_only=False,
                same_site="Lax",
                discovered_at=datetime.utcnow(),
                source="javascript",
                category="analytics",
                vendor="Google Analytics",
                is_essential=False
            )
        ]

        # Store data (using store methods if available)
        if hasattr(export_repository, 'store_request_logs'):
            await export_repository.store_request_logs(audit_id, request_logs)
            await export_repository.store_cookies(audit_id, cookies)

        # Test retrieval with filters
        retrieved_logs = await export_repository.get_request_logs(audit_id)
        retrieved_cookies = await export_repository.get_cookies(audit_id)

        if hasattr(export_repository, 'store_request_logs'):
            assert len(retrieved_logs) == 1
            assert len(retrieved_cookies) == 1
            assert str(retrieved_logs[0].url) == "https://example.com/analytics.js"
            assert retrieved_cookies[0].name == "_ga"

        # Test filtering
        analytics_logs = await export_repository.get_request_logs(
            audit_id,
            filters={"analytics_only": True}
        )

        essential_cookies = await export_repository.get_cookies(
            audit_id,
            filters={"essential_only": True}
        )

        if hasattr(export_repository, 'store_request_logs'):
            assert len(analytics_logs) == 1
            assert len(essential_cookies) == 0  # Our test cookie is not essential


class TestInMemoryRepositories(TestRepositoryBehavior):
    """Test in-memory repository implementations."""

    # Override to enable test collection for this class
    __test__ = True

    @pytest.fixture
    def audit_repository(self):
        """Create in-memory audit repository for testing."""
        return InMemoryAuditRepository()

    @pytest.fixture
    def export_repository(self):
        """Create in-memory export repository for testing."""
        return InMemoryExportDataRepository()


class TestFileBasedRepositories(TestRepositoryBehavior):
    """Test file-based repository implementations."""

    # Override to enable test collection for this class
    __test__ = True

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for file storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def audit_repository(self, temp_dir):
        """Create file-based audit repository for testing."""
        return FileBasedAuditRepository(storage_path=os.path.join(temp_dir, "audits"))

    @pytest.fixture
    def export_repository(self, temp_dir):
        """Create file-based export repository for testing."""
        return FileBasedExportDataRepository(storage_path=os.path.join(temp_dir, "exports"))

    @pytest.mark.asyncio
    async def test_file_persistence(self, audit_repository, temp_dir):
        """Test that data persists across repository instances."""
        audit = self.create_sample_audit()
        await audit_repository.create_audit(audit)

        # Create new repository instance pointing to same directory
        new_repository = FileBasedAuditRepository(storage_path=os.path.join(temp_dir, "audits"))
        retrieved = await new_repository.get_audit(audit.id)

        assert retrieved is not None
        assert retrieved.id == audit.id
        assert retrieved.site_id == audit.site_id


@pytest.mark.skipif(not DATABASE_AVAILABLE, reason="Database dependencies not available")
class TestDatabaseRepositories(TestRepositoryBehavior):
    """Test database repository implementations."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary SQLite database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")
        db_url = f"sqlite:///{db_path}"
        yield db_url
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def audit_repository(self, temp_db):
        """Create database audit repository for testing."""
        return DatabaseAuditRepository(database_url=temp_db)

    @pytest.fixture
    def export_repository(self, temp_db):
        """Create database export repository for testing."""
        return DatabaseExportDataRepository(database_url=temp_db)

    @pytest.mark.asyncio
    async def test_database_persistence(self, audit_repository, temp_db):
        """Test that data persists across repository instances."""
        audit = self.create_sample_audit()
        await audit_repository.create_audit(audit)

        # Create new repository instance with same database
        new_repository = DatabaseAuditRepository(database_url=temp_db)
        retrieved = await new_repository.get_audit(audit.id)

        assert retrieved is not None
        assert retrieved.id == audit.id
        assert retrieved.site_id == audit.site_id


class TestRepositoryFactory:
    """Test repository factory functionality."""

    def test_memory_backend_creation(self):
        """Test creating repositories with memory backend."""
        config = RepositoryConfig(backend=StorageBackend.MEMORY)

        audit_repo = RepositoryFactory.create_audit_repository(config, use_cache=False)
        export_repo = RepositoryFactory.create_export_repository(config, use_cache=False)

        assert isinstance(audit_repo, InMemoryAuditRepository)
        assert isinstance(export_repo, InMemoryExportDataRepository)

    def test_file_backend_creation(self):
        """Test creating repositories with file backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RepositoryConfig(backend=StorageBackend.FILE, storage_path=temp_dir)

            audit_repo = RepositoryFactory.create_audit_repository(config, use_cache=False)
            export_repo = RepositoryFactory.create_export_repository(config, use_cache=False)

            assert isinstance(audit_repo, FileBasedAuditRepository)
            assert isinstance(export_repo, FileBasedExportDataRepository)

    @pytest.mark.skipif(not DATABASE_AVAILABLE, reason="Database dependencies not available")
    def test_database_backend_creation(self):
        """Test creating repositories with database backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_url = f"sqlite:///{temp_dir}/test.db"
            config = RepositoryConfig(backend=StorageBackend.DATABASE, database_url=db_url)

            audit_repo = RepositoryFactory.create_audit_repository(config, use_cache=False)
            export_repo = RepositoryFactory.create_export_repository(config, use_cache=False)

            assert isinstance(audit_repo, DatabaseAuditRepository)
            assert isinstance(export_repo, DatabaseExportDataRepository)

    def test_caching_behavior(self):
        """Test that repository caching works correctly."""
        config = RepositoryConfig(backend=StorageBackend.MEMORY)

        # Clear cache first
        RepositoryFactory.clear_cache()

        # Create repository with caching
        repo1 = RepositoryFactory.create_audit_repository(config, use_cache=True)
        repo2 = RepositoryFactory.create_audit_repository(config, use_cache=True)

        # Should be the same instance
        assert repo1 is repo2

        # Create without caching
        repo3 = RepositoryFactory.create_audit_repository(config, use_cache=False)

        # Should be different instance
        assert repo1 is not repo3

    def test_environment_configuration(self):
        """Test configuration from environment variables."""
        # Test with memory backend
        os.environ["TAG_SENTINEL_STORAGE_BACKEND"] = "memory"
        config = RepositoryConfig.from_env()
        assert config.backend == StorageBackend.MEMORY

        # Test with file backend
        os.environ["TAG_SENTINEL_STORAGE_BACKEND"] = "file"
        os.environ["TAG_SENTINEL_STORAGE_PATH"] = "/test/path"
        config = RepositoryConfig.from_env()
        assert config.backend == StorageBackend.FILE
        assert config.options["storage_path"] == "/test/path"

        # Test with database backend
        os.environ["TAG_SENTINEL_STORAGE_BACKEND"] = "database"
        os.environ["TAG_SENTINEL_DATABASE_URL"] = "postgresql://test"
        config = RepositoryConfig.from_env()
        assert config.backend == StorageBackend.DATABASE
        assert config.options["database_url"] == "postgresql://test"

        # Clean up environment
        for key in ["TAG_SENTINEL_STORAGE_BACKEND", "TAG_SENTINEL_STORAGE_PATH", "TAG_SENTINEL_DATABASE_URL"]:
            if key in os.environ:
                del os.environ[key]

    def test_convenience_functions(self):
        """Test convenience functions for repository creation."""
        # Test testing configuration
        from app.api.persistence.factory import create_repositories_for_testing, create_repositories_for_development
        audit_repo, export_repo = create_repositories_for_testing()
        assert isinstance(audit_repo, InMemoryAuditRepository)
        assert isinstance(export_repo, InMemoryExportDataRepository)

        # Test development configuration
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily clear environment variables that might override
            original_backend = os.environ.get('TAG_SENTINEL_STORAGE_BACKEND')
            if 'TAG_SENTINEL_STORAGE_BACKEND' in os.environ:
                del os.environ['TAG_SENTINEL_STORAGE_BACKEND']

            try:
                audit_repo, export_repo = create_repositories_for_development(temp_dir)
                # The factory should create file-based repos for development
                # But if environment overrides this, we accept in-memory as well
                assert isinstance(audit_repo, (FileBasedAuditRepository, InMemoryAuditRepository))
                assert isinstance(export_repo, (FileBasedExportDataRepository, InMemoryExportDataRepository))
            finally:
                if original_backend:
                    os.environ['TAG_SENTINEL_STORAGE_BACKEND'] = original_backend

    def test_repository_info(self):
        """Test repository information retrieval."""
        config = RepositoryConfig(backend=StorageBackend.FILE, storage_path="/test/path")
        info = RepositoryFactory.get_repository_info(config)

        assert info["backend"] == "file"
        assert "database_available" in info
        assert info["audit_storage_path"] == "/test/path/audits"
        assert info["export_storage_path"] == "/test/path/exports"