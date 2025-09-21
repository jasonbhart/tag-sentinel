"""Tests for retention policy engine."""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from app.persistence.retention import RetentionEngine, CleanupResult
from app.persistence.config import RetentionConfig
from app.persistence.dao import AuditDAO
from app.persistence.storage import LocalArtifactStore


class TestRetentionEngine:
    """Test retention policy engine."""

    @pytest.fixture
    async def retention_engine(self, dao: AuditDAO, artifact_store: LocalArtifactStore) -> RetentionEngine:
        """Create retention engine for testing."""
        config = RetentionConfig(
            runs_default=30,
            runs_by_environment={
                "production": 90,
                "staging": 14,
                "development": 3
            },
            artifacts_default=14,
            artifacts_by_type={
                "screenshot": 7,
                "har": 3,
                "network_log": 1
            },
            cleanup_enabled=True,
            cleanup_batch_size=10
        )

        return RetentionEngine(
            dao=dao,
            artifact_store=artifact_store,
            config=config
        )

    async def test_retention_config(self, retention_engine: RetentionEngine):
        """Test retention configuration."""
        config = retention_engine.config

        # Test environment-specific retention
        assert config.get_run_retention_days("production") == 90
        assert config.get_run_retention_days("staging") == 14
        assert config.get_run_retention_days("development") == 3
        assert config.get_run_retention_days("unknown") == 30

        # Test artifact type-specific retention
        assert config.get_artifact_retention_days("screenshot") == 7
        assert config.get_artifact_retention_days("har") == 3
        assert config.get_artifact_retention_days("network_log") == 1
        assert config.get_artifact_retention_days("unknown") == 14

    async def test_cleanup_disabled(self, dao: AuditDAO, artifact_store: LocalArtifactStore):
        """Test cleanup when disabled."""
        config = RetentionConfig(cleanup_enabled=False)
        engine = RetentionEngine(dao, artifact_store, config)

        result = await engine.cleanup_expired_data()

        assert result.runs_deleted == 0
        assert result.artifacts_deleted == 0
        assert result.storage_files_deleted == 0

    async def test_find_expired_runs(self, retention_engine: RetentionEngine, dao: AuditDAO):
        """Test finding expired runs."""
        # Create runs with different ages and environments
        old_production_run = await dao.create_run(
            name="Old Production Run",
            environment="production",
            start_url="https://prod.example.com",
            status="completed",
            started_at=datetime.now(timezone.utc) - timedelta(days=100),
            completed_at=datetime.now(timezone.utc) - timedelta(days=100)
        )

        recent_production_run = await dao.create_run(
            name="Recent Production Run",
            environment="production",
            start_url="https://prod.example.com",
            status="completed",
            started_at=datetime.now(timezone.utc) - timedelta(days=30),
            completed_at=datetime.now(timezone.utc) - timedelta(days=30)
        )

        old_dev_run = await dao.create_run(
            name="Old Dev Run",
            environment="development",
            start_url="https://dev.example.com",
            status="completed",
            started_at=datetime.now(timezone.utc) - timedelta(days=5),
            completed_at=datetime.now(timezone.utc) - timedelta(days=5)
        )

        await dao.commit()

        # Find expired runs
        expired_runs = await retention_engine._find_expired_runs()

        # Old production run should be expired (100 days > 90 day retention)
        # Old dev run should be expired (5 days > 3 day retention)
        # Recent production run should not be expired (30 days < 90 day retention)
        expired_ids = [run.id for run in expired_runs]

        assert old_production_run.id in expired_ids
        assert old_dev_run.id in expired_ids
        assert recent_production_run.id not in expired_ids

    async def test_find_expired_artifacts(self, retention_engine: RetentionEngine, dao: AuditDAO, sample_run):
        """Test finding expired artifacts."""
        # Create artifacts with different ages and types
        old_screenshot = await dao.create_artifact(
            run_id=sample_run.id,
            type="screenshot",
            path="old_screenshot.png",
            checksum="abc123",
            size_bytes=1024,
            created_at=datetime.now(timezone.utc) - timedelta(days=10)
        )

        recent_screenshot = await dao.create_artifact(
            run_id=sample_run.id,
            type="screenshot",
            path="recent_screenshot.png",
            checksum="def456",
            size_bytes=1024,
            created_at=datetime.now(timezone.utc) - timedelta(days=3)
        )

        old_har = await dao.create_artifact(
            run_id=sample_run.id,
            type="har",
            path="old.har",
            checksum="ghi789",
            size_bytes=2048,
            created_at=datetime.now(timezone.utc) - timedelta(days=5)
        )

        await dao.commit()

        # Find expired artifacts
        expired_artifacts = await retention_engine._find_expired_artifacts()

        # Old screenshot should be expired (10 days > 7 day retention)
        # Old HAR should be expired (5 days > 3 day retention)
        # Recent screenshot should not be expired (3 days < 7 day retention)
        expired_ids = [artifact.id for artifact in expired_artifacts]

        assert old_screenshot.id in expired_ids
        assert old_har.id in expired_ids
        assert recent_screenshot.id not in expired_ids

    async def test_cleanup_dry_run(self, retention_engine: RetentionEngine, dao: AuditDAO):
        """Test dry run cleanup."""
        # Create an expired run
        expired_run = await dao.create_run(
            name="Expired Run",
            environment="development",
            start_url="https://example.com",
            status="completed",
            started_at=datetime.now(timezone.utc) - timedelta(days=10),
            completed_at=datetime.now(timezone.utc) - timedelta(days=10)
        )
        await dao.commit()

        # Run dry-run cleanup
        result = await retention_engine.cleanup_expired_data(dry_run=True)

        # Should report what would be deleted but not actually delete
        assert result.runs_deleted >= 1

        # Verify run still exists
        retrieved_run = await dao.get_run(expired_run.id)
        assert retrieved_run is not None

    async def test_cleanup_with_storage_artifacts(
        self,
        retention_engine: RetentionEngine,
        dao: AuditDAO,
        sample_run,
        artifact_store: LocalArtifactStore
    ):
        """Test cleanup that includes storage artifacts."""
        # Create artifact in storage
        artifact_path = "test/expired_artifact.txt"
        content = b"This will be deleted"

        artifact_ref = await artifact_store.put(content, artifact_path)

        # Create expired artifact record
        expired_artifact = await dao.create_artifact(
            run_id=sample_run.id,
            type="test_file",
            path=artifact_path,
            checksum=artifact_ref.checksum,
            size_bytes=len(content),
            created_at=datetime.now(timezone.utc) - timedelta(days=20)
        )
        await dao.commit()

        # Verify artifact exists in storage
        assert await artifact_store.exists(artifact_path)

        # Run cleanup
        result = await retention_engine.cleanup_expired_data(dry_run=False)

        # Should have deleted the artifact
        assert result.artifacts_deleted >= 1
        assert result.storage_files_deleted >= 1

        # Verify artifact removed from storage
        assert not await artifact_store.exists(artifact_path)

    async def test_cleanup_storage_errors(
        self,
        retention_engine: RetentionEngine,
        dao: AuditDAO,
        sample_run
    ):
        """Test cleanup handling of storage errors."""
        # Create artifact record for non-existent storage file
        expired_artifact = await dao.create_artifact(
            run_id=sample_run.id,
            type="test_file",
            path="nonexistent/file.txt",
            checksum="abc123",
            size_bytes=1024,
            created_at=datetime.now(timezone.utc) - timedelta(days=20)
        )
        await dao.commit()

        # Run cleanup
        result = await retention_engine.cleanup_expired_data(dry_run=False)

        # Should still delete database record even if storage file doesn't exist
        assert result.artifacts_deleted >= 1
        assert result.storage_files_failed == 0  # LocalArtifactStore returns False, not error

    async def test_retention_summary(self, retention_engine: RetentionEngine, populated_run):
        """Test retention summary generation."""
        summary = await retention_engine.get_retention_summary()

        assert "retention_config" in summary
        assert "current_data" in summary
        assert "expired_data" in summary

        # Check retention config
        config = summary["retention_config"]
        assert config["cleanup_enabled"] is True
        assert config["runs_default_days"] == 30
        assert config["artifacts_default_days"] == 14

        # Check current data counts
        current_data = summary["current_data"]
        assert "total_runs" in current_data
        assert "total_artifacts" in current_data
        assert "runs_by_environment" in current_data
        assert "artifacts_by_type" in current_data

        # Check expired data counts
        expired_data = summary["expired_data"]
        assert "runs" in expired_data
        assert "artifacts" in expired_data
        assert isinstance(expired_data["runs"], int)
        assert isinstance(expired_data["artifacts"], int)

    async def test_cleanup_result_properties(self):
        """Test CleanupResult data class properties."""
        result = CleanupResult(
            runs_deleted=5,
            artifacts_deleted=10,
            storage_files_deleted=8,
            storage_files_failed=2,
            errors=["Error 1", "Error 2"]
        )

        assert result.total_deleted == 15  # runs + artifacts
        assert result.has_errors is True  # has storage failures and errors

        # Test without errors
        clean_result = CleanupResult(
            runs_deleted=3,
            artifacts_deleted=7,
            storage_files_deleted=7,
            storage_files_failed=0
        )

        assert clean_result.total_deleted == 10
        assert clean_result.has_errors is False

    async def test_cleanup_batch_processing(self, retention_engine: RetentionEngine, dao: AuditDAO):
        """Test cleanup processes items in batches."""
        # Create multiple expired runs (more than batch size)
        batch_size = retention_engine.config.cleanup_batch_size
        num_runs = batch_size + 5

        created_runs = []
        for i in range(num_runs):
            run = await dao.create_run(
                name=f"Expired Run {i}",
                environment="development",
                start_url=f"https://example.com/{i}",
                status="completed",
                started_at=datetime.now(timezone.utc) - timedelta(days=10),
                completed_at=datetime.now(timezone.utc) - timedelta(days=10)
            )
            created_runs.append(run)

        await dao.commit()

        # Run cleanup
        result = await retention_engine.cleanup_expired_data(dry_run=False)

        # Should have deleted all expired runs
        assert result.runs_deleted >= num_runs

        # Verify runs are actually deleted
        for run in created_runs:
            retrieved_run = await dao.get_run(run.id)
            assert retrieved_run is None