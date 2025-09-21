"""Integration tests for persistence layer components."""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from app.persistence import (
    PersistenceManager,
    PersistenceConfig,
    RetentionConfig,
    PersistentRunner,
    ExportService,
    ExportFormat
)


class TestPersistenceIntegration:
    """Test integration between persistence components."""

    async def test_full_audit_workflow(self, temp_artifact_dir):
        """Test complete audit workflow with persistence."""
        # Configure persistence for testing
        config = PersistenceConfig(
            storage_backend="local",
            storage_config={"base_path": str(temp_artifact_dir)},
            retention=RetentionConfig(
                runs_default=30,
                artifacts_default=14
            )
        )

        persistence_manager = PersistenceManager(config)

        # Create a persistent audit run
        async with PersistentRunner(
            persistence_manager=persistence_manager,
            run_name="Integration Test Run",
            environment="test",
            config={"test_mode": True}
        ) as runner:

            # Store page results
            page_result = await runner.store_page_result(
                url="https://example.com",
                final_url="https://example.com",
                status_code=200,
                success=True,
                load_time_ms=1500,
                metadata={"test": True}
            )

            assert page_result.id is not None
            assert runner.run_id is not None

            # Verify run summary
            summary = await runner.get_run_summary()
            assert summary is not None
            assert summary["run_id"] == runner.run_id
            assert summary["name"] == "Integration Test Run"
            assert summary["environment"] == "test"

        # Verify persistence manager health
        health = await persistence_manager.health_check()
        assert health["database"] is True
        assert health["storage"] is True

        await persistence_manager.close()

    async def test_export_workflow(self, populated_run, temp_artifact_dir):
        """Test export workflow integration."""
        config = PersistenceConfig(
            storage_backend="local",
            storage_config={"base_path": str(temp_artifact_dir)}
        )

        persistence_manager = PersistenceManager(config)

        # Get export service
        async with persistence_manager.database.session() as session:
            from app.persistence.dao import AuditDAO
            dao = AuditDAO(session)
            export_service = ExportService(dao)

            # Test export summary
            summary = await export_service.get_export_summary(populated_run.id)
            assert summary["run_id"] == populated_run.id
            assert "available_exports" in summary

            # Test streaming exports
            chunks = []
            async for chunk in export_service.export_request_logs(
                populated_run.id,
                format=ExportFormat.JSON,
                batch_size=2
            ):
                chunks.append(chunk)

            assert len(chunks) > 0

        await persistence_manager.close()

    async def test_retention_integration(self, sample_run, temp_artifact_dir):
        """Test retention engine integration."""
        config = PersistenceConfig(
            storage_backend="local",
            storage_config={"base_path": str(temp_artifact_dir)},
            retention=RetentionConfig(
                runs_default=1,  # Very short retention for testing
                artifacts_default=1,
                cleanup_enabled=True
            )
        )

        persistence_manager = PersistenceManager(config)

        # Get retention engine
        async with persistence_manager.database.session() as session:
            from app.persistence.dao import AuditDAO
            from app.persistence.retention import RetentionEngine

            dao = AuditDAO(session)
            retention_engine = RetentionEngine(
                dao=dao,
                artifact_store=persistence_manager.artifact_store,
                config=persistence_manager.retention
            )

            # Test retention summary
            summary = await retention_engine.get_retention_summary()
            assert "retention_config" in summary
            assert "current_data" in summary
            assert "expired_data" in summary

            # Test dry-run cleanup
            result = await retention_engine.cleanup_expired_data(dry_run=True)
            assert result.runs_deleted >= 0
            assert result.artifacts_deleted >= 0

        await persistence_manager.close()


class TestErrorHandling:
    """Test error handling in persistence integration."""

    async def test_invalid_storage_backend(self):
        """Test handling of invalid storage backend."""
        config = PersistenceConfig(storage_backend="invalid")

        with pytest.raises(ValueError, match="Unsupported storage backend"):
            config.validate()

    async def test_missing_database_connection(self):
        """Test handling of database connection issues."""
        # This would require mocking database failures
        # For now, we'll test basic configuration validation
        config = PersistenceConfig()
        config.validate()  # Should not raise

    async def test_persistent_runner_without_initialization(self):
        """Test PersistentRunner error handling."""
        config = PersistenceConfig()
        persistence_manager = PersistenceManager(config)

        runner = PersistentRunner(
            persistence_manager=persistence_manager,
            run_name="Test",
            environment="test"
        )

        # Should raise error when not properly initialized
        with pytest.raises(RuntimeError, match="not properly initialized"):
            await runner.store_page_result(
                url="https://example.com",
                final_url="https://example.com",
                status_code=200,
                success=True
            )


class TestConfigurationManagement:
    """Test configuration management."""

    def test_persistence_config_from_environment(self, monkeypatch, temp_artifact_dir):
        """Test creating configuration from environment variables."""
        # Set environment variables
        monkeypatch.setenv("POSTGRES_URL", "postgresql://test:test@localhost/test")
        monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
        monkeypatch.setenv("ARTIFACT_STORAGE_PATH", str(temp_artifact_dir))
        monkeypatch.setenv("RETENTION_RUNS_DAYS", "60")
        monkeypatch.setenv("RETENTION_ARTIFACTS_DAYS", "21")
        monkeypatch.setenv("EXPORT_BATCH_SIZE", "500")

        config = PersistenceConfig.from_environment()

        assert config.storage_backend == "local"
        assert config.storage_config["base_path"] == str(temp_artifact_dir)
        assert config.retention.runs_default == 60
        assert config.retention.artifacts_default == 21
        assert config.export_batch_size == 500

    def test_retention_config_policies(self):
        """Test retention configuration policies."""
        config = RetentionConfig(
            runs_by_environment={
                "production": 365,
                "staging": 30,
                "development": 7
            },
            artifacts_by_type={
                "screenshot": 14,
                "har": 7
            }
        )

        # Test environment-specific retention
        assert config.get_run_retention_days("production") == 365
        assert config.get_run_retention_days("staging") == 30
        assert config.get_run_retention_days("unknown") == config.runs_default

        # Test artifact type-specific retention
        assert config.get_artifact_retention_days("screenshot") == 14
        assert config.get_artifact_retention_days("har") == 7
        assert config.get_artifact_retention_days("unknown") == config.artifacts_default

    async def test_persistence_manager_lifecycle(self, temp_artifact_dir):
        """Test persistence manager lifecycle."""
        config = PersistenceConfig(
            storage_backend="local",
            storage_config={"base_path": str(temp_artifact_dir)}
        )

        manager = PersistenceManager(config)

        # Test properties access
        assert manager.database is not None
        assert manager.artifact_store is not None
        assert manager.retention is not None

        # Test health check
        health = await manager.health_check()
        assert isinstance(health, dict)
        assert "database" in health
        assert "storage" in health

        # Test cleanup
        await manager.close()

    def test_global_persistence_manager(self, temp_artifact_dir):
        """Test global persistence manager functions."""
        from app.persistence.config import (
            get_persistence_manager,
            configure_persistence,
            close_persistence
        )

        # Configure global manager
        config = PersistenceConfig(
            storage_backend="local",
            storage_config={"base_path": str(temp_artifact_dir)}
        )
        configure_persistence(config)

        # Get global manager
        manager = get_persistence_manager()
        assert manager is not None
        assert manager.config.storage_backend == "local"

        # Test that it returns the same instance
        manager2 = get_persistence_manager()
        assert manager is manager2