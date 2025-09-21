"""Tag Sentinel persistence layer.

This package provides comprehensive data persistence capabilities for Tag Sentinel,
including database operations, artifact storage, exports, and retention policies.

Key Components:
- Models: SQLAlchemy ORM models for audit data
- DAO: Data Access Object for database operations
- Storage: Artifact storage backends (local, S3)
- Exports: Streaming export services (JSON, CSV, NDJSON)
- Retention: Automated data cleanup policies
- Config: Configuration management
- Integration: Integration with existing audit components

Usage:
    # Basic setup
    from app.persistence import get_persistence_manager

    persistence = get_persistence_manager()

    # Create a persistent audit run
    from app.persistence.integration import PersistentRunner

    async with PersistentRunner(
        persistence_manager=persistence,
        run_name="My Audit",
        environment="production"
    ) as runner:
        # Store audit results
        await runner.store_page_result(...)
        await runner.store_rule_results(...)

    # Export data
    from app.persistence.exports import ExportService, ExportFormat

    async with persistence.database.session() as session:
        dao = AuditDAO(session)
        export_service = ExportService(dao)

        async for chunk in export_service.export_request_logs(
            run_id=123, format=ExportFormat.JSON
        ):
            print(chunk)
"""

# Core models and database
from .models import (
    Base,
    Run,
    PageResult,
    RequestLog,
    Cookie,
    DataLayerSnapshot,
    RuleFailure,
    Artifact,
    RunStatus,
    PageStatus,
    ArtifactKind,
    SeverityLevel,
)

from .database import DatabaseConfig, get_session, init_database, close_database

# Data access
from .dao import AuditDAO

# Storage backends
from .storage import (
    ArtifactStore,
    LocalArtifactStore,
    S3ArtifactStore,
    ArtifactRef,
    create_artifact_store,
    create_default_artifact_store
)

# Export services
from .exports import ExportService, ExportFormat

# Retention policies
from .retention import RetentionEngine, CleanupResult

# Configuration and management
from .config import (
    PersistenceConfig,
    RetentionConfig,
    PersistenceManager,
    get_persistence_manager,
    configure_persistence,
    close_persistence
)

# Integration helpers
from .integration import PersistentRunner, PersistenceIntegrator

__all__ = [
    # Models
    "Base",
    "Run",
    "PageResult",
    "RequestLog",
    "Cookie",
    "DataLayerSnapshot",
    "RuleFailure",
    "Artifact",
    "RunStatus",
    "PageStatus",
    "ArtifactKind",
    "SeverityLevel",

    # Database
    "DatabaseConfig",
    "get_session",
    "init_database",
    "close_database",

    # Data access
    "AuditDAO",

    # Storage
    "ArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "ArtifactRef",
    "create_artifact_store",
    "create_default_artifact_store",

    # Exports
    "ExportService",
    "ExportFormat",

    # Retention
    "RetentionEngine",
    "CleanupResult",

    # Configuration
    "PersistenceConfig",
    "RetentionConfig",
    "PersistenceManager",
    "get_persistence_manager",
    "configure_persistence",
    "close_persistence",

    # Integration
    "PersistentRunner",
    "PersistenceIntegrator"
]