"""Configuration management for persistence layer."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from pathlib import Path

from .database import DatabaseConfig
from .storage import ArtifactStore, create_artifact_store


@dataclass
class RetentionConfig:
    """Configuration for data retention policies."""

    # Run retention (days)
    runs_default: int = 90
    runs_by_environment: Dict[str, int] = field(default_factory=lambda: {
        "production": 365,
        "staging": 30,
        "development": 7
    })

    # Artifact retention (days)
    artifacts_default: int = 30
    artifacts_by_type: Dict[str, int] = field(default_factory=lambda: {
        "screenshot": 14,
        "har": 7,
        "network_log": 7
    })

    # Cleanup settings
    cleanup_batch_size: int = 100
    cleanup_enabled: bool = True

    def get_run_retention_days(self, environment: str) -> int:
        """Get retention days for runs in specific environment."""
        return self.runs_by_environment.get(environment, self.runs_default)

    def get_artifact_retention_days(self, artifact_type: str) -> int:
        """Get retention days for specific artifact type."""
        return self.artifacts_by_type.get(artifact_type, self.artifacts_default)


@dataclass
class PersistenceConfig:
    """Complete persistence layer configuration."""

    # Database configuration
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Storage backend
    storage_backend: str = "local"
    storage_config: Dict[str, Any] = field(default_factory=dict)

    # Retention policies
    retention: RetentionConfig = field(default_factory=RetentionConfig)

    # Export settings
    export_batch_size: int = 1000
    export_timeout_seconds: int = 300

    # Performance settings
    batch_insert_size: int = 100
    connection_pool_size: int = 10

    @classmethod
    def from_environment(cls) -> "PersistenceConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Database configuration
        config.database = DatabaseConfig()

        # Storage configuration
        config.storage_backend = os.getenv("ARTIFACT_STORAGE_BACKEND", "local").lower()

        if config.storage_backend == "local":
            config.storage_config = {
                "base_path": os.getenv("ARTIFACT_STORAGE_PATH", "./artifacts")
            }
        elif config.storage_backend == "s3":
            config.storage_config = {
                "bucket": os.getenv("ARTIFACT_S3_BUCKET", "tag-sentinel-artifacts"),
                "prefix": os.getenv("ARTIFACT_S3_PREFIX", ""),
                "region": os.getenv("ARTIFACT_S3_REGION", "us-east-1"),
                "endpoint_url": os.getenv("ARTIFACT_S3_ENDPOINT_URL"),
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY")
            }

        # Retention configuration
        config.retention.runs_default = int(os.getenv("RETENTION_RUNS_DAYS", "90"))
        config.retention.artifacts_default = int(os.getenv("RETENTION_ARTIFACTS_DAYS", "30"))
        config.retention.cleanup_enabled = os.getenv("RETENTION_CLEANUP_ENABLED", "true").lower() == "true"

        # Performance settings
        config.export_batch_size = int(os.getenv("EXPORT_BATCH_SIZE", "1000"))
        config.batch_insert_size = int(os.getenv("BATCH_INSERT_SIZE", "100"))
        config.connection_pool_size = int(os.getenv("DB_POOL_SIZE", "10"))

        return config

    def create_artifact_store(self) -> ArtifactStore:
        """Create artifact store instance from configuration."""
        return create_artifact_store(self.storage_backend, **self.storage_config)

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.storage_backend not in ["local", "s3"]:
            raise ValueError(f"Unsupported storage backend: {self.storage_backend}")

        if self.export_batch_size <= 0:
            raise ValueError("Export batch size must be positive")

        if self.batch_insert_size <= 0:
            raise ValueError("Batch insert size must be positive")

        if self.retention.runs_default <= 0:
            raise ValueError("Run retention days must be positive")

        if self.retention.artifacts_default <= 0:
            raise ValueError("Artifact retention days must be positive")


class PersistenceManager:
    """Main manager for persistence layer components."""

    def __init__(self, config: Optional[PersistenceConfig] = None):
        """Initialize persistence manager with configuration."""
        self.config = config or PersistenceConfig.from_environment()
        self.config.validate()

        self._artifact_store: Optional[ArtifactStore] = None

    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.config.database

    @property
    def artifact_store(self) -> ArtifactStore:
        """Get or create artifact store instance."""
        if self._artifact_store is None:
            self._artifact_store = self.config.create_artifact_store()
        return self._artifact_store

    @property
    def retention(self) -> RetentionConfig:
        """Get retention configuration."""
        return self.config.retention

    async def health_check(self) -> Dict[str, bool]:
        """Check health of persistence components."""
        results = {}

        # Check database connectivity
        try:
            results["database"] = await self.database.health_check()
        except Exception:
            results["database"] = False

        # Check artifact store (basic existence test)
        try:
            # For local storage, check if base directory exists
            if self.config.storage_backend == "local":
                base_path = Path(self.config.storage_config.get("base_path", "./artifacts"))
                results["storage"] = base_path.exists()
            else:
                # For S3, we'd need to test bucket access
                # For now, assume it's available
                results["storage"] = True
        except Exception:
            results["storage"] = False

        return results

    async def close(self) -> None:
        """Close all persistence connections."""
        await self.database.close()


# Global persistence manager instance
_persistence_manager: Optional[PersistenceManager] = None


def get_persistence_manager() -> PersistenceManager:
    """Get global persistence manager instance."""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = PersistenceManager()
    return _persistence_manager


def configure_persistence(config: PersistenceConfig) -> None:
    """Configure global persistence manager."""
    global _persistence_manager
    _persistence_manager = PersistenceManager(config)


async def close_persistence() -> None:
    """Close global persistence manager."""
    global _persistence_manager
    if _persistence_manager is not None:
        await _persistence_manager.close()
        _persistence_manager = None