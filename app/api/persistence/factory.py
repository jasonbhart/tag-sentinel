"""Repository factory for Tag Sentinel API.

This module provides factory functions to create repository instances
based on configuration settings, supporting multiple storage backends.
"""

import logging
import os
from typing import Optional, Dict, Any
from enum import Enum

from .repositories import AuditRepository, ExportDataRepository
from .repositories import InMemoryAuditRepository, InMemoryExportDataRepository
from .file_repositories import FileBasedAuditRepository, FileBasedExportDataRepository

try:
    from .database_repositories import DatabaseAuditRepository, DatabaseExportDataRepository
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Supported storage backends."""
    MEMORY = "memory"
    FILE = "file"
    DATABASE = "database"


class RepositoryConfig:
    """Configuration for repository creation."""

    def __init__(
        self,
        backend: StorageBackend = StorageBackend.MEMORY,
        **kwargs
    ):
        """Initialize repository configuration.

        Args:
            backend: Storage backend type
            **kwargs: Backend-specific configuration options

        Backend-specific options:
        - file: storage_path (str) - Directory for file storage
        - database: database_url (str) - Database connection URL
        """
        self.backend = backend
        self.options = kwargs

    @classmethod
    def from_env(cls) -> "RepositoryConfig":
        """Create configuration from environment variables.

        Environment variables:
        - TAG_SENTINEL_STORAGE_BACKEND: Backend type (memory, file, database)
        - TAG_SENTINEL_STORAGE_PATH: File storage path (for file backend)
        - TAG_SENTINEL_DATABASE_URL: Database URL (for database backend)
        """
        backend_str = os.getenv("TAG_SENTINEL_STORAGE_BACKEND", "memory").lower()

        try:
            backend = StorageBackend(backend_str)
        except ValueError:
            logger.warning(f"Invalid storage backend '{backend_str}', defaulting to memory")
            backend = StorageBackend.MEMORY

        options = {}

        if backend == StorageBackend.FILE:
            storage_path = os.getenv("TAG_SENTINEL_STORAGE_PATH", "./data")
            options["storage_path"] = storage_path

        elif backend == StorageBackend.DATABASE:
            database_url = os.getenv(
                "TAG_SENTINEL_DATABASE_URL",
                "sqlite:///./data/tag_sentinel.db"
            )
            options["database_url"] = database_url

        return cls(backend=backend, **options)

    @classmethod
    def for_testing(cls) -> "RepositoryConfig":
        """Create configuration optimized for testing."""
        return cls(backend=StorageBackend.MEMORY)

    @classmethod
    def for_development(cls, storage_path: str = "./data") -> "RepositoryConfig":
        """Create configuration for development with file storage."""
        return cls(backend=StorageBackend.FILE, storage_path=storage_path)

    @classmethod
    def for_production(cls, database_url: str) -> "RepositoryConfig":
        """Create configuration for production with database storage."""
        if not DATABASE_AVAILABLE:
            logger.warning("Database backend not available, falling back to file storage")
            return cls.for_development()

        return cls(backend=StorageBackend.DATABASE, database_url=database_url)


class RepositoryFactory:
    """Factory for creating repository instances."""

    _audit_repository_cache: Optional[AuditRepository] = None
    _export_repository_cache: Optional[ExportDataRepository] = None

    @classmethod
    def create_audit_repository(
        self,
        config: Optional[RepositoryConfig] = None,
        use_cache: bool = True
    ) -> AuditRepository:
        """Create audit repository instance.

        Args:
            config: Repository configuration (defaults to environment-based config)
            use_cache: Whether to cache and reuse repository instance

        Returns:
            Configured audit repository instance
        """
        if use_cache and self._audit_repository_cache is not None:
            return self._audit_repository_cache

        if config is None:
            config = RepositoryConfig.from_env()

        logger.info(f"Creating audit repository with backend: {config.backend}")

        if config.backend == StorageBackend.MEMORY:
            repository = InMemoryAuditRepository()

        elif config.backend == StorageBackend.FILE:
            storage_path = config.options.get("storage_path", "./data/audits")
            repository = FileBasedAuditRepository(storage_path=storage_path)

        elif config.backend == StorageBackend.DATABASE:
            if not DATABASE_AVAILABLE:
                logger.error("Database backend requested but not available, falling back to file")
                storage_path = config.options.get("storage_path", "./data/audits")
                repository = FileBasedAuditRepository(storage_path=storage_path)
            else:
                database_url = config.options.get("database_url", "sqlite:///./data/tag_sentinel.db")
                repository = DatabaseAuditRepository(database_url=database_url)

        else:
            raise ValueError(f"Unknown storage backend: {config.backend}")

        if use_cache:
            self._audit_repository_cache = repository

        return repository

    @classmethod
    def create_export_repository(
        self,
        config: Optional[RepositoryConfig] = None,
        use_cache: bool = True
    ) -> ExportDataRepository:
        """Create export data repository instance.

        Args:
            config: Repository configuration (defaults to environment-based config)
            use_cache: Whether to cache and reuse repository instance

        Returns:
            Configured export data repository instance
        """
        if use_cache and self._export_repository_cache is not None:
            return self._export_repository_cache

        if config is None:
            config = RepositoryConfig.from_env()

        logger.info(f"Creating export repository with backend: {config.backend}")

        if config.backend == StorageBackend.MEMORY:
            repository = InMemoryExportDataRepository()

        elif config.backend == StorageBackend.FILE:
            storage_path = config.options.get("storage_path", "./data/exports")
            repository = FileBasedExportDataRepository(storage_path=storage_path)

        elif config.backend == StorageBackend.DATABASE:
            if not DATABASE_AVAILABLE:
                logger.error("Database backend requested but not available, falling back to file")
                storage_path = config.options.get("storage_path", "./data/exports")
                repository = FileBasedExportDataRepository(storage_path=storage_path)
            else:
                database_url = config.options.get("database_url", "sqlite:///./data/tag_sentinel.db")
                repository = DatabaseExportDataRepository(database_url=database_url)

        else:
            raise ValueError(f"Unknown storage backend: {config.backend}")

        if use_cache:
            self._export_repository_cache = repository

        return repository

    @classmethod
    def clear_cache(self) -> None:
        """Clear cached repository instances."""
        self._audit_repository_cache = None
        self._export_repository_cache = None

    @classmethod
    def get_repository_info(self, config: Optional[RepositoryConfig] = None) -> Dict[str, Any]:
        """Get information about the configured repositories.

        Args:
            config: Repository configuration to inspect

        Returns:
            Dictionary with repository configuration details
        """
        if config is None:
            config = RepositoryConfig.from_env()

        info = {
            "backend": config.backend.value,
            "database_available": DATABASE_AVAILABLE,
            "options": config.options
        }

        # Add backend-specific information
        if config.backend == StorageBackend.FILE:
            audit_path = config.options.get("storage_path", "./data") + "/audits"
            export_path = config.options.get("storage_path", "./data") + "/exports"
            info["audit_storage_path"] = audit_path
            info["export_storage_path"] = export_path

        elif config.backend == StorageBackend.DATABASE:
            info["database_url"] = config.options.get("database_url", "sqlite:///./data/tag_sentinel.db")

        return info


# Convenience functions for common use cases

def create_repositories_for_testing() -> tuple[AuditRepository, ExportDataRepository]:
    """Create repository instances optimized for testing.

    Returns:
        Tuple of (audit_repository, export_repository)
    """
    config = RepositoryConfig.for_testing()
    return (
        RepositoryFactory.create_audit_repository(config, use_cache=False),
        RepositoryFactory.create_export_repository(config, use_cache=False)
    )


def create_repositories_for_development(storage_path: str = "./data") -> tuple[AuditRepository, ExportDataRepository]:
    """Create repository instances for development.

    Args:
        storage_path: Base path for file storage

    Returns:
        Tuple of (audit_repository, export_repository)
    """
    config = RepositoryConfig.for_development(storage_path)
    return (
        RepositoryFactory.create_audit_repository(config),
        RepositoryFactory.create_export_repository(config)
    )


def create_repositories_for_production(database_url: str) -> tuple[AuditRepository, ExportDataRepository]:
    """Create repository instances for production.

    Args:
        database_url: Database connection URL

    Returns:
        Tuple of (audit_repository, export_repository)
    """
    config = RepositoryConfig.for_production(database_url)
    return (
        RepositoryFactory.create_audit_repository(config),
        RepositoryFactory.create_export_repository(config)
    )


def get_default_repositories() -> tuple[AuditRepository, ExportDataRepository]:
    """Get repository instances using default configuration from environment.

    Returns:
        Tuple of (audit_repository, export_repository)
    """
    return (
        RepositoryFactory.create_audit_repository(),
        RepositoryFactory.create_export_repository()
    )