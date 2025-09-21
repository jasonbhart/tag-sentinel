"""Retention policy engine for automatic data cleanup."""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from sqlalchemy import and_, select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Run, PageResult, RequestLog, Cookie, DataLayerSnapshot, RuleFailure, Artifact
from .dao import AuditDAO
from .config import RetentionConfig
from .storage import ArtifactStore


logger = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""

    runs_deleted: int = 0
    artifacts_deleted: int = 0
    storage_files_deleted: int = 0
    storage_files_failed: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def total_deleted(self) -> int:
        """Total number of database records deleted."""
        return self.runs_deleted + self.artifacts_deleted

    @property
    def has_errors(self) -> bool:
        """Whether any errors occurred during cleanup."""
        return len(self.errors) > 0 or self.storage_files_failed > 0


class RetentionEngine:
    """Engine for enforcing data retention policies."""

    def __init__(
        self,
        dao: AuditDAO,
        artifact_store: ArtifactStore,
        config: RetentionConfig
    ):
        """Initialize retention engine.

        Args:
            dao: Database access object
            artifact_store: Artifact storage backend
            config: Retention configuration
        """
        self.dao = dao
        self.artifact_store = artifact_store
        self.config = config

    async def cleanup_expired_data(self, dry_run: bool = False) -> CleanupResult:
        """Clean up expired runs and artifacts.

        Args:
            dry_run: If True, only report what would be deleted without actually deleting

        Returns:
            CleanupResult with details of cleanup operation
        """
        if not self.config.cleanup_enabled:
            logger.info("Cleanup is disabled, skipping retention cleanup")
            return CleanupResult()

        logger.info(f"Starting retention cleanup (dry_run={dry_run})")
        result = CleanupResult()

        try:
            # Clean up expired runs (which cascades to related data)
            run_result = await self._cleanup_expired_runs(dry_run)
            result.runs_deleted = run_result

            # Clean up orphaned artifacts
            artifact_result = await self._cleanup_expired_artifacts(dry_run)
            result.artifacts_deleted = artifact_result[0]
            result.storage_files_deleted = artifact_result[1]
            result.storage_files_failed = artifact_result[2]

            if not dry_run:
                await self.dao.commit()

            logger.info(
                f"Cleanup completed: {result.runs_deleted} runs, "
                f"{result.artifacts_deleted} artifacts, "
                f"{result.storage_files_deleted} storage files deleted"
            )

        except Exception as e:
            error_msg = f"Cleanup failed: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)

            if not dry_run:
                await self.dao.rollback()

        return result

    async def _cleanup_expired_runs(self, dry_run: bool = False) -> int:
        """Clean up runs that have exceeded their retention period."""
        # Get expired runs by environment
        expired_runs = await self._find_expired_runs()

        if not expired_runs:
            logger.info("No expired runs found")
            return 0

        if dry_run:
            logger.info(f"Would delete {len(expired_runs)} expired runs")
            return len(expired_runs)

        # Delete runs in batches (cascading deletes will handle related data)
        deleted_count = 0
        batch_size = self.config.cleanup_batch_size

        for i in range(0, len(expired_runs), batch_size):
            batch = expired_runs[i:i + batch_size]
            run_ids = [run.id for run in batch]

            # Delete batch of runs
            delete_stmt = delete(Run).where(Run.id.in_(run_ids))
            result = await self.dao.session.execute(delete_stmt)
            deleted_count += result.rowcount

            logger.info(f"Deleted {result.rowcount} runs in batch {i//batch_size + 1}")

        return deleted_count

    async def _cleanup_expired_artifacts(self, dry_run: bool = False) -> Tuple[int, int, int]:
        """Clean up artifacts that have exceeded their retention period.

        Returns:
            Tuple of (db_artifacts_deleted, storage_files_deleted, storage_files_failed)
        """
        # Find expired artifacts
        expired_artifacts = await self._find_expired_artifacts()

        if not expired_artifacts:
            logger.info("No expired artifacts found")
            return 0, 0, 0

        if dry_run:
            logger.info(f"Would delete {len(expired_artifacts)} expired artifacts")
            return len(expired_artifacts), 0, 0

        # Delete from storage first, then from database
        storage_deleted = 0
        storage_failed = 0

        for artifact in expired_artifacts:
            try:
                deleted = await self.artifact_store.delete(artifact.path)
                if deleted:
                    storage_deleted += 1
                # Note: If storage delete fails, we might still want to clean up DB record
            except Exception as e:
                logger.warning(f"Failed to delete storage artifact {artifact.path}: {e}")
                storage_failed += 1

        # Delete from database
        artifact_ids = [a.id for a in expired_artifacts]
        delete_stmt = delete(Artifact).where(Artifact.id.in_(artifact_ids))
        result = await self.dao.session.execute(delete_stmt)

        logger.info(
            f"Deleted {result.rowcount} artifact records, "
            f"{storage_deleted} storage files, "
            f"{storage_failed} storage failures"
        )

        return result.rowcount, storage_deleted, storage_failed

    async def _find_expired_runs(self) -> List[Run]:
        """Find runs that have exceeded their retention period."""
        expired_runs = []

        # Check each environment's retention policy
        for environment, retention_days in self.config.runs_by_environment.items():
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

            query = (
                select(Run)
                .where(
                    and_(
                        Run.environment == environment,
                        Run.finished_at.isnot(None),  # Guard against in-flight runs
                        Run.finished_at < cutoff_date
                    )
                )
                .order_by(Run.finished_at)
            )

            result = await self.dao.session.execute(query)
            env_expired = list(result.scalars().all())
            expired_runs.extend(env_expired)

            if env_expired:
                logger.info(
                    f"Found {len(env_expired)} expired runs for environment '{environment}' "
                    f"(retention: {retention_days} days, cutoff: {cutoff_date.date()})"
                )

        # Check runs not in specific environment policies (use default)
        known_environments = set(self.config.runs_by_environment.keys())
        default_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.runs_default)

        query = (
            select(Run)
            .where(
                and_(
                    ~Run.environment.in_(known_environments),
                    Run.finished_at.isnot(None),  # Guard against in-flight runs
                    Run.finished_at < default_cutoff
                )
            )
            .order_by(Run.finished_at)
        )

        result = await self.dao.session.execute(query)
        default_expired = list(result.scalars().all())
        expired_runs.extend(default_expired)

        if default_expired:
            logger.info(
                f"Found {len(default_expired)} expired runs for other environments "
                f"(default retention: {self.config.runs_default} days)"
            )

        return expired_runs

    async def _find_expired_artifacts(self) -> List[Artifact]:
        """Find artifacts that have exceeded their retention period."""
        expired_artifacts = []

        # Check each artifact type's retention policy
        for artifact_type, retention_days in self.config.artifacts_by_type.items():
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

            query = (
                select(Artifact)
                .where(
                    and_(
                        Artifact.kind == artifact_type,
                        Artifact.created_at < cutoff_date
                    )
                )
                .order_by(Artifact.created_at)
            )

            result = await self.dao.session.execute(query)
            type_expired = list(result.scalars().all())
            expired_artifacts.extend(type_expired)

            if type_expired:
                logger.info(
                    f"Found {len(type_expired)} expired artifacts of type '{artifact_type}' "
                    f"(retention: {retention_days} days)"
                )

        # Check artifacts not in specific type policies (use default)
        known_types = set(self.config.artifacts_by_type.keys())
        default_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.artifacts_default)

        query = (
            select(Artifact)
            .where(
                and_(
                    ~Artifact.kind.in_(known_types),
                    Artifact.created_at < default_cutoff
                )
            )
            .order_by(Artifact.created_at)
        )

        result = await self.dao.session.execute(query)
        default_expired = list(result.scalars().all())
        expired_artifacts.extend(default_expired)

        if default_expired:
            logger.info(
                f"Found {len(default_expired)} expired artifacts of other types "
                f"(default retention: {self.config.artifacts_default} days)"
            )

        return expired_artifacts

    async def get_retention_summary(self) -> Dict[str, Any]:
        """Get summary of data subject to retention policies."""
        summary = {
            "retention_config": {
                "cleanup_enabled": self.config.cleanup_enabled,
                "runs_default_days": self.config.runs_default,
                "artifacts_default_days": self.config.artifacts_default,
                "runs_by_environment": dict(self.config.runs_by_environment),
                "artifacts_by_type": dict(self.config.artifacts_by_type)
            },
            "current_data": {},
            "expired_data": {}
        }

        try:
            # Current data counts
            summary["current_data"] = await self._get_current_data_counts()

            # Expired data counts
            expired_runs = await self._find_expired_runs()
            expired_artifacts = await self._find_expired_artifacts()

            summary["expired_data"] = {
                "runs": len(expired_runs),
                "artifacts": len(expired_artifacts)
            }

        except Exception as e:
            logger.error(f"Failed to generate retention summary: {e}")
            summary["error"] = str(e)

        return summary

    async def _get_current_data_counts(self) -> Dict[str, Any]:
        """Get counts of current data in the system."""
        counts = {}

        # Total runs by environment
        query = (
            select(Run.environment, func.count(Run.id))
            .group_by(Run.environment)
        )
        result = await self.dao.session.execute(query)
        counts["runs_by_environment"] = dict(result.all())

        # Total artifacts by type
        query = (
            select(Artifact.kind, func.count(Artifact.id))
            .group_by(Artifact.kind)
        )
        result = await self.dao.session.execute(query)
        counts["artifacts_by_type"] = dict(result.all())

        # Overall totals
        counts["total_runs"] = sum(counts["runs_by_environment"].values())
        counts["total_artifacts"] = sum(counts["artifacts_by_type"].values())

        return counts