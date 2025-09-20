"""Audit management service layer for Tag Sentinel API.

This module provides business logic for audit creation, tracking, and management,
abstracting the integration with audit runners, persistence, and state management.
"""

import asyncio
import logging
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterable

from app.api.schemas import (
    CreateAuditRequest,
    ListAuditsRequest,
    AuditDetail,
    AuditRef,
    AuditList,
    AuditSummary,
    AuditLinks
)
from app.api.schemas.responses import AuditStatus
from app.audit.models.crawl import CrawlConfig
from app.api.persistence.repositories import AuditRepository
from app.api.persistence.factory import RepositoryFactory
from app.api.persistence.models import PersistentAuditRecord
from app.api.runner_integration import RunnerIntegrationService

logger = logging.getLogger(__name__)



@dataclass
class AuditRecord:
    """Internal audit record for service layer operations.

    This represents the internal data structure used by the service layer
    to track audit state and metadata.
    """
    id: str
    site_id: str
    env: str
    status: AuditStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    params: Optional[Dict[str, Any]] = None
    rules_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None
    priority: int = 0
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    progress_percent: Optional[float] = None
    summary: Optional[Dict[str, Any]] = None


class IdempotencyError(Exception):
    """Raised when an idempotent operation is attempted with different parameters."""
    def __init__(self, existing_audit_id: str, message: str = "Duplicate audit creation"):
        self.existing_audit_id = existing_audit_id
        super().__init__(message)


class AuditNotFoundError(Exception):
    """Raised when a requested audit cannot be found."""
    pass


class AuditService:
    """Service layer for audit management operations.

    This service provides high-level business logic for audit operations,
    including creation, tracking, and lifecycle management. It abstracts
    the underlying audit runner integration and persistence concerns.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        idempotency_window_hours: int = 24,
        max_audit_age_days: int = 90,
        repository: Optional[AuditRepository] = None,
        runner_service: Optional[RunnerIntegrationService] = None
    ):
        """Initialize the audit service.

        Args:
            base_url: Base URL for generating audit links
            idempotency_window_hours: Time window for idempotency key validity
            max_audit_age_days: Maximum age for audit retention
            repository: Audit repository instance (defaults to in-memory)
            runner_service: Runner integration service (defaults to mock)
        """
        self.base_url = base_url.rstrip('/')
        self.idempotency_window = timedelta(hours=idempotency_window_hours)
        self.max_audit_age = timedelta(days=max_audit_age_days)

        # Use provided repository or default from factory
        self.repository = repository or RepositoryFactory.create_audit_repository()

        # Use provided runner service or default to mock
        self.runner_service = runner_service or RunnerIntegrationService()


        logger.info(f"AuditService initialized with base_url={base_url}, repository={type(self.repository).__name__}, runner={type(self.runner_service).__name__}")

    async def create_audit(self, request: CreateAuditRequest) -> AuditRef:
        """Create a new audit run.

        Args:
            request: Audit creation request

        Returns:
            Created audit reference

        Raises:
            IdempotencyError: If idempotency key already exists with different params
            ValueError: If parameters are invalid
        """
        logger.info(f"Creating audit for site_id={request.site_id}, env={request.env}")

        # Check idempotency
        existing_audit_id = None
        if request.idempotency_key:
            existing_audit_id = await self._check_idempotency(request)
            if existing_audit_id:
                # Idempotency key matches existing audit - return it as AuditRef
                logger.info(f"Returning existing audit {existing_audit_id} for idempotency key {request.idempotency_key}")
                existing_audit = await self.repository.get_audit(existing_audit_id)
                if not existing_audit:
                    raise AuditNotFoundError(f"Audit {existing_audit_id} not found")
                return self._persistent_to_audit_ref(existing_audit)

        # Generate audit ID
        audit_id = self._generate_audit_id(request.site_id, request.env)

        # Validate and normalize parameters
        validated_params = await self._validate_audit_params(request.params)

        # Create persistent audit record
        now = datetime.now(timezone.utc)
        persistent_record = PersistentAuditRecord(
            id=audit_id,
            site_id=request.site_id,
            env=request.env,
            status=AuditStatus.QUEUED,
            created_at=now,
            updated_at=now,
            params=validated_params,
            rules_path=request.rules_path,
            metadata=request.metadata,
            idempotency_key=request.idempotency_key,
            priority=request.priority or 0
        )

        # Store audit record in repository
        await self.repository.create_audit(persistent_record)

        try:
            # Convert to audit record for dispatch
            audit_record = self._persistent_to_audit_record(persistent_record)

            # Dispatch audit to runner
            await self._dispatch_audit_to_runner(audit_record)

            logger.info(f"Successfully created and dispatched audit {audit_id}")

        except Exception as e:
            # Update audit status on dispatch failure
            persistent_record.status = AuditStatus.FAILED
            persistent_record.error_message = f"Failed to dispatch audit: {str(e)}"
            persistent_record.updated_at = datetime.now(timezone.utc)
            await self.repository.update_audit(persistent_record)

            logger.error(f"Failed to dispatch audit {audit_id}: {e}")
            raise

        return self._persistent_to_audit_ref(persistent_record)

    async def get_audit(self, audit_id: str) -> AuditDetail:
        """Get detailed audit information.

        Args:
            audit_id: Unique audit identifier

        Returns:
            Detailed audit information

        Raises:
            AuditNotFoundError: If audit is not found
        """
        persistent_record = await self.repository.get_audit(audit_id)
        if not persistent_record:
            raise AuditNotFoundError(f"Audit {audit_id} not found")

        return persistent_record.to_api_model(self.base_url)

    async def list_audits(self, request: ListAuditsRequest) -> AuditList:
        """List audits with filtering and pagination.

        Args:
            request: List request with filters and pagination

        Returns:
            Paginated list of audits with metadata
        """
        logger.debug(f"Listing audits with filters: {request.model_dump(exclude_none=True)}")

        # Ensure date filters are timezone-aware
        normalized_request = self._normalize_date_filters(request)

        # Handle cursor-based pagination (simplified implementation)
        offset = 0
        if normalized_request.cursor:
            try:
                offset = int(normalized_request.cursor)
            except ValueError:
                offset = 0

        limit = normalized_request.limit or 20

        # Get audits from repository with filtering and pagination
        page_audits, total_count = await self.repository.list_audits(
            normalized_request, limit=limit, offset=offset
        )

        # Generate audit references
        audit_refs = [self._persistent_to_audit_ref(audit) for audit in page_audits]

        # Prepare pagination metadata
        has_more = offset + limit < total_count
        next_cursor = str(offset + limit) if has_more else None

        # Generate summary stats if requested
        summary_stats = None
        if normalized_request.include_stats:
            # Get all matching audits for stats (without pagination)
            all_audits, _ = await self.repository.list_audits(
                normalized_request, limit=total_count, offset=0
            )
            audit_records = [self._persistent_to_audit_record(audit) for audit in all_audits]
            summary_stats = await self._generate_summary_stats(audit_records)

        return AuditList(
            audits=audit_refs,
            total_count=total_count,
            has_more=has_more,
            next_cursor=next_cursor,
            summary_stats=summary_stats
        )

    async def update_audit_status(
        self,
        audit_id: str,
        status: AuditStatus,
        progress_percent: Optional[float] = None,
        error_message: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update audit status and progress.

        Args:
            audit_id: Unique audit identifier
            status: New audit status
            progress_percent: Progress percentage (0-100)
            error_message: Error message if status is FAILED
            error_details: Additional error details

        Raises:
            AuditNotFoundError: If audit is not found
        """
        persistent_record = await self.repository.get_audit(audit_id)
        if not persistent_record:
            raise AuditNotFoundError(f"Audit {audit_id} not found")

        # Update status and timing
        old_status = persistent_record.status
        persistent_record.status = status
        persistent_record.updated_at = datetime.now(timezone.utc)

        if status == AuditStatus.RUNNING and old_status == AuditStatus.QUEUED:
            persistent_record.started_at = persistent_record.updated_at

        if status in (AuditStatus.COMPLETED, AuditStatus.FAILED):
            persistent_record.finished_at = persistent_record.updated_at

        # Update progress
        if progress_percent is not None:
            persistent_record.progress_percent = max(0, min(100, progress_percent))

        # Update error information
        if error_message:
            persistent_record.error_message = error_message
        if error_details:
            persistent_record.error_details = error_details

        # Save to repository
        await self.repository.update_audit(persistent_record)

        logger.info(f"Updated audit {audit_id} status: {old_status} -> {status}")

    async def _check_idempotency(self, request: CreateAuditRequest) -> Optional[str]:
        """Check idempotency key for duplicate creation.

        Args:
            request: Audit creation request

        Returns:
            Existing audit ID if parameters match, None if no conflict

        Raises:
            IdempotencyError: If key exists with different parameters
        """
        if not request.idempotency_key:
            return None

        # Find existing audit by idempotency key within time window
        existing_audit = await self.repository.find_by_idempotency_key(
            request.idempotency_key,
            max_age_hours=int(self.idempotency_window.total_seconds() / 3600)
        )

        if not existing_audit:
            return None

        # Check if parameters match
        if await self._audit_params_match_persistent(existing_audit, request):
            # Parameters match - return existing audit ID
            return existing_audit.id
        else:
            # Parameters don't match - raise error for conflicting parameters
            raise IdempotencyError(
                existing_audit.id,
                f"Idempotency key {request.idempotency_key} exists with different parameters"
            )

    async def _audit_params_match_persistent(self, existing: PersistentAuditRecord, request: CreateAuditRequest) -> bool:
        """Check if audit parameters match for idempotency."""
        # Validate request parameters to normalize them for comparison
        try:
            validated_request_params = await self._validate_audit_params(request.params)
        except Exception as e:
            logger.debug(f"Failed to validate request params for idempotency check: {e}")
            return False

        return (
            existing.site_id == request.site_id
            and existing.env == request.env
            and existing.params == validated_request_params
            and existing.rules_path == request.rules_path
            and existing.metadata == request.metadata
            and existing.priority == (request.priority or 0)
        )

    async def _validate_audit_params(self, params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate and normalize audit parameters.

        Args:
            params: Raw parameters from request

        Returns:
            Validated and normalized parameters

        Raises:
            ValueError: If parameters are invalid
        """
        if not params:
            return None

        # Validate against CrawlConfig model
        try:
            config = CrawlConfig(**params)
            return config.model_dump(exclude_unset=True)
        except Exception as e:
            raise ValueError(f"Invalid audit parameters: {e}")

    async def _dispatch_audit_to_runner(self, audit_record: AuditRecord) -> None:
        """Dispatch audit to the audit runner.

        Args:
            audit_record: Audit to dispatch

        Raises:
            RuntimeError: If dispatch fails
        """
        logger.info(f"Dispatching audit {audit_record.id} to runner")

        try:
            # Dispatch audit using the runner integration service
            success = await self.runner_service.dispatch_audit(
                audit_id=audit_record.id,
                site_id=audit_record.site_id,
                env=audit_record.env,
                params=audit_record.params or {}
            )

            if not success:
                raise RuntimeError("Runner dispatch returned failure")

            logger.info(f"Successfully dispatched audit {audit_record.id} to runner")

        except Exception as e:
            logger.error(f"Failed to dispatch audit {audit_record.id}: {e}")
            raise RuntimeError(f"Failed to dispatch audit: {str(e)}")

    def _generate_audit_id(self, site_id: str, env: str) -> str:
        """Generate a unique audit identifier.

        Args:
            site_id: Site identifier
            env: Environment

        Returns:
            Unique audit identifier
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(uuid.uuid4())[:8]
        return f"audit_{timestamp}_{site_id}_{env}_{random_suffix}"

    def _normalize_date_filters(self, request: ListAuditsRequest) -> ListAuditsRequest:
        """Normalize date filters to timezone-aware UTC datetimes.

        Args:
            request: Original request with potentially naive datetime filters

        Returns:
            Request with timezone-aware UTC datetime filters
        """
        # Create a copy of the request
        normalized_data = request.model_dump()

        # Normalize date_from
        if request.date_from:
            if request.date_from.tzinfo is None:
                # Naive datetime - assume UTC
                normalized_data['date_from'] = request.date_from.replace(tzinfo=timezone.utc)
            else:
                # Already timezone-aware - convert to UTC
                normalized_data['date_from'] = request.date_from.astimezone(timezone.utc)

        # Normalize date_to
        if request.date_to:
            if request.date_to.tzinfo is None:
                # Naive datetime - assume UTC
                normalized_data['date_to'] = request.date_to.replace(tzinfo=timezone.utc)
            else:
                # Already timezone-aware - convert to UTC
                normalized_data['date_to'] = request.date_to.astimezone(timezone.utc)

        return ListAuditsRequest(**normalized_data)

    def _persistent_to_audit_record(self, persistent: PersistentAuditRecord) -> AuditRecord:
        """Convert PersistentAuditRecord to AuditRecord for backwards compatibility."""
        return AuditRecord(
            id=persistent.id,
            site_id=persistent.site_id,
            env=persistent.env,
            status=persistent.status,
            created_at=persistent.created_at,
            updated_at=persistent.updated_at,
            started_at=persistent.started_at,
            finished_at=persistent.finished_at,
            params=persistent.params,
            rules_path=persistent.rules_path,
            metadata=persistent.metadata,
            idempotency_key=persistent.idempotency_key,
            priority=persistent.priority,
            error_message=persistent.error_message,
            error_details=persistent.error_details,
            progress_percent=persistent.progress_percent,
            summary=persistent.summary
        )

    def _persistent_to_audit_ref(self, persistent: PersistentAuditRecord) -> AuditRef:
        """Convert PersistentAuditRecord to AuditRef."""
        return AuditRef(
            id=persistent.id,
            site_id=persistent.site_id,
            env=persistent.env,
            status=persistent.status.value,
            created_at=persistent.created_at,
            updated_at=persistent.updated_at
        )



    async def _generate_summary_stats(self, audits: List[AuditRecord]) -> Dict[str, Any]:
        """Generate summary statistics for a list of audits."""
        if not audits:
            return {}

        total_audits = len(audits)
        completed_audits = [a for a in audits if a.status == AuditStatus.COMPLETED]
        failed_audits = [a for a in audits if a.status == AuditStatus.FAILED]

        # Calculate average duration for completed audits
        avg_duration = 0.0
        if completed_audits:
            durations = []
            for audit in completed_audits:
                if audit.started_at and audit.finished_at:
                    duration = (audit.finished_at - audit.started_at).total_seconds()
                    durations.append(duration)
            if durations:
                avg_duration = sum(durations) / len(durations)

        return {
            "total_audits": total_audits,
            "completed_audits": len(completed_audits),
            "failed_audits": len(failed_audits),
            "success_rate": (len(completed_audits) / total_audits * 100) if total_audits > 0 else 0,
            "avg_duration_seconds": avg_duration,
        }