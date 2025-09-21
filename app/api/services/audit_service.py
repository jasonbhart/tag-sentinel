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
from urllib.parse import urlparse, parse_qs

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
from app.api.persistence.repositories import AuditRepository, ExportDataRepository
from app.api.schemas.exports import TagExport, CookieExport
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
        export_repository: Optional[ExportDataRepository] = None,
        runner_service: Optional[RunnerIntegrationService] = None
    ):
        """Initialize the audit service.

        Args:
            base_url: Base URL for generating audit links
            idempotency_window_hours: Time window for idempotency key validity
            max_audit_age_days: Maximum age for audit retention
            repository: Audit repository instance (defaults to in-memory)
            export_repository: Export data repository instance (defaults to in-memory)
            runner_service: Runner integration service (defaults to mock)
        """
        self.base_url = base_url.rstrip('/')
        self.idempotency_window = timedelta(hours=idempotency_window_hours)
        self.max_audit_age = timedelta(days=max_audit_age_days)

        # Use provided repository or default from factory
        self.repository = repository or RepositoryFactory.create_audit_repository()

        # Use provided export repository or default from factory
        self.export_repository = export_repository or RepositoryFactory.create_export_repository()

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

        # Get base audit detail
        audit_detail = persistent_record.to_api_model(self.base_url)

        # Populate detailed report data for completed audits
        if persistent_record.status == AuditStatus.COMPLETED:
            audit_detail = await self._populate_report_data(audit_detail, audit_id)

        return audit_detail

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
        prev_cursor = str(max(0, offset - limit)) if offset > 0 else None

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
            prev_cursor=prev_cursor,
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

    async def _populate_report_data(self, audit_detail: AuditDetail, audit_id: str) -> AuditDetail:
        """Populate detailed report data for a completed audit.

        Args:
            audit_detail: Base audit detail to enhance
            audit_id: Audit identifier

        Returns:
            Enhanced audit detail with report data
        """
        try:
            # Fetch detailed report data from export repository
            pages_data = await self._get_pages_data(audit_id)
            tags_data = await self._get_tags_data(audit_id)
            health_data = await self._get_health_data(audit_id)
            cookies_data = await self._get_cookies_data(audit_id)

            # Create a new AuditDetail with populated report data using model_copy
            return audit_detail.model_copy(update={
                'pages': pages_data,
                'tags': tags_data,
                'health': health_data,
                'duplicates': [],  # TODO: Implement duplicate detection data
                'variables': [],   # TODO: Implement data layer variables data
                'cookies': cookies_data,
                'rules': [],       # TODO: Implement rules violation data
                'privacy_summary': self._generate_privacy_summary(cookies_data)
            })

        except Exception as e:
            logger.warning(f"Failed to populate report data for audit {audit_id}: {e}")
            # Return original audit detail if report data population fails
            return audit_detail

    async def _get_pages_data(self, audit_id: str) -> List[Dict[str, Any]]:
        """Get page-level results data aggregated from real audit data."""
        try:
            # Get real data from export repository
            tags = await self.export_repository.get_tags(audit_id)
            cookies = await self.export_repository.get_cookies(audit_id)
            request_logs = await self.export_repository.get_request_logs(audit_id)

            # Group data by page URL
            page_data = {}

            # Process tags by page
            for tag in tags:
                page_url = str(tag.page_url)
                if page_url not in page_data:
                    # Generate stable, deterministic page ID using UUID5 (full hex for uniqueness)
                    page_id = uuid.uuid5(uuid.NAMESPACE_URL, page_url).hex
                    page_data[page_url] = {
                        "id": f"page_{page_id}",
                        "url": page_url,
                        "status": "completed",
                        "load_time": 0,
                        "tags_count": 0,
                        "issues_count": 0,
                        "cookies_count": 0,
                        "_load_times": [],  # Temporary field for calculations
                        "_has_errors": False
                    }

                page_data[page_url]["tags_count"] += 1

                # Track load times if available
                if hasattr(tag, 'load_time') and tag.load_time:
                    page_data[page_url]["_load_times"].append(tag.load_time)

                # Count issues (tags with errors or validation problems)
                if (hasattr(tag, 'has_errors') and tag.has_errors) or \
                   (hasattr(tag, 'validation_issues') and tag.validation_issues):
                    page_data[page_url]["issues_count"] += 1
                    page_data[page_url]["_has_errors"] = True

            # Process cookies by page
            for cookie in cookies:
                page_url = str(cookie.page_url)
                if page_url not in page_data:
                    # Generate stable, deterministic page ID using UUID5 (full hex for uniqueness)
                    page_id = uuid.uuid5(uuid.NAMESPACE_URL, page_url).hex
                    page_data[page_url] = {
                        "id": f"page_{page_id}",
                        "url": page_url,
                        "status": "completed",
                        "load_time": 0,
                        "tags_count": 0,
                        "issues_count": 0,
                        "cookies_count": 0,
                        "_load_times": [],
                        "_has_errors": False
                    }

                page_data[page_url]["cookies_count"] += 1

            # Process request logs by page for additional timing data
            for request in request_logs:
                page_url = str(request.page_url)
                if page_url not in page_data:
                    # Generate stable, deterministic page ID using UUID5 (full hex for uniqueness)
                    page_id = uuid.uuid5(uuid.NAMESPACE_URL, page_url).hex
                    page_data[page_url] = {
                        "id": f"page_{page_id}",
                        "url": page_url,
                        "status": "completed",
                        "load_time": 0,
                        "tags_count": 0,
                        "issues_count": 0,
                        "cookies_count": 0,
                        "_load_times": [],
                        "_has_errors": False
                    }

                # Add request timing data if available
                if hasattr(request, 'response_time') and request.response_time:
                    page_data[page_url]["_load_times"].append(request.response_time)

                # Check for failed requests
                if hasattr(request, 'status') and request.status in ['failed', 'timeout', 'aborted']:
                    page_data[page_url]["issues_count"] += 1
                    page_data[page_url]["_has_errors"] = True

            # Calculate final metrics and clean up temporary fields
            result = []
            for page_url, data in page_data.items():
                # Calculate average load time
                if data["_load_times"]:
                    data["load_time"] = int(sum(data["_load_times"]) / len(data["_load_times"]))
                else:
                    data["load_time"] = 0

                # Set status based on whether there were errors
                if data["_has_errors"]:
                    data["status"] = "completed_with_issues"
                else:
                    data["status"] = "completed"

                # Remove temporary fields
                del data["_load_times"]
                del data["_has_errors"]

                result.append(data)

            # Sort by URL for consistent ordering
            result.sort(key=lambda x: x["url"])

            logger.debug(f"Retrieved {len(result)} pages for audit {audit_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to get pages data for audit {audit_id}: {e}")
            return []

    async def _get_tags_data(self, audit_id: str) -> List[Dict[str, Any]]:
        """Get tag detection results data."""
        try:
            tags = await self.export_repository.get_tags(audit_id)
            request_logs = await self.export_repository.get_request_logs(audit_id)

            # Build a map of analytics requests per tracking ID to derive event counts
            analytics_requests_by_tracking_id = {}
            for request in request_logs:
                if hasattr(request, 'is_analytics') and request.is_analytics:
                    # Try to extract tracking ID from request URL or headers
                    url = str(request.url)
                    tracking_id = None

                    # Parse URL properly to handle all analytics patterns
                    parsed_url = urlparse(url)
                    query_params = parse_qs(parsed_url.query)

                    # Helper function to extract valid container IDs
                    def extract_container_id(param_values):
                        """Extract valid GTM/Optimize container ID from parameter values."""
                        return next((v.strip() for v in param_values
                                   if v.strip().upper().startswith(('GTM-', 'OPT-'))), None)

                    # Check for GTM container ID first (works for any domain including custom)
                    # Scan all 'id' parameters to handle middleware/CDN scenarios with multiple values
                    if 'id' in query_params:
                        gtm_id = extract_container_id(query_params.get('id', []))
                        if gtm_id:
                            tracking_id = gtm_id

                    # Only continue to other checks if we haven't found a tracking_id yet
                    if not tracking_id:
                        # Look for Google Analytics tracking ID
                        if 'google-analytics.com' in parsed_url.netloc and 'tid' in query_params:
                            tracking_id = query_params['tid'][0]

                        # Look for GTM parameter in analytics requests (Google or custom domains)
                        elif 'gtm' in query_params:
                            gtm_id = extract_container_id(query_params.get('gtm', []))
                            if gtm_id:
                                tracking_id = gtm_id

                        # Fall back to request.tracking_id attribute if present
                        elif hasattr(request, 'tracking_id') and request.tracking_id:
                            tracking_id = request.tracking_id

                    if tracking_id:
                        # Normalize tracking ID to uppercase for consistent mapping
                        normalized_tracking_id = tracking_id.upper()
                        analytics_requests_by_tracking_id[normalized_tracking_id] = analytics_requests_by_tracking_id.get(normalized_tracking_id, 0) + 1

            result = []
            for tag in tags:
                tag_data = {
                    "vendor": tag.vendor,
                    "tag_type": tag.tag_type,
                    "measurement_id": tag.tracking_id or 'unknown',
                    "confidence": "high" if tag.confidence_score >= 0.9 else "medium" if tag.confidence_score >= 0.7 else "low"
                }

                # Only include events_count if we can derive it from real data
                tracking_id = tag.tracking_id
                if tracking_id:
                    # Normalize tracking ID for case-insensitive lookup
                    normalized_tag_id = tracking_id.upper()
                    if normalized_tag_id in analytics_requests_by_tracking_id:
                        tag_data["events_count"] = analytics_requests_by_tracking_id[normalized_tag_id]
                # Note: Omitting events_count when no real data is available instead of showing misleading values

                result.append(tag_data)

            return result
        except Exception as e:
            logger.error(f"Failed to get tags data for audit {audit_id}: {e}")
            return []

    async def _get_health_data(self, audit_id: str) -> Dict[str, Any]:
        """Get performance and health metrics calculated from real audit data."""
        try:
            # Get real data from export repository
            tags = await self.export_repository.get_tags(audit_id)
            cookies = await self.export_repository.get_cookies(audit_id)
            request_logs = await self.export_repository.get_request_logs(audit_id)

            # Calculate load performance based on tag load times
            load_performance = "Unknown"
            if tags:
                tag_load_times = [tag.load_time for tag in tags if hasattr(tag, 'load_time') and tag.load_time]
                if tag_load_times:
                    avg_load_time = sum(tag_load_times) / len(tag_load_times)
                    if avg_load_time < 200:
                        load_performance = "Excellent"
                    elif avg_load_time < 500:
                        load_performance = "Good"
                    elif avg_load_time < 1000:
                        load_performance = "Fair"
                    else:
                        load_performance = "Poor"

            # Calculate error rate from tags and requests
            total_items = len(tags) + len(request_logs)
            error_items = 0

            # Count tag errors
            for tag in tags:
                if (hasattr(tag, 'has_errors') and tag.has_errors) or \
                   (hasattr(tag, 'validation_issues') and tag.validation_issues):
                    error_items += 1

            # Count failed requests
            for request in request_logs:
                if hasattr(request, 'status') and request.status in ['failed', 'timeout', 'aborted']:
                    error_items += 1

            if total_items > 0:
                error_rate_percent = (error_items / total_items) * 100
                error_rate = f"{error_rate_percent:.1f}%"
            else:
                error_rate = "0.0%"

            # Calculate tag coverage (pages with tags vs total audited pages)
            # Try to get the canonical count of audited pages from the audit record
            audit_record = await self.repository.get_audit(audit_id)
            canonical_pages_count = None
            if audit_record and audit_record.pages_count > 0:
                canonical_pages_count = audit_record.pages_count

            # Count pages that actually have tags
            pages_with_tags = set()
            for tag in tags:
                pages_with_tags.add(str(tag.page_url))

            # Count artifact-based pages (which may miss pages with no artifacts)
            artifact_pages = set()
            for tag in tags:
                artifact_pages.add(str(tag.page_url))
            for cookie in cookies:
                artifact_pages.add(str(cookie.page_url))
            for request in request_logs:
                artifact_pages.add(str(request.page_url))

            # Use canonical pages count if available, otherwise fall back to artifact count
            # but acknowledge this limitation
            if canonical_pages_count:
                total_pages_count = canonical_pages_count
                coverage_percent = (len(pages_with_tags) / total_pages_count) * 100
                tag_coverage = f"{coverage_percent:.0f}%"
            elif artifact_pages:
                # Fall back to artifact-based counting but note this may inflate coverage
                coverage_percent = (len(pages_with_tags) / len(artifact_pages)) * 100
                tag_coverage = f"{coverage_percent:.0f}%*"  # Asterisk indicates potential inflation
            else:
                tag_coverage = "0%"

            # Collect issues from real data
            issues = []

            # Add tag-specific issues
            for tag in tags:
                if hasattr(tag, 'has_errors') and tag.has_errors:
                    issues.append({
                        "type": "tag_error",
                        "severity": "high",
                        "description": f"Tag {tag.tag_id} on {tag.page_url} has errors",
                        "page_url": str(tag.page_url),
                        "tag_id": tag.tag_id
                    })

                if hasattr(tag, 'validation_issues') and tag.validation_issues:
                    for issue in tag.validation_issues:
                        issues.append({
                            "type": "validation_issue",
                            "severity": "medium",
                            "description": f"Tag {tag.tag_id}: {issue}",
                            "page_url": str(tag.page_url),
                            "tag_id": tag.tag_id
                        })

            # Add performance issues
            if load_performance in ["Poor", "Fair"]:
                issues.append({
                    "type": "performance_issue",
                    "severity": "medium" if load_performance == "Fair" else "high",
                    "description": f"Poor tag load performance detected (average load time indicates {load_performance.lower()} performance)",
                    "metric": "load_performance",
                    "value": load_performance
                })

            return {
                "load_performance": load_performance,
                "error_rate": error_rate,
                "tag_coverage": tag_coverage,
                "issues": issues
            }
        except Exception as e:
            logger.error(f"Failed to get health data for audit {audit_id}: {e}")
            return {
                "load_performance": "Unknown",
                "error_rate": "Unknown",
                "tag_coverage": "Unknown",
                "issues": []
            }

    async def _get_cookies_data(self, audit_id: str) -> List[Dict[str, Any]]:
        """Get cookie usage analysis data."""
        try:
            cookies = await self.export_repository.get_cookies(audit_id)
            return [
                {
                    "name": cookie.name,
                    "domain": cookie.domain,
                    "category": cookie.category or "unknown",
                    "max_age": cookie.max_age or 0,
                    "privacy_impact": "high" if cookie.category == "marketing" else "medium" if cookie.category == "analytics" else "low"
                }
                for cookie in cookies
            ]
        except Exception as e:
            logger.error(f"Failed to get cookies data for audit {audit_id}: {e}")
            return []

    def _generate_privacy_summary(self, cookies_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate privacy compliance summary from cookie data."""
        try:
            total_cookies = len(cookies_data)
            analytics_cookies = len([c for c in cookies_data if c.get("category") == "analytics"])
            marketing_cookies = len([c for c in cookies_data if c.get("category") == "marketing"])
            functional_cookies = len([c for c in cookies_data if c.get("category") == "functional"])

            return {
                "total_cookies": total_cookies,
                "analytics_cookies": analytics_cookies,
                "marketing_cookies": marketing_cookies,
                "functional_cookies": functional_cookies
            }
        except Exception as e:
            logger.error(f"Failed to generate privacy summary: {e}")
            return {}

    async def populate_sample_export_data_for_testing(self, audit_id: str) -> None:
        """Populate sample export data for testing purposes only.

        This method creates mock export data that the UI can display.
        Only use this for testing - in production, data should come from the audit runner.

        Args:
            audit_id: Audit identifier to populate data for
        """
        try:
            # Only populate if we have an in-memory export repository
            if hasattr(self.export_repository, '_tags') and hasattr(self.export_repository, '_cookies'):
                # Populate sample tags data
                from datetime import datetime, timezone, timedelta
                sample_tags = [
                    TagExport(
                        audit_id=audit_id,
                        page_url="https://example.com/",
                        tag_id="ga4_001",
                        vendor="Google Analytics 4",
                        tag_type="pageview",
                        implementation_method="gtm",
                        detected_at=datetime.now(timezone.utc),
                        confidence_score=0.95,
                        detection_method="network_request_analysis",
                        tracking_id="G-XXXXXXXXXX",
                        parameters={"page_title": "Home Page", "currency": "USD"},
                        load_order=1,
                        load_time=250.0,
                        blocking=False,
                        uses_data_layer=True,
                        data_layer_variables=["pageTitle", "currency"],
                        collects_pii=False,
                        consent_required=True,
                        consent_status="granted",
                        has_errors=False,
                        validation_issues=None,
                        privacy_scenario_detected=True,
                        baseline_scenario_detected=True
                    ),
                    TagExport(
                        audit_id=audit_id,
                        page_url="https://example.com/checkout",
                        tag_id="fb_001",
                        vendor="Facebook Pixel",
                        tag_type="event",
                        implementation_method="script_tag",
                        detected_at=datetime.now(timezone.utc),
                        confidence_score=0.85,
                        detection_method="script_tag_analysis",
                        tracking_id="123456789",
                        parameters={"event": "purchase", "value": 99.99},
                        load_order=2,
                        load_time=350.0,
                        blocking=True,
                        uses_data_layer=False,
                        data_layer_variables=None,
                        collects_pii=True,
                        consent_required=True,
                        consent_status="granted",
                        has_errors=False,
                        validation_issues=None,
                        privacy_scenario_detected=True,
                        baseline_scenario_detected=True
                    )
                ]

                # Populate sample cookies data
                sample_cookies = [
                    CookieExport(
                        audit_id=audit_id,
                        page_url="https://example.com/",
                        name="_ga",
                        domain=".example.com",
                        path="/",
                        value="GA1.2.XXXXXXXXXX",
                        http_only=False,
                        secure=True,
                        same_site="Lax",
                        expires=datetime.now(timezone.utc) + timedelta(days=730),
                        max_age=63072000,
                        discovered_at=datetime.now(timezone.utc),
                        source="javascript",
                        category="analytics",
                        vendor="Google Analytics",
                        purpose="Analytics tracking",
                        gdpr_compliance={"requires_consent": True, "purpose": "Analytics tracking"},
                        is_essential=False,
                        privacy_scenario_detected=True,
                        baseline_scenario_detected=True
                    ),
                    CookieExport(
                        audit_id=audit_id,
                        page_url="https://example.com/",
                        name="_fbp",
                        domain=".example.com",
                        path="/",
                        value="fb.1.XXXXXXXXXX",
                        http_only=False,
                        secure=True,
                        same_site="Lax",
                        expires=datetime.now(timezone.utc) + timedelta(days=365),
                        max_age=7776000,
                        discovered_at=datetime.now(timezone.utc),
                        source="javascript",
                        category="marketing",
                        vendor="Facebook",
                        purpose="Marketing and advertising",
                        gdpr_compliance={"requires_consent": True, "purpose": "Marketing and advertising"},
                        is_essential=False,
                        privacy_scenario_detected=True,
                        baseline_scenario_detected=True
                    )
                ]

                # Atomically check and store data to prevent race conditions
                async with self.export_repository._lock:
                    # Re-check existence inside the lock to prevent race conditions
                    # Use key presence instead of truthiness to preserve legitimate empty results
                    if audit_id in self.export_repository._tags or audit_id in self.export_repository._cookies:
                        logger.info(f"Audit {audit_id} already has export data, skipping sample population")
                        return

                    # Safe to populate since we hold the lock and confirmed no existing data
                    self.export_repository._tags[audit_id] = sample_tags
                    self.export_repository._cookies[audit_id] = sample_cookies

                logger.info(f"Populated sample export data for audit {audit_id}")

        except Exception as e:
            logger.warning(f"Failed to populate sample export data for audit {audit_id}: {e}")