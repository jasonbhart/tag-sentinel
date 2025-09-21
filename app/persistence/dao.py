"""Data Access Objects for Tag Sentinel persistence layer.

This module provides high-level data access operations for audit runs,
page results, and related entities with transaction management and
optimized queries.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Sequence
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from .models import (
    Run, PageResult, RequestLog, Cookie, DataLayerSnapshot,
    RuleFailure, Artifact, RunStatus, PageStatus, SeverityLevel
)
from .database import db_config


class AuditDAO:
    """Data Access Object for audit-related operations."""

    def __init__(self, session: AsyncSession):
        """Initialize DAO with database session."""
        self.session = session

    # ============= Run Operations =============

    async def create_run(
        self,
        site_id: str,
        environment: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Run:
        """Create a new audit run."""
        run = Run(
            site_id=site_id,
            environment=environment,
            status=RunStatus.PENDING,
            config_json=config
        )

        self.session.add(run)
        await self.session.flush()
        return run

    async def get_run_by_id(self, run_id: int) -> Optional[Run]:
        """Get run by ID with basic info."""
        result = await self.session.execute(
            select(Run).where(Run.id == run_id)
        )
        return result.scalar_one_or_none()

    async def get_run_with_results(self, run_id: int) -> Optional[Run]:
        """Get run with all related page results loaded."""
        result = await self.session.execute(
            select(Run)
            .options(selectinload(Run.page_results))
            .where(Run.id == run_id)
        )
        return result.scalar_one_or_none()

    async def get_run_with_full_details(self, run_id: int) -> Optional[Run]:
        """Get run with all related data loaded."""
        result = await self.session.execute(
            select(Run)
            .options(
                selectinload(Run.page_results).selectinload(PageResult.request_logs),
                selectinload(Run.page_results).selectinload(PageResult.cookies),
                selectinload(Run.page_results).selectinload(PageResult.datalayer_snapshots),
                selectinload(Run.rule_failures),
                selectinload(Run.artifacts)
            )
            .where(Run.id == run_id)
        )
        return result.scalar_one_or_none()

    async def list_runs(
        self,
        site_id: Optional[str] = None,
        environment: Optional[str] = None,
        status: Optional[RunStatus] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "started_at",
        order_desc: bool = True
    ) -> List[Run]:
        """List runs with optional filtering and pagination."""
        query = select(Run)

        # Apply filters
        conditions = []
        if site_id:
            conditions.append(Run.site_id == site_id)
        if environment:
            conditions.append(Run.environment == environment)
        if status:
            conditions.append(Run.status == status)

        if conditions:
            query = query.where(and_(*conditions))

        # Apply ordering
        order_column = getattr(Run, order_by, Run.started_at)
        if order_desc:
            query = query.order_by(desc(order_column))
        else:
            query = query.order_by(asc(order_column))

        # Apply pagination
        query = query.offset(offset).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_run_status(
        self,
        run_id: int,
        status: RunStatus,
        error_message: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update run status and optionally set completion time."""
        run = await self.get_run_by_id(run_id)
        if not run:
            return False

        run.status = status
        if error_message:
            run.error_message = error_message
        if summary:
            run.summary_json = summary

        # Set finished_at for terminal states
        if status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
            run.finished_at = func.now()

        return True

    async def get_run_statistics(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive statistics for a run."""
        run = await self.get_run_by_id(run_id)
        if not run:
            return None

        # Get page result counts
        page_stats = await self.session.execute(
            select(
                func.count(PageResult.id).label('total_pages'),
                func.count().filter(PageResult.status == PageStatus.SUCCESS).label('successful_pages'),
                func.count().filter(PageResult.status == PageStatus.FAILED).label('failed_pages'),
                func.avg(PageResult.load_time_ms).label('avg_load_time')
            ).where(PageResult.run_id == run_id)
        )
        page_result = page_stats.first()

        # Get request log counts
        request_stats = await self.session.execute(
            select(
                func.count(RequestLog.id).label('total_requests'),
                func.count().filter(RequestLog.success == True).label('successful_requests')
            )
            .select_from(RequestLog)
            .join(PageResult)
            .where(PageResult.run_id == run_id)
        )
        request_result = request_stats.first()

        # Get cookie counts
        cookie_stats = await self.session.execute(
            select(
                func.count(Cookie.id).label('total_cookies'),
                func.count().filter(Cookie.first_party == True).label('first_party_cookies'),
                func.count().filter(Cookie.first_party == False).label('third_party_cookies')
            )
            .select_from(Cookie)
            .join(PageResult)
            .where(PageResult.run_id == run_id)
        )
        cookie_result = cookie_stats.first()

        # Get rule failure counts by severity
        failure_stats = await self.session.execute(
            select(
                RuleFailure.severity,
                func.count(RuleFailure.id).label('count')
            )
            .where(RuleFailure.run_id == run_id)
            .group_by(RuleFailure.severity)
        )
        failure_counts = {row.severity: row.count for row in failure_stats.all()}

        return {
            'run_id': run_id,
            'status': run.status,
            'started_at': run.started_at,
            'finished_at': run.finished_at,
            'duration_seconds': run.duration_seconds,
            'pages': {
                'total': page_result.total_pages or 0,
                'successful': page_result.successful_pages or 0,
                'failed': page_result.failed_pages or 0,
                'avg_load_time_ms': float(page_result.avg_load_time) if page_result.avg_load_time else None
            },
            'requests': {
                'total': request_result.total_requests or 0,
                'successful': request_result.successful_requests or 0
            },
            'cookies': {
                'total': cookie_result.total_cookies or 0,
                'first_party': cookie_result.first_party_cookies or 0,
                'third_party': cookie_result.third_party_cookies or 0
            },
            'rule_failures': {
                'total': sum(failure_counts.values()),
                'by_severity': failure_counts
            }
        }

    # ============= Page Result Operations =============

    async def create_page_result(
        self,
        run_id: int,
        url: str,
        status: PageStatus = PageStatus.SUCCESS,
        final_url: Optional[str] = None,
        title: Optional[str] = None,
        load_time_ms: Optional[float] = None,
        timings: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
        capture_error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> PageResult:
        """Create a new page result."""
        page_result = PageResult(
            run_id=run_id,
            url=url,
            final_url=final_url,
            title=title,
            status=status,
            load_time_ms=load_time_ms,
            timings_json=timings,
            errors_json=errors,
            capture_error=capture_error,
            metrics_json=metrics
        )

        self.session.add(page_result)
        await self.session.flush()
        return page_result

    async def get_page_result_by_id(self, page_result_id: int) -> Optional[PageResult]:
        """Get page result by ID."""
        result = await self.session.execute(
            select(PageResult).where(PageResult.id == page_result_id)
        )
        return result.scalar_one_or_none()

    async def get_page_result_with_details(self, page_result_id: int) -> Optional[PageResult]:
        """Get page result with all related data loaded."""
        result = await self.session.execute(
            select(PageResult)
            .options(
                selectinload(PageResult.request_logs),
                selectinload(PageResult.cookies),
                selectinload(PageResult.datalayer_snapshots)
            )
            .where(PageResult.id == page_result_id)
        )
        return result.scalar_one_or_none()

    async def list_page_results_for_run(
        self,
        run_id: int,
        status: Optional[PageStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[PageResult]:
        """List page results for a run with optional status filtering."""
        query = select(PageResult).where(PageResult.run_id == run_id)

        if status:
            query = query.where(PageResult.status == status)

        query = query.order_by(PageResult.capture_time).offset(offset).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    # ============= Batch Operations =============

    async def bulk_create_request_logs(
        self,
        page_result_id: int,
        request_logs: List[Dict[str, Any]]
    ) -> List[RequestLog]:
        """Bulk create request logs for efficiency."""
        logs = []
        for log_data in request_logs:
            log = RequestLog(page_result_id=page_result_id, **log_data)
            logs.append(log)

        self.session.add_all(logs)
        await self.session.flush()
        return logs

    async def bulk_create_cookies(
        self,
        page_result_id: int,
        cookies: List[Dict[str, Any]]
    ) -> List[Cookie]:
        """Bulk create cookies for efficiency with deduplication."""
        # Deduplicate cookies by name/domain/path combination
        unique_cookies = {}
        for cookie_data in cookies:
            # Create deduplication key
            dedup_key = (
                cookie_data.get('name', ''),
                cookie_data.get('domain', ''),
                cookie_data.get('path', '/')
            )

            # Keep the last occurrence (or merge as needed)
            if dedup_key not in unique_cookies:
                unique_cookies[dedup_key] = cookie_data
            else:
                # Update with latest data, preserving important fields
                existing = unique_cookies[dedup_key]
                existing.update(cookie_data)

        cookie_objects = []
        for cookie_data in unique_cookies.values():
            cookie = Cookie(page_result_id=page_result_id, **cookie_data)
            cookie_objects.append(cookie)

        self.session.add_all(cookie_objects)
        await self.session.flush()
        return cookie_objects

    # ============= RequestLog & Cookie Operations =============

    async def get_request_logs_for_page(
        self,
        page_result_id: int,
        limit: int = 1000,
        offset: int = 0
    ) -> List[RequestLog]:
        """Get request logs for a specific page result."""
        result = await self.session.execute(
            select(RequestLog)
            .where(RequestLog.page_result_id == page_result_id)
            .order_by(RequestLog.start_time)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_cookies_for_page(
        self,
        page_result_id: int,
        first_party_only: Optional[bool] = None
    ) -> List[Cookie]:
        """Get cookies for a specific page result."""
        query = select(Cookie).where(Cookie.page_result_id == page_result_id)

        if first_party_only is not None:
            query = query.where(Cookie.first_party == first_party_only)

        query = query.order_by(Cookie.domain, Cookie.name)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_cookies_for_run(
        self,
        run_id: int,
        first_party_only: Optional[bool] = None,
        essential_only: Optional[bool] = None
    ) -> List[Cookie]:
        """Get all cookies for a run with optional filtering."""
        query = (
            select(Cookie)
            .select_from(Cookie)
            .join(PageResult)
            .where(PageResult.run_id == run_id)
        )

        if first_party_only is not None:
            query = query.where(Cookie.first_party == first_party_only)

        if essential_only is not None:
            query = query.where(Cookie.essential == essential_only)

        query = query.order_by(Cookie.domain, Cookie.name)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create_datalayer_snapshot(
        self,
        page_result_id: int,
        exists: bool,
        size_bytes: int = 0,
        truncated: bool = False,
        sample_data: Optional[Dict[str, Any]] = None,
        schema_valid: Optional[bool] = None,
        validation_errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataLayerSnapshot:
        """Create a data layer snapshot."""
        snapshot = DataLayerSnapshot(
            page_result_id=page_result_id,
            exists=exists,
            size_bytes=size_bytes,
            truncated=truncated,
            sample_json=sample_data,
            schema_valid=schema_valid,
            validation_errors_json=validation_errors,
            metadata_json=metadata
        )

        self.session.add(snapshot)
        await self.session.flush()
        return snapshot

    # ============= Rule Failure Operations =============

    async def create_rule_failure(
        self,
        run_id: int,
        rule_id: str,
        severity: SeverityLevel,
        message: str,
        rule_name: Optional[str] = None,
        page_url: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> RuleFailure:
        """Create a rule failure record."""
        failure = RuleFailure(
            run_id=run_id,
            rule_id=rule_id,
            rule_name=rule_name,
            severity=severity,
            message=message,
            page_url=page_url,
            details_json=details
        )

        self.session.add(failure)
        await self.session.flush()
        return failure

    async def get_rule_failures_for_run(
        self,
        run_id: int,
        severity: Optional[SeverityLevel] = None,
        rule_id: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[RuleFailure]:
        """Get rule failures for a run with optional filtering."""
        query = select(RuleFailure).where(RuleFailure.run_id == run_id)

        if severity:
            query = query.where(RuleFailure.severity == severity)

        if rule_id:
            query = query.where(RuleFailure.rule_id == rule_id)

        query = query.order_by(
            desc(RuleFailure.severity),
            RuleFailure.detected_at
        ).offset(offset).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def bulk_create_rule_failures(
        self,
        failures: List[Dict[str, Any]]
    ) -> List[RuleFailure]:
        """Bulk create rule failures for efficiency."""
        failure_objects = []
        for failure_data in failures:
            failure = RuleFailure(**failure_data)
            failure_objects.append(failure)

        self.session.add_all(failure_objects)
        await self.session.flush()
        return failure_objects

    # ============= Artifact Operations =============

    async def create_artifact(
        self,
        run_id: int,
        kind: str,
        path: str,
        checksum: str,
        size_bytes: int,
        content_type: Optional[str] = None,
        page_url: Optional[str] = None,
        description: Optional[str] = None,
        storage_backend: str = "local",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Artifact:
        """Create an artifact record."""
        artifact = Artifact(
            run_id=run_id,
            kind=kind,
            path=path,
            checksum=checksum,
            size_bytes=size_bytes,
            content_type=content_type,
            page_url=page_url,
            description=description,
            storage_backend=storage_backend,
            metadata_json=metadata
        )

        self.session.add(artifact)
        await self.session.flush()
        return artifact

    async def get_artifacts_for_run(
        self,
        run_id: int,
        kind: Optional[str] = None
    ) -> List[Artifact]:
        """Get artifacts for a run with optional kind filtering."""
        query = select(Artifact).where(Artifact.run_id == run_id)

        if kind:
            query = query.where(Artifact.kind == kind)

        query = query.order_by(Artifact.created_at)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_artifact_by_path(self, path: str) -> Optional[Artifact]:
        """Get artifact by storage path."""
        result = await self.session.execute(
            select(Artifact).where(Artifact.path == path)
        )
        return result.scalar_one_or_none()

    async def delete_artifact(self, artifact_id: int) -> bool:
        """Delete an artifact record."""
        artifact = await self.session.get(Artifact, artifact_id)
        if not artifact:
            return False

        await self.session.delete(artifact)
        return True

    # ============= Export Query Operations =============

    async def stream_request_logs_for_run(
        self,
        run_id: int,
        batch_size: int = 1000
    ):
        """Stream request logs for a run in batches for export."""
        offset = 0
        while True:
            query = (
                select(RequestLog)
                .select_from(RequestLog)
                .join(PageResult)
                .where(PageResult.run_id == run_id)
                .order_by(RequestLog.start_time)
                .offset(offset)
                .limit(batch_size)
            )

            result = await self.session.execute(query)
            logs = list(result.scalars().all())

            if not logs:
                break

            yield logs
            offset += batch_size

    async def stream_cookies_for_run(
        self,
        run_id: int,
        batch_size: int = 1000
    ):
        """Stream cookies for a run in batches for export."""
        offset = 0
        while True:
            query = (
                select(Cookie)
                .select_from(Cookie)
                .join(PageResult)
                .where(PageResult.run_id == run_id)
                .order_by(Cookie.domain, Cookie.name)
                .offset(offset)
                .limit(batch_size)
            )

            result = await self.session.execute(query)
            cookies = list(result.scalars().all())

            if not cookies:
                break

            yield cookies
            offset += batch_size

    async def get_tag_inventory_for_run(self, run_id: int) -> List[Dict[str, Any]]:
        """Get aggregated tag inventory for a run."""
        # This aggregates vendor tags from request logs
        query = (
            select(RequestLog.vendor_tags_json, PageResult.id.label('page_id'))
            .select_from(RequestLog)
            .join(PageResult)
            .where(
                and_(
                    PageResult.run_id == run_id,
                    RequestLog.vendor_tags_json.isnot(None)
                )
            )
        )

        result = await self.session.execute(query)

        # Aggregate tags by vendor/name/id
        tag_inventory = {}
        for row in result.all():
            if row.vendor_tags_json:
                for tag in row.vendor_tags_json:
                    key = f"{tag.get('vendor', 'unknown')}:{tag.get('name', 'unknown')}:{tag.get('id', 'unknown')}"
                    if key not in tag_inventory:
                        tag_inventory[key] = {
                            'vendor': tag.get('vendor'),
                            'name': tag.get('name'),
                            'id': tag.get('id'),
                            'count': 0,
                            'pages': set()
                        }
                    tag_inventory[key]['count'] += 1
                    tag_inventory[key]['pages'].add(row.page_id)

        # Convert to list and clean up
        inventory = []
        for tag_data in tag_inventory.values():
            tag_data['pages'] = len(tag_data['pages'])
            inventory.append(tag_data)

        return sorted(inventory, key=lambda x: (x['vendor'], x['name']))

    # ============= Transaction Management =============

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()

    async def flush(self) -> None:
        """Flush pending changes without committing."""
        await self.session.flush()


# ============= Factory Functions =============

def create_audit_dao_factory():
    """Create DAO factory that manages its own session lifecycle.

    Returns a function that creates DAO instances with fresh sessions.
    Callers are responsible for session management.
    """
    def _create_dao():
        session = db_config.session_factory()
        return AuditDAO(session), session

    return _create_dao

# Deprecated: Use session dependency injection instead
async def create_audit_dao():
    """Create DAO with new database session.

    DEPRECATED: This creates a closed session. Use dependency injection instead.
    """
    raise DeprecationWarning(
        "create_audit_dao() creates closed sessions. "
        "Use get_session() dependency injection or manage sessions explicitly."
    )