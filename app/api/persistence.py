"""REST API endpoints for persistence layer operations."""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..persistence import (
    get_session,
    AuditDAO,
    ExportService,
    ExportFormat,
    RetentionEngine,
    get_persistence_manager
)

router = APIRouter(prefix="/api/persistence", tags=["persistence"])


# ============= Request/Response Models =============

class RunSummary(BaseModel):
    """Summary of an audit run."""
    id: int
    site_id: str
    environment: str
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    summary_json: Optional[Dict[str, Any]] = None


class RunListResponse(BaseModel):
    """Paginated list of runs."""
    runs: List[RunSummary]
    total: int
    page: int
    limit: int


class ExportSummaryResponse(BaseModel):
    """Summary of available exports for a run."""
    run_id: int
    run_status: str
    export_timestamp: str
    available_exports: Dict[str, Any]
    run_statistics: Optional[Dict[str, Any]] = None


class RetentionSummaryResponse(BaseModel):
    """Summary of retention policies and data."""
    retention_config: Dict[str, Any]
    current_data: Dict[str, Any]
    expired_data: Dict[str, Any]
    error: Optional[str] = None


class CleanupRequest(BaseModel):
    """Request for data cleanup."""
    dry_run: bool = Field(default=True, description="If true, only simulate cleanup")


class CleanupResponse(BaseModel):
    """Response from cleanup operation."""
    runs_deleted: int
    artifacts_deleted: int
    storage_files_deleted: int
    storage_files_failed: int
    errors: List[str]
    has_errors: bool
    total_deleted: int


# ============= Dependencies =============

async def get_dao(session: AsyncSession = Depends(get_session)) -> AuditDAO:
    """Dependency to get DAO instance."""
    return AuditDAO(session)


async def get_export_service(dao: AuditDAO = Depends(get_dao)) -> ExportService:
    """Dependency to get export service."""
    return ExportService(dao)


async def get_retention_engine(dao: AuditDAO = Depends(get_dao)) -> RetentionEngine:
    """Dependency to get retention engine."""
    persistence_manager = get_persistence_manager()

    return RetentionEngine(
        dao=dao,
        artifact_store=persistence_manager.artifact_store,
        config=persistence_manager.retention
    )


# ============= Run Management Endpoints =============

@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    environment: Optional[str] = Query(None, description="Filter by environment"),
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=1000, description="Items per page"),
    dao: AuditDAO = Depends(get_dao)
):
    """List audit runs with pagination and filtering."""
    offset = (page - 1) * limit

    runs = await dao.list_runs(
        environment=environment,
        status=status,
        limit=limit,
        offset=offset
    )

    # For now, we don't have a separate count query, so we'll use the number of returned runs
    # In production, you'd want a separate count query for accurate pagination
    total = len(runs)

    return RunListResponse(
        runs=[
            RunSummary(
                id=run.id,
                site_id=run.site_id,
                environment=run.environment,
                status=run.status.value,
                started_at=run.started_at,
                finished_at=run.finished_at,
                summary_json=run.summary_json
            ) for run in runs
        ],
        total=total,
        page=page,
        limit=limit
    )


@router.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(
    run_id: int,
    dao: AuditDAO = Depends(get_dao)
):
    """Get details of a specific run."""
    run = await dao.get_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunSummary(
        id=run.id,
        site_id=run.site_id,
        environment=run.environment,
        status=run.status.value,
        started_at=run.started_at,
        finished_at=run.finished_at,
        summary_json=run.summary_json
    )


@router.get("/runs/{run_id}/statistics")
async def get_run_statistics(
    run_id: int,
    dao: AuditDAO = Depends(get_dao)
):
    """Get statistics for a specific run."""
    stats = await dao.get_run_statistics(run_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Run not found or no statistics available")

    return stats


# ============= Export Endpoints =============

@router.get("/runs/{run_id}/exports/summary", response_model=ExportSummaryResponse)
async def get_export_summary(
    run_id: int,
    export_service: ExportService = Depends(get_export_service)
):
    """Get summary of available exports for a run."""
    summary = await export_service.get_export_summary(run_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Run not found")

    return ExportSummaryResponse(**summary)


@router.get("/runs/{run_id}/exports/request-logs")
async def export_request_logs(
    run_id: int,
    format: ExportFormat = Query(ExportFormat.NDJSON, description="Export format"),
    batch_size: int = Query(1000, ge=100, le=10000, description="Batch size for streaming"),
    export_service: ExportService = Depends(get_export_service)
):
    """Export request logs for a run."""

    async def generate_export():
        async for chunk in export_service.export_request_logs(
            run_id=run_id,
            format=format,
            batch_size=batch_size
        ):
            yield chunk

    content_type = export_service.get_content_type(format)
    file_extension = export_service.get_file_extension(format)
    filename = f"request_logs_run_{run_id}{file_extension}"

    return StreamingResponse(
        generate_export(),
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/runs/{run_id}/exports/cookies")
async def export_cookies(
    run_id: int,
    format: ExportFormat = Query(ExportFormat.NDJSON, description="Export format"),
    first_party_only: Optional[bool] = Query(None, description="Filter to first-party cookies only"),
    batch_size: int = Query(1000, ge=100, le=10000, description="Batch size for streaming"),
    export_service: ExportService = Depends(get_export_service)
):
    """Export cookies for a run."""

    async def generate_export():
        async for chunk in export_service.export_cookies(
            run_id=run_id,
            format=format,
            first_party_only=first_party_only,
            batch_size=batch_size
        ):
            yield chunk

    content_type = export_service.get_content_type(format)
    file_extension = export_service.get_file_extension(format)
    filename = f"cookies_run_{run_id}{file_extension}"

    return StreamingResponse(
        generate_export(),
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/runs/{run_id}/exports/tag-inventory")
async def export_tag_inventory(
    run_id: int,
    format: ExportFormat = Query(ExportFormat.JSON, description="Export format"),
    export_service: ExportService = Depends(get_export_service)
):
    """Export tag inventory for a run."""

    try:
        content = await export_service.export_tag_inventory(run_id, format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    content_type = export_service.get_content_type(format)
    file_extension = export_service.get_file_extension(format)
    filename = f"tag_inventory_run_{run_id}{file_extension}"

    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/runs/{run_id}/exports/rule-failures")
async def export_rule_failures(
    run_id: int,
    format: ExportFormat = Query(ExportFormat.NDJSON, description="Export format"),
    severity_filter: Optional[str] = Query(None, description="Filter by severity level"),
    export_service: ExportService = Depends(get_export_service)
):
    """Export rule failures for a run."""

    async def generate_export():
        async for chunk in export_service.export_rule_failures(
            run_id=run_id,
            format=format,
            severity_filter=severity_filter
        ):
            yield chunk

    content_type = export_service.get_content_type(format)
    file_extension = export_service.get_file_extension(format)
    filename = f"rule_failures_run_{run_id}{file_extension}"

    return StreamingResponse(
        generate_export(),
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ============= Retention Management Endpoints =============

@router.get("/retention/summary", response_model=RetentionSummaryResponse)
async def get_retention_summary(
    retention_engine: RetentionEngine = Depends(get_retention_engine)
):
    """Get summary of retention policies and current data."""
    summary = await retention_engine.get_retention_summary()
    return RetentionSummaryResponse(**summary)


@router.post("/retention/cleanup", response_model=CleanupResponse)
async def cleanup_expired_data(
    request: CleanupRequest,
    retention_engine: RetentionEngine = Depends(get_retention_engine)
):
    """Clean up expired data according to retention policies."""

    try:
        result = await retention_engine.cleanup_expired_data(dry_run=request.dry_run)

        return CleanupResponse(
            runs_deleted=result.runs_deleted,
            artifacts_deleted=result.artifacts_deleted,
            storage_files_deleted=result.storage_files_deleted,
            storage_files_failed=result.storage_files_failed,
            errors=result.errors,
            has_errors=result.has_errors,
            total_deleted=result.total_deleted
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# ============= Health Check Endpoints =============

@router.get("/health")
async def health_check():
    """Check health of persistence components."""
    persistence_manager = get_persistence_manager()

    try:
        health_status = await persistence_manager.health_check()

        overall_healthy = all(health_status.values())
        status_code = 200 if overall_healthy else 503

        return Response(
            content={
                "status": "healthy" if overall_healthy else "unhealthy",
                "components": health_status,
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=status_code
        )

    except Exception as e:
        return Response(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=500
        )


# ============= Artifact Management Endpoints =============

@router.get("/runs/{run_id}/artifacts")
async def list_artifacts(
    run_id: int,
    artifact_type: Optional[str] = Query(None, description="Filter by artifact type"),
    dao: AuditDAO = Depends(get_dao)
):
    """List artifacts for a run."""
    artifacts = await dao.get_artifacts_for_run(run_id, artifact_type=artifact_type)

    return [
        {
            "id": artifact.id,
            "type": artifact.type,
            "path": artifact.path,
            "checksum": artifact.checksum,
            "size_bytes": artifact.size_bytes,
            "content_type": artifact.content_type,
            "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
            "metadata": artifact.metadata_json
        }
        for artifact in artifacts
    ]


@router.get("/artifacts/{artifact_id}/url")
async def get_artifact_url(
    artifact_id: int,
    signed: bool = Query(True, description="Generate signed URL"),
    ttl_seconds: int = Query(3600, ge=60, le=86400, description="URL time-to-live"),
    dao: AuditDAO = Depends(get_dao)
):
    """Get URL to access an artifact."""
    # Get artifact from database
    artifact = await dao.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Generate URL from storage backend
    persistence_manager = get_persistence_manager()
    try:
        url = await persistence_manager.artifact_store.get_url(
            artifact.path,
            signed=signed,
            ttl_seconds=ttl_seconds
        )

        return {
            "url": url,
            "signed": signed,
            "expires_in_seconds": ttl_seconds if signed else None,
            "artifact": {
                "id": artifact.id,
                "type": artifact.type,
                "size_bytes": artifact.size_bytes,
                "content_type": artifact.content_type
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate URL: {str(e)}")