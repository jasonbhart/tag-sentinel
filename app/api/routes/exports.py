"""Export API routes for Tag Sentinel.

This module implements FastAPI routes for audit data exports including
request logs, cookie inventories, tag detection results, and data layer snapshots.
"""

import logging
from typing import Optional, Dict, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import StreamingResponse

from app.api.services.export_service import ExportService
from app.api.services.audit_service import AuditService, AuditNotFoundError
from app.api.routes.audits import get_audit_service
from app.api.schemas import ErrorResponse

logger = logging.getLogger(__name__)

# Create router with tags for OpenAPI documentation
router = APIRouter(
    prefix="/exports",
    tags=["Exports"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Audit Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

# Shared export service instance
_export_service_instance = None

def get_export_service() -> ExportService:
    """Dependency to provide export service instance."""
    global _export_service_instance
    if _export_service_instance is None:
        _export_service_instance = ExportService()
    return _export_service_instance


@router.get(
    "/{audit_id}/requests.{format}",
    summary="Export request logs",
    description="""
    Export network request logs from an audit in streaming format.

    ## Supported Formats

    - **JSON** (`.json`): NDJSON format with one request record per line
    - **CSV** (`.csv`): Comma-separated values with headers

    ## Filtering

    Use query parameters to filter the exported data:
    - `status`: Filter by request status (success, failed, timeout, etc.)
    - `resource_type`: Filter by resource type (script, xhr, image, etc.)
    - `analytics_only`: Only include analytics-related requests

    ## Export Features

    - Streaming response for efficient memory usage
    - Comprehensive request/response metadata
    - Analytics detection and vendor identification
    - Timing information (DNS, connect, response times)
    - Privacy-filtered headers when configured

    ## Examples

    ```bash
    # Export all requests as JSON
    curl "http://localhost:8000/api/exports/audit123/requests.json"

    # Export only analytics requests as CSV
    curl "http://localhost:8000/api/exports/audit123/requests.csv?analytics_only=true"

    # Export failed requests for debugging
    curl "http://localhost:8000/api/exports/audit123/requests.json?status=failed"
    ```
    """,
    responses={
        200: {
            "description": "Request log export stream",
            "content": {
                "application/x-ndjson": {
                    "example": '{"id":"req_001","audit_id":"audit123","url":"https://example.com"}\n'
                },
                "text/csv": {
                    "example": "id,audit_id,url\nreq_001,audit123,https://example.com\n"
                }
            }
        }
    }
)
async def export_requests(
    audit_id: str,
    format: Literal["json", "csv"],
    http_request: Request,
    status: Optional[str] = Query(None, description="Filter by request status"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    analytics_only: Optional[bool] = Query(False, description="Only include analytics requests"),
    audit_service: AuditService = Depends(get_audit_service),
    export_service: ExportService = Depends(get_export_service)
) -> StreamingResponse:
    """Export network request logs for an audit."""
    request_id = getattr(http_request.state, "request_id", None)

    try:
        # Verify audit exists
        await audit_service.get_audit(audit_id)

        # Build filters
        filters = {}
        if status:
            filters["status"] = status
        if resource_type:
            filters["resource_type"] = resource_type
        if analytics_only:
            filters["analytics_only"] = True

        logger.info(
            f"Starting request export for audit {audit_id} in {format} format",
            extra={"request_id": request_id, "audit_id": audit_id, "filters": filters}
        )

        # Generate export stream
        export_generator = export_service.export_requests(audit_id, format, filters)

        # Set appropriate content type and filename
        if format == "csv":
            media_type = "text/csv"
            filename = f"requests_{audit_id}.csv"
        else:
            media_type = "application/x-ndjson"
            filename = f"requests_{audit_id}.json"

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }

        return StreamingResponse(
            export_generator,
            media_type=media_type,
            headers=headers
        )

    except AuditNotFoundError:
        logger.warning(
            f"Audit {audit_id} not found for export",
            extra={"request_id": request_id, "audit_id": audit_id}
        )
        raise HTTPException(
            status_code=404,
            detail=f"Audit '{audit_id}' not found"
        )

    except Exception as e:
        logger.error(
            f"Failed to export requests for audit {audit_id}: {e}",
            extra={"request_id": request_id, "audit_id": audit_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to export request data"
        )


@router.get(
    "/{audit_id}/cookies.{format}",
    summary="Export cookie inventory",
    description="""
    Export cookie inventory from an audit with privacy compliance analysis.

    ## Supported Formats

    - **JSON** (`.json`): NDJSON format with one cookie record per line
    - **CSV** (`.csv`): Comma-separated values with headers

    ## Filtering

    Use query parameters to filter the exported data:
    - `category`: Filter by cookie category (analytics, marketing, functional, etc.)
    - `vendor`: Filter by detected vendor
    - `essential_only`: Only include essential cookies
    - `privacy_compliant`: Only include privacy-compliant cookies

    ## Export Features

    - Comprehensive cookie attributes and metadata
    - Privacy compliance analysis (GDPR, consent requirements)
    - Cookie classification and vendor identification
    - Cross-scenario comparison (baseline vs privacy signals)
    - Configurable value redaction for sensitive data

    ## Examples

    ```bash
    # Export all cookies as JSON
    curl "http://localhost:8000/api/exports/audit123/cookies.json"

    # Export only marketing cookies as CSV
    curl "http://localhost:8000/api/exports/audit123/cookies.csv?category=marketing"

    # Export essential cookies only
    curl "http://localhost:8000/api/exports/audit123/cookies.json?essential_only=true"
    ```
    """,
    responses={
        200: {
            "description": "Cookie inventory export stream",
            "content": {
                "application/x-ndjson": {
                    "example": '{"audit_id":"audit123","name":"_ga","domain":".example.com","category":"analytics"}\n'
                },
                "text/csv": {
                    "example": "audit_id,name,domain,category\naudit123,_ga,.example.com,analytics\n"
                }
            }
        }
    }
)
async def export_cookies(
    audit_id: str,
    format: Literal["json", "csv"],
    http_request: Request,
    category: Optional[str] = Query(None, description="Filter by cookie category"),
    vendor: Optional[str] = Query(None, description="Filter by vendor"),
    essential_only: Optional[bool] = Query(False, description="Only include essential cookies"),
    privacy_compliant: Optional[bool] = Query(False, description="Only include privacy-compliant cookies"),
    audit_service: AuditService = Depends(get_audit_service),
    export_service: ExportService = Depends(get_export_service)
) -> StreamingResponse:
    """Export cookie inventory for an audit."""
    request_id = getattr(http_request.state, "request_id", None)

    try:
        # Verify audit exists
        await audit_service.get_audit(audit_id)

        # Build filters
        filters = {}
        if category:
            filters["category"] = category
        if vendor:
            filters["vendor"] = vendor
        if essential_only:
            filters["essential_only"] = True
        if privacy_compliant:
            filters["privacy_compliant"] = True

        logger.info(
            f"Starting cookie export for audit {audit_id} in {format} format",
            extra={"request_id": request_id, "audit_id": audit_id, "filters": filters}
        )

        # Generate export stream
        export_generator = export_service.export_cookies(audit_id, format, filters)

        # Set appropriate content type and filename
        if format == "csv":
            media_type = "text/csv"
            filename = f"cookies_{audit_id}.csv"
        else:
            media_type = "application/x-ndjson"
            filename = f"cookies_{audit_id}.json"

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }

        return StreamingResponse(
            export_generator,
            media_type=media_type,
            headers=headers
        )

    except AuditNotFoundError:
        logger.warning(
            f"Audit {audit_id} not found for cookie export",
            extra={"request_id": request_id, "audit_id": audit_id}
        )
        raise HTTPException(
            status_code=404,
            detail=f"Audit '{audit_id}' not found"
        )

    except Exception as e:
        logger.error(
            f"Failed to export cookies for audit {audit_id}: {e}",
            extra={"request_id": request_id, "audit_id": audit_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to export cookie data"
        )


@router.get(
    "/{audit_id}/tags.{format}",
    summary="Export tag detection results",
    description="""
    Export analytics tag detection results with comprehensive metadata.

    ## Supported Formats

    - **JSON** (`.json`): NDJSON format with one tag record per line
    - **CSV** (`.csv`): Comma-separated values with headers

    ## Filtering

    Use query parameters to filter the exported data:
    - `vendor`: Filter by tag vendor (Google Analytics, Facebook, etc.)
    - `tag_type`: Filter by tag type (pageview, event, conversion, etc.)
    - `implementation_method`: Filter by implementation (script_tag, gtm, etc.)
    - `has_errors`: Only include tags with validation errors

    ## Export Features

    - Comprehensive tag detection metadata
    - Confidence scores and detection methods
    - Load performance and timing analysis
    - Data layer integration details
    - Privacy and consent analysis
    - Cross-scenario comparison data

    ## Examples

    ```bash
    # Export all tags as JSON
    curl "http://localhost:8000/api/exports/audit123/tags.json"

    # Export only Google Analytics tags as CSV
    curl "http://localhost:8000/api/exports/audit123/tags.csv?vendor=Google Analytics"

    # Export tags with errors for debugging
    curl "http://localhost:8000/api/exports/audit123/tags.json?has_errors=true"
    ```
    """,
    responses={
        200: {
            "description": "Tag detection export stream",
            "content": {
                "application/x-ndjson": {
                    "example": '{"audit_id":"audit123","tag_id":"ga4_001","vendor":"Google Analytics 4","tag_type":"pageview"}\n'
                },
                "text/csv": {
                    "example": "audit_id,tag_id,vendor,tag_type\naudit123,ga4_001,Google Analytics 4,pageview\n"
                }
            }
        }
    }
)
async def export_tags(
    audit_id: str,
    format: Literal["json", "csv"],
    http_request: Request,
    vendor: Optional[str] = Query(None, description="Filter by tag vendor"),
    tag_type: Optional[str] = Query(None, description="Filter by tag type"),
    implementation_method: Optional[str] = Query(None, description="Filter by implementation method"),
    has_errors: Optional[bool] = Query(False, description="Only include tags with errors"),
    audit_service: AuditService = Depends(get_audit_service),
    export_service: ExportService = Depends(get_export_service)
) -> StreamingResponse:
    """Export tag detection results for an audit."""
    request_id = getattr(http_request.state, "request_id", None)

    try:
        # Verify audit exists
        await audit_service.get_audit(audit_id)

        # Build filters
        filters = {}
        if vendor:
            filters["vendor"] = vendor
        if tag_type:
            filters["tag_type"] = tag_type
        if implementation_method:
            filters["implementation_method"] = implementation_method
        if has_errors:
            filters["has_errors"] = True

        logger.info(
            f"Starting tag export for audit {audit_id} in {format} format",
            extra={"request_id": request_id, "audit_id": audit_id, "filters": filters}
        )

        # Generate export stream
        export_generator = export_service.export_tags(audit_id, format, filters)

        # Set appropriate content type and filename
        if format == "csv":
            media_type = "text/csv"
            filename = f"tags_{audit_id}.csv"
        else:
            media_type = "application/x-ndjson"
            filename = f"tags_{audit_id}.json"

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }

        return StreamingResponse(
            export_generator,
            media_type=media_type,
            headers=headers
        )

    except AuditNotFoundError:
        logger.warning(
            f"Audit {audit_id} not found for tag export",
            extra={"request_id": request_id, "audit_id": audit_id}
        )
        raise HTTPException(
            status_code=404,
            detail=f"Audit '{audit_id}' not found"
        )

    except Exception as e:
        logger.error(
            f"Failed to export tags for audit {audit_id}: {e}",
            extra={"request_id": request_id, "audit_id": audit_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to export tag data"
        )


@router.get(
    "/{audit_id}/data-layer.{format}",
    summary="Export data layer snapshots",
    description="""
    Export data layer snapshots with structure analysis and validation results.

    ## Supported Formats

    - **JSON** (`.json`): NDJSON format with one snapshot record per line
    - **CSV** (`.csv`): Comma-separated values with headers (data flattened)

    ## Filtering

    Use query parameters to filter the exported data:
    - `trigger_event`: Filter by trigger event type
    - `contains_pii`: Only include snapshots containing PII
    - `schema_valid`: Only include schema-valid snapshots
    - `min_properties`: Minimum number of properties in snapshot

    ## Export Features

    - Complete data layer content (with privacy redaction)
    - Structure analysis (property count, nesting depth)
    - PII detection and classification
    - Schema validation results
    - Cross-scenario comparison data
    - Privacy-compliant data handling

    ## Examples

    ```bash
    # Export all data layer snapshots as JSON
    curl "http://localhost:8000/api/exports/audit123/data-layer.json"

    # Export only page load snapshots as CSV
    curl "http://localhost:8000/api/exports/audit123/data-layer.csv?trigger_event=page_load"

    # Export snapshots with PII for privacy analysis
    curl "http://localhost:8000/api/exports/audit123/data-layer.json?contains_pii=true"
    ```
    """,
    responses={
        200: {
            "description": "Data layer export stream",
            "content": {
                "application/x-ndjson": {
                    "example": '{"audit_id":"audit123","snapshot_id":"dl_001","trigger_event":"page_load","total_properties":5}\n'
                },
                "text/csv": {
                    "example": "audit_id,snapshot_id,trigger_event,total_properties\naudit123,dl_001,page_load,5\n"
                }
            }
        }
    }
)
async def export_data_layer(
    audit_id: str,
    format: Literal["json", "csv"],
    http_request: Request,
    trigger_event: Optional[str] = Query(None, description="Filter by trigger event"),
    contains_pii: Optional[bool] = Query(False, description="Only include snapshots with PII"),
    schema_valid: Optional[bool] = Query(False, description="Only include schema-valid snapshots"),
    min_properties: Optional[int] = Query(None, ge=0, description="Minimum number of properties"),
    audit_service: AuditService = Depends(get_audit_service),
    export_service: ExportService = Depends(get_export_service)
) -> StreamingResponse:
    """Export data layer snapshots for an audit."""
    request_id = getattr(http_request.state, "request_id", None)

    try:
        # Verify audit exists
        await audit_service.get_audit(audit_id)

        # Build filters
        filters = {}
        if trigger_event:
            filters["trigger_event"] = trigger_event
        if contains_pii:
            filters["contains_pii"] = True
        if schema_valid:
            filters["schema_valid"] = True
        if min_properties is not None:
            filters["min_properties"] = min_properties

        logger.info(
            f"Starting data layer export for audit {audit_id} in {format} format",
            extra={"request_id": request_id, "audit_id": audit_id, "filters": filters}
        )

        # Generate export stream
        export_generator = export_service.export_data_layer(audit_id, format, filters)

        # Set appropriate content type and filename
        if format == "csv":
            media_type = "text/csv"
            filename = f"data-layer_{audit_id}.csv"
        else:
            media_type = "application/x-ndjson"
            filename = f"data-layer_{audit_id}.json"

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }

        return StreamingResponse(
            export_generator,
            media_type=media_type,
            headers=headers
        )

    except AuditNotFoundError:
        logger.warning(
            f"Audit {audit_id} not found for data layer export",
            extra={"request_id": request_id, "audit_id": audit_id}
        )
        raise HTTPException(
            status_code=404,
            detail=f"Audit '{audit_id}' not found"
        )

    except Exception as e:
        logger.error(
            f"Failed to export data layer for audit {audit_id}: {e}",
            extra={"request_id": request_id, "audit_id": audit_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to export data layer data"
        )