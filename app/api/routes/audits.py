"""Audit management API routes for Tag Sentinel.

This module implements FastAPI routes for audit creation, status tracking,
and audit lifecycle management with comprehensive validation and error handling.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import JSONResponse

from app.api.schemas import (
    CreateAuditRequest,
    ListAuditsRequest,
    AuditDetail,
    AuditRef,
    AuditList,
    ErrorResponse
)
from app.api.services import AuditService, AuditNotFoundError, IdempotencyError

logger = logging.getLogger(__name__)

# Create router with tags for OpenAPI documentation
router = APIRouter(
    prefix="/audits",
    tags=["Audits"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Audit Not Found"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)


# Shared audit service instance to maintain state across requests
# In production, this would be replaced with proper dependency injection
# and database persistence
_audit_service_instance = None

def get_audit_service() -> AuditService:
    """Dependency to provide audit service instance.

    Returns:
        Configured audit service instance

    Note:
        In production, this would inject configuration and dependencies
        from the application context (database connections, etc.).
    """
    global _audit_service_instance
    if _audit_service_instance is None:
        _audit_service_instance = AuditService(base_url="http://localhost:8000")
    return _audit_service_instance


@router.post(
    "",
    response_model=AuditRef,
    status_code=201,
    summary="Create new audit",
    description="""
    Create a new audit run for the specified site and environment.

    ## Idempotency

    Use the `idempotency_key` parameter to prevent duplicate audit creation.
    If an audit with the same idempotency key already exists within the
    configured time window (24 hours by default), the existing audit
    will be returned instead of creating a duplicate.

    ## Parameters

    The `params` field allows overriding default audit configuration:
    - `max_pages`: Maximum number of pages to crawl
    - `discovery_mode`: URL discovery method (seeds, sitemap, dom, hybrid)
    - `include_patterns`: Regex patterns for URLs to include
    - `exclude_patterns`: Regex patterns for URLs to exclude
    - And many more crawling configuration options

    ## Priority

    Higher priority audits will be scheduled before lower priority ones.
    Priority range is -100 to 100, with 0 being the default.

    ## Examples

    Basic audit creation:
    ```json
    {
        "site_id": "ecommerce",
        "env": "production"
    }
    ```

    Advanced audit with custom parameters:
    ```json
    {
        "site_id": "ecommerce",
        "env": "staging",
        "params": {
            "max_pages": 50,
            "discovery_mode": "hybrid",
            "include_patterns": [".*\\/checkout\\/.*"]
        },
        "idempotency_key": "staging-checkout-test-001",
        "priority": 10,
        "metadata": {
            "triggered_by": "ci_pipeline",
            "test_suite": "checkout_flow"
        }
    }
    ```
    """,
    responses={
        201: {
            "description": "Audit created successfully",
            "model": AuditRef
        },
        409: {
            "description": "Idempotency conflict - audit already exists",
            "model": ErrorResponse
        }
    }
)
async def create_audit(
    request: CreateAuditRequest,
    http_request: Request,
    audit_service: AuditService = Depends(get_audit_service)
) -> AuditRef:
    """Create a new audit run with idempotency support."""
    request_id = getattr(http_request.state, "request_id", None)

    try:
        logger.info(
            f"Creating audit for site_id={request.site_id}, env={request.env}",
            extra={"request_id": request_id}
        )

        audit = await audit_service.create_audit(request)

        logger.info(
            f"Successfully created audit {audit.id}",
            extra={"request_id": request_id, "audit_id": audit.id}
        )

        return audit

    except IdempotencyError as e:
        logger.warning(
            f"Idempotency conflict for key {request.idempotency_key}: {e}",
            extra={"request_id": request_id}
        )
        raise HTTPException(
            status_code=409,
            detail=f"Idempotency key '{request.idempotency_key}' exists with different parameters"
        )

    except ValueError as e:
        logger.warning(
            f"Invalid audit parameters: {e}",
            extra={"request_id": request_id}
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameters: {str(e)}"
        )

    except Exception as e:
        logger.error(
            f"Failed to create audit: {e}",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to create audit"
        )


@router.get(
    "/{audit_id}",
    response_model=AuditDetail,
    summary="Get audit details",
    description="""
    Retrieve detailed information about a specific audit run.

    ## Response Details

    The response includes:
    - **Basic Information**: ID, site, environment, status
    - **Timing**: Creation, start, and completion timestamps
    - **Progress**: Real-time progress percentage for running audits
    - **Summary**: Comprehensive statistics for completed audits
    - **Configuration**: Parameters and rules used for the audit
    - **Links**: URLs for accessing exports and artifacts
    - **Error Information**: Details about any failures

    ## Status Values

    - `queued`: Audit is waiting to start
    - `running`: Audit is currently executing
    - `completed`: Audit finished successfully
    - `failed`: Audit encountered an error

    ## Examples

    ```bash
    curl -X GET "http://localhost:8000/api/audits/audit_20240115_ecommerce_prod_abc123"
    ```
    """,
    responses={
        200: {
            "description": "Audit details retrieved successfully",
            "model": AuditDetail
        }
    }
)
async def get_audit(
    audit_id: str,
    http_request: Request,
    audit_service: AuditService = Depends(get_audit_service)
) -> AuditDetail:
    """Get detailed information about a specific audit."""
    request_id = getattr(http_request.state, "request_id", None)

    try:
        logger.debug(
            f"Retrieving audit {audit_id}",
            extra={"request_id": request_id, "audit_id": audit_id}
        )

        audit = await audit_service.get_audit(audit_id)
        return audit

    except AuditNotFoundError:
        logger.warning(
            f"Audit {audit_id} not found",
            extra={"request_id": request_id, "audit_id": audit_id}
        )
        raise HTTPException(
            status_code=404,
            detail=f"Audit '{audit_id}' not found"
        )

    except Exception as e:
        logger.error(
            f"Failed to retrieve audit {audit_id}: {e}",
            extra={"request_id": request_id, "audit_id": audit_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve audit"
        )


@router.get(
    "",
    response_model=AuditList,
    summary="List audits",
    description="""
    List audits with comprehensive filtering, sorting, and pagination support.

    ## Filtering

    Supports filtering by:
    - **Site ID**: Filter by specific site identifier
    - **Environment**: Filter by deployment environment
    - **Status**: Filter by audit status (multiple values allowed)
    - **Date Range**: Filter by creation date range
    - **Tags**: Filter by metadata tags
    - **Search**: Full-text search across audit metadata

    ## Sorting

    Sort results by:
    - `created_at`: Creation timestamp (default)
    - `updated_at`: Last update timestamp
    - `duration`: Audit execution duration
    - `status`: Audit status
    - `site_id`: Site identifier

    Order can be `asc` (ascending) or `desc` (descending, default).

    ## Pagination

    Uses cursor-based pagination for consistent results:
    - `limit`: Number of results per page (1-100, default 20)
    - `cursor`: Opaque cursor for next page (returned in previous response)

    ## Summary Statistics

    Set `include_stats=true` to include summary statistics across all
    matching audits (not just the current page).

    ## Examples

    List recent audits:
    ```bash
    curl -X GET "http://localhost:8000/api/audits?limit=10"
    ```

    Filter by site and status:
    ```bash
    curl -X GET "http://localhost:8000/api/audits?site_id=ecommerce&status=completed&status=failed"
    ```

    Search with date range:
    ```bash
    curl -X GET "http://localhost:8000/api/audits?search=checkout&date_from=2024-01-01T00:00:00Z&include_stats=true"
    ```
    """,
    responses={
        200: {
            "description": "Audit list retrieved successfully",
            "model": AuditList
        }
    }
)
async def list_audits(
    http_request: Request,
    site_id: Optional[str] = Query(None, description="Filter by site identifier"),
    env: Optional[str] = Query(None, description="Filter by environment"),
    status: Optional[list[str]] = Query(None, description="Filter by status (multiple allowed)"),
    tags: Optional[list[str]] = Query(None, description="Filter by metadata tags"),
    search: Optional[str] = Query(None, description="Full-text search"),
    sort_by: Optional[str] = Query("created_at", description="Sort field"),
    sort_order: Optional[str] = Query("desc", description="Sort order (asc/desc)"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    limit: Optional[int] = Query(20, ge=1, le=100, description="Results per page"),
    include_stats: Optional[bool] = Query(False, description="Include summary statistics"),
    audit_service: AuditService = Depends(get_audit_service)
) -> AuditList:
    """List audits with filtering, sorting, and pagination."""
    request_id = getattr(http_request.state, "request_id", None)

    try:
        # Build list request from query parameters
        list_request = ListAuditsRequest(
            site_id=site_id,
            env=env,
            status=status,
            tags=tags,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            cursor=cursor,
            limit=limit,
            include_stats=include_stats
        )

        logger.debug(
            f"Listing audits with filters: {list_request.model_dump(exclude_none=True)}",
            extra={"request_id": request_id}
        )

        audit_list = await audit_service.list_audits(list_request)

        logger.debug(
            f"Retrieved {len(audit_list.audits)} audits (total: {audit_list.total_count})",
            extra={"request_id": request_id}
        )

        return audit_list

    except Exception as e:
        logger.error(
            f"Failed to list audits: {e}",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to list audits"
        )