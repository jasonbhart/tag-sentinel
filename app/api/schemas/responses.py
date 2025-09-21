"""API response schemas for Tag Sentinel REST API.

This module defines Pydantic models for all API response payloads,
including audit details, lists, exports, and error responses.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Literal, Union
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


class AuditStatus(str, Enum):
    """Audit execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AuditRef(BaseModel):
    """Minimal audit reference for lists and relationships.

    This schema provides a lightweight representation of an audit
    suitable for inclusion in lists and cross-references.
    """

    id: str = Field(
        ...,
        description="Unique audit identifier",
        examples=["audit_2024011501_ecommerce_prod"]
    )

    site_id: str = Field(
        ...,
        description="Site identifier this audit targets"
    )

    env: str = Field(
        ...,
        description="Environment this audit targets"
    )

    status: Literal['queued', 'running', 'completed', 'failed'] = Field(
        ...,
        description="Current audit status"
    )

    created_at: datetime = Field(
        ...,
        description="When the audit was created"
    )

    updated_at: datetime = Field(
        ...,
        description="When the audit was last updated"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "id": "audit_2024011501_ecommerce_prod",
                "site_id": "ecommerce",
                "env": "production",
                "status": "completed",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:45:23Z"
            }
        }


class AuditSummary(BaseModel):
    """Summary statistics for an audit run.

    This schema provides key metrics and counts without detailed
    breakdown data, suitable for overview displays.
    """

    pages_discovered: int = Field(
        ...,
        ge=0,
        description="Total number of pages discovered during crawl"
    )

    pages_processed: int = Field(
        ...,
        ge=0,
        description="Number of pages successfully processed"
    )

    pages_failed: int = Field(
        ...,
        ge=0,
        description="Number of pages that failed processing"
    )

    requests_captured: int = Field(
        ...,
        ge=0,
        description="Total number of network requests captured"
    )

    cookies_found: int = Field(
        ...,
        ge=0,
        description="Total number of unique cookies discovered"
    )

    tags_detected: int = Field(
        ...,
        ge=0,
        description="Total number of analytics tags detected"
    )

    errors_count: int = Field(
        ...,
        ge=0,
        description="Total number of errors encountered"
    )

    duration_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total audit duration in seconds"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "pages_discovered": 47,
                "pages_processed": 45,
                "pages_failed": 2,
                "requests_captured": 1247,
                "cookies_found": 23,
                "tags_detected": 15,
                "errors_count": 3,
                "duration_seconds": 892.5
            }
        }


class AuditLinks(BaseModel):
    """Links to audit-related resources and exports.

    This schema provides URLs for accessing audit artifacts,
    exports, and related resources.
    """

    self: HttpUrl = Field(
        ...,
        description="Link to this audit's detail endpoint"
    )

    requests_export: HttpUrl = Field(
        ...,
        description="Link to request log export endpoint"
    )

    cookies_export: HttpUrl = Field(
        ...,
        description="Link to cookie inventory export endpoint"
    )

    tags_export: HttpUrl = Field(
        ...,
        description="Link to tag detection export endpoint"
    )

    artifacts: Optional[HttpUrl] = Field(
        default=None,
        description="Link to static artifacts (HAR files, screenshots, etc.)"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "self": "https://api.example.com/api/audits/audit_2024011501_ecommerce_prod",
                "requests_export": "https://api.example.com/api/exports/audit_2024011501_ecommerce_prod/requests.json",
                "cookies_export": "https://api.example.com/api/exports/audit_2024011501_ecommerce_prod/cookies.csv",
                "tags_export": "https://api.example.com/api/exports/audit_2024011501_ecommerce_prod/tags.json",
                "artifacts": "https://api.example.com/artifacts/audit_2024011501_ecommerce_prod/"
            }
        }


class AuditDetail(BaseModel):
    """Comprehensive audit details with full metadata and results.

    This schema provides complete information about an audit run,
    including configuration, progress, results, and links to exports.
    """

    # Basic audit information
    id: str = Field(
        ...,
        description="Unique audit identifier"
    )

    site_id: str = Field(
        ...,
        description="Site identifier this audit targets"
    )

    env: str = Field(
        ...,
        description="Environment this audit targets"
    )

    status: Literal['queued', 'running', 'completed', 'failed'] = Field(
        ...,
        description="Current audit status"
    )

    # Timing information
    created_at: datetime = Field(
        ...,
        description="When the audit was created"
    )

    started_at: Optional[datetime] = Field(
        default=None,
        description="When the audit execution started"
    )

    finished_at: Optional[datetime] = Field(
        default=None,
        description="When the audit execution finished"
    )

    updated_at: datetime = Field(
        ...,
        description="When the audit was last updated"
    )

    # Progress and results
    progress_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Completion percentage for running audits"
    )

    summary: Optional[AuditSummary] = Field(
        default=None,
        description="Summary statistics (available when audit completes)"
    )

    # Configuration and metadata
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parameters used for this audit"
    )

    rules_path: Optional[str] = Field(
        default=None,
        description="Rules configuration file used"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata associated with this audit"
    )

    # Error information
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if audit failed"
    )

    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed error information for debugging"
    )

    # Detailed audit results (only populated for completed audits)
    pages: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Detailed page-level results"
    )

    tags: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tag detection results"
    )

    health: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Performance and health metrics"
    )

    duplicates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Duplicate tag detection results"
    )

    variables: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Data layer variables analysis"
    )

    cookies: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Cookie usage analysis"
    )

    rules: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Rule violation results"
    )

    # Privacy analysis results
    privacy_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Privacy compliance summary"
    )

    # Links to related resources
    links: AuditLinks = Field(
        ...,
        description="Links to exports and artifacts"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "id": "audit_2024011501_ecommerce_prod",
                "site_id": "ecommerce",
                "env": "production",
                "status": "completed",
                "created_at": "2024-01-15T10:30:00Z",
                "started_at": "2024-01-15T10:30:15Z",
                "finished_at": "2024-01-15T10:45:23Z",
                "updated_at": "2024-01-15T10:45:23Z",
                "progress_percent": 100.0,
                "summary": {
                    "pages_discovered": 47,
                    "pages_processed": 45,
                    "pages_failed": 2,
                    "requests_captured": 1247,
                    "cookies_found": 23,
                    "tags_detected": 15,
                    "errors_count": 3,
                    "duration_seconds": 892.5
                },
                "params": {
                    "max_pages": 200,
                    "discovery_mode": "hybrid"
                },
                "metadata": {
                    "triggered_by": "scheduled_job",
                    "campaign": "holiday_checkout_monitoring"
                },
                "pages": [
                    {
                        "id": "page_1",
                        "url": "https://example.com/",
                        "status": "completed",
                        "load_time": 1250,
                        "tags_count": 5,
                        "issues_count": 0
                    }
                ],
                "tags": [
                    {
                        "vendor": "google",
                        "tag_type": "analytics",
                        "measurement_id": "G-XXXXXXXXXX",
                        "confidence": "high",
                        "events_count": 12
                    }
                ],
                "health": {
                    "load_performance": "Good",
                    "error_rate": "2.1%",
                    "tag_coverage": "94%",
                    "issues": []
                },
                "cookies": [
                    {
                        "name": "_ga",
                        "domain": ".example.com",
                        "category": "analytics",
                        "max_age": 63072000,
                        "privacy_impact": "medium"
                    }
                ],
                "privacy_summary": {
                    "total_cookies": 23,
                    "analytics_cookies": 8,
                    "marketing_cookies": 12,
                    "functional_cookies": 3
                },
                "links": {
                    "self": "https://api.example.com/api/audits/audit_2024011501_ecommerce_prod",
                    "requests_export": "https://api.example.com/api/exports/audit_2024011501_ecommerce_prod/requests.json",
                    "cookies_export": "https://api.example.com/api/exports/audit_2024011501_ecommerce_prod/cookies.csv",
                    "tags_export": "https://api.example.com/api/exports/audit_2024011501_ecommerce_prod/tags.json",
                    "artifacts": "https://api.example.com/artifacts/audit_2024011501_ecommerce_prod/"
                }
            }
        }


class AuditList(BaseModel):
    """Paginated list of audits with metadata.

    This schema provides a paginated collection of audit references
    along with pagination metadata and optional summary statistics.
    """

    audits: List[AuditRef] = Field(
        ...,
        description="List of audit references"
    )

    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of audits matching the filter criteria"
    )

    has_more: bool = Field(
        ...,
        description="Whether there are more results available"
    )

    next_cursor: Optional[str] = Field(
        default=None,
        description="Cursor for retrieving the next page of results"
    )

    prev_cursor: Optional[str] = Field(
        default=None,
        description="Cursor for retrieving the previous page of results"
    )

    summary_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Summary statistics across all matching audits"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "audits": [
                    {
                        "id": "audit_2024011501_ecommerce_prod",
                        "site_id": "ecommerce",
                        "env": "production",
                        "status": "completed",
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-01-15T10:45:23Z"
                    },
                    {
                        "id": "audit_2024011401_ecommerce_prod",
                        "site_id": "ecommerce",
                        "env": "production",
                        "status": "completed",
                        "created_at": "2024-01-14T10:30:00Z",
                        "updated_at": "2024-01-14T10:42:15Z"
                    }
                ],
                "total_count": 127,
                "has_more": True,
                "next_cursor": "eyJpZCI6ImF1ZGl0XzIwMjQwMTE0MDFfZWNvbW1lcmNlX3Byb2QiLCJzb3J0IjoiY3JlYXRlZF9hdCJ9",
                "prev_cursor": "eyJpZCI6ImF1ZGl0XzIwMjQwMTE2MDFfZWNvbW1lcmNlX3Byb2QiLCJzb3J0IjoiY3JlYXRlZF9hdCJ9",
                "summary_stats": {
                    "total_pages_processed": 15420,
                    "avg_duration_seconds": 654.2,
                    "success_rate": 97.3
                }
            }
        }


class ExportMetadata(BaseModel):
    """Metadata for export operations.

    This schema provides information about export generation,
    including timing, record counts, and format details.
    """

    audit_id: str = Field(
        ...,
        description="Audit identifier this export is for"
    )

    export_type: str = Field(
        ...,
        description="Type of export (requests, cookies, tags, etc.)"
    )

    format: str = Field(
        ...,
        description="Export format (json, ndjson, csv)"
    )

    generated_at: datetime = Field(
        ...,
        description="When this export was generated"
    )

    record_count: int = Field(
        ...,
        ge=0,
        description="Number of records in the export"
    )

    size_bytes: int = Field(
        ...,
        ge=0,
        description="Export size in bytes"
    )

    filters_applied: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filters that were applied to generate this export"
    )

    fields_included: Optional[List[str]] = Field(
        default=None,
        description="Fields included in the export"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "audit_id": "audit_2024011501_ecommerce_prod",
                "export_type": "requests",
                "format": "csv",
                "generated_at": "2024-01-15T11:00:00Z",
                "record_count": 1247,
                "size_bytes": 324567,
                "filters_applied": {
                    "status": ["success"],
                    "resource_type": ["script", "xhr"]
                },
                "fields_included": ["url", "status", "response_time", "resource_type"]
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response schema.

    This schema provides consistent error information across
    all API endpoints, including error codes, messages, and debugging details.
    """

    error: str = Field(
        ...,
        description="Error code or type"
    )

    message: str = Field(
        ...,
        description="Human-readable error message"
    )

    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details for debugging"
    )

    request_id: Optional[str] = Field(
        default=None,
        description="Unique request identifier for tracking"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid site_id format. Must contain only alphanumeric characters, hyphens, underscores, and dots.",
                "details": {
                    "field": "site_id",
                    "value": "invalid@site",
                    "pattern": "^[a-zA-Z0-9_\\-\\.]+$"
                },
                "request_id": "req_2024011501_abc123",
                "timestamp": "2024-01-15T11:00:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema.

    This schema provides system health and status information
    for monitoring and operational purposes.
    """

    status: Literal['healthy', 'degraded', 'unhealthy'] = Field(
        ...,
        description="Overall system health status"
    )

    version: str = Field(
        ...,
        description="API version"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )

    services: Dict[str, Literal['healthy', 'degraded', 'unhealthy']] = Field(
        ...,
        description="Health status of individual services"
    )

    uptime_seconds: float = Field(
        ...,
        ge=0,
        description="Application uptime in seconds"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T11:00:00Z",
                "services": {
                    "database": "healthy",
                    "cache": "healthy",
                    "audit_runner": "healthy",
                    "browser_engine": "degraded"
                },
                "uptime_seconds": 86400.5
            }
        }