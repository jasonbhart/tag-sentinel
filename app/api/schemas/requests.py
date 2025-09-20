"""API request schemas for Tag Sentinel REST API.

This module defines Pydantic models for all API request payloads,
including audit creation, filtering, and export requests.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
import re


class CreateAuditRequest(BaseModel):
    """Request schema for creating a new audit run.

    This schema validates all parameters needed to trigger a new audit,
    including site configuration, environment targeting, and optional
    custom parameters and rules.
    """

    site_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_\-\.]+$',
        description="Unique identifier for the site to audit",
        examples=["ecommerce", "marketing-site", "docs.example.com"]
    )

    env: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r'^[a-zA-Z0-9_\-]+$',
        description="Target environment for the audit",
        examples=["production", "staging", "dev", "qa"]
    )

    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom parameters to override default audit configuration",
        examples=[{
            "max_pages": 100,
            "max_depth": 3,
            "include_patterns": [".*\\.html$"],
            "discovery_mode": "hybrid"
        }]
    )

    rules_path: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="Path to custom rules configuration file",
        examples=["rules/ecommerce-rules.yaml", "/custom/rules/privacy-focused.json"]
    )

    idempotency_key: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_\-\.]+$',
        description="Optional idempotency key to prevent duplicate audit creation",
        examples=["audit-2024-01-15-001", "daily-check-20240115"]
    )

    priority: Optional[int] = Field(
        default=0,
        ge=-100,
        le=100,
        description="Audit priority (-100 to 100, higher = more important)"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata to associate with this audit",
        examples=[{
            "triggered_by": "ci_pipeline",
            "build_id": "build-12345",
            "branch": "feature/new-checkout"
        }]
    )

    @field_validator('params')
    @classmethod
    def validate_params(cls, v):
        """Validate that params contains only safe parameter names."""
        if v is None:
            return v

        # List of allowed parameter keys for security
        allowed_keys = {
            'max_pages', 'max_depth', 'max_concurrency', 'requests_per_second',
            'page_timeout', 'navigation_timeout', 'max_retries', 'retry_delay',
            'include_patterns', 'exclude_patterns', 'same_site_only',
            'discovery_mode', 'load_wait_strategy', 'load_wait_timeout',
            'user_agent', 'extra_headers', 'respect_robots', 'download_delay_ms',
            'seeds', 'sitemap_url', 'load_wait_selector', 'load_wait_js', 'metadata'
        }

        invalid_keys = set(v.keys()) - allowed_keys
        if invalid_keys:
            raise ValueError(f"Invalid parameter keys: {', '.join(invalid_keys)}")

        # Validate regex patterns if present
        for pattern_key in ['include_patterns', 'exclude_patterns']:
            if pattern_key in v and v[pattern_key]:
                for pattern in v[pattern_key]:
                    try:
                        re.compile(pattern)
                    except re.error as e:
                        raise ValueError(f"Invalid regex pattern in {pattern_key}: {pattern} - {e}")

        return v

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "site_id": "ecommerce",
                "env": "production",
                "params": {
                    "max_pages": 200,
                    "discovery_mode": "hybrid",
                    "include_patterns": [".*\\/checkout\\/.*", ".*\\/product\\/.*"]
                },
                "idempotency_key": "audit-2024-01-15-001",
                "priority": 10,
                "metadata": {
                    "triggered_by": "scheduled_job",
                    "campaign": "holiday_checkout_monitoring"
                }
            }
        }


class ListAuditsRequest(BaseModel):
    """Request schema for listing and filtering audits.

    This schema supports comprehensive filtering, sorting, and pagination
    for audit listing endpoints.
    """

    # Filtering parameters
    site_id: Optional[str] = Field(
        default=None,
        pattern=r'^[a-zA-Z0-9_\-\.]+$',
        description="Filter by site identifier"
    )

    env: Optional[str] = Field(
        default=None,
        pattern=r'^[a-zA-Z0-9_\-]+$',
        description="Filter by environment"
    )

    status: Optional[List[Literal['queued', 'running', 'completed', 'failed']]] = Field(
        default=None,
        description="Filter by audit status (multiple values allowed)",
        examples=[["completed", "failed"], ["running"]]
    )

    date_from: Optional[datetime] = Field(
        default=None,
        description="Filter audits created after this date"
    )

    date_to: Optional[datetime] = Field(
        default=None,
        description="Filter audits created before this date"
    )

    tags: Optional[List[str]] = Field(
        default=None,
        max_length=10,
        description="Filter by metadata tags",
        examples=[["ci_pipeline", "regression_test"]]
    )

    search: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Full-text search across audit metadata"
    )

    # Sorting parameters
    sort_by: Optional[Literal['created_at', 'updated_at', 'duration', 'status', 'site_id']] = Field(
        default='created_at',
        description="Field to sort results by"
    )

    sort_order: Optional[Literal['asc', 'desc']] = Field(
        default='desc',
        description="Sort order"
    )

    # Pagination parameters
    cursor: Optional[str] = Field(
        default=None,
        description="Cursor for pagination (opaque string)"
    )

    limit: Optional[int] = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )

    include_stats: Optional[bool] = Field(
        default=False,
        description="Include summary statistics in response"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "site_id": "ecommerce",
                "env": "production",
                "status": ["completed", "failed"],
                "date_from": "2024-01-01T00:00:00Z",
                "date_to": "2024-01-31T23:59:59Z",
                "sort_by": "created_at",
                "sort_order": "desc",
                "limit": 50,
                "include_stats": True
            }
        }


class ExportRequest(BaseModel):
    """Request schema for data export operations.

    This schema supports filtering and format selection for various
    export endpoints (requests, cookies, tags, etc.).
    """

    format: Literal['json', 'ndjson', 'csv'] = Field(
        default='json',
        description="Export format"
    )

    fields: Optional[List[str]] = Field(
        default=None,
        description="Specific fields to include in export (default: all)",
        examples=[["url", "status", "response_time"], ["cookie_name", "domain", "secure"]]
    )

    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filters to apply to export data",
        examples=[{
            "status": ["success"],
            "resource_type": ["script", "xhr"],
            "response_time_gt": 1000
        }]
    )

    compress: Optional[bool] = Field(
        default=False,
        description="Whether to compress the export (gzip)"
    )

    include_metadata: Optional[bool] = Field(
        default=True,
        description="Include export metadata in response headers"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "format": "csv",
                "fields": ["url", "status", "response_time", "resource_type"],
                "filters": {
                    "status": ["success"],
                    "response_time_gt": 500
                },
                "compress": True,
                "include_metadata": True
            }
        }