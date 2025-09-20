"""Export data schemas for Tag Sentinel REST API.

This module defines Pydantic models for structured export data,
including request logs, cookie inventories, and tag detection results.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Literal, Union
from pydantic import BaseModel, Field, HttpUrl


class RequestLogExport(BaseModel):
    """Schema for network request log export records.

    This schema defines the structure of individual request records
    in exported data, providing comprehensive request/response information.
    """

    # Request identification
    id: str = Field(
        ...,
        description="Unique request identifier within the audit"
    )

    audit_id: str = Field(
        ...,
        description="Audit this request belongs to"
    )

    page_url: HttpUrl = Field(
        ...,
        description="URL of the page that initiated this request"
    )

    # Request details
    url: HttpUrl = Field(
        ...,
        description="Request URL"
    )

    method: str = Field(
        ...,
        description="HTTP method (GET, POST, etc.)"
    )

    resource_type: Literal[
        'document', 'stylesheet', 'image', 'media', 'font', 'script',
        'texttrack', 'xhr', 'fetch', 'eventsource', 'websocket', 'manifest', 'other'
    ] = Field(
        ...,
        description="Type of resource requested"
    )

    # Response details
    status: Literal['pending', 'success', 'failed', 'timeout', 'aborted'] = Field(
        ...,
        description="Request status"
    )

    status_code: Optional[int] = Field(
        default=None,
        ge=100,
        le=599,
        description="HTTP status code"
    )

    response_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Response body size in bytes"
    )

    # Timing information
    timestamp: datetime = Field(
        ...,
        description="When the request was initiated"
    )

    response_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total response time in milliseconds"
    )

    dns_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="DNS lookup time in milliseconds"
    )

    connect_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="Connection establishment time in milliseconds"
    )

    # Headers and metadata
    request_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Request headers (may be filtered for privacy)"
    )

    response_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Response headers (may be filtered for privacy)"
    )

    # Analytics detection
    is_analytics: bool = Field(
        default=False,
        description="Whether this request was identified as analytics-related"
    )

    analytics_vendor: Optional[str] = Field(
        default=None,
        description="Detected analytics vendor (GA4, GTM, etc.)"
    )

    analytics_type: Optional[str] = Field(
        default=None,
        description="Type of analytics request (pageview, event, etc.)"
    )

    # Error information
    error_type: Optional[str] = Field(
        default=None,
        description="Error type if request failed"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if request failed"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "id": "req_001_page_001",
                "audit_id": "audit_2024011501_ecommerce_prod",
                "page_url": "https://shop.example.com/checkout",
                "url": "https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID",
                "method": "GET",
                "resource_type": "script",
                "status": "success",
                "status_code": 200,
                "response_size": 45678,
                "timestamp": "2024-01-15T10:35:00Z",
                "response_time": 234.5,
                "dns_time": 12.3,
                "connect_time": 45.7,
                "request_headers": {
                    "User-Agent": "Mozilla/5.0...",
                    "Accept": "application/javascript"
                },
                "response_headers": {
                    "Content-Type": "application/javascript",
                    "Cache-Control": "public, max-age=3600"
                },
                "is_analytics": True,
                "analytics_vendor": "Google Analytics",
                "analytics_type": "script_load"
            }
        }


class CookieExport(BaseModel):
    """Schema for cookie inventory export records.

    This schema defines the structure of individual cookie records
    in exported data, providing comprehensive cookie analysis information.
    """

    # Cookie identification
    audit_id: str = Field(
        ...,
        description="Audit this cookie was discovered in"
    )

    page_url: HttpUrl = Field(
        ...,
        description="URL where this cookie was first observed"
    )

    # Cookie details
    name: str = Field(
        ...,
        description="Cookie name"
    )

    domain: str = Field(
        ...,
        description="Cookie domain"
    )

    path: str = Field(
        default="/",
        description="Cookie path"
    )

    value: Optional[str] = Field(
        default=None,
        description="Cookie value (may be redacted for privacy)"
    )

    # Cookie attributes
    secure: bool = Field(
        default=False,
        description="Whether cookie has Secure flag"
    )

    http_only: bool = Field(
        default=False,
        description="Whether cookie has HttpOnly flag"
    )

    same_site: Optional[Literal['Strict', 'Lax', 'None']] = Field(
        default=None,
        description="SameSite attribute value"
    )

    expires: Optional[datetime] = Field(
        default=None,
        description="Cookie expiration date"
    )

    max_age: Optional[int] = Field(
        default=None,
        description="Max-Age in seconds"
    )

    # Discovery metadata
    discovered_at: datetime = Field(
        ...,
        description="When this cookie was first discovered"
    )

    source: Literal['http_header', 'javascript', 'document_cookie'] = Field(
        ...,
        description="How this cookie was set"
    )

    # Classification and analysis
    category: Optional[str] = Field(
        default=None,
        description="Cookie category (analytics, marketing, functional, etc.)"
    )

    vendor: Optional[str] = Field(
        default=None,
        description="Identified vendor/service that set this cookie"
    )

    purpose: Optional[str] = Field(
        default=None,
        description="Identified purpose of this cookie"
    )

    # Privacy analysis
    is_essential: Optional[bool] = Field(
        default=None,
        description="Whether cookie is classified as essential"
    )

    gdpr_compliance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="GDPR compliance analysis"
    )

    # Cross-scenario comparison
    privacy_scenario_detected: bool = Field(
        default=False,
        description="Whether cookie was also detected in privacy-focused scenario"
    )

    baseline_scenario_detected: bool = Field(
        default=False,
        description="Whether cookie was detected in baseline scenario"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "audit_id": "audit_2024011501_ecommerce_prod",
                "page_url": "https://shop.example.com/checkout",
                "name": "_ga",
                "domain": ".example.com",
                "path": "/",
                "value": "[REDACTED]",
                "secure": True,
                "http_only": False,
                "same_site": "Lax",
                "expires": "2026-01-15T10:35:00Z",
                "max_age": 63072000,
                "discovered_at": "2024-01-15T10:35:00Z",
                "source": "javascript",
                "category": "analytics",
                "vendor": "Google Analytics",
                "purpose": "User tracking and analytics",
                "is_essential": False,
                "gdpr_compliance": {
                    "requires_consent": True,
                    "lawful_basis": "consent"
                },
                "privacy_scenario_detected": False,
                "baseline_scenario_detected": True
            }
        }


class TagExport(BaseModel):
    """Schema for analytics tag detection export records.

    This schema defines the structure of individual tag detection records
    in exported data, providing comprehensive tag analysis information.
    """

    # Tag identification
    audit_id: str = Field(
        ...,
        description="Audit this tag was detected in"
    )

    page_url: HttpUrl = Field(
        ...,
        description="URL where this tag was detected"
    )

    tag_id: str = Field(
        ...,
        description="Unique identifier for this tag instance"
    )

    # Tag details
    vendor: str = Field(
        ...,
        description="Tag vendor (Google Analytics, GTM, Facebook, etc.)"
    )

    tag_type: str = Field(
        ...,
        description="Type of tag (pageview, event, conversion, etc.)"
    )

    implementation_method: Literal['script_tag', 'gtm', 'direct_api', 'pixel'] = Field(
        ...,
        description="How the tag is implemented"
    )

    # Detection metadata
    detected_at: datetime = Field(
        ...,
        description="When this tag was detected"
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0.0 to 1.0)"
    )

    detection_method: str = Field(
        ...,
        description="Method used to detect this tag"
    )

    # Tag configuration
    tracking_id: Optional[str] = Field(
        default=None,
        description="Tracking/measurement ID associated with tag"
    )

    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tag parameters and configuration"
    )

    # Load and performance analysis
    load_order: Optional[int] = Field(
        default=None,
        description="Order in which this tag loaded relative to others"
    )

    load_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="Tag load time in milliseconds"
    )

    blocking: bool = Field(
        default=False,
        description="Whether tag blocks page rendering"
    )

    # Data layer integration
    uses_data_layer: bool = Field(
        default=False,
        description="Whether tag reads from data layer"
    )

    data_layer_variables: Optional[List[str]] = Field(
        default=None,
        description="Data layer variables accessed by this tag"
    )

    # Compliance and privacy
    collects_pii: Optional[bool] = Field(
        default=None,
        description="Whether tag potentially collects PII"
    )

    consent_required: bool = Field(
        default=False,
        description="Whether tag requires user consent"
    )

    consent_status: Optional[str] = Field(
        default=None,
        description="Detected consent status when tag fired"
    )

    # Error and validation
    has_errors: bool = Field(
        default=False,
        description="Whether tag had any errors or warnings"
    )

    validation_issues: Optional[List[str]] = Field(
        default=None,
        description="List of validation issues found"
    )

    # Cross-scenario comparison
    privacy_scenario_detected: bool = Field(
        default=False,
        description="Whether tag was also detected in privacy-focused scenario"
    )

    baseline_scenario_detected: bool = Field(
        default=False,
        description="Whether tag was detected in baseline scenario"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "audit_id": "audit_2024011501_ecommerce_prod",
                "page_url": "https://shop.example.com/checkout",
                "tag_id": "ga4_pageview_checkout_001",
                "vendor": "Google Analytics 4",
                "tag_type": "pageview",
                "implementation_method": "gtm",
                "detected_at": "2024-01-15T10:35:12Z",
                "confidence_score": 0.95,
                "detection_method": "network_request_analysis",
                "tracking_id": "G-XXXXXXXXXX",
                "parameters": {
                    "page_title": "Checkout - Step 1",
                    "page_location": "https://shop.example.com/checkout",
                    "currency": "USD"
                },
                "load_order": 3,
                "load_time": 145.2,
                "blocking": False,
                "uses_data_layer": True,
                "data_layer_variables": ["page_title", "user_id", "cart_value"],
                "collects_pii": True,
                "consent_required": True,
                "consent_status": "granted",
                "has_errors": False,
                "validation_issues": None,
                "privacy_scenario_detected": False,
                "baseline_scenario_detected": True
            }
        }


class DataLayerExport(BaseModel):
    """Schema for data layer snapshot export records.

    This schema defines the structure of data layer snapshot records
    in exported data, providing comprehensive data layer analysis.
    """

    # Data layer identification
    audit_id: str = Field(
        ...,
        description="Audit this data layer snapshot belongs to"
    )

    page_url: HttpUrl = Field(
        ...,
        description="URL where this data layer was captured"
    )

    snapshot_id: str = Field(
        ...,
        description="Unique identifier for this snapshot"
    )

    # Snapshot metadata
    captured_at: datetime = Field(
        ...,
        description="When this snapshot was captured"
    )

    trigger_event: Optional[str] = Field(
        default=None,
        description="Event that triggered this snapshot"
    )

    # Data layer content
    data: Dict[str, Any] = Field(
        ...,
        description="Data layer content (may be redacted for privacy)"
    )

    # Structure analysis
    total_properties: int = Field(
        ...,
        ge=0,
        description="Total number of properties in data layer"
    )

    nested_levels: int = Field(
        ...,
        ge=0,
        description="Maximum nesting depth of data layer structure"
    )

    # Data classification
    contains_pii: bool = Field(
        default=False,
        description="Whether data layer contains potential PII"
    )

    pii_fields: Optional[List[str]] = Field(
        default=None,
        description="Fields identified as potential PII"
    )

    # Validation results
    schema_valid: Optional[bool] = Field(
        default=None,
        description="Whether data layer validates against expected schema"
    )

    validation_errors: Optional[List[str]] = Field(
        default=None,
        description="Schema validation errors if any"
    )

    # Cross-scenario comparison
    privacy_scenario_captured: bool = Field(
        default=False,
        description="Whether similar data was captured in privacy scenario"
    )

    baseline_scenario_captured: bool = Field(
        default=False,
        description="Whether data was captured in baseline scenario"
    )

    differences_from_baseline: Optional[List[str]] = Field(
        default=None,
        description="Differences from baseline scenario data layer"
    )

    class Config:
        """Pydantic configuration with examples for OpenAPI documentation."""
        json_schema_extra = {
            "example": {
                "audit_id": "audit_2024011501_ecommerce_prod",
                "page_url": "https://shop.example.com/checkout",
                "snapshot_id": "dl_snapshot_checkout_001",
                "captured_at": "2024-01-15T10:35:12Z",
                "trigger_event": "page_load",
                "data": {
                    "event": "page_view",
                    "page_title": "Checkout - Step 1",
                    "page_location": "https://shop.example.com/checkout",
                    "user_id": "[REDACTED]",
                    "cart": {
                        "items": 3,
                        "value": 125.99,
                        "currency": "USD"
                    }
                },
                "total_properties": 6,
                "nested_levels": 2,
                "contains_pii": True,
                "pii_fields": ["user_id", "email"],
                "schema_valid": True,
                "validation_errors": None,
                "privacy_scenario_captured": False,
                "baseline_scenario_captured": True,
                "differences_from_baseline": None
            }
        }