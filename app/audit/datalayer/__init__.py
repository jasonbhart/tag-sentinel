"""DataLayer Integrity module for Tag Sentinel.

This module provides comprehensive dataLayer capture, validation, and integrity
checking capabilities for web analytics auditing.

Key Components:
- DataLayerService: Main orchestration service
- Snapshotter: Safe JavaScript-based dataLayer capture  
- Redactor: Privacy-compliant sensitive data redaction
- Validator: JSON Schema-based validation with detailed reporting
- Models: Pydantic data models for all operations

Example Usage:
    from app.audit.datalayer import DataLayerService, capture_page_datalayer
    
    # Simple page capture
    result = await capture_page_datalayer(page, schema_path="schema.json")
    
    # Full service with configuration
    service = DataLayerService()
    result = await service.capture_and_validate(page, site_domain="example.com")
"""

from .models import (
    DataLayerSnapshot,
    ValidationIssue, 
    DLContext,
    DLResult,
    DLAggregate,
    ValidationSeverity,
    RedactionMethod
)

from .config import (
    DataLayerConfig,
    CaptureConfig,
    RedactionConfig,
    SchemaConfig,
    get_datalayer_config,
    get_site_datalayer_config
)

from .snapshot import Snapshotter, BatchSnapshotter
from .redaction import Redactor, RedactionManager
from .validation import Validator, SchemaManager
from .service import DataLayerService, capture_page_datalayer, create_dataLayer_service

__version__ = "1.0.0"

__all__ = [
    # Main service
    "DataLayerService",
    "capture_page_datalayer", 
    "create_dataLayer_service",
    
    # Core components
    "Snapshotter",
    "BatchSnapshotter",
    "Redactor", 
    "RedactionManager",
    "Validator",
    "SchemaManager",
    
    # Data models
    "DataLayerSnapshot",
    "ValidationIssue",
    "DLContext", 
    "DLResult",
    "DLAggregate",
    "ValidationSeverity",
    "RedactionMethod",
    
    # Configuration
    "DataLayerConfig",
    "CaptureConfig",
    "RedactionConfig", 
    "SchemaConfig",
    "get_datalayer_config",
    "get_site_datalayer_config"
]