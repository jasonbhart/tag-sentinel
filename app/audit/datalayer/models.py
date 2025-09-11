"""Pydantic models for dataLayer snapshots, validation, and configuration.

This module defines the core data models used by the DataLayer Integrity system,
including snapshot capture, schema validation, configuration, and aggregation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Literal, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, AnyHttpUrl


class RedactionMethod(str, Enum):
    """Methods for redacting sensitive data."""
    REMOVE = "remove"        # Delete the field entirely
    HASH = "hash"           # Replace with SHA-256 hash
    MASK = "mask"           # Replace with asterisks
    TRUNCATE = "truncate"   # Keep first N characters


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"


class DLContext(BaseModel):
    """Configuration and context for dataLayer capture operations."""
    
    env: str = Field(description="Environment name (e.g., dev, staging, prod)")
    data_layer_object: str = Field(
        default="dataLayer",
        description="Name of the global object to capture (default: dataLayer)"
    )
    
    # Capture limits
    max_depth: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum nesting depth to capture"
    )
    max_entries: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum number of entries to capture"
    )
    max_size_bytes: int | None = Field(
        default=1048576,  # 1MB
        ge=1024,
        description="Maximum capture size in bytes"
    )
    
    # Redaction configuration
    redact_paths: List[str] = Field(
        default_factory=list,
        description="JSON Pointer paths or glob patterns for redaction"
    )
    redaction_method: RedactionMethod = Field(
        default=RedactionMethod.HASH,
        description="Default redaction method"
    )
    
    # Schema validation
    schema_path: str | None = Field(
        default=None,
        description="Path to JSON/YAML schema file for validation"
    )
    
    # Site-specific overrides
    site_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Site-specific configuration overrides"
    )


class DataLayerSnapshot(BaseModel):
    """Captured dataLayer state from a single page."""
    
    # Page identification
    page_url: AnyHttpUrl = Field(description="URL of the page where dataLayer was captured")
    capture_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the snapshot was taken"
    )
    
    # DataLayer state
    exists: bool = Field(description="Whether dataLayer object exists on the page")
    latest: Dict[str, Any] | None = Field(
        default=None,
        description="Latest state of dataLayer variables (flattened from pushes)"
    )
    events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Event pushes extracted from dataLayer"
    )
    
    # Capture metadata
    object_name: str = Field(
        default="dataLayer",
        description="Name of the captured object"
    )
    size_bytes: int | None = Field(
        default=None,
        description="Estimated size of captured data in bytes"
    )
    truncated: bool = Field(
        default=False,
        description="Whether capture was truncated due to size/depth limits"
    )
    depth_reached: int | None = Field(
        default=None,
        description="Maximum nesting depth found in the data"
    )
    entries_captured: int = Field(
        default=0,
        description="Total number of entries captured"
    )
    
    # Redaction tracking
    redacted_paths: List[str] = Field(
        default_factory=list,
        description="Paths that were redacted during capture"
    )
    redaction_method_used: RedactionMethod | None = Field(
        default=None,
        description="Redaction method applied"
    )
    
    @field_validator('page_url', mode='before')
    @classmethod
    def validate_page_url(cls, v):
        """Validate and normalize page URL."""
        if isinstance(v, str):
            # Handle URLs without protocol
            if not v.startswith(('http://', 'https://')):
                v = f'https://{v}'
        return v
    
    @property
    def host(self) -> str:
        """Extract host from page URL."""
        return urlparse(str(self.page_url)).netloc
    
    @property
    def variable_count(self) -> int:
        """Count of variables in latest state."""
        return len(self.latest) if self.latest else 0
    
    @property
    def event_count(self) -> int:
        """Count of events captured."""
        return len(self.events)
    
    @property
    def is_complete(self) -> bool:
        """Check if capture was complete (not truncated)."""
        return not self.truncated
    
    def get_variable_names(self) -> List[str]:
        """Get list of all variable names in latest state."""
        if not self.latest:
            return []
        
        def _extract_keys(obj: Dict[str, Any], prefix: str = '') -> List[str]:
            keys = []
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                keys.append(full_key)
                if isinstance(value, dict):
                    keys.extend(_extract_keys(value, full_key))
            return keys
        
        return _extract_keys(self.latest)
    
    def get_event_types(self) -> List[str]:
        """Get list of unique event types/names."""
        event_types = set()
        for event in self.events:
            if 'event' in event:
                event_types.add(event['event'])
            elif 'eventName' in event:
                event_types.add(event['eventName'])
            elif 'type' in event:
                event_types.add(event['type'])
        return list(event_types)


class ValidationIssue(BaseModel):
    """A validation issue found during schema validation."""
    
    # Issue identification
    page_url: AnyHttpUrl = Field(description="URL where the issue was found")
    path: str = Field(description="JSON Pointer path to the problematic field")
    
    # Issue details
    severity: ValidationSeverity = Field(description="Severity level of the issue")
    message: str = Field(description="Human-readable error message")
    
    # Schema validation context
    schema_rule: str | None = Field(
        default=None,
        description="JSON Schema rule that was violated"
    )
    expected: Any | None = Field(
        default=None,
        description="Expected value or type according to schema"
    )
    observed: Any | None = Field(
        default=None,
        description="Actual value that was observed"
    )
    
    # Additional context
    variable_name: str | None = Field(
        default=None,
        description="Variable name if issue is related to a specific variable"
    )
    event_type: str | None = Field(
        default=None,
        description="Event type if issue is related to a specific event"
    )
    
    @field_validator('path')
    @classmethod
    def validate_json_pointer_path(cls, v):
        """Validate that path is a valid JSON Pointer."""
        if not v.startswith('/'):
            # Convert dot notation to JSON Pointer if needed
            if '.' in v:
                parts = v.split('.')
                v = '/' + '/'.join(parts)
            else:
                v = f'/{v}'
        return v
    
    @property
    def field_name(self) -> str:
        """Extract field name from JSON Pointer path."""
        if '/' not in self.path:
            return self.path
        return self.path.split('/')[-1]
    
    @property
    def is_critical(self) -> bool:
        """Check if this is a critical issue."""
        return self.severity == ValidationSeverity.CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'page_url': str(self.page_url),
            'path': self.path,
            'severity': self.severity,
            'message': self.message,
            'schema_rule': self.schema_rule,
            'expected': self.expected,
            'observed': self.observed,
            'variable_name': self.variable_name,
            'event_type': self.event_type
        }


class DLResult(BaseModel):
    """Comprehensive result from dataLayer capture and validation."""
    
    # Core components
    snapshot: DataLayerSnapshot = Field(description="Captured dataLayer snapshot")
    issues: List[ValidationIssue] = Field(
        default_factory=list,
        description="Validation issues found"
    )
    
    # Aggregation delta for run-level tracking
    aggregate_delta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Incremental data for run-level aggregation"
    )
    
    # Processing metadata
    processing_time_ms: float | None = Field(
        default=None,
        description="Total processing time in milliseconds"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Processing notes and warnings"
    )
    
    # Error tracking
    capture_error: str | None = Field(
        default=None,
        description="Error message if capture failed"
    )
    validation_error: str | None = Field(
        default=None,
        description="Error message if validation failed"
    )
    
    @property
    def is_successful(self) -> bool:
        """Check if overall operation was successful."""
        return self.snapshot.exists and not self.capture_error
    
    @property
    def has_validation_errors(self) -> bool:
        """Check if there are any validation issues."""
        return len(self.issues) > 0
    
    @property
    def critical_issues_count(self) -> int:
        """Count of critical validation issues."""
        return len([issue for issue in self.issues if issue.is_critical])
    
    @property
    def warning_issues_count(self) -> int:
        """Count of warning-level validation issues."""
        return len([issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING])
    
    def add_note(self, note: str) -> None:
        """Add a processing note."""
        self.notes.append(note)
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def export_summary(self) -> Dict[str, Any]:
        """Export summary for reporting."""
        return {
            'page_url': str(self.snapshot.page_url),
            'capture_successful': self.is_successful,
            'dataLayer_exists': self.snapshot.exists,
            'variable_count': self.snapshot.variable_count,
            'event_count': self.snapshot.event_count,
            'total_issues': len(self.issues),
            'critical_issues': self.critical_issues_count,
            'warning_issues': self.warning_issues_count,
            'capture_truncated': self.snapshot.truncated,
            'processing_time_ms': self.processing_time_ms,
            'has_errors': bool(self.capture_error or self.validation_error)
        }


class VariablePresence(BaseModel):
    """Tracking of variable presence across pages."""
    
    name: str = Field(description="Variable name")
    path: str = Field(description="JSON Pointer path to variable")
    pages_with_variable: int = Field(default=0, description="Number of pages with this variable")
    total_pages: int = Field(default=0, description="Total pages processed")
    example_values: List[Any] = Field(
        default_factory=list,
        description="Sample values observed"
    )
    types_observed: List[str] = Field(
        default_factory=list,
        description="Data types observed for this variable"
    )
    
    @property
    def presence_percentage(self) -> float:
        """Calculate presence percentage."""
        if self.total_pages == 0:
            return 0.0
        return (self.pages_with_variable / self.total_pages) * 100.0
    
    @property
    def is_consistent_type(self) -> bool:
        """Check if variable has consistent type across pages."""
        return len(set(self.types_observed)) <= 1


class EventFrequency(BaseModel):
    """Tracking of event frequency across pages."""
    
    event_type: str = Field(description="Event type/name")
    total_occurrences: int = Field(default=0, description="Total event occurrences")
    pages_with_event: int = Field(default=0, description="Number of pages with this event")
    total_pages: int = Field(default=0, description="Total pages processed")
    example_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sample event objects"
    )
    
    @property
    def average_per_page(self) -> float:
        """Calculate average occurrences per page."""
        if self.total_pages == 0:
            return 0.0
        return self.total_occurrences / self.total_pages
    
    @property
    def presence_percentage(self) -> float:
        """Calculate presence percentage across pages."""
        if self.total_pages == 0:
            return 0.0
        return (self.pages_with_event / self.total_pages) * 100.0


class DLAggregate(BaseModel):
    """Run-level aggregation of dataLayer data across all pages."""
    
    # Run metadata
    run_id: str | None = Field(default=None, description="Unique run identifier")
    start_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When aggregation started"
    )
    end_time: datetime | None = Field(default=None, description="When aggregation completed")
    
    # Page statistics
    total_pages: int = Field(default=0, description="Total pages processed")
    pages_with_datalayer: int = Field(default=0, description="Pages with dataLayer present")
    pages_successful: int = Field(default=0, description="Pages successfully captured")
    pages_failed: int = Field(default=0, description="Pages with capture failures")
    
    # Variable analysis
    variables: Dict[str, VariablePresence] = Field(
        default_factory=dict,
        description="Variable presence tracking"
    )
    
    # Event analysis
    events: Dict[str, EventFrequency] = Field(
        default_factory=dict,
        description="Event frequency tracking"
    )
    
    # Validation statistics
    total_validation_issues: int = Field(default=0, description="Total validation issues")
    critical_issues: int = Field(default=0, description="Critical validation issues")
    warning_issues: int = Field(default=0, description="Warning validation issues")
    
    # Performance metrics
    average_processing_time_ms: float | None = Field(
        default=None,
        description="Average processing time per page"
    )
    total_data_captured_bytes: int = Field(
        default=0,
        description="Total bytes of dataLayer data captured"
    )
    
    @property
    def datalayer_presence_rate(self) -> float:
        """Calculate dataLayer presence rate as percentage."""
        if self.total_pages == 0:
            return 0.0
        return (self.pages_with_datalayer / self.total_pages) * 100.0
    
    @property
    def success_rate(self) -> float:
        """Calculate capture success rate as percentage."""
        if self.total_pages == 0:
            return 0.0
        return (self.pages_successful / self.total_pages) * 100.0
    
    @property
    def unique_variables_count(self) -> int:
        """Count of unique variables across all pages."""
        return len(self.variables)
    
    @property
    def unique_events_count(self) -> int:
        """Count of unique event types across all pages."""
        return len(self.events)
    
    @property
    def most_common_variables(self) -> List[str]:
        """Get variables sorted by presence rate (most common first)."""
        return sorted(
            self.variables.keys(),
            key=lambda var: self.variables[var].presence_percentage,
            reverse=True
        )
    
    @property
    def most_frequent_events(self) -> List[str]:
        """Get events sorted by frequency (most frequent first)."""
        return sorted(
            self.events.keys(),
            key=lambda event: self.events[event].total_occurrences,
            reverse=True
        )
    
    def add_page_result(self, result: DLResult) -> None:
        """Add a page result to the aggregation."""
        self.total_pages += 1
        
        if result.snapshot.exists:
            self.pages_with_datalayer += 1
        
        if result.is_successful:
            self.pages_successful += 1
        else:
            self.pages_failed += 1
        
        # Track validation issues
        self.total_validation_issues += len(result.issues)
        self.critical_issues += result.critical_issues_count
        self.warning_issues += result.warning_issues_count
        
        # Update performance metrics
        if result.processing_time_ms:
            if self.average_processing_time_ms is None:
                self.average_processing_time_ms = result.processing_time_ms
            else:
                # Running average
                total_time = self.average_processing_time_ms * (self.total_pages - 1)
                self.average_processing_time_ms = (total_time + result.processing_time_ms) / self.total_pages
        
        # Track data size
        if result.snapshot.size_bytes:
            self.total_data_captured_bytes += result.snapshot.size_bytes
        
        # Track variables
        if result.snapshot.latest:
            for var_name in result.snapshot.get_variable_names():
                if var_name not in self.variables:
                    self.variables[var_name] = VariablePresence(
                        name=var_name,
                        path=f"/{var_name.replace('.', '/')}",
                        total_pages=self.total_pages
                    )
                
                var_presence = self.variables[var_name]
                var_presence.pages_with_variable += 1
                var_presence.total_pages = self.total_pages
                
                # Update example values and types
                value = self._get_nested_value(result.snapshot.latest, var_name)
                if value is not None:
                    value_type = type(value).__name__
                    if value_type not in var_presence.types_observed:
                        var_presence.types_observed.append(value_type)
                    
                    # Keep up to 3 example values
                    if len(var_presence.example_values) < 3:
                        var_presence.example_values.append(value)
        
        # Track events
        for event in result.snapshot.events:
            event_types = result.snapshot.get_event_types()
            for event_type in event_types:
                if event_type not in self.events:
                    self.events[event_type] = EventFrequency(
                        event_type=event_type,
                        total_pages=self.total_pages
                    )
                
                event_freq = self.events[event_type]
                event_freq.total_occurrences += 1
                event_freq.pages_with_event += 1
                event_freq.total_pages = self.total_pages
                
                # Keep up to 3 example events
                if len(event_freq.example_events) < 3:
                    event_freq.example_events.append(event)
    
    def _get_nested_value(self, obj: Dict[str, Any], path: str) -> Any:
        """Get value from nested object using dot notation path."""
        keys = path.split('.')
        current = obj
        
        try:
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current
        except (KeyError, TypeError):
            return None
    
    def finalize(self) -> None:
        """Finalize aggregation (called when run is complete)."""
        self.end_time = datetime.utcnow()
        
        # Update total_pages for all variables and events
        for var_presence in self.variables.values():
            var_presence.total_pages = self.total_pages
        
        for event_freq in self.events.values():
            event_freq.total_pages = self.total_pages
    
    def export_summary(self) -> Dict[str, Any]:
        """Export comprehensive summary for reporting."""
        return {
            'run_id': self.run_id,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            'total_pages': self.total_pages,
            'pages_with_datalayer': self.pages_with_datalayer,
            'datalayer_presence_rate': self.datalayer_presence_rate,
            'success_rate': self.success_rate,
            'unique_variables': self.unique_variables_count,
            'unique_events': self.unique_events_count,
            'total_validation_issues': self.total_validation_issues,
            'critical_issues': self.critical_issues,
            'warning_issues': self.warning_issues,
            'average_processing_time_ms': self.average_processing_time_ms,
            'total_data_captured_mb': round(self.total_data_captured_bytes / 1024 / 1024, 2),
            'most_common_variables': self.most_common_variables[:10],
            'most_frequent_events': self.most_frequent_events[:10]
        }