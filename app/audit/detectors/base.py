"""Base detector protocol and data models for analytics tag detection.

This module defines the core interfaces and data structures that all analytics 
detectors must implement. It provides standardized output formats and detection 
context to ensure consistent behavior across different vendor-specific detectors.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Set, Union, Awaitable
from pydantic import BaseModel, Field

from ..models.capture import PageResult


class Vendor(str, Enum):
    """Supported analytics vendors."""
    GA4 = "ga4"
    GTM = "gtm"
    ADOBE = "adobe"
    FACEBOOK = "facebook"
    UNKNOWN = "unknown"


class TagStatus(str, Enum):
    """Status of a detected tag event."""
    OK = "ok"
    ERROR = "error" 
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    """Confidence level for tag detection."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NoteSeverity(str, Enum):
    """Severity levels for detector notes."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"


class NoteCategory(str, Enum):
    """Categories for detector notes."""
    DUPLICATE = "duplicate"
    SEQUENCING = "sequencing"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"


class TagEvent(BaseModel):
    """Standardized representation of a detected analytics tag event."""
    
    # Core identification
    vendor: Vendor = Field(description="Analytics vendor (ga4, gtm, etc)")
    name: str = Field(description="Event or tag name")
    category: Optional[str] = Field(
        default=None,
        description="Event category for grouping"
    )
    id: Optional[str] = Field(
        default=None,
        description="Tracking ID, measurement ID, or container ID"
    )
    
    # Context
    page_url: str = Field(description="URL where event was detected")
    request_url: Optional[str] = Field(
        default=None,
        description="Network request URL that triggered detection"
    )
    
    # Timing and status
    timing_ms: Optional[int] = Field(
        default=None,
        description="Event timing in milliseconds from page load start"
    )
    status: TagStatus = Field(
        default=TagStatus.UNKNOWN,
        description="Event status"
    )
    confidence: Confidence = Field(
        default=Confidence.MEDIUM,
        description="Detection confidence level"
    )
    
    # Event parameters and data
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event parameters and data"
    )
    
    # Detection metadata
    detection_method: str = Field(
        description="Method used to detect this event (e.g., 'network_request', 'dom_scan')"
    )
    detector_version: str = Field(
        default="1.0.0",
        description="Version of detector that found this event"
    )
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this event was detected"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class DetectorNote(BaseModel):
    """Informational messages, warnings, and errors from detector analysis."""
    
    severity: NoteSeverity = Field(description="Severity level of the note")
    category: NoteCategory = Field(description="Category for grouping notes")
    message: str = Field(description="Human-readable note message")
    
    # Context for the note
    page_url: str = Field(description="URL where note applies")
    related_events: List[str] = Field(
        default_factory=list,
        description="Event names or IDs this note relates to"
    )
    
    # Optional detailed information
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context and debug information"
    )
    
    # Metadata
    detector_name: str = Field(description="Name of detector that generated this note")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this note was generated"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class DetectContext(BaseModel):
    """Environment and configuration context for detector execution."""
    
    # Environment information
    environment: str = Field(
        default="production",
        description="Environment name (production, staging, development)"
    )
    is_production: bool = Field(
        default=True,
        description="Whether running in production environment"
    )
    
    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detector-specific configuration settings"
    )
    
    # Site and audit context
    site_domain: Optional[str] = Field(
        default=None,
        description="Primary domain being audited"
    )
    audit_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the current audit run"
    )
    
    # Processing preferences
    enable_debug: bool = Field(
        default=False,
        description="Enable debug-level processing (may impact performance)"
    )
    enable_external_validation: bool = Field(
        default=False,
        description="Allow external API calls for validation (non-production only)"
    )
    
    # Performance constraints
    max_processing_time_ms: int = Field(
        default=5000,
        description="Maximum time to spend processing per page"
    )
    max_events_per_page: int = Field(
        default=1000,
        description="Maximum events to process per page"
    )


class DetectResult(BaseModel):
    """Standardized output from detector analysis."""
    
    # Detection results
    events: List[TagEvent] = Field(
        default_factory=list,
        description="Analytics events detected on the page"
    )
    notes: List[DetectorNote] = Field(
        default_factory=list,
        description="Analysis notes, warnings, and recommendations"
    )
    
    # Processing metadata
    detector_name: str = Field(description="Name of the detector")
    detector_version: str = Field(
        default="1.0.0",
        description="Version of the detector"
    )
    processing_time_ms: Optional[int] = Field(
        default=None,
        description="Time spent processing in milliseconds"
    )
    processed_requests: int = Field(
        default=0,
        description="Number of network requests processed"
    )
    
    # Status and errors
    success: bool = Field(
        default=True,
        description="Whether detection completed successfully"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if detection failed"
    )
    
    # Analysis timestamp
    analyzed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When analysis was performed"
    )
    
    # Additional metrics and metadata
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metrics and metadata from detection"
    )
    
    def add_event(self, event: TagEvent) -> None:
        """Add a detected event to the results."""
        self.events.append(event)
    
    def add_note(self, note: DetectorNote) -> None:
        """Add an analysis note to the results."""
        self.notes.append(note)
    
    def add_info_note(self, message: str, category: NoteCategory = NoteCategory.DATA_QUALITY, 
                     related_events: Optional[List[str]] = None, **details) -> None:
        """Convenience method to add an info-level note."""
        note = DetectorNote(
            severity=NoteSeverity.INFO,
            category=category,
            message=message,
            page_url="",  # Will be set by detector
            related_events=related_events or [],
            details=details,
            detector_name=self.detector_name
        )
        self.add_note(note)
    
    def add_warning_note(self, message: str, category: NoteCategory = NoteCategory.DATA_QUALITY,
                        related_events: Optional[List[str]] = None, **details) -> None:
        """Convenience method to add a warning-level note.""" 
        note = DetectorNote(
            severity=NoteSeverity.WARNING,
            category=category,
            message=message,
            page_url="",  # Will be set by detector
            related_events=related_events or [],
            details=details,
            detector_name=self.detector_name
        )
        self.add_note(note)
    
    def add_error_note(self, message: str, category: NoteCategory = NoteCategory.VALIDATION,
                      related_events: Optional[List[str]] = None, **details) -> None:
        """Convenience method to add an error-level note."""
        note = DetectorNote(
            severity=NoteSeverity.ERROR,
            category=category, 
            message=message,
            page_url="",  # Will be set by detector
            related_events=related_events or [],
            details=details,
            detector_name=self.detector_name
        )
        self.add_note(note)
    
    @property
    def has_errors(self) -> bool:
        """Check if any error-level notes were generated."""
        return any(note.severity == NoteSeverity.ERROR for note in self.notes)
    
    @property
    def has_warnings(self) -> bool:
        """Check if any warning-level notes were generated."""
        return any(note.severity == NoteSeverity.WARNING for note in self.notes)
    
    @property
    def event_count_by_vendor(self) -> Dict[str, int]:
        """Count events by vendor."""
        counts = {}
        for event in self.events:
            vendor = event.vendor.value if hasattr(event.vendor, 'value') else str(event.vendor)
            counts[vendor] = counts.get(vendor, 0) + 1
        return counts


class Detector(Protocol):
    """Protocol that all analytics detectors must implement."""
    
    @property
    def name(self) -> str:
        """Unique name for this detector."""
        ...
    
    @property 
    def version(self) -> str:
        """Version of this detector."""
        ...
    
    @property
    def supported_vendors(self) -> Set[Vendor]:
        """Set of vendors this detector can analyze."""
        ...
    
    def detect(self, page: PageResult, ctx: DetectContext) -> Union[DetectResult, Awaitable[DetectResult]]:
        """Analyze a page result and return detected analytics events."""
        ...


class BaseDetector(ABC):
    """Abstract base class providing common detector functionality."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
    
    @property
    def name(self) -> str:
        """Unique name for this detector."""
        return self._name
    
    @property
    def version(self) -> str:
        """Version of this detector."""
        return self._version
    
    @property
    @abstractmethod
    def supported_vendors(self) -> Set[Vendor]:
        """Set of vendors this detector can analyze."""
        ...
    
    @abstractmethod
    def detect(self, page: PageResult, ctx: DetectContext) -> Union[DetectResult, Awaitable[DetectResult]]:
        """Analyze a page result and return detected analytics events."""
        ...
    
    def _create_result(self) -> DetectResult:
        """Create a new DetectResult with metadata populated."""
        return DetectResult(
            detector_name=self.name,
            detector_version=self.version
        )
    
    def _set_note_page_urls(self, result: DetectResult, page_url: str) -> None:
        """Set page_url for all notes that don't have one set."""
        for note in result.notes:
            if not note.page_url:
                note.page_url = page_url


# Registry for detector discovery and management
class DetectorRegistry:
    """Registry for managing detector discovery and instantiation.
    
    Supports configuration-driven detector activation, automatic discovery,
    and graceful error handling for detector initialization.
    """
    
    def __init__(self):
        self._detectors: Dict[str, type] = {}
        self._instances: Dict[str, Detector] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._enabled: Dict[str, bool] = {}
        self._initialization_errors: Dict[str, str] = {}
    
    def register(self, detector_class: type, name: Optional[str] = None, 
                metadata: Optional[Dict[str, Any]] = None,
                enabled: bool = True) -> None:
        """Register a detector class.
        
        Args:
            detector_class: The detector class to register
            name: Optional name override for the detector
            metadata: Optional metadata dict with version, description, etc.
            enabled: Whether the detector is enabled by default
        """
        if name is None:
            name = getattr(detector_class, 'name', detector_class.__name__)
        
        self._detectors[name] = detector_class
        self._metadata[name] = metadata or {}
        self._enabled[name] = enabled
        
        # Clear any existing instance if re-registering
        if name in self._instances:
            del self._instances[name]
        
        # Clear any previous initialization errors
        if name in self._initialization_errors:
            del self._initialization_errors[name]
    
    def get_detector(self, name: str, **kwargs) -> Optional[Detector]:
        """Get a detector instance by name.
        
        Returns cached instance if available, otherwise creates new instance.
        Returns None if detector is not registered, disabled, or fails to initialize.
        """
        # Check if detector exists and is enabled
        if name not in self._detectors or not self._enabled.get(name, False):
            return None
        
        # Return cached instance if available
        if name in self._instances:
            return self._instances[name]
        
        # Check for previous initialization errors
        if name in self._initialization_errors:
            return None
        
        # Try to create new instance
        try:
            detector_class = self._detectors[name]
            instance = detector_class(**kwargs)
            self._instances[name] = instance
            return instance
        except Exception as e:
            # Store error for future reference and debugging
            self._initialization_errors[name] = str(e)
            return None
    
    def get_enabled_detectors(self, config: Optional[Dict[str, Any]] = None) -> List[Detector]:
        """Get all enabled detector instances.
        
        Args:
            config: Optional configuration dict that can override enabled state
            
        Returns:
            List of successfully initialized detector instances
        """
        detectors = []
        enabled_names = self._get_enabled_names(config)
        
        for name in enabled_names:
            detector = self.get_detector(name)
            if detector is not None:
                detectors.append(detector)
        
        return detectors
    
    def _get_enabled_names(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get list of enabled detector names based on registry and config."""
        enabled_names = []
        
        for name in self._detectors.keys():
            # Start with registry default
            enabled = self._enabled.get(name, False)
            
            # Override with config if provided
            if config and 'detectors' in config:
                detector_config = config['detectors'].get(name, {})
                if 'enabled' in detector_config:
                    enabled = detector_config['enabled']
            
            if enabled:
                enabled_names.append(name)
        
        return enabled_names
    
    def is_enabled(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a detector is enabled."""
        if name not in self._detectors:
            return False
        
        enabled = self._enabled.get(name, False)
        
        if config and 'detectors' in config:
            detector_config = config['detectors'].get(name, {})
            if 'enabled' in detector_config:
                enabled = detector_config['enabled']
        
        return enabled
    
    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a detector.
        
        Args:
            name: Detector name
            enabled: Whether to enable the detector
            
        Returns:
            True if detector exists, False otherwise
        """
        if name not in self._detectors:
            return False
        
        self._enabled[name] = enabled
        
        # Clear cached instance if disabling
        if not enabled and name in self._instances:
            del self._instances[name]
        
        return True
    
    def list_detectors(self, enabled_only: bool = False) -> List[str]:
        """List registered detector names.
        
        Args:
            enabled_only: If True, only return enabled detectors
            
        Returns:
            List of detector names
        """
        if enabled_only:
            return [name for name, enabled in self._enabled.items() if enabled]
        return list(self._detectors.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a registered detector."""
        return self._metadata.get(name, {})
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all detectors for debugging and monitoring.
        
        Returns:
            Dict mapping detector names to status information
        """
        status = {}
        
        for name in self._detectors.keys():
            status[name] = {
                'registered': True,
                'enabled': self._enabled.get(name, False),
                'instantiated': name in self._instances,
                'initialization_error': self._initialization_errors.get(name),
                'metadata': self._metadata.get(name, {})
            }
        
        return status
    
    def clear(self) -> None:
        """Clear all registered detectors and instances."""
        self._detectors.clear()
        self._instances.clear()
        self._metadata.clear()
        self._enabled.clear()
        self._initialization_errors.clear()
    
    def discover_detectors(self) -> None:
        """Discover and register detectors from entry points.
        
        This method would use setuptools entry points in a real implementation
        to automatically discover detector plugins. For now, it's a placeholder.
        """
        # In a real implementation, this would use:
        # import pkg_resources
        # for entry_point in pkg_resources.iter_entry_points('tag_sentinel.detectors'):
        #     try:
        #         detector_class = entry_point.load()
        #         self.register(detector_class, entry_point.name)
        #     except Exception as e:
        #         # Log error but continue with other detectors
        #         pass
        pass


# Global registry instance
registry = DetectorRegistry()


def create_detect_context(detector_name: str, **kwargs) -> DetectContext:
    """Create a DetectContext with configuration for a specific detector.
    
    Args:
        detector_name: Name of the detector
        **kwargs: Additional context parameters to override
        
    Returns:
        Configured DetectContext
    """
    # Import here to avoid circular imports
    from .config import get_config
    
    config = get_config()
    detector_config = config.get_detector_config(detector_name)
    
    # Create context with detector-specific config
    context_params = {
        "environment": config.environment,
        "is_production": config.is_production,
        "config": detector_config,
        "site_domain": config.site_domain,
        "enable_debug": detector_config.get("enable_debug", False),
        "enable_external_validation": detector_config.get("enable_external_validation", False),
        "max_processing_time_ms": detector_config.get("max_processing_time_ms", 5000),
        "max_events_per_page": detector_config.get("max_events_per_page", 1000)
    }
    
    # Apply any overrides
    context_params.update(kwargs)
    
    return DetectContext(**context_params)