"""Analytics tag detection framework.

This module provides a pluggable framework for detecting and analyzing 
analytics tags from web page capture data. It includes base protocols,
data models, and vendor-specific detector implementations.
"""

from .base import (
    Detector,
    BaseDetector, 
    TagEvent,
    DetectorNote,
    DetectContext,
    DetectResult,
    DetectorRegistry,
    Vendor,
    TagStatus,
    Confidence,
    NoteSeverity,
    NoteCategory,
    registry,
    create_detect_context
)

# Import detector implementations
from .ga4 import GA4Detector
from .gtm import GTMDetector
from .duplicates import DuplicateAnalyzer, analyze_events_for_duplicates
from .sequencing import SequencingAnalyzer

# Import utilities
from .utils import patterns, ParameterParser

# Import configuration system
from .config import (
    DetectorConfig,
    ConfigManager,
    get_config,
    load_config,
    configure_detectors,
    ConfigurationError
)

# Import performance utilities
from .performance import (
    get_performance_summary,
    reset_performance_metrics,
    PerformanceMetrics,
    monitor_performance
)

# Register detectors with the global registry
def _register_default_detectors():
    """Register default detector implementations."""
    registry.register(GA4Detector, enabled=True)
    registry.register(GTMDetector, enabled=True)
    registry.register(DuplicateAnalyzer, enabled=True)
    registry.register(SequencingAnalyzer, enabled=True)

# Auto-register on import
_register_default_detectors()

__all__ = [
    # Base framework
    "Detector",
    "BaseDetector",
    "TagEvent", 
    "DetectorNote",
    "DetectContext",
    "DetectResult",
    "DetectorRegistry",
    "Vendor",
    "TagStatus", 
    "Confidence",
    "NoteSeverity",
    "NoteCategory",
    "registry",
    "create_detect_context",
    
    # Configuration system
    "DetectorConfig",
    "ConfigManager", 
    "get_config",
    "load_config",
    "configure_detectors",
    "ConfigurationError",
    
    # Detector implementations
    "GA4Detector",
    "GTMDetector", 
    "DuplicateAnalyzer",
    "SequencingAnalyzer",
    
    # Utilities
    "patterns",
    "ParameterParser",
    "analyze_events_for_duplicates",
    
    # Performance utilities
    "get_performance_summary",
    "reset_performance_metrics", 
    "PerformanceMetrics",
    "monitor_performance"
]