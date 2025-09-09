"""Browser Capture Engine for Tag Sentinel.

This module provides comprehensive web page capture capabilities using Playwright,
including network request tracking, cookie analysis, console log capture,
and debug artifact generation.

Main Components:
- Core Data Models: Pydantic models for captured data (models/capture.py)
- Browser Factory: Browser context creation and management
- Network Observer: Complete network request lifecycle tracking
- Console Observer: Console logs and page error capture
- Cookie Collector: Privacy-conscious cookie analysis
- Pre-Steps Executor: Scripted actions before capture (login, CMP, etc.)
- Page Session: Orchestrated capture workflow
- Capture Engine: Main coordination and batch processing

Usage:
    from app.audit.capture import CaptureEngine
    
    engine = CaptureEngine(config)
    result = await engine.capture_page("https://example.com")
"""

__version__ = "1.0.0"

# Main exports
__all__ = [
    # Data models
    "RequestLog",
    "CookieRecord", 
    "ConsoleLog",
    "PageResult",
    "TimingData",
    "ArtifactPaths",
    
    # Enums
    "RequestStatus",
    "ResourceType", 
    "ConsoleLevel",
    "CaptureStatus",
    
    # Main components
    "CaptureEngine",
    "CaptureEngineConfig",
    "BrowserFactory",
    "BrowserConfig",
    "PageSession",
    "PageSessionConfig",
    "WaitStrategy",
    
    # Observers
    "NetworkObserver",
    "CombinedObserver",
    "CookieCollector",
    "PreStepsExecutor",
    
    # Convenience functions
    "create_capture_engine",
    "create_debug_capture_engine",
    "create_default_factory",
]

# Import data models
from ..models.capture import (
    RequestLog,
    CookieRecord,
    ConsoleLog, 
    PageResult,
    TimingData,
    ArtifactPaths,
    RequestStatus,
    ResourceType,
    ConsoleLevel,
    CaptureStatus,
)

# Import main components
from .engine import (
    CaptureEngine,
    CaptureEngineConfig,
    create_capture_engine,
    create_debug_capture_engine,
)

from .browser_factory import (
    BrowserFactory,
    BrowserConfig,
    create_default_factory,
)

from .page_session import (
    PageSession,
    PageSessionConfig,
    WaitStrategy,
)

from .network_observer import NetworkObserver
from .console_observer import CombinedObserver
from .cookie_collector import CookieCollector
from .presteps import PreStepsExecutor