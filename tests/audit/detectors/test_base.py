"""Unit tests for base detector functionality."""

import pytest
from datetime import datetime
from typing import Set

from app.audit.detectors.base import (
    BaseDetector,
    DetectorRegistry,
    TagEvent,
    DetectorNote,
    DetectContext,
    DetectResult,
    Vendor,
    TagStatus,
    Confidence,
    NoteSeverity,
    NoteCategory
)
from app.audit.models.capture import PageResult, CaptureStatus


class TestDetector(BaseDetector):
    """Test detector implementation for testing."""
    
    def __init__(self):
        super().__init__("TestDetector", "1.0.0")
    
    @property
    def supported_vendors(self) -> Set[Vendor]:
        return {Vendor.GA4}
    
    def detect(self, page: PageResult, ctx: DetectContext) -> DetectResult:
        result = self._create_result()
        
        # Create a test event
        event = TagEvent(
            vendor=Vendor.GA4,
            name="test_event",
            page_url=page.url,
            detection_method="test",
            detector_version=self.version
        )
        result.add_event(event)
        
        # Add a test note
        result.add_info_note("Test note", NoteCategory.DATA_QUALITY)
        
        return result


class TestTagEvent:
    """Test TagEvent model."""
    
    def test_tag_event_creation(self):
        """Test basic TagEvent creation."""
        event = TagEvent(
            vendor=Vendor.GA4,
            name="page_view",
            page_url="https://example.com",
            detection_method="network_request",
            detector_version="1.0.0"
        )
        
        assert event.vendor == Vendor.GA4
        assert event.name == "page_view"
        assert event.page_url == "https://example.com"
        assert event.status == TagStatus.UNKNOWN  # Default
        assert event.confidence == Confidence.MEDIUM  # Default
        assert isinstance(event.params, dict)
        assert isinstance(event.detected_at, datetime)
    
    def test_tag_event_with_params(self):
        """Test TagEvent with parameters."""
        params = {
            "measurement_id": "G-1234567890",
            "client_id": "12345.67890",
            "page_title": "Test Page"
        }
        
        event = TagEvent(
            vendor=Vendor.GA4,
            name="page_view",
            page_url="https://example.com",
            params=params,
            detection_method="network_request",
            detector_version="1.0.0"
        )
        
        assert event.params == params
        assert event.params["measurement_id"] == "G-1234567890"


class TestDetectorNote:
    """Test DetectorNote model."""
    
    def test_detector_note_creation(self):
        """Test basic DetectorNote creation."""
        note = DetectorNote(
            severity=NoteSeverity.WARNING,
            category=NoteCategory.DUPLICATE,
            message="Duplicate events detected",
            page_url="https://example.com",
            detector_name="TestDetector"
        )
        
        assert note.severity == NoteSeverity.WARNING
        assert note.category == NoteCategory.DUPLICATE
        assert note.message == "Duplicate events detected"
        assert note.page_url == "https://example.com"
        assert note.detector_name == "TestDetector"
        assert isinstance(note.created_at, datetime)
        assert isinstance(note.details, dict)
        assert isinstance(note.related_events, list)


class TestDetectContext:
    """Test DetectContext model."""
    
    def test_detect_context_creation(self):
        """Test basic DetectContext creation."""
        ctx = DetectContext(
            environment="development",
            is_production=False,
            site_domain="example.com"
        )
        
        assert ctx.environment == "development"
        assert ctx.is_production is False
        assert ctx.site_domain == "example.com"
        assert isinstance(ctx.config, dict)
        assert ctx.enable_debug is False  # Default
    
    def test_detect_context_with_config(self):
        """Test DetectContext with configuration."""
        config = {
            "ga4": {"enabled": True},
            "gtm": {"enabled": False}
        }
        
        ctx = DetectContext(
            environment="production",
            config=config,
            enable_debug=True
        )
        
        assert ctx.config == config
        assert ctx.config["ga4"]["enabled"] is True
        assert ctx.enable_debug is True


class TestDetectResult:
    """Test DetectResult model."""
    
    def test_detect_result_creation(self):
        """Test basic DetectResult creation."""
        result = DetectResult(
            detector_name="TestDetector",
            detector_version="1.0.0"
        )
        
        assert result.detector_name == "TestDetector"
        assert result.detector_version == "1.0.0"
        assert result.success is True  # Default
        assert len(result.events) == 0
        assert len(result.notes) == 0
        assert isinstance(result.analyzed_at, datetime)
    
    def test_detect_result_add_event(self):
        """Test adding events to DetectResult."""
        result = DetectResult(
            detector_name="TestDetector",
            detector_version="1.0.0"
        )
        
        event = TagEvent(
            vendor=Vendor.GA4,
            name="page_view",
            page_url="https://example.com",
            detection_method="test",
            detector_version="1.0.0"
        )
        
        result.add_event(event)
        
        assert len(result.events) == 1
        assert result.events[0] == event
    
    def test_detect_result_add_notes(self):
        """Test adding notes to DetectResult."""
        result = DetectResult(
            detector_name="TestDetector",
            detector_version="1.0.0"
        )
        
        # Add info note
        result.add_info_note("Info message", NoteCategory.DATA_QUALITY)
        
        # Add warning note
        result.add_warning_note("Warning message", NoteCategory.PERFORMANCE)
        
        # Add error note
        result.add_error_note("Error message", NoteCategory.VALIDATION)
        
        assert len(result.notes) == 3
        
        # Check note severities
        severities = [note.severity for note in result.notes]
        assert NoteSeverity.INFO in severities
        assert NoteSeverity.WARNING in severities
        assert NoteSeverity.ERROR in severities
        
        # Test convenience properties
        assert result.has_warnings is True
        assert result.has_errors is True
    
    def test_event_count_by_vendor(self):
        """Test event counting by vendor."""
        result = DetectResult(
            detector_name="TestDetector",
            detector_version="1.0.0"
        )
        
        # Add GA4 events
        for i in range(3):
            event = TagEvent(
                vendor=Vendor.GA4,
                name=f"ga4_event_{i}",
                page_url="https://example.com",
                detection_method="test",
                detector_version="1.0.0"
            )
            result.add_event(event)
        
        # Add GTM event
        gtm_event = TagEvent(
            vendor=Vendor.GTM,
            name="container_load",
            page_url="https://example.com",
            detection_method="test",
            detector_version="1.0.0"
        )
        result.add_event(gtm_event)
        
        counts = result.event_count_by_vendor
        
        assert counts["ga4"] == 3
        assert counts["gtm"] == 1
        assert len(counts) == 2


class TestBaseDetector:
    """Test BaseDetector abstract class."""
    
    def test_detector_creation(self):
        """Test detector instantiation."""
        detector = TestDetector()
        
        assert detector.name == "TestDetector"
        assert detector.version == "1.0.0"
        assert Vendor.GA4 in detector.supported_vendors
    
    def test_detector_custom_name_version(self):
        """Test detector with custom name and version."""
        detector = TestDetector()
        detector._name = "CustomTestDetector"  
        detector._version = "2.0.0"
        
        assert detector.name == "CustomTestDetector"
        assert detector.version == "2.0.0"
    
    def test_detector_detect_method(self):
        """Test detector detect method."""
        detector = TestDetector()
        
        # Create test page result
        page = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS
        )
        
        # Create test context
        ctx = DetectContext()
        
        # Run detection
        result = detector.detect(page, ctx)
        
        assert isinstance(result, DetectResult)
        assert result.detector_name == "TestDetector"
        assert result.success is True
        assert len(result.events) == 1
        assert len(result.notes) == 1
        
        # Check the test event
        event = result.events[0]
        assert event.vendor == Vendor.GA4
        assert event.name == "test_event"
        assert event.page_url == page.url
    
    def test_create_result(self):
        """Test _create_result helper method."""
        detector = TestDetector()
        result = detector._create_result()
        
        assert isinstance(result, DetectResult)
        assert result.detector_name == detector.name
        assert result.detector_version == detector.version
    
    def test_set_note_page_urls(self):
        """Test _set_note_page_urls helper method."""
        detector = TestDetector()
        result = detector._create_result()
        
        # Add notes without page_url set
        result.add_info_note("Test note 1")
        result.add_warning_note("Test note 2")
        
        # Verify page_url is empty
        assert all(note.page_url == "" for note in result.notes)
        
        # Set page URLs
        test_url = "https://example.com"
        detector._set_note_page_urls(result, test_url)
        
        # Verify page_url is now set
        assert all(note.page_url == test_url for note in result.notes)


class TestDetectorRegistry:
    """Test DetectorRegistry functionality."""
    
    def setup_method(self):
        """Set up test with fresh registry."""
        self.registry = DetectorRegistry()
    
    def test_registry_creation(self):
        """Test registry instantiation."""
        assert len(self.registry.list_detectors()) == 0
    
    def test_register_detector(self):
        """Test registering a detector."""
        self.registry.register(TestDetector)
        
        detectors = self.registry.list_detectors()
        assert "TestDetector" in detectors
        
        # Check metadata
        status = self.registry.get_status()
        assert "TestDetector" in status
        assert status["TestDetector"]["registered"] is True
        assert status["TestDetector"]["enabled"] is True  # Default
    
    def test_register_detector_with_custom_name(self):
        """Test registering detector with custom name."""
        self.registry.register(TestDetector, name="CustomName", enabled=False)
        
        detectors = self.registry.list_detectors()
        assert "CustomName" in detectors
        assert "TestDetector" not in detectors
        
        # Check enabled state
        assert self.registry.is_enabled("CustomName") is False
    
    def test_get_detector(self):
        """Test getting detector instance."""
        self.registry.register(TestDetector)
        
        detector = self.registry.get_detector("TestDetector")
        assert isinstance(detector, TestDetector)
        assert detector.name == "TestDetector"
        
        # Should return same instance on second call
        detector2 = self.registry.get_detector("TestDetector")
        assert detector is detector2
    
    def test_get_nonexistent_detector(self):
        """Test getting detector that doesn't exist."""
        detector = self.registry.get_detector("NonExistent")
        assert detector is None
    
    def test_enabled_disabled_detectors(self):
        """Test enabled/disabled detector management."""
        self.registry.register(TestDetector, enabled=False)
        
        # Should not return disabled detector
        detector = self.registry.get_detector("TestDetector")
        assert detector is None
        
        # Enable it
        self.registry.set_enabled("TestDetector", True)
        detector = self.registry.get_detector("TestDetector")
        assert isinstance(detector, TestDetector)
        
        # Disable it again
        self.registry.set_enabled("TestDetector", False)
        detector2 = self.registry.get_detector("TestDetector")
        assert detector2 is None
    
    def test_get_enabled_detectors(self):
        """Test getting all enabled detectors."""
        # Register multiple detectors
        self.registry.register(TestDetector, name="Detector1", enabled=True)
        self.registry.register(TestDetector, name="Detector2", enabled=False)
        self.registry.register(TestDetector, name="Detector3", enabled=True)
        
        enabled_detectors = self.registry.get_enabled_detectors()
        assert len(enabled_detectors) == 2
        
        detector_names = [d.name for d in enabled_detectors]
        assert "Detector1" in detector_names
        assert "Detector3" in detector_names
        assert "Detector2" not in detector_names
    
    def test_config_override(self):
        """Test configuration overrides for enabled state."""
        self.registry.register(TestDetector, enabled=False)
        
        # Config that enables the detector
        config = {
            "detectors": {
                "TestDetector": {"enabled": True}
            }
        }
        
        # Should be enabled via config
        assert self.registry.is_enabled("TestDetector", config) is True
        
        enabled_detectors = self.registry.get_enabled_detectors(config)
        assert len(enabled_detectors) == 1
        assert isinstance(enabled_detectors[0], TestDetector)
    
    def test_registry_status(self):
        """Test registry status reporting."""
        self.registry.register(TestDetector, enabled=True)
        
        # Get an instance to test instantiation tracking
        self.registry.get_detector("TestDetector")
        
        status = self.registry.get_status()
        
        assert "TestDetector" in status
        detector_status = status["TestDetector"]
        
        assert detector_status["registered"] is True
        assert detector_status["enabled"] is True
        assert detector_status["instantiated"] is True
        assert detector_status["initialization_error"] is None
    
    def test_registry_clear(self):
        """Test clearing registry."""
        self.registry.register(TestDetector)
        self.registry.get_detector("TestDetector")  # Create instance
        
        assert len(self.registry.list_detectors()) == 1
        
        self.registry.clear()
        
        assert len(self.registry.list_detectors()) == 0
        assert len(self.registry.get_status()) == 0