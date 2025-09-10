"""Unit tests for duplicate detection functionality."""

import pytest
from datetime import datetime, timedelta

from app.audit.detectors.duplicates import (
    EventCanonicalizer,
    DuplicateGroup,
    DuplicateAnalyzer,
    analyze_events_for_duplicates
)
from app.audit.detectors.base import (
    TagEvent,
    DetectContext,
    Vendor,
    TagStatus,
    Confidence
)
from app.audit.models.capture import PageResult, CaptureStatus


class TestEventCanonicalizer:
    """Test EventCanonicalizer functionality."""
    
    def setup_method(self):
        """Set up test with fresh canonicalizer."""
        self.canonicalizer = EventCanonicalizer()
    
    def test_canonicalize_event_basic(self):
        """Test basic event canonicalization."""
        event = TagEvent(
            vendor=Vendor.GA4,
            name="page_view",
            category="analytics",
            id="G-1234567890",
            page_url="https://example.com",
            params={
                "measurement_id": "G-1234567890",
                "client_id": "12345.67890",
                "page_title": "Test Page"
            },
            detection_method="network_request",
            detector_version="1.0.0"
        )
        
        canonical = self.canonicalizer.canonicalize_event(event)
        
        assert canonical["vendor"] == "ga4"
        assert canonical["name"] == "page_view"
        assert canonical["category"] == "analytics"
        assert canonical["id"] == "G-1234567890"
        
        # Should include key parameters
        assert "measurement_id" in canonical["params"]
        assert "client_id" in canonical["params"]
        assert "page_title" in canonical["params"]
    
    def test_canonicalize_event_excludes_timing(self):
        """Test that timing parameters are excluded from canonicalization."""
        event = TagEvent(
            vendor=Vendor.GA4,
            name="page_view",
            page_url="https://example.com",
            params={
                "measurement_id": "G-1234567890",
                "timing_ms": 1500,
                "request_url": "https://google-analytics.com/mp/collect",
                "status_code": 200,
                "cache_buster": "12345"
            },
            detection_method="network_request",
            detector_version="1.0.0"
        )
        
        canonical = self.canonicalizer.canonicalize_event(event)
        
        # Should include measurement_id (always include)
        assert "measurement_id" in canonical["params"]
        
        # Should exclude timing and request-specific params
        assert "timing_ms" not in canonical["params"]
        assert "request_url" not in canonical["params"]
        assert "status_code" not in canonical["params"]
        assert "cache_buster" not in canonical["params"]
    
    def test_canonicalize_event_custom_config(self):
        """Test canonicalization with custom configuration."""
        config = {
            "exclude_params": {"custom_exclude"},
            "always_include": {"measurement_id", "custom_exclude"}
        }
        canonicalizer = EventCanonicalizer(config)
        
        event = TagEvent(
            vendor=Vendor.GA4,
            name="page_view", 
            page_url="https://example.com",
            params={
                "measurement_id": "G-1234567890",
                "custom_exclude": "should_be_included",  # In always_include
                "timing_ms": 1500  # Not in exclude_params, so should be included
            },
            detection_method="network_request",
            detector_version="1.0.0"
        )
        
        canonical = canonicalizer.canonicalize_event(event)
        
        # Should include both due to always_include override
        assert "measurement_id" in canonical["params"] 
        assert "custom_exclude" in canonical["params"]
        assert "timing_ms" in canonical["params"]  # Not in custom exclude list
    
    def test_normalize_value(self):
        """Test value normalization."""
        canonicalizer = EventCanonicalizer()
        
        # String normalization
        assert canonicalizer._normalize_value("  test  ") == "test"
        assert canonicalizer._normalize_value("true") is True
        assert canonicalizer._normalize_value("false") is False
        assert canonicalizer._normalize_value("1") is True
        assert canonicalizer._normalize_value("0") is False
        
        # Number normalization
        assert canonicalizer._normalize_value(123) == 123
        assert canonicalizer._normalize_value(45.6) == 45.6
        
        # Boolean normalization
        assert canonicalizer._normalize_value(True) is True
        assert canonicalizer._normalize_value(False) is False
        
        # None normalization
        assert canonicalizer._normalize_value(None) is None
    
    def test_generate_hash(self):
        """Test hash generation."""
        canonical1 = {
            "vendor": "ga4",
            "name": "page_view",
            "params": {"measurement_id": "G-1234567890"}
        }
        
        canonical2 = {
            "vendor": "ga4", 
            "name": "page_view",
            "params": {"measurement_id": "G-1234567890"}
        }
        
        # Same canonical events should generate same hash
        hash1 = self.canonicalizer.generate_hash(canonical1)
        hash2 = self.canonicalizer.generate_hash(canonical2)
        assert hash1 == hash2
        
        # Different canonical events should generate different hashes
        canonical3 = {
            "vendor": "ga4",
            "name": "scroll",  # Different name
            "params": {"measurement_id": "G-1234567890"}
        }
        
        hash3 = self.canonicalizer.generate_hash(canonical3)
        assert hash1 != hash3
    
    def test_get_event_hash(self):
        """Test convenience method for getting event hash."""
        event = TagEvent(
            vendor=Vendor.GA4,
            name="page_view",
            page_url="https://example.com",
            params={"measurement_id": "G-1234567890"},
            detection_method="network_request", 
            detector_version="1.0.0"
        )
        
        hash_value = self.canonicalizer.get_event_hash(event)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex string length


class TestDuplicateGroup:
    """Test DuplicateGroup functionality."""
    
    def create_test_event(self, name="page_view", timestamp=None):
        """Helper to create test events."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        event = TagEvent(
            vendor=Vendor.GA4,
            name=name,
            page_url="https://example.com",
            params={"measurement_id": "G-1234567890"},
            detection_method="network_request",
            detector_version="1.0.0"
        )
        event.detected_at = timestamp
        return event
    
    def test_duplicate_group_creation(self):
        """Test creating a duplicate group."""
        event = self.create_test_event()
        group = DuplicateGroup("hash123", event)
        
        assert group.canonical_hash == "hash123"
        assert group.count == 1
        assert len(group.events) == 1
        assert group.events[0] == event
        assert not group.is_duplicate  # Single event is not a duplicate
        assert group.representative_event == event
    
    def test_add_event_to_group(self):
        """Test adding events to a group."""
        base_time = datetime.utcnow()
        
        event1 = self.create_test_event("page_view", base_time)
        event2 = self.create_test_event("page_view", base_time + timedelta(seconds=1))
        
        group = DuplicateGroup("hash123", event1)
        group.add_event(event2)
        
        assert group.count == 2
        assert len(group.events) == 2
        assert group.is_duplicate  # Multiple events make it a duplicate
        assert group.time_span_ms == 1000  # 1 second difference
    
    def test_group_analysis_data(self):
        """Test group analysis data tracking.""" 
        base_time = datetime.utcnow()
        
        event1 = self.create_test_event("page_view", base_time)
        event1.request_url = "https://example.com/req1"
        event1.timing_ms = 100
        event1.status = TagStatus.OK
        
        event2 = self.create_test_event("page_view", base_time + timedelta(seconds=1))
        event2.request_url = "https://example.com/req2"
        event2.timing_ms = 150
        event2.status = TagStatus.ERROR
        
        group = DuplicateGroup("hash123", event1)
        group.add_event(event2)
        
        # Should track unique URLs and timing values
        assert len(group.unique_request_urls) == 2
        assert len(group.unique_timing_values) == 2
        
        # Should track status distribution
        assert group.status_counts[TagStatus.OK] == 1
        assert group.status_counts[TagStatus.ERROR] == 1
    
    def test_group_to_dict(self):
        """Test converting group to dictionary."""
        event = self.create_test_event()
        group = DuplicateGroup("hash123", event)
        
        group_dict = group.to_dict()
        
        assert group_dict["canonical_hash"] == "hash123"
        assert group_dict["count"] == 1
        assert group_dict["time_span_ms"] == 0
        assert "representative_event" in group_dict
        assert group_dict["representative_event"]["name"] == "page_view"


class TestDuplicateAnalyzer:
    """Test DuplicateAnalyzer functionality."""
    
    def setup_method(self):
        """Set up test with fresh analyzer."""
        self.analyzer = DuplicateAnalyzer()
    
    def test_analyzer_properties(self):
        """Test analyzer basic properties."""
        assert self.analyzer.name == "DuplicateAnalyzer"
        assert self.analyzer.version == "1.0.0" 
        assert Vendor.GA4 in self.analyzer.supported_vendors
        assert Vendor.GTM in self.analyzer.supported_vendors
    
    def create_test_events(self, count=3, same_content=True):
        """Helper to create test events."""
        events = []
        base_time = datetime.utcnow()
        
        for i in range(count):
            name = "page_view" if same_content else f"event_{i}"
            event = TagEvent(
                vendor=Vendor.GA4,
                name=name,
                page_url="https://example.com",
                params={"measurement_id": "G-1234567890"},
                detection_method="network_request",
                detector_version="1.0.0"
            )
            event.detected_at = base_time + timedelta(milliseconds=i * 500)
            events.append(event)
        
        return events
    
    def test_analyze_duplicates_with_duplicates(self):
        """Test analyzing events with actual duplicates."""
        # Create 3 identical events within time window
        events = self.create_test_events(count=3, same_content=True)
        
        # Initialize canonicalizer
        self.analyzer.canonicalizer = EventCanonicalizer()
        
        # Analyze with 2 second window (should group all events)
        duplicate_groups = self.analyzer._analyze_duplicates(events, 2000)
        
        assert len(duplicate_groups) == 1
        group = duplicate_groups[0]
        assert group.count == 3
        assert group.is_duplicate
    
    def test_analyze_duplicates_no_duplicates(self):
        """Test analyzing events with no duplicates."""
        # Create different events
        events = self.create_test_events(count=3, same_content=False)
        
        # Initialize canonicalizer
        self.analyzer.canonicalizer = EventCanonicalizer()
        
        duplicate_groups = self.analyzer._analyze_duplicates(events, 2000)
        
        # Should have 3 groups, each with 1 event (no duplicates)
        assert len(duplicate_groups) == 3
        for group in duplicate_groups:
            assert group.count == 1
            assert not group.is_duplicate
    
    def test_analyze_duplicates_time_window(self):
        """Test time window behavior in duplicate analysis."""
        base_time = datetime.utcnow()
        
        # Create events with larger time gaps
        events = []
        for i in range(3):
            event = TagEvent(
                vendor=Vendor.GA4,
                name="page_view",
                page_url="https://example.com", 
                params={"measurement_id": "G-1234567890"},
                detection_method="network_request",
                detector_version="1.0.0"
            )
            event.detected_at = base_time + timedelta(seconds=i * 3)  # 3 second gaps
            events.append(event)
        
        # Initialize canonicalizer
        self.analyzer.canonicalizer = EventCanonicalizer()
        
        # With 2 second window, should create separate groups
        groups_small_window = self.analyzer._analyze_duplicates(events, 2000)
        duplicate_groups_small = [g for g in groups_small_window if g.is_duplicate]
        assert len(duplicate_groups_small) == 0  # No duplicates within 2 seconds
        
        # With 10 second window, should group all events
        groups_large_window = self.analyzer._analyze_duplicates(events, 10000)
        assert len(groups_large_window) == 1
        assert groups_large_window[0].count == 3
    
    def test_detect_no_events(self):
        """Test detect method with no events available."""
        page = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS
        )
        ctx = DetectContext()
        
        result = self.analyzer.detect(page, ctx)
        
        assert result.success is True
        assert len(result.events) == 0
        assert len(result.notes) > 0
        
        # Should have info note about no events
        info_notes = [note for note in result.notes if note.severity == "info"]
        assert len(info_notes) > 0


class TestAnalyzeEventsForDuplicates:
    """Test standalone analyze_events_for_duplicates function."""
    
    def create_test_events(self, count=3, same_content=True):
        """Helper to create test events."""
        events = []
        base_time = datetime.utcnow()
        
        for i in range(count):
            name = "page_view" if same_content else f"event_{i}"
            event = TagEvent(
                vendor=Vendor.GA4,
                name=name,
                page_url="https://example.com",
                params={"measurement_id": "G-1234567890"},
                detection_method="network_request", 
                detector_version="1.0.0"
            )
            event.detected_at = base_time + timedelta(milliseconds=i * 100)
            events.append(event)
        
        return events
    
    def test_analyze_events_with_duplicates(self):
        """Test analyzing events with duplicates using standalone function."""
        events = self.create_test_events(count=4, same_content=True)
        
        config = {"window_ms": 1000}  # 1 second window
        result = analyze_events_for_duplicates(events, config)
        
        assert "duplicate_groups" in result
        assert "summary" in result
        
        summary = result["summary"]
        assert summary["duplicates_found"] == 1
        assert summary["total_duplicate_events"] == 3  # 4 events - 1 original = 3 duplicates
        assert summary["total_events_analyzed"] == 4
        
        # Check duplicate group details
        duplicate_groups = result["duplicate_groups"]
        assert len(duplicate_groups) == 1
        group = duplicate_groups[0]
        assert group.count == 4
        assert group.is_duplicate
    
    def test_analyze_events_no_duplicates(self):
        """Test analyzing events with no duplicates."""
        events = self.create_test_events(count=3, same_content=False)
        
        result = analyze_events_for_duplicates(events)
        
        summary = result["summary"]
        assert summary["duplicates_found"] == 0
        assert summary["total_duplicate_events"] == 0
        assert summary["total_events_analyzed"] == 3
        
        assert len(result["duplicate_groups"]) == 0
    
    def test_analyze_empty_events(self):
        """Test analyzing empty event list.""" 
        result = analyze_events_for_duplicates([])
        
        assert result["duplicate_groups"] == []
        summary = result["summary"]
        assert summary["duplicates_found"] == 0
    
    def test_analyze_with_custom_config(self):
        """Test analyzing with custom configuration."""
        events = self.create_test_events(count=3, same_content=True)
        
        config = {
            "window_ms": 5000,  # Large window
            "exclude_params": {"custom_param"},
            "always_include": {"measurement_id"}
        }
        
        result = analyze_events_for_duplicates(events, config)
        
        # Should still detect duplicates with custom config
        assert result["summary"]["duplicates_found"] == 1
        assert result["summary"]["total_duplicate_events"] == 2