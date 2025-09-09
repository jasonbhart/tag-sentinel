"""Unit tests for GA4 detector."""

import pytest
from datetime import datetime

from app.audit.detectors.ga4 import GA4Detector
from app.audit.detectors.base import DetectContext, Vendor, TagStatus, Confidence
from app.audit.models.capture import (
    PageResult, 
    RequestLog, 
    ResourceType, 
    RequestStatus,
    CaptureStatus
)


class TestGA4Detector:
    """Test GA4Detector functionality."""
    
    def setup_method(self):
        """Set up test with fresh GA4 detector."""
        self.detector = GA4Detector()
    
    def test_detector_properties(self):
        """Test basic detector properties."""
        assert self.detector.name == "GA4Detector"
        assert self.detector.version == "1.0.0"
        assert Vendor.GA4 in self.detector.supported_vendors
    
    def test_is_ga4_request(self):
        """Test GA4 request pattern matching."""
        # Test GA4 MP collect endpoint
        assert self.detector._is_ga4_request("https://www.google-analytics.com/mp/collect") is True
        assert self.detector._is_ga4_request("https://region1.google-analytics.com/mp/collect") is True
        
        # Test GA4 g/collect endpoint
        assert self.detector._is_ga4_request("https://www.google-analytics.com/g/collect") is True
        
        # Test non-GA4 requests
        assert self.detector._is_ga4_request("https://www.example.com/api") is False
        assert self.detector._is_ga4_request("https://www.facebook.com/tr") is False
        assert self.detector._is_ga4_request("https://www.googletagmanager.com/gtm.js") is False
    
    def test_find_ga4_requests(self):
        """Test finding GA4 requests from request list."""
        requests = [
            RequestLog(
                url="https://www.google-analytics.com/mp/collect?measurement_id=G-1234567890",
                method="POST",
                resource_type=ResourceType.XHR,
                status=RequestStatus.SUCCESS
            ),
            RequestLog(
                url="https://www.example.com/api/data",
                method="GET", 
                resource_type=ResourceType.XHR,
                status=RequestStatus.SUCCESS
            ),
            RequestLog(
                url="https://region1.google-analytics.com/mp/collect",
                method="POST",
                resource_type=ResourceType.XHR, 
                status=RequestStatus.SUCCESS
            )
        ]
        
        ga4_requests = self.detector._find_ga4_requests(requests)
        
        assert len(ga4_requests) == 2
        assert "google-analytics.com" in ga4_requests[0].url
        assert "google-analytics.com" in ga4_requests[1].url
    
    def test_parse_url_parameters(self):
        """Test URL parameter parsing."""
        # Test URL with query parameters
        url = "https://www.google-analytics.com/mp/collect?measurement_id=G-1234567890&tid=G-1234567890"
        params = self.detector._parse_url_parameters(url)
        
        assert "measurement_id" in params
        assert params["measurement_id"] == "G-1234567890"
        assert "tid" in params
        
        # Test URL without parameters
        url_no_params = "https://www.google-analytics.com/mp/collect"
        params_empty = self.detector._parse_url_parameters(url_no_params)
        assert len(params_empty) == 0
    
    def test_parse_body_parameters_json(self):
        """Test parsing JSON body parameters."""
        request = RequestLog(
            url="https://www.google-analytics.com/mp/collect",
            method="POST",
            resource_type=ResourceType.XHR,
            request_headers={"content-type": "application/json"},
            request_body='{"measurement_id": "G-1234567890", "events": [{"name": "page_view"}]}'
        )
        
        params = self.detector._parse_body_parameters(request)
        
        assert "measurement_id" in params
        assert params["measurement_id"] == "G-1234567890"
        assert "events" in params
        assert len(params["events"]) == 1
        assert params["events"][0]["name"] == "page_view"
    
    def test_parse_body_parameters_form_encoded(self):
        """Test parsing form-encoded body parameters."""
        request = RequestLog(
            url="https://www.google-analytics.com/g/collect",
            method="POST", 
            resource_type=ResourceType.XHR,
            request_headers={"content-type": "application/x-www-form-urlencoded"},
            request_body="tid=G-1234567890&t=pageview&dp=%2F&dt=Test%20Page"
        )
        
        params = self.detector._parse_body_parameters(request)
        
        assert "tid" in params
        assert params["tid"] == "G-1234567890"
        assert "t" in params
        assert params["t"] == "pageview"
        assert "dp" in params
        assert params["dt"] == "Test Page"  # Should be URL decoded
    
    def test_classify_endpoint(self):
        """Test endpoint classification."""
        mp_url = "https://www.google-analytics.com/mp/collect"
        g_url = "https://www.google-analytics.com/g/collect"
        regional_url = "https://region1.google-analytics.com/mp/collect"
        
        assert self.detector._classify_endpoint(mp_url) == "measurement_protocol"
        assert self.detector._classify_endpoint(g_url) == "gtag_collect"
        assert self.detector._classify_endpoint(regional_url) == "regional_mp"
    
    def test_determine_confidence(self):
        """Test confidence level determination."""
        # High confidence: valid measurement ID and known event
        confidence_high = self.detector._determine_confidence("G-1234567890", "page_view")
        assert confidence_high == Confidence.HIGH
        
        # Medium confidence: valid measurement ID but unknown event
        confidence_med = self.detector._determine_confidence("G-1234567890", "custom_event")
        assert confidence_med == Confidence.MEDIUM
        
        # Medium confidence: has measurement ID but invalid format
        confidence_med2 = self.detector._determine_confidence("GA-12345", "page_view")
        assert confidence_med2 == Confidence.MEDIUM
        
        # Low confidence: no measurement ID
        confidence_low = self.detector._determine_confidence(None, "page_view")
        assert confidence_low == Confidence.LOW
    
    def test_extract_events_from_request_mp(self):
        """Test extracting events from Measurement Protocol request."""
        request = RequestLog(
            url="https://www.google-analytics.com/mp/collect",
            method="POST",
            resource_type=ResourceType.XHR,
            status_code=200,
            request_headers={"content-type": "application/json"},
            request_body='{"measurement_id": "G-1234567890", "client_id": "12345.67890", "events": [{"name": "page_view", "params": {"page_title": "Test Page", "page_location": "https://example.com"}}]}'
        )
        
        page_url = "https://example.com"
        ctx = DetectContext()
        
        events = self.detector._extract_events_from_request(request, page_url, ctx)
        
        assert len(events) == 1
        
        event = events[0]
        assert event.vendor == Vendor.GA4
        assert event.name == "page_view"
        assert event.id == "G-1234567890"
        assert event.page_url == page_url
        assert event.request_url == request.url
        assert event.status == TagStatus.OK  # Successful request
        assert event.confidence == Confidence.HIGH
        
        # Check enriched parameters
        assert "measurement_id" in event.params
        assert event.params["measurement_id"] == "G-1234567890"
        assert "client_id" in event.params
        assert event.params["client_id"] == "12345.67890"
    
    def test_extract_events_from_request_gtag(self):
        """Test extracting events from gtag format request."""
        request = RequestLog(
            url="https://www.google-analytics.com/g/collect?tid=G-1234567890&t=event&ea=click&el=button",
            method="GET",
            resource_type=ResourceType.XHR,
            status_code=200
        )
        
        page_url = "https://example.com"
        ctx = DetectContext()
        
        events = self.detector._extract_events_from_request(request, page_url, ctx)
        
        assert len(events) == 1
        
        event = events[0]
        assert event.vendor == Vendor.GA4
        assert event.name == "click"  # Event action becomes name
        assert event.id == "G-1234567890"
        assert "measurement_id" in event.params
    
    def test_extract_events_generic(self):
        """Test creating generic event when specific events can't be identified."""
        request = RequestLog(
            url="https://www.google-analytics.com/mp/collect?measurement_id=G-1234567890",
            method="POST",
            resource_type=ResourceType.XHR,
            status_code=200,
            request_body='{"measurement_id": "G-1234567890", "client_id": "12345.67890"}'
        )
        
        page_url = "https://example.com"
        ctx = DetectContext()
        
        events = self.detector._extract_events_from_request(request, page_url, ctx)
        
        assert len(events) == 1
        
        event = events[0]
        assert event.vendor == Vendor.GA4
        assert event.name == "measurement_protocol"  # Generic name for MP
        assert event.category == "generic"
        assert event.id == "G-1234567890"
    
    def test_calculate_timing(self):
        """Test timing calculation from request."""
        from app.audit.models.capture import TimingData
        
        # Test with timing data
        timing = TimingData(request_start=100.0, response_end=350.0)
        request_with_timing = RequestLog(
            url="https://www.google-analytics.com/mp/collect",
            method="POST",
            resource_type=ResourceType.XHR,
            timing=timing
        )
        
        calculated_timing = self.detector._calculate_timing(request_with_timing)
        assert calculated_timing == 250  # 350 - 100
        
        # Test with duration_ms
        request_with_duration = RequestLog(
            url="https://www.google-analytics.com/mp/collect", 
            method="POST",
            resource_type=ResourceType.XHR,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow()
        )
        # Simulate duration_ms property
        request_with_duration.end_time = request_with_duration.start_time
        
        # Test with no timing info
        request_no_timing = RequestLog(
            url="https://www.google-analytics.com/mp/collect",
            method="POST",
            resource_type=ResourceType.XHR
        )
        
        timing_no_data = self.detector._calculate_timing(request_no_timing)
        assert timing_no_data is None
    
    def test_detect_with_ga4_requests(self):
        """Test full detect method with GA4 requests."""
        # Create page with GA4 requests
        requests = [
            RequestLog(
                url="https://www.google-analytics.com/mp/collect",
                method="POST",
                resource_type=ResourceType.XHR,
                status_code=200,
                request_headers={"content-type": "application/json"},
                request_body='{"measurement_id": "G-1234567890", "events": [{"name": "page_view"}]}'
            ),
            RequestLog(
                url="https://www.google-analytics.com/mp/collect", 
                method="POST",
                resource_type=ResourceType.XHR,
                status_code=200,
                request_headers={"content-type": "application/json"},
                request_body='{"measurement_id": "G-1234567890", "events": [{"name": "scroll"}]}'
            )
        ]
        
        page = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=requests
        )
        
        ctx = DetectContext()
        result = self.detector.detect(page, ctx)
        
        assert result.success is True
        assert result.detector_name == "GA4Detector"
        assert result.processed_requests == 2
        assert len(result.events) == 2
        
        # Check events
        event_names = [event.name for event in result.events]
        assert "page_view" in event_names
        assert "scroll" in event_names
        
        # Should have analysis notes
        assert len(result.notes) > 0
    
    def test_detect_no_ga4_requests(self):
        """Test detect method with no GA4 requests."""
        # Create page with non-GA4 requests
        requests = [
            RequestLog(
                url="https://www.example.com/api/data",
                method="GET",
                resource_type=ResourceType.XHR,
                status_code=200
            )
        ]
        
        page = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=requests
        )
        
        ctx = DetectContext()
        result = self.detector.detect(page, ctx)
        
        assert result.success is True
        assert len(result.events) == 0
        assert len(result.notes) > 0
        
        # Should have a note about no GA4 requests
        note_messages = [note.message for note in result.notes]
        assert any("No GA4 requests detected" in msg for msg in note_messages)
    
    def test_detect_with_failed_requests(self):
        """Test detect method with failed GA4 requests."""
        requests = [
            RequestLog(
                url="https://www.google-analytics.com/mp/collect",
                method="POST",
                resource_type=ResourceType.XHR,
                status_code=500,
                status=RequestStatus.FAILED,
                error_text="Server Error"
            )
        ]
        
        page = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=requests
        )
        
        ctx = DetectContext()
        result = self.detector.detect(page, ctx)
        
        assert result.success is True
        assert len(result.notes) > 0
        
        # Should have error note about failed requests
        error_notes = [note for note in result.notes if note.severity.value == "error"]
        assert len(error_notes) > 0
    
    def test_detect_multiple_measurement_ids(self):
        """Test detection with multiple measurement IDs."""
        requests = [
            RequestLog(
                url="https://www.google-analytics.com/mp/collect",
                method="POST",
                resource_type=ResourceType.XHR,
                status_code=200,
                request_headers={"content-type": "application/json"},
                request_body='{"measurement_id": "G-1111111111", "events": [{"name": "page_view"}]}'
            ),
            RequestLog(
                url="https://www.google-analytics.com/mp/collect",
                method="POST", 
                resource_type=ResourceType.XHR,
                status_code=200,
                request_headers={"content-type": "application/json"},
                request_body='{"measurement_id": "G-2222222222", "events": [{"name": "page_view"}]}'
            )
        ]
        
        page = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=requests
        )
        
        ctx = DetectContext()
        result = self.detector.detect(page, ctx)
        
        assert result.success is True
        assert len(result.events) == 2
        
        # Should have warning about multiple measurement IDs
        warning_notes = [note for note in result.notes if note.severity.value == "warning"]
        multiple_id_warnings = [note for note in warning_notes 
                               if "Multiple GA4 measurement IDs" in note.message]
        assert len(multiple_id_warnings) > 0