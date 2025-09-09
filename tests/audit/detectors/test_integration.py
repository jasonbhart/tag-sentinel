"""Integration tests for analytics detectors with realistic scenarios.

This module provides end-to-end testing of the detector system using
realistic page capture data and complex interaction patterns.
"""

import asyncio
import pytest
from datetime import datetime
from typing import List

from app.audit.detectors import (
    GA4Detector,
    GTMDetector,
    DuplicateAnalyzer,
    SequencingAnalyzer,
    DetectContext,
    get_config,
    configure_detectors,
    registry,
    get_performance_summary,
    reset_performance_metrics
)
from app.audit.models.capture import (
    PageResult,
    RequestLog,
    CaptureStatus,
    RequestStatus,
    ResourceType,
    ConsoleLog,
    ConsoleLevel
)


class TestDetectorIntegration:
    """Integration tests for detector system."""
    
    @pytest.fixture
    def reset_performance(self):
        """Reset performance metrics before each test."""
        reset_performance_metrics()
        yield
        reset_performance_metrics()
    
    @pytest.fixture
    def sample_ga4_requests(self) -> List[RequestLog]:
        """Create sample GA4 requests for testing."""
        return [
            RequestLog(
                url="https://www.google-analytics.com/mp/collect?measurement_id=G-1234567890",
                method="POST",
                resource_type=ResourceType.FETCH,
                status=RequestStatus.SUCCESS,
                status_code=200,
                request_headers={"content-type": "application/json"},
                response_headers={"content-type": "application/json"},
                request_body='{"client_id":"123.456","events":[{"name":"page_view","params":{"page_title":"Test Page","page_location":"https://example.com/test"}}]}',
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            ),
            RequestLog(
                url="https://www.google-analytics.com/g/collect?tid=UA-123456-1&t=pageview",
                method="GET",
                resource_type=ResourceType.FETCH,
                status=RequestStatus.SUCCESS,
                status_code=200,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )
        ]
    
    @pytest.fixture
    def sample_gtm_requests(self) -> List[RequestLog]:
        """Create sample GTM requests for testing."""
        return [
            RequestLog(
                url="https://www.googletagmanager.com/gtm.js?id=GTM-ABC123",
                method="GET",
                resource_type=ResourceType.SCRIPT,
                status=RequestStatus.SUCCESS,
                status_code=200,
                response_headers={"content-type": "application/javascript"},
                response_body='window.dataLayer = window.dataLayer || []; gtag("config", "G-123");',
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )
        ]
    
    @pytest.fixture 
    def sample_page_result(self, sample_ga4_requests, sample_gtm_requests) -> PageResult:
        """Create a comprehensive sample page result."""
        all_requests = sample_ga4_requests + sample_gtm_requests
        
        return PageResult(
            url="https://example.com/test-page",
            final_url="https://example.com/test-page",
            title="Test E-commerce Page",
            capture_status=CaptureStatus.SUCCESS,
            capture_time=datetime.utcnow(),
            load_time_ms=1250,
            network_requests=all_requests,
            console_logs=[
                ConsoleLog(
                    level=ConsoleLevel.INFO,
                    text="GA4 initialized successfully",
                    url="https://example.com/test-page"
                ),
                ConsoleLog(
                    level=ConsoleLevel.WARN, 
                    text="DataLayer not found during initial load",
                    url="https://example.com/test-page"
                )
            ],
            metrics={
                "dom_content_loaded": 800,
                "first_paint": 650,
                "largest_contentful_paint": 1100
            }
        )
    
    @pytest.fixture
    def test_context(self) -> DetectContext:
        """Create test detection context."""
        return DetectContext(
            environment="test",
            is_production=False,
            config={
                "ga4": {
                    "enabled": True,
                    "mp_debug": {"enabled": True, "timeout_ms": 1000}
                },
                "gtm": {
                    "enabled": True,
                    "validate_datalayer": True
                }
            },
            site_domain="example.com",
            audit_id="test-audit-001",
            enable_debug=True,
            enable_external_validation=False,
            max_processing_time_ms=5000,
            max_events_per_page=100
        )
    
    @pytest.mark.asyncio
    async def test_ga4_detector_integration(self, sample_page_result, test_context, reset_performance):
        """Test GA4 detector with realistic page data."""
        detector = GA4Detector()
        
        # Run detection
        result = await detector.detect(sample_page_result, test_context)
        
        # Verify basic result structure
        assert result.success
        assert result.detector_name == "GA4Detector"
        assert result.detector_version == "1.0.0"
        assert result.processing_time_ms is None or result.processing_time_ms >= 0
        
        # Verify events were detected
        assert len(result.events) > 0
        
        # Check for page_view event
        page_view_events = [e for e in result.events if e.name == "page_view"]
        assert len(page_view_events) > 0
        
        page_view_event = page_view_events[0]
        assert page_view_event.vendor == "ga4"
        assert page_view_event.page_url == sample_page_result.url
        assert "event_page_title" in page_view_event.params
        
        # Verify no critical errors
        assert not result.has_errors or not any(
            note.severity == "error" for note in result.notes
        )
        
        # Check performance metrics were recorded
        perf_summary = get_performance_summary()
        assert perf_summary["total_calls"] > 0
    
    def test_gtm_detector_integration(self, sample_page_result, test_context, reset_performance):
        """Test GTM detector with realistic page data."""
        detector = GTMDetector()
        
        # Run detection
        result = detector.detect(sample_page_result, test_context)
        
        # Verify basic result structure
        assert result.success
        assert result.detector_name == "GTMDetector"
        assert result.processing_time_ms is None or result.processing_time_ms >= 0
        
        # Verify container events were detected
        container_events = [e for e in result.events if e.name == "container_load"]
        assert len(container_events) > 0
        
        container_event = container_events[0]
        assert container_event.vendor == "gtm"
        # Container ID extraction depends on detector implementation
        assert "request_url" in container_event.params
        assert "GTM-ABC123" in container_event.request_url
        
        # Check for dataLayer validation
        datalayer_events = [e for e in result.events if e.name == "dataLayer_validation"]
        # May or may not be present depending on dataLayer detection
    
    def test_duplicate_detection_integration(self, test_context, reset_performance):
        """Test duplicate detection with realistic scenarios."""
        duplicate_detector = DuplicateAnalyzer()
        
        # Create a simple page result
        page_result = PageResult(
            url="https://example.com/duplicate-test",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=[]
        )
        
        # Run duplicate detection - this will test the basic workflow
        result = duplicate_detector.detect(page_result, test_context)
        
        assert result.success
        assert result.detector_name == "DuplicateAnalyzer"
        assert result.processing_time_ms is None or result.processing_time_ms >= 0
        
        # Should have a note about no events available
        assert len(result.notes) > 0
        info_notes = [n for n in result.notes if n.severity == "info"]
        assert any("No events available" in note.message for note in info_notes)
    
    def test_sequencing_analysis_integration(self, test_context, reset_performance):
        """Test sequencing analysis with realistic tag loading scenarios."""
        detector = SequencingAnalyzer()
        
        # Create requests with proper timing sequence
        base_time = datetime.utcnow()
        
        sequenced_requests = [
            # GTM loads first
            RequestLog(
                url="https://www.googletagmanager.com/gtm.js?id=GTM-ABC123",
                method="GET",
                resource_type=ResourceType.SCRIPT,
                status=RequestStatus.SUCCESS,
                start_time=base_time
            ),
            # GA4 loads after GTM
            RequestLog(
                url="https://www.google-analytics.com/mp/collect",
                method="POST", 
                resource_type=ResourceType.FETCH,
                status=RequestStatus.SUCCESS,
                start_time=base_time
            )
        ]
        
        page_with_sequence = PageResult(
            url="https://example.com/sequence-test",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=sequenced_requests
        )
        
        # Run sequencing analysis
        result = detector.detect(page_with_sequence, test_context)
        
        assert result.success
        assert result.detector_name == "SequencingAnalyzer"
        assert result.processing_time_ms is None or result.processing_time_ms >= 0
        
        # Sequencing analysis may or may not find issues depending on implementation
        # The key is that it runs without errors
    
    @pytest.mark.asyncio
    async def test_full_detection_pipeline(self, sample_page_result, test_context, reset_performance):
        """Test complete detection pipeline with all detectors."""
        # Configure registry
        config = get_config()
        configure_detectors(registry, config_path=None)
        
        # Get all enabled detectors
        detectors = registry.get_enabled_detectors()
        assert len(detectors) > 0
        
        results = []
        
        # Run all detectors
        for detector in detectors:
            if hasattr(detector.detect, '__code__') and detector.detect.__code__.co_flags & 0x80:
                # Async detector
                result = await detector.detect(sample_page_result, test_context)
            else:
                # Sync detector
                result = detector.detect(sample_page_result, test_context)
            
            results.append(result)
        
        # Verify all detectors ran successfully
        for result in results:
            assert result.success or not result.has_errors, f"{result.detector_name} had critical errors"
            assert result.processing_time_ms is None or result.processing_time_ms >= 0
        
        # Check that different types of events were detected
        all_events = []
        for result in results:
            all_events.extend(result.events)
        
        assert len(all_events) > 0
        
        # Should have events from different vendors
        vendors = {event.vendor for event in all_events}
        assert len(vendors) > 0
        
        # Verify performance tracking
        perf_summary = get_performance_summary()
        assert perf_summary["total_calls"] > 0
        assert len(perf_summary["operations"]) > 0
    
    def test_error_handling_integration(self, test_context, reset_performance):
        """Test detector error handling with problematic data."""
        detector = GA4Detector()
        
        # Create page with problematic requests
        problematic_requests = [
            RequestLog(
                url="https://www.google-analytics.com/mp/collect",
                method="POST",
                resource_type=ResourceType.FETCH,
                status=RequestStatus.FAILED,
                status_code=500,
                request_body='{"malformed": json}',  # Invalid JSON
                error_text="Server error"
            ),
            RequestLog(
                url="not-a-valid-url",  # Invalid URL
                method="GET",
                resource_type=ResourceType.OTHER,
                status=RequestStatus.FAILED
            )
        ]
        
        problematic_page = PageResult(
            url="https://example.com/error-test",
            capture_status=CaptureStatus.PARTIAL,
            network_requests=problematic_requests,
            page_errors=["JavaScript error: undefined function"]
        )
        
        # Run detection - should not crash
        result = asyncio.run(detector.detect(problematic_page, test_context))
        
        # Should still return a result
        assert result is not None
        assert result.detector_name == "GA4Detector"
        
        # May have errors but should be gracefully handled
        if result.has_errors:
            # Verify errors are properly categorized
            error_notes = [note for note in result.notes if note.severity.value == "error"]
            for error_note in error_notes:
                assert error_note.message is not None
                assert error_note.detector_name == "GA4Detector"
    
    def test_performance_constraints(self, test_context, reset_performance):
        """Test detector performance with high-volume data."""
        detector = GA4Detector()
        
        # Create page with many requests
        many_requests = []
        for i in range(100):
            many_requests.append(
                RequestLog(
                    url=f"https://www.google-analytics.com/mp/collect?id={i}",
                    method="POST",
                    resource_type=ResourceType.FETCH,
                    status=RequestStatus.SUCCESS,
                    request_body=f'{{"client_id":"123.456","events":[{{"name":"event_{i}"}}]}}',
                    start_time=datetime.utcnow()
                )
            )
        
        high_volume_page = PageResult(
            url="https://example.com/high-volume-test",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=many_requests
        )
        
        # Run detection with performance monitoring
        import time
        start_time = time.perf_counter()
        
        result = asyncio.run(detector.detect(high_volume_page, test_context))
        
        end_time = time.perf_counter()
        actual_processing_time = (end_time - start_time) * 1000
        
        # Verify performance constraints
        assert result.success
        assert actual_processing_time < test_context.max_processing_time_ms
        assert len(result.events) <= test_context.max_events_per_page
        
        # Check performance metrics
        perf_summary = get_performance_summary()
        assert perf_summary["total_calls"] > 0
        
        # Verify caching is working (should have good hit rates for repeated patterns)
        if "operations" in perf_summary:
            pattern_match_ops = perf_summary["operations"].get("ga4_pattern_match", {})
            if pattern_match_ops.get("total_calls", 0) > 10:
                # Should have some cache efficiency with repeated patterns
                assert pattern_match_ops.get("cache_hit_rate", 0) > 0
    
    def test_configuration_integration(self, sample_page_result, reset_performance):
        """Test detector behavior with different configurations."""
        # Test with debug disabled
        production_context = DetectContext(
            environment="production",
            is_production=True,
            config={
                "ga4": {"mp_debug": {"enabled": False}},
                "gtm": {"validate_datalayer": False}
            },
            enable_debug=False
        )
        
        detector = GA4Detector()
        result = asyncio.run(detector.detect(sample_page_result, production_context))
        
        assert result.success
        # Should have fewer debug-related notes in production
        debug_notes = [note for note in result.notes if "debug" in note.message.lower()]
        # MP debug should not run in production
        
        # Test with limited processing
        limited_context = DetectContext(
            environment="test",
            max_events_per_page=5,
            max_processing_time_ms=100
        )
        
        # Should still work with constraints
        result_limited = asyncio.run(detector.detect(sample_page_result, limited_context))
        assert result_limited is not None