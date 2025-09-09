#!/usr/bin/env python3
"""Example script to test the GA4 & GTM detectors with sample data."""

import asyncio
from datetime import datetime

from app.audit.detectors import (
    GA4Detector, 
    GTMDetector, 
    DuplicateAnalyzer,
    SequencingAnalyzer,
    DetectContext,
    registry,
    get_config,
    configure_detectors
)
from app.audit.models.capture import (
    PageResult,
    RequestLog, 
    CaptureStatus,
    RequestStatus,
    ResourceType
)


async def test_ga4_detection():
    """Test GA4 detector with sample requests."""
    print("🔍 Testing GA4 Detector...")
    
    # Create sample GA4 requests
    ga4_requests = [
        RequestLog(
            url="https://www.google-analytics.com/mp/collect?measurement_id=G-XXXXXXXXXX",
            method="POST",
            resource_type=ResourceType.FETCH,
            status=RequestStatus.SUCCESS,
            status_code=200,
            request_body='{"client_id":"123.456","events":[{"name":"page_view","params":{"page_title":"Test Page","page_location":"https://example.com/test"}}]}',
            start_time=datetime.utcnow()
        ),
        RequestLog(
            url="https://www.google-analytics.com/g/collect?tid=G-XXXXXXXXXX&t=pageview&tid=G-XXXXXXXXXX",
            method="GET", 
            resource_type=ResourceType.FETCH,
            status=RequestStatus.SUCCESS,
            status_code=200,
            start_time=datetime.utcnow()
        )
    ]
    
    # Create page result
    page_result = PageResult(
        url="https://example.com/test",
        title="Test Page",
        capture_status=CaptureStatus.SUCCESS,
        network_requests=ga4_requests
    )
    
    # Create detection context
    context = DetectContext(
        environment="test",
        is_production=False,
        config={
            "ga4": {
                "enabled": True,
                "mp_debug": {"enabled": False}  # Disable for basic test
            }
        },
        enable_debug=True
    )
    
    # Run detector
    detector = GA4Detector()
    result = await detector.detect(page_result, context)
    
    print(f"✅ GA4 Detection Results:")
    print(f"   - Success: {result.success}")
    print(f"   - Events detected: {len(result.events)}")
    print(f"   - Processing time: {result.processing_time_ms}ms")
    
    for event in result.events:
        print(f"   - {event.vendor} {event.name}: {event.id}")
        
    for note in result.notes:
        print(f"   📝 {note.severity}: {note.message}")


def test_gtm_detection():
    """Test GTM detector with sample requests."""
    print("\n🔍 Testing GTM Detector...")
    
    # Create sample GTM requests
    gtm_requests = [
        RequestLog(
            url="https://www.googletagmanager.com/gtm.js?id=GTM-ABC123",
            method="GET",
            resource_type=ResourceType.SCRIPT,
            status=RequestStatus.SUCCESS,
            status_code=200,
            response_body='window.dataLayer = window.dataLayer || []; gtag("config", "G-123");',
            start_time=datetime.utcnow()
        )
    ]
    
    # Create page result
    page_result = PageResult(
        url="https://example.com/test",
        title="Test Page",
        capture_status=CaptureStatus.SUCCESS,
        network_requests=gtm_requests
    )
    
    # Create detection context
    context = DetectContext(
        environment="test",
        config={
            "gtm": {
                "enabled": True,
                "validate_datalayer": True
            }
        },
        enable_debug=True
    )
    
    # Run detector
    detector = GTMDetector()
    result = detector.detect(page_result, context)
    
    print(f"✅ GTM Detection Results:")
    print(f"   - Success: {result.success}")
    print(f"   - Events detected: {len(result.events)}")
    print(f"   - Processing time: {result.processing_time_ms}ms")
    
    for event in result.events:
        print(f"   - {event.vendor} {event.name}: {event.id}")
        
    for note in result.notes:
        print(f"   📝 {note.severity}: {note.message}")


def test_duplicate_detection():
    """Test duplicate detection with sample data."""
    print("\n🔍 Testing Duplicate Detection...")
    
    # Create page result (duplicate analyzer doesn't use network requests directly)
    page_result = PageResult(
        url="https://example.com/test",
        capture_status=CaptureStatus.SUCCESS,
        network_requests=[]
    )
    
    context = DetectContext(
        environment="test",
        config={
            "duplicates": {
                "enabled": True,
                "window_ms": 4000
            }
        }
    )
    
    detector = DuplicateAnalyzer()
    result = detector.detect(page_result, context)
    
    print(f"✅ Duplicate Detection Results:")
    print(f"   - Success: {result.success}")
    print(f"   - Events detected: {len(result.events)}")
    
    for note in result.notes:
        print(f"   📝 {note.severity}: {note.message}")


async def test_full_pipeline():
    """Test complete detector pipeline."""
    print("\n🚀 Testing Full Detection Pipeline...")
    
    # Configure registry
    config = get_config()
    configure_detectors(registry, config_path=None)
    
    # Create comprehensive page result
    all_requests = [
        # GA4 requests
        RequestLog(
            url="https://www.google-analytics.com/mp/collect?measurement_id=G-XXXXXXXXXX",
            method="POST",
            resource_type=ResourceType.FETCH,
            status=RequestStatus.SUCCESS,
            request_body='{"client_id":"123.456","events":[{"name":"purchase","params":{"transaction_id":"12345","value":99.99}}]}',
            start_time=datetime.utcnow()
        ),
        # GTM requests
        RequestLog(
            url="https://www.googletagmanager.com/gtm.js?id=GTM-ABC123",
            method="GET",
            resource_type=ResourceType.SCRIPT,
            status=RequestStatus.SUCCESS,
            start_time=datetime.utcnow()
        )
    ]
    
    page_result = PageResult(
        url="https://example.com/checkout-complete",
        title="Order Complete",
        capture_status=CaptureStatus.SUCCESS,
        network_requests=all_requests
    )
    
    context = DetectContext(
        environment="test",
        is_production=False,
        enable_debug=True
    )
    
    # Get all enabled detectors
    detectors = registry.get_enabled_detectors()
    print(f"   Running {len(detectors)} detectors...")
    
    all_results = []
    
    for detector in detectors:
        print(f"   - Running {detector.name}...")
        
        # Handle both sync and async detectors
        if hasattr(detector.detect, '__code__') and detector.detect.__code__.co_flags & 0x80:
            # Async detector
            result = await detector.detect(page_result, context)
        else:
            # Sync detector  
            result = detector.detect(page_result, context)
            
        all_results.append(result)
        print(f"     ✅ {result.detector_name}: {len(result.events)} events, {len(result.notes)} notes")
    
    # Summary
    total_events = sum(len(r.events) for r in all_results)
    total_notes = sum(len(r.notes) for r in all_results)
    
    print(f"\n📊 Pipeline Summary:")
    print(f"   - Total events detected: {total_events}")
    print(f"   - Total notes generated: {total_notes}")
    print(f"   - All detectors successful: {all(r.success for r in all_results)}")


async def main():
    """Run all detector tests."""
    print("🧪 Testing GA4 & GTM Detectors\n")
    
    await test_ga4_detection()
    test_gtm_detection()
    test_duplicate_detection()
    await test_full_pipeline()
    
    print("\n✨ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())