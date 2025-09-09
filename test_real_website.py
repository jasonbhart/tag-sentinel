#!/usr/bin/env python3
"""Example of how to test detectors with real website capture data."""

import asyncio
from datetime import datetime

# You would import your browser capture system here
# from app.audit.capture import BrowserEngine

from app.audit.detectors import GA4Detector, GTMDetector, DetectContext
from app.audit.models.capture import PageResult, CaptureStatus


async def test_real_website():
    """Test detectors with a real website that has GA4/GTM."""
    
    # This is pseudocode - you'd need to implement the browser capture
    # browser_engine = BrowserEngine()
    # page_result = await browser_engine.capture_page("https://example-ecommerce.com")
    
    # For now, create a mock result that represents real capture data
    page_result = PageResult(
        url="https://shop.example.com/product/123",
        title="Product Page - Example Shop",
        capture_status=CaptureStatus.SUCCESS,
        network_requests=[
            # Real GA4 request captured from network
            # RequestLog(...),
        ]
    )
    
    context = DetectContext(
        environment="test",
        is_production=False,
        site_domain="example.com",
        audit_id="test-audit-001",
        enable_external_validation=True  # Enable MP debug validation
    )
    
    # Test GA4 detector
    ga4_detector = GA4Detector()
    ga4_result = await ga4_detector.detect(page_result, context)
    
    print(f"Real Website GA4 Detection:")
    print(f"  Events: {len(ga4_result.events)}")
    print(f"  Notes: {len(ga4_result.notes)}")
    
    # Test GTM detector  
    gtm_detector = GTMDetector()
    gtm_result = gtm_detector.detect(page_result, context)
    
    print(f"Real Website GTM Detection:")
    print(f"  Events: {len(gtm_result.events)}")
    print(f"  Notes: {len(gtm_result.notes)}")


if __name__ == "__main__":
    print("To test with real websites, you need to:")
    print("1. Implement browser capture (EPIC 2)")
    print("2. Capture a page with GA4/GTM")
    print("3. Pass the PageResult to detectors")
    print("\nCurrently testing with mock data...")
    # asyncio.run(test_real_website())