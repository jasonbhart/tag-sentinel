#!/usr/bin/env python3
"""Interactive testing script for GA4 & GTM detectors."""

import asyncio
import json
from datetime import datetime
from typing import List

from app.audit.detectors import (
    GA4Detector, 
    GTMDetector, 
    DuplicateAnalyzer,
    DetectContext,
    registry
)
from app.audit.models.capture import (
    PageResult,
    RequestLog, 
    CaptureStatus,
    RequestStatus,
    ResourceType
)


class DetectorTester:
    """Interactive testing class for detectors."""
    
    def __init__(self):
        self.context = DetectContext(
            environment="test",
            is_production=False,
            config={
                "ga4": {
                    "enabled": True,
                    "mp_debug": {"enabled": False}  # Set to True to test MP debug
                },
                "gtm": {
                    "enabled": True,
                    "validate_datalayer": True
                },
                "duplicates": {
                    "enabled": True,
                    "window_ms": 4000
                }
            },
            enable_debug=True,
            enable_external_validation=False  # Set to True for MP debug
        )
    
    def create_ga4_requests(self, scenario: str) -> List[RequestLog]:
        """Create different GA4 request scenarios."""
        scenarios = {
            "ecommerce": [
                RequestLog(
                    url="https://www.google-analytics.com/mp/collect?measurement_id=G-ABCD1234",
                    method="POST",
                    resource_type=ResourceType.FETCH,
                    status=RequestStatus.SUCCESS,
                    status_code=200,
                    request_body=json.dumps({
                        "client_id": "123.456",
                        "events": [{
                            "name": "purchase",
                            "params": {
                                "transaction_id": "T12345",
                                "value": 99.99,
                                "currency": "USD",
                                "items": [{
                                    "item_id": "SKU123",
                                    "item_name": "Test Product",
                                    "category": "Electronics",
                                    "quantity": 1,
                                    "price": 99.99
                                }]
                            }
                        }]
                    }),
                    start_time=datetime.utcnow()
                )
            ],
            "pageview": [
                RequestLog(
                    url="https://www.google-analytics.com/mp/collect?measurement_id=G-ABCD1234",
                    method="POST",
                    resource_type=ResourceType.FETCH,
                    status=RequestStatus.SUCCESS,
                    status_code=200,
                    request_body=json.dumps({
                        "client_id": "789.012",
                        "events": [{
                            "name": "page_view",
                            "params": {
                                "page_title": "Home Page",
                                "page_location": "https://example.com/",
                                "page_referrer": "https://google.com"
                            }
                        }]
                    }),
                    start_time=datetime.utcnow()
                )
            ],
            "custom_events": [
                RequestLog(
                    url="https://www.google-analytics.com/mp/collect?measurement_id=G-ABCD1234",
                    method="POST",
                    resource_type=ResourceType.FETCH,
                    status=RequestStatus.SUCCESS,
                    status_code=200,
                    request_body=json.dumps({
                        "client_id": "345.678",
                        "events": [{
                            "name": "video_play",
                            "params": {
                                "video_title": "Product Demo",
                                "video_duration": 120,
                                "video_percent": 25
                            }
                        }]
                    }),
                    start_time=datetime.utcnow()
                )
            ],
            "multiple": [
                RequestLog(
                    url="https://www.google-analytics.com/mp/collect?measurement_id=G-ABCD1234",
                    method="POST",
                    resource_type=ResourceType.FETCH,
                    status=RequestStatus.SUCCESS,
                    request_body=json.dumps({
                        "client_id": "111.222",
                        "events": [
                            {"name": "page_view", "params": {"page_title": "Product Page"}},
                            {"name": "view_item", "params": {"item_id": "PROD123"}}
                        ]
                    }),
                    start_time=datetime.utcnow()
                ),
                RequestLog(
                    url="https://www.google-analytics.com/g/collect?tid=G-ABCD1234&t=event&ea=click&ec=button",
                    method="GET",
                    resource_type=ResourceType.FETCH,
                    status=RequestStatus.SUCCESS,
                    start_time=datetime.utcnow()
                )
            ]
        }
        return scenarios.get(scenario, scenarios["pageview"])
    
    def create_gtm_requests(self, scenario: str) -> List[RequestLog]:
        """Create different GTM request scenarios."""
        scenarios = {
            "basic": [
                RequestLog(
                    url="https://www.googletagmanager.com/gtm.js?id=GTM-ABC123",
                    method="GET",
                    resource_type=ResourceType.SCRIPT,
                    status=RequestStatus.SUCCESS,
                    status_code=200,
                    response_body='window.dataLayer = window.dataLayer || []; gtag("config", "G-ABCD1234");',
                    start_time=datetime.utcnow()
                )
            ],
            "multiple_containers": [
                RequestLog(
                    url="https://www.googletagmanager.com/gtm.js?id=GTM-ABC123",
                    method="GET",
                    resource_type=ResourceType.SCRIPT,
                    status=RequestStatus.SUCCESS,
                    response_body='window.dataLayer = window.dataLayer || [];',
                    start_time=datetime.utcnow()
                ),
                RequestLog(
                    url="https://www.googletagmanager.com/gtm.js?id=GTM-XYZ789",
                    method="GET", 
                    resource_type=ResourceType.SCRIPT,
                    status=RequestStatus.SUCCESS,
                    response_body='gtag("config", "G-SECOND123");',
                    start_time=datetime.utcnow()
                )
            ]
        }
        return scenarios.get(scenario, scenarios["basic"])
    
    async def test_ga4_scenario(self, scenario: str):
        """Test GA4 detector with specific scenario."""
        print(f"\n🔍 Testing GA4 Scenario: {scenario.upper()}")
        print("-" * 50)
        
        requests = self.create_ga4_requests(scenario)
        
        page_result = PageResult(
            url=f"https://example.com/{scenario}",
            title=f"Test {scenario.title()} Page",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=requests
        )
        
        detector = GA4Detector()
        result = await detector.detect(page_result, self.context)
        
        print(f"✅ Detection Results:")
        print(f"   Success: {result.success}")
        print(f"   Processing time: {result.processing_time_ms}ms")
        print(f"   Events detected: {len(result.events)}")
        print(f"   Notes: {len(result.notes)}")
        
        if result.events:
            print(f"\n📊 Events:")
            for i, event in enumerate(result.events, 1):
                print(f"   {i}. {event.name} ({event.vendor})")
                print(f"      ID: {event.id}")
                print(f"      Confidence: {event.confidence}")
                print(f"      Status: {event.status}")
                if event.params:
                    key_params = {k: v for k, v in event.params.items() if k in ['event_name', 'measurement_id', 'client_id', 'page_title']}
                    if key_params:
                        print(f"      Key params: {key_params}")
        
        if result.notes:
            print(f"\n📝 Notes:")
            for note in result.notes:
                print(f"   {note.severity}: {note.message}")
        
        print()
    
    def test_gtm_scenario(self, scenario: str):
        """Test GTM detector with specific scenario."""
        print(f"\n🔍 Testing GTM Scenario: {scenario.upper()}")
        print("-" * 50)
        
        requests = self.create_gtm_requests(scenario)
        
        page_result = PageResult(
            url=f"https://example.com/{scenario}",
            title=f"Test {scenario.title()} Page", 
            capture_status=CaptureStatus.SUCCESS,
            network_requests=requests
        )
        
        detector = GTMDetector()
        result = detector.detect(page_result, self.context)
        
        print(f"✅ Detection Results:")
        print(f"   Success: {result.success}")
        print(f"   Processing time: {result.processing_time_ms}ms")
        print(f"   Events detected: {len(result.events)}")
        print(f"   Notes: {len(result.notes)}")
        
        if result.events:
            print(f"\n📊 Events:")
            for i, event in enumerate(result.events, 1):
                print(f"   {i}. {event.name} ({event.vendor})")
                print(f"      ID: {event.id}")
                print(f"      Status: {event.status}")
        
        if result.notes:
            print(f"\n📝 Notes:")
            for note in result.notes:
                print(f"   {note.severity}: {note.message}")
        
        print()
    
    async def test_performance_scenario(self):
        """Test detector performance with high-volume requests."""
        print(f"\n🚀 Testing Performance Scenario")
        print("-" * 50)
        
        # Create many GA4 requests
        requests = []
        for i in range(20):
            requests.append(RequestLog(
                url=f"https://www.google-analytics.com/mp/collect?measurement_id=G-PERF{i:03d}",
                method="POST",
                resource_type=ResourceType.FETCH,
                status=RequestStatus.SUCCESS,
                request_body=json.dumps({
                    "client_id": f"{i}.{i*2}",
                    "events": [{
                        "name": "test_event",
                        "params": {"test_param": f"value_{i}"}
                    }]
                }),
                start_time=datetime.utcnow()
            ))
        
        page_result = PageResult(
            url="https://example.com/performance-test",
            title="Performance Test Page",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=requests
        )
        
        # Test with time measurement
        import time
        start_time = time.perf_counter()
        
        detector = GA4Detector()
        result = await detector.detect(page_result, self.context)
        
        end_time = time.perf_counter()
        actual_time = (end_time - start_time) * 1000
        
        print(f"✅ Performance Results:")
        print(f"   Requests processed: {len(requests)}")
        print(f"   Events detected: {len(result.events)}")
        print(f"   Actual processing time: {actual_time:.2f}ms")
        print(f"   Reported processing time: {result.processing_time_ms}ms")
        print(f"   Average per request: {actual_time/len(requests):.2f}ms")
        
        if result.metrics:
            print(f"   Metrics available: {list(result.metrics.keys())}")
        
        print()


async def main():
    """Run interactive detector tests."""
    tester = DetectorTester()
    
    print("🧪 GA4 & GTM Detector Interactive Tests")
    print("=" * 60)
    
    # Test different GA4 scenarios
    await tester.test_ga4_scenario("pageview")
    await tester.test_ga4_scenario("ecommerce") 
    await tester.test_ga4_scenario("custom_events")
    await tester.test_ga4_scenario("multiple")
    
    # Test GTM scenarios
    tester.test_gtm_scenario("basic")
    tester.test_gtm_scenario("multiple_containers")
    
    # Test performance
    await tester.test_performance_scenario()
    
    print("✨ All interactive tests completed!")
    print("\n💡 To test with MP Debug validation:")
    print("   1. Set enable_external_validation=True in DetectContext")
    print("   2. Set mp_debug.enabled=True in config")
    print("   3. Run in non-production environment")


if __name__ == "__main__":
    asyncio.run(main())