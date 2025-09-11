#!/usr/bin/env python3
"""
DataLayer Integrity Example

This example demonstrates how to use the DataLayer integrity system to capture,
validate, and audit dataLayer objects from web pages.

Prerequisites:
- Playwright browser automation
- JSON Schema validation (optional: pip install jsonschema)

Usage:
    python examples/datalayer_example.py
"""

import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

# Import the DataLayer integrity system
from app.audit.datalayer import (
    DataLayerService, 
    capture_page_datalayer,
    DataLayerConfig,
    DLContext,
    ValidationSeverity,
    RedactionMethod
)


async def simple_capture_example():
    """Simple example of capturing dataLayer from a page."""
    print("=== Simple DataLayer Capture Example ===")
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Create test page with dataLayer
        await page.set_content("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
        </head>
        <body>
            <h1>DataLayer Test Page</h1>
            <script>
                // Initialize dataLayer with typical GTM pattern
                window.dataLayer = window.dataLayer || [];
                
                // Push initial data
                dataLayer.push({
                    'page_type': 'example',
                    'user': {
                        'id': '12345',
                        'email': 'user@example.com',
                        'preferences': {
                            'newsletter': true,
                            'analytics': true
                        }
                    },
                    'site': {
                        'name': 'Example Site',
                        'version': '1.0.0'
                    }
                });
                
                // Push some events
                dataLayer.push({
                    'event': 'page_view',
                    'page_title': 'Test Page',
                    'page_location': window.location.href
                });
                
                dataLayer.push({
                    'event': 'user_engagement',
                    'engagement_time_msec': 1500
                });
            </script>
        </body>
        </html>
        """)
        
        # Wait for page to load
        await page.wait_for_timeout(1000)
        
        # Capture dataLayer using simple function
        result = await capture_page_datalayer(page)
        
        print(f"DataLayer exists: {result.snapshot.exists}")
        print(f"Variables found: {result.snapshot.variable_count}")
        print(f"Events found: {result.snapshot.event_count}")
        print(f"Variable names: {result.snapshot.get_variable_names()}")
        print(f"Event types: {result.snapshot.get_event_types()}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
        
        if result.snapshot.latest:
            print("\\nLatest dataLayer state:")
            print(json.dumps(result.snapshot.latest, indent=2))
        
        if result.snapshot.events:
            print("\\nCaptured events:")
            for i, event in enumerate(result.snapshot.events):
                print(f"  Event {i+1}: {json.dumps(event, indent=2)}")
        
        await browser.close()


async def advanced_service_example():
    """Advanced example using DataLayerService with configuration."""
    print("\\n=== Advanced DataLayer Service Example ===")
    
    # Create custom configuration
    config = DataLayerConfig()
    
    # Configure redaction for privacy
    config.redaction.enabled = True
    config.redaction.default_method = RedactionMethod.HASH
    
    # Configure capture limits
    config.capture.max_depth = 4
    config.capture.max_entries = 100
    
    # Create service with configuration
    service = DataLayerService(config)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Create test page with sensitive data
        await page.set_content("""
        <!DOCTYPE html>
        <html>
        <head><title>E-commerce Test</title></head>
        <body>
            <script>
                window.dataLayer = [{
                    'page_type': 'product',
                    'user': {
                        'id': 'user123',
                        'email': 'customer@example.com',
                        'phone': '555-123-4567',
                        'address': {
                            'street': '123 Main St',
                            'city': 'Anytown',
                            'zip': '12345'
                        }
                    },
                    'product': {
                        'id': 'prod456',
                        'name': 'Test Product',
                        'category': 'electronics',
                        'price': 99.99
                    },
                    'cart': {
                        'value': 199.98,
                        'items': 2,
                        'currency': 'USD'
                    }
                }];
                
                // Push purchase event
                dataLayer.push({
                    'event': 'purchase',
                    'transaction_id': 'tx789',
                    'value': 199.98,
                    'currency': 'USD',
                    'payment_method': 'credit_card'
                });
            </script>
        </body>
        </html>
        """)
        
        await page.wait_for_timeout(1000)
        
        # Start aggregation for multiple page processing
        service.start_aggregation("example_run_001")
        
        # Capture with service
        result = await service.capture_and_validate(
            page, 
            page_url="https://example.com/product",
            site_domain="example.com"
        )
        
        print(f"DataLayer exists: {result.snapshot.exists}")
        print(f"Variables: {result.snapshot.variable_count}")
        print(f"Events: {result.snapshot.event_count}")
        print(f"Redacted paths: {len(result.snapshot.redacted_paths)}")
        print(f"Validation issues: {len(result.issues)}")
        print(f"Processing notes: {result.notes}")
        
        # Show redacted data (sensitive info should be hashed)
        if result.snapshot.latest:
            print("\\nRedacted dataLayer (sensitive data hashed):")
            print(json.dumps(result.snapshot.latest, indent=2))
        
        # Get service health status
        health = service.health_check()
        print(f"\\nService health: {health['status']}")
        print(f"Components: {list(health['components'].keys())}")
        
        # Finalize aggregation
        aggregation = service.finalize_aggregation()
        if aggregation:
            summary = aggregation.export_summary()
            print(f"\\nAggregation summary:")
            print(f"  Pages processed: {summary['total_pages']}")
            print(f"  DataLayer presence rate: {summary['datalayer_presence_rate']:.1f}%")
            print(f"  Success rate: {summary['success_rate']:.1f}%")
            print(f"  Unique variables: {summary['unique_variables']}")
            print(f"  Unique events: {summary['unique_events']}")
        
        await browser.close()


async def batch_processing_example():
    """Example of processing multiple pages in batch."""
    print("\\n=== Batch Processing Example ===")
    
    service = DataLayerService()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        
        # Create multiple test pages
        page_configs = [
            {
                'url': 'https://example.com/home',
                'content': '''
                <script>
                    window.dataLayer = [{'page_type': 'home', 'user_id': 'user1'}];
                    dataLayer.push({'event': 'page_view', 'page_title': 'Home'});
                </script>
                '''
            },
            {
                'url': 'https://example.com/product',
                'content': '''
                <script>
                    window.dataLayer = [{'page_type': 'product', 'product_id': 'p123'}];
                    dataLayer.push({'event': 'view_item', 'item_id': 'p123'});
                </script>
                '''
            },
            {
                'url': 'https://example.com/cart',
                'content': '''
                <script>
                    window.dataLayer = [{'page_type': 'cart', 'cart_value': 99.99}];
                    dataLayer.push({'event': 'view_cart', 'value': 99.99});
                </script>
                '''
            }
        ]
        
        # Create pages
        page_contexts = []
        for config in page_configs:
            page = await context.new_page()
            await page.set_content(f'<html><body>{config["content"]}</body></html>')
            await page.wait_for_timeout(500)
            page_contexts.append((page, config['url'], 'example.com'))
        
        # Progress callback
        async def progress_callback(current, total, result):
            print(f"  Processed {current}/{total}: {result.snapshot.page_url}")
        
        # Process all pages in batch
        results, aggregation = await service.process_multiple_pages(
            page_contexts,
            run_id="batch_example",
            progress_callback=progress_callback
        )
        
        print(f"\\nBatch processing complete!")
        print(f"Pages processed: {len(results)}")
        
        # Summary statistics
        successful = len([r for r in results if r.is_successful])
        total_variables = sum(r.snapshot.variable_count for r in results)
        total_events = sum(r.snapshot.event_count for r in results)
        
        print(f"Successful captures: {successful}/{len(results)}")
        print(f"Total variables found: {total_variables}")
        print(f"Total events captured: {total_events}")
        
        if aggregation:
            summary = aggregation.export_summary()
            print(f"DataLayer presence rate: {summary['datalayer_presence_rate']:.1f}%")
            print(f"Most common variables: {summary['most_common_variables'][:3]}")
            print(f"Most frequent events: {summary['most_frequent_events'][:3]}")
        
        # Cleanup
        for page, _, _ in page_contexts:
            await page.close()
        await browser.close()


async def main():
    """Run all examples."""
    print("DataLayer Integrity System Examples")
    print("=" * 40)
    
    try:
        await simple_capture_example()
        await advanced_service_example()
        await batch_processing_example()
        
        print("\\n" + "=" * 40)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())