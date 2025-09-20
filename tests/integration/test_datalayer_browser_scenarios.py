"""Integration tests for specific browser scenarios and edge cases."""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.audit.datalayer.models import DLContext


@pytest.mark.integration
class TestSinglePageApplicationScenarios:
    """Integration tests for Single Page Application (SPA) scenarios."""
    
    @pytest.mark.asyncio
    async def test_spa_navigation_datalayer_changes(self, integration_service):
        """Test dataLayer changes during SPA navigation."""
        # Simulate SPA page transitions with dataLayer updates
        spa_scenarios = [
            {
                "route": "/",
                "datalayer": {
                    "page": {"title": "Home", "route": "/", "spa": True},
                    "user": {"id": "user123", "session": "sess_abc"},
                    "navigation": {"type": "initial_load"}
                },
                "events": [{"event": "page_view", "method": "initial"}]
            },
            {
                "route": "/products",
                "datalayer": {
                    "page": {"title": "Products", "route": "/products", "spa": True},
                    "user": {"id": "user123", "session": "sess_abc"},  # User persists
                    "navigation": {"type": "spa_navigation", "from": "/"},
                    "category": {"name": "all_products", "count": 150}
                },
                "events": [{"event": "page_view", "method": "spa_navigation"}]
            },
            {
                "route": "/products/123",
                "datalayer": {
                    "page": {"title": "Product Detail", "route": "/products/123", "spa": True},
                    "user": {"id": "user123", "session": "sess_abc"},
                    "navigation": {"type": "spa_navigation", "from": "/products"},
                    "product": {"id": "123", "name": "Wireless Mouse", "price": 29.99}
                },
                "events": [
                    {"event": "page_view", "method": "spa_navigation"},
                    {"event": "view_item", "item_id": "123"}
                ]
            }
        ]
        
        results = []
        
        for scenario in spa_scenarios:
            mock_page = AsyncMock()
            mock_page.url = f"https://spa-example.com{scenario['route']}"
            mock_page.evaluate.return_value = {
                'exists': True,
                'objectName': 'dataLayer',
                'latest': scenario['datalayer'],
                'events': scenario.get('events', []),
                'truncated': False
            }
            
            context = DLContext(
                env='test',
                url=f"https://spa-example.com{scenario['route']}",
                page_title=scenario['datalayer']['page']['title']
            )
            
            result = await integration_service.capture_and_validate(mock_page, context)
            results.append(result)
        
        # Verify all captures successful
        assert all(result.snapshot.exists for result in results)
        
        # Verify user session persistence across navigation
        user_sessions = [r.snapshot.latest['user']['session'] for r in results]
        assert all(session == "sess_abc" for session in user_sessions)
        
        # Verify navigation metadata is captured
        nav_types = [r.snapshot.latest['navigation']['type'] for r in results]
        assert nav_types[0] == "initial_load"
        assert nav_types[1] == "spa_navigation"
        assert nav_types[2] == "spa_navigation"
    
    @pytest.mark.asyncio
    async def test_dynamic_datalayer_updates(self, integration_service):
        """Test dynamic updates to dataLayer after page load."""
        # Simulate dynamic dataLayer updates that happen after initial page load
        initial_datalayer = {
            "page": {"title": "Dynamic Page", "loaded": True},
            "user": {"id": "user456"},
            "loading_state": "initial"
        }
        
        # Simulate user interactions that update dataLayer
        updated_datalayer = {
            "page": {"title": "Dynamic Page", "loaded": True},
            "user": {"id": "user456", "interacted": True},
            "loading_state": "interactive",
            "form_data": {"field1": "value1", "field2": "value2"},
            "interactions": [
                {"type": "click", "element": "button", "timestamp": "10:30:15"},
                {"type": "scroll", "depth": 50, "timestamp": "10:30:20"}
            ]
        }
        
        # Test initial capture
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/dynamic"
        mock_page.evaluate.return_value = {
            'exists': True,
            'objectName': 'dataLayer',
            'latest': initial_datalayer,
            'events': [],
            'truncated': False
        }
        
        context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://example.com/dynamic"})
        initial_result = await integration_service.capture_and_validate(mock_page, context)
        
        # Test updated capture (simulating later capture)
        mock_page.evaluate.return_value = {
            'exists': True,
            'objectName': 'dataLayer',
            'latest': updated_datalayer,
            'events': [],
            'truncated': False
        }
        updated_result = await integration_service.capture_and_validate(mock_page, context)
        
        # Verify both captures successful
        assert initial_result.snapshot.exists
        assert updated_result.snapshot.exists
        
        # Verify data evolution
        assert initial_result.snapshot.latest['loading_state'] == "initial"
        assert updated_result.snapshot.latest['loading_state'] == "interactive"
        
        # Verify new data added
        assert 'form_data' not in initial_result.snapshot.latest
        assert 'form_data' in updated_result.snapshot.latest
        assert len(updated_result.snapshot.latest['interactions']) == 2


@pytest.mark.integration
class TestEcommerceScenarios:
    """Integration tests for e-commerce specific scenarios."""
    
    @pytest.mark.asyncio
    async def test_ecommerce_funnel_tracking(self, integration_service, ecommerce_schema):
        """Test complete e-commerce funnel with dataLayer tracking."""
        funnel_steps = [
            {
                "step": "product_view",
                "url": "https://shop.example.com/products/headphones",
                "datalayer": {
                    "page": {"title": "Wireless Headphones", "type": "product"},
                    "user": {"id": "user789", "segment": "premium"},
                    "product": {
                        "id": "headphones001",
                        "name": "Wireless Headphones",
                        "brand": "AudioBrand",
                        "price": 159.99,
                        "currency": "USD",
                        "category": "electronics"
                    }
                },
                "events": [
                    {"event": "page_view", "page_type": "product"},
                    {"event": "view_item", "item_id": "headphones001", "value": 159.99}
                ]
            },
            {
                "step": "add_to_cart",
                "url": "https://shop.example.com/products/headphones",
                "datalayer": {
                    "page": {"title": "Wireless Headphones", "type": "product"},
                    "user": {"id": "user789", "segment": "premium"},
                    "product": {
                        "id": "headphones001",
                        "name": "Wireless Headphones",
                        "price": 159.99,
                        "currency": "USD"
                    },
                    "cart": {"items": 1, "value": 159.99}
                },
                "events": [
                    {"event": "add_to_cart", "item_id": "headphones001", "value": 159.99, "quantity": 1}
                ]
            },
            {
                "step": "cart_view",
                "url": "https://shop.example.com/cart",
                "datalayer": {
                    "page": {"title": "Shopping Cart", "type": "cart"},
                    "user": {"id": "user789", "segment": "premium"},
                    "cart": {
                        "items": [
                            {"id": "headphones001", "name": "Wireless Headphones", "price": 159.99, "quantity": 1}
                        ],
                        "total_items": 1,
                        "total_value": 159.99,
                        "currency": "USD"
                    }
                },
                "events": [
                    {"event": "page_view", "page_type": "cart"},
                    {"event": "view_cart", "value": 159.99}
                ]
            },
            {
                "step": "checkout",
                "url": "https://shop.example.com/checkout",
                "datalayer": {
                    "page": {"title": "Checkout", "type": "checkout"},
                    "user": {"id": "user789", "segment": "premium"},
                    "ecommerce": {
                        "currency": "USD",
                        "value": 159.99,
                        "items": [
                            {
                                "item_id": "headphones001",
                                "item_name": "Wireless Headphones",
                                "category": "electronics",
                                "brand": "AudioBrand",
                                "price": 159.99,
                                "quantity": 1
                            }
                        ]
                    }
                },
                "events": [
                    {"event": "page_view", "page_type": "checkout"},
                    {"event": "begin_checkout", "value": 159.99, "currency": "USD"}
                ]
            }
        ]
        
        results = []
        
        for step in funnel_steps:
            mock_page = AsyncMock()
            mock_page.url = step["url"]
            mock_page.evaluate.return_value = {
                'exists': True,
                'objectName': 'dataLayer',
                'latest': step["datalayer"],
                'events': [],
                'truncated': False
            }
            
            context = DLContext(env='test')
            result = await integration_service.capture_and_validate(
                mock_page, context, ecommerce_schema
            )
            results.append((step["step"], result))
        
        # Verify all steps captured successfully
        assert all(result.snapshot.exists for _, result in results)
        
        # Verify no validation errors for valid e-commerce data
        validation_errors = [len(result.issues) for _, result in results]
        assert all(errors == 0 for errors in validation_errors)
        
        # Verify funnel progression
        user_ids = [result.snapshot.latest["user"]["id"] for _, result in results]
        assert all(uid == "user789" for uid in user_ids)  # Same user throughout
        
        # Verify cart value progression
        step_values = []
        for step_name, result in results:
            latest = result.snapshot.latest
            if "cart" in latest:
                step_values.append(latest["cart"].get("total_value", latest["cart"].get("value")))
            elif "ecommerce" in latest:
                step_values.append(latest["ecommerce"]["value"])
        
        # Cart values should be consistent
        assert all(value == 159.99 for value in step_values)
    
    @pytest.mark.asyncio
    async def test_enhanced_ecommerce_events(self, integration_service):
        """Test enhanced e-commerce events with complex data structures."""
        enhanced_ecommerce_data = {
            "page": {"title": "Purchase Complete", "type": "checkout"},
            "user": {"id": "user999", "segment": "vip"},
            "ecommerce": {
                "transaction_id": "TXN-2024-001",
                "affiliation": "Online Store",
                "value": 459.97,
                "tax": 36.80,
                "shipping": 9.99,
                "currency": "USD",
                "coupon": "SAVE10",
                "items": [
                    {
                        "item_id": "laptop001",
                        "item_name": "Gaming Laptop",
                        "category": "computers",
                        "category2": "gaming",
                        "category3": "laptops",
                        "brand": "TechBrand",
                        "variant": "16GB-1TB",
                        "price": 399.99,
                        "quantity": 1,
                        "discount": 40.00,
                        "position": 1
                    },
                    {
                        "item_id": "mouse001", 
                        "item_name": "Gaming Mouse",
                        "category": "accessories",
                        "category2": "gaming",
                        "brand": "TechBrand",
                        "price": 59.98,
                        "quantity": 2,
                        "position": 2
                    }
                ]
            },
            "promotion": {
                "id": "PROMO-2024-JAN",
                "name": "January Sale",
                "creative": "banner_top",
                "position": "homepage"
            }
        }
        
        mock_page = AsyncMock()
        mock_page.url = "https://shop.example.com/checkout/success"
        mock_page.evaluate.return_value = {
            'exists': True,
            'objectName': 'dataLayer',
            'latest': enhanced_ecommerce_data,
            'events': [],
            'truncated': False
        }
        
        context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://shop.example.com/checkout/success"})
        result = await integration_service.capture_and_validate(mock_page, context)
        
        # Verify successful capture
        assert result.snapshot.exists
        
        # Verify complex nested structure preserved
        ecommerce = result.snapshot.latest["ecommerce"]
        assert ecommerce["transaction_id"] == "TXN-2024-001"
        assert len(ecommerce["items"]) == 2
        assert ecommerce["items"][0]["category3"] == "laptops"
        assert ecommerce["items"][1]["quantity"] == 2
        
        # Verify promotion data captured
        assert result.snapshot.latest["promotion"]["name"] == "January Sale"


@pytest.mark.integration
class TestErrorRecoveryScenarios:
    """Integration tests for error handling and recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_partial_datalayer_corruption(self, integration_service):
        """Test handling of partially corrupted dataLayer data."""
        # Simulate dataLayer with various data integrity issues
        corrupted_scenarios = [
            {
                "name": "circular_reference",
                "datalayer": {
                    "page": {"title": "Test Page"},
                    "user": {"id": "user123"},
                    "metadata": {}  # Will add circular reference programmatically
                }
            },
            {
                "name": "extremely_nested", 
                "datalayer": self._create_deeply_nested_object(20)  # Very deep nesting
            },
            {
                "name": "mixed_types",
                "datalayer": {
                    "page": {"title": "Mixed Types"},
                    "strange_data": {
                        "function": "function() { return 'test'; }",  # String representation of function
                        "undefined_val": None,
                        "numbers": [1, 2.5, float('inf')],
                        "mixed_array": ["string", 123, True, None, {"nested": "object"}]
                    }
                }
            }
        ]
        
        results = []
        
        for scenario in corrupted_scenarios:
            mock_page = AsyncMock()
            mock_page.url = f"https://example.com/{scenario['name']}"
            
            # For circular reference scenario, create the circular reference
            if scenario['name'] == "circular_reference":
                data = scenario['datalayer']
                data['metadata']['self_reference'] = data  # Create circular reference
                
            mock_page.evaluate.return_value = {
                'exists': True,
                'objectName': 'dataLayer',
                'latest': scenario['datalayer'],
                'events': [],
                'truncated': False
            }
            
            context = DLContext(env='test')
            result = await integration_service.capture_and_validate(mock_page, context)
            results.append((scenario['name'], result))
        
        # System should handle all corrupted scenarios gracefully
        successful_captures = [result.snapshot.exists for _, result in results]
        
        # At least some should succeed (system should be resilient)
        assert sum(successful_captures) >= len(successful_captures) // 2
        
        # None should cause system crashes
        assert all(result is not None for _, result in results)
    
    def _create_deeply_nested_object(self, depth: int) -> Dict[str, Any]:
        """Create deeply nested object for testing depth limits."""
        if depth <= 0:
            return {"value": "deep_value", "depth": 0}
        
        return {
            "level": depth,
            "data": f"level_{depth}_data",
            "nested": self._create_deeply_nested_object(depth - 1)
        }
    
    @pytest.mark.asyncio
    async def test_timeout_and_retry_scenarios(self, integration_service):
        """Test timeout handling and retry mechanisms."""
        timeout_scenarios = [
            {
                "name": "immediate_timeout",
                "side_effect": asyncio.TimeoutError("Immediate timeout")
            },
            {
                "name": "delayed_response",
                "side_effect": lambda: asyncio.sleep(0.1) or {"page": {"title": "Delayed"}}
            },
            {
                "name": "intermittent_failure", 
                "responses": [
                    Exception("First attempt fails"),
                    Exception("Second attempt fails"), 
                    {"page": {"title": "Third attempt succeeds"}}
                ]
            }
        ]
        
        results = []
        
        for scenario in timeout_scenarios:
            mock_page = AsyncMock()
            mock_page.url = f"https://example.com/{scenario['name']}"
            
            if "side_effect" in scenario:
                mock_page.evaluate.side_effect = scenario["side_effect"]
            elif "responses" in scenario:
                mock_page.evaluate.side_effect = scenario["responses"]
            
            context = DLContext(env='test')
            result = await integration_service.capture_and_validate(mock_page, context)
            results.append((scenario['name'], result))
        
        # Verify graceful handling of timeouts and failures
        for name, result in results:
            assert result is not None  # Should not crash
            # Specific behavior depends on retry configuration
    
    @pytest.mark.asyncio
    async def test_memory_pressure_scenarios(self, integration_service):
        """Test behavior under memory pressure conditions."""
        # Create very large dataLayer objects
        large_data_scenarios = [
            {
                "name": "large_array",
                "datalayer": {
                    "page": {"title": "Large Array Test"},
                    "large_array": [f"item_{i}" for i in range(10000)]
                }
            },
            {
                "name": "large_object",
                "datalayer": {
                    "page": {"title": "Large Object Test"}, 
                    "large_object": {f"key_{i}": f"value_{i}" for i in range(5000)}
                }
            },
            {
                "name": "nested_arrays",
                "datalayer": {
                    "page": {"title": "Nested Arrays Test"},
                    "nested_data": [
                        [f"item_{i}_{j}" for j in range(100)]
                        for i in range(100)
                    ]
                }
            }
        ]
        
        results = []
        
        for scenario in large_data_scenarios:
            mock_page = AsyncMock()
            mock_page.url = f"https://example.com/{scenario['name']}"
            mock_page.evaluate.return_value = {
                'exists': True,
                'objectName': 'dataLayer',
                'latest': scenario['datalayer'],
                'events': [],
                'truncated': False
            }
            
            context = DLContext(env='test')
            
            # Monitor processing time
            import time
            start_time = time.time()
            result = await integration_service.capture_and_validate(mock_page, context)
            processing_time = time.time() - start_time
            
            results.append((scenario['name'], result, processing_time))
        
        # System should handle large data without excessive processing time
        for name, result, processing_time in results:
            assert result is not None
            assert processing_time < 30.0  # Should complete within 30 seconds
            
            # If capture succeeded, verify data integrity
            if result.snapshot.exists:
                assert result.snapshot.latest is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])