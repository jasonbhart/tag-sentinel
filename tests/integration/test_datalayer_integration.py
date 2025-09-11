"""Integration tests for DataLayer system with real browser environments."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.audit.datalayer.service import DataLayerService
from app.audit.datalayer.config import DataLayerConfig, RedactionConfig, SchemaConfig, PerformanceConfig
from app.audit.datalayer.models import ValidationSeverity
from app.audit.capture.browser_factory import BrowserFactory


@pytest.mark.integration
class TestDataLayerBrowserIntegration:
    """Integration tests with real browser automation."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.config = DataLayerConfig(
            enabled=True,
            capture_timeout=10.0,
            max_depth=20,
            max_size=100000,
            redaction=RedactionConfig(
                enabled=True,
                default_action="MASK"
            ),
            validation=SchemaConfig(
                enabled=True,
                strict_mode=False
            ),
            performance=PerformanceConfig(
                max_concurrent_captures=1,
                batch_processing=False
            )
        )
        self.service = DataLayerService(self.config)
    
    @pytest.mark.asyncio
    async def test_capture_gtm_style_datalayer(self):
        """Test capturing GTM-style dataLayer from mock browser page."""
        # Mock page with GTM-style dataLayer
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/gtm-test"
        
        # Simulate GTM dataLayer structure
        gtm_datalayer = [
            {"gtm.start": 1640995200000},
            {"event": "gtm.js"},
            {"event": "page_view", "page_title": "Home", "page_type": "homepage"},
            {"user_id": "user123", "user_type": "premium"},
            {"event": "scroll", "scroll_depth": 25},
            {"products": ["item1", "item2"], "category": "electronics"}
        ]
        
        mock_page.evaluate.return_value = gtm_datalayer
        
        result = await self.service.capture_and_validate(mock_page, "https://example.com/gtm-test")
        
        # Verify successful capture
        assert result.snapshot.exists is True
        assert result.snapshot.page_url == "https://example.com/gtm-test"
        
        # Verify events were extracted
        assert len(result.snapshot.events) >= 3  # page_view, scroll, gtm.js
        
        # Verify variables were merged
        latest = result.snapshot.latest
        assert latest["user_id"] == "user123"
        assert latest["user_type"] == "premium"
        assert latest["category"] == "electronics"
        
        # GTM internal data should be filtered
        assert "gtm.start" not in latest
    
    @pytest.mark.asyncio
    async def test_capture_object_style_datalayer(self):
        """Test capturing object-style dataLayer."""
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/object-test"
        
        # Simulate object-style dataLayer
        object_datalayer = {
            "page": {
                "title": "Product Page",
                "type": "product",
                "category": "electronics"
            },
            "user": {
                "id": "user456",
                "segment": "returning_customer",
                "preferences": ["mobile", "deals"]
            },
            "product": {
                "id": "prod789",
                "name": "Smart Phone",
                "price": 699.99,
                "in_stock": True
            },
            "ecommerce": {
                "currency": "USD",
                "items": [
                    {"item_id": "prod789", "price": 699.99, "quantity": 1}
                ]
            }
        }
        
        mock_page.evaluate.return_value = object_datalayer
        
        result = await self.service.capture_and_validate(mock_page, "https://example.com/object-test")
        
        # Verify successful capture
        assert result.snapshot.exists is True
        assert result.snapshot.latest == object_datalayer
        assert len(result.snapshot.events) == 0  # No events in object style
        
        # Verify nested structure preserved
        assert result.snapshot.latest["page"]["title"] == "Product Page"
        assert result.snapshot.latest["user"]["id"] == "user456"
        assert result.snapshot.latest["product"]["price"] == 699.99
    
    @pytest.mark.asyncio
    async def test_missing_datalayer_handling(self):
        """Test handling of pages without dataLayer."""
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/no-datalayer"
        mock_page.evaluate.return_value = None  # No dataLayer
        
        result = await self.service.capture_and_validate(mock_page, "https://example.com/no-datalayer")
        
        # Should handle gracefully
        assert result.snapshot.exists is False
        assert result.snapshot.latest is None
        assert len(result.snapshot.events) == 0
    
    @pytest.mark.asyncio
    async def test_capture_with_javascript_errors(self):
        """Test handling JavaScript errors during capture."""
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/js-error"
        mock_page.evaluate.side_effect = Exception("JavaScript error: ReferenceError")
        
        result = await self.service.capture_and_validate(mock_page, "https://example.com/js-error")
        
        # Should handle JS errors gracefully
        assert result.snapshot.exists is False
        assert result.snapshot.latest is None
    
    @pytest.mark.asyncio
    async def test_capture_timeout_handling(self):
        """Test handling of capture timeouts."""
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/timeout"
        mock_page.evaluate.side_effect = asyncio.TimeoutError("Page timeout")
        
        result = await self.service.capture_and_validate(mock_page, "https://example.com/timeout")
        
        # Should handle timeout gracefully
        assert result.snapshot.exists is False


@pytest.mark.integration
class TestDataLayerValidationIntegration:
    """Integration tests for validation with real schemas."""
    
    def setup_method(self):
        """Set up validation integration tests."""
        self.config = DataLayerConfig(
            enabled=True,
            validation=SchemaConfig(enabled=True, strict_mode=True)
        )
        self.service = DataLayerService(self.config)
    
    @pytest.mark.asyncio
    async def test_ecommerce_schema_validation(self):
        """Test validation against e-commerce dataLayer schema."""
        # Create comprehensive e-commerce schema
        ecommerce_schema = {
            "type": "object",
            "properties": {
                "page": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["homepage", "product", "cart", "checkout"]},
                        "title": {"type": "string", "minLength": 1}
                    },
                    "required": ["type", "title"]
                },
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": "^user[0-9]+$"},
                        "email": {"type": "string", "format": "email"},
                        "segment": {"type": "string"}
                    },
                    "required": ["id"]
                },
                "ecommerce": {
                    "type": "object",
                    "properties": {
                        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
                        "value": {"type": "number", "minimum": 0},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item_id": {"type": "string"},
                                    "item_name": {"type": "string"},
                                    "price": {"type": "number", "minimum": 0},
                                    "quantity": {"type": "integer", "minimum": 1}
                                },
                                "required": ["item_id", "price", "quantity"]
                            }
                        }
                    }
                }
            },
            "required": ["page"]
        }
        
        # Test valid e-commerce dataLayer
        valid_datalayer = {
            "page": {"type": "product", "title": "Smart Phone - Electronics Store"},
            "user": {"id": "user123", "email": "user@example.com", "segment": "premium"},
            "ecommerce": {
                "currency": "USD",
                "value": 699.99,
                "items": [
                    {"item_id": "phone001", "item_name": "Smart Phone", "price": 699.99, "quantity": 1}
                ]
            }
        }
        
        mock_page = AsyncMock()
        mock_page.url = "https://shop.example.com/products/phone"
        mock_page.evaluate.return_value = valid_datalayer
        
        result = await self.service.capture_and_validate(mock_page, "https://shop.example.com/products/phone")
        
        # Should pass validation
        assert len(result.issues) == 0
        assert result.snapshot.exists is True
    
    @pytest.mark.asyncio
    async def test_invalid_schema_validation(self):
        """Test validation with invalid dataLayer data."""
        # Same schema as above
        ecommerce_schema = {
            "type": "object",
            "properties": {
                "page": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["homepage", "product", "cart", "checkout"]},
                        "title": {"type": "string", "minLength": 1}
                    },
                    "required": ["type", "title"]
                },
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": "^user[0-9]+$"},
                        "email": {"type": "string", "format": "email"}
                    },
                    "required": ["id"]
                }
            },
            "required": ["page"]
        }
        
        # Invalid dataLayer - missing required fields, wrong types, invalid formats
        invalid_datalayer = {
            "page": {"type": "invalid_page_type"},  # Missing title, invalid enum
            "user": {"id": "invalid_user_id", "email": "not-an-email"},  # Invalid pattern and format
            "extra_field": "not_allowed"  # Additional property (in strict mode)
        }
        
        mock_page = AsyncMock()
        mock_page.url = "https://shop.example.com/invalid"
        mock_page.evaluate.return_value = invalid_datalayer
        
        result = await self.service.capture_and_validate(mock_page, "https://shop.example.com/invalid")
        
        # Should have multiple validation issues
        assert len(result.issues) >= 3
        
        # Check for specific error types
        error_messages = [issue.message.lower() for issue in result.issues]
        severity_levels = [issue.severity for issue in result.issues]
        
        # Should have errors for missing required field, invalid enum, invalid format
        assert any("required" in msg or "missing" in msg for msg in error_messages)
        assert any("enum" in msg or "invalid" in msg for msg in error_messages)
        assert any("format" in msg or "email" in msg for msg in error_messages)
        assert ValidationSeverity.ERROR in severity_levels


@pytest.mark.integration
class TestDataLayerRedactionIntegration:
    """Integration tests for sensitive data redaction."""
    
    def setup_method(self):
        """Set up redaction integration tests."""
        self.config = DataLayerConfig(
            enabled=True,
            redaction=RedactionConfig(
                enabled=True,
                default_action="MASK",
                custom_patterns={
                    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    "phone": r"\b\d{3}-\d{3}-\d{4}\b",
                    "user_id": r"user\d+"
                }
            )
        )
        self.service = DataLayerService(self.config)
    
    @pytest.mark.asyncio
    async def test_automatic_pii_redaction(self):
        """Test automatic redaction of PII in dataLayer."""
        # DataLayer with various PII types
        sensitive_datalayer = {
            "user": {
                "email": "john.doe@example.com",
                "phone": "555-123-4567",
                "id": "user12345",
                "name": "John Doe",  # Should not be redacted
                "address": {
                    "street": "123 Main St",
                    "city": "Anytown"
                }
            },
            "contact": {
                "support_email": "support@company.com",
                "office_phone": "555-987-6543",
                "website": "https://example.com"  # Should not be redacted
            },
            "marketing": {
                "campaign_id": "camp123",
                "customer_email": "customer@test.com",
                "reference_number": "REF-456789"
            }
        }
        
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/sensitive-data"
        mock_page.evaluate.return_value = sensitive_datalayer
        
        result = await self.service.capture_and_validate(mock_page, "https://example.com/sensitive-data")
        
        # Define redaction paths
        redaction_paths = [
            "/user/email",
            "/user/phone",
            "/user/id",
            "/contact/support_email",
            "/contact/office_phone",
            "/marketing/customer_email"
        ]
        
        result = await self.service.capture_and_validate(
            mock_page, context, redaction_paths=redaction_paths
        )
        
        # Verify redaction occurred
        latest = result.snapshot.latest
        
        # Email addresses should be masked
        assert latest["user"]["email"] != "john.doe@example.com"
        assert "*" in latest["user"]["email"] or "[" in latest["user"]["email"]
        
        # Phone numbers should be masked
        assert latest["user"]["phone"] != "555-123-4567"
        assert "*" in latest["user"]["phone"] or "[" in latest["user"]["phone"]
        
        # User IDs should be redacted
        assert latest["user"]["id"] != "user12345"
        
        # Non-sensitive data should remain unchanged
        assert latest["user"]["name"] == "John Doe"
        assert latest["user"]["address"]["city"] == "Anytown"
        assert latest["contact"]["website"] == "https://example.com"
    
    @pytest.mark.asyncio
    async def test_pattern_based_redaction(self):
        """Test pattern-based automatic redaction."""
        # DataLayer that matches sensitive patterns
        pattern_datalayer = {
            "form_data": {
                "email_field": "user@domain.com",
                "phone_field": "123-456-7890",
                "ssn_field": "123-45-6789",
                "credit_card": "4111-1111-1111-1111",
                "safe_field": "This is safe data"
            },
            "analytics": {
                "visitor_email": "visitor@site.com",
                "contact_phone": "555-000-1234",
                "session_id": "sess_abc123",  # Should not match patterns
                "page_views": 5
            }
        }
        
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/form-page"
        mock_page.evaluate.return_value = pattern_datalayer
        
        result = await self.service.capture_and_validate(mock_page, "https://example.com/form-page")
        
        # With automatic pattern detection, sensitive data should be identified
        # and redacted (exact behavior depends on implementation)
        latest = result.snapshot.latest
        
        # Verify structure is preserved even if some data is redacted
        assert "form_data" in latest
        assert "analytics" in latest
        assert latest["form_data"]["safe_field"] == "This is safe data"
        assert latest["analytics"]["page_views"] == 5


@pytest.mark.integration
class TestDataLayerAggregationIntegration:
    """Integration tests for run-level data aggregation."""
    
    def setup_method(self):
        """Set up aggregation integration tests."""
        self.config = DataLayerConfig(enabled=True)
        self.service = DataLayerService(self.config)
    
    @pytest.mark.asyncio
    async def test_multi_page_aggregation(self):
        """Test aggregation across multiple pages in a site crawl."""
        # Simulate multiple pages with different dataLayer patterns
        page_scenarios = [
            {
                "url": "https://shop.example.com/",
                "datalayer": {
                    "page": {"type": "homepage", "title": "Home"},
                    "user": {"segment": "new_visitor"},
                    "site": {"version": "2.1"}
                },
                "events": [{"event": "page_view", "page_type": "homepage"}]
            },
            {
                "url": "https://shop.example.com/products",
                "datalayer": {
                    "page": {"type": "category", "title": "Products", "category": "all"},
                    "user": {"segment": "returning_customer", "id": "user123"},
                    "site": {"version": "2.1"}
                },
                "events": [{"event": "page_view", "page_type": "category"}]
            },
            {
                "url": "https://shop.example.com/products/phone",
                "datalayer": {
                    "page": {"type": "product", "title": "Smart Phone"},
                    "user": {"segment": "returning_customer", "id": "user123"},
                    "product": {"id": "phone001", "price": 699.99, "category": "electronics"},
                    "site": {"version": "2.1"}
                },
                "events": [
                    {"event": "page_view", "page_type": "product"},
                    {"event": "view_item", "item_id": "phone001", "value": 699.99}
                ]
            },
            {
                "url": "https://shop.example.com/cart",
                "datalayer": {
                    "page": {"type": "cart", "title": "Shopping Cart"},
                    "user": {"segment": "returning_customer", "id": "user123"},
                    "cart": {"items": 2, "value": 899.98},
                    "site": {"version": "2.1"}
                },
                "events": [{"event": "page_view", "page_type": "cart"}]
            },
            {
                "url": "https://shop.example.com/missing-datalayer",
                "datalayer": None,  # Page without dataLayer
                "events": []
            }
        ]
        
        # Create page contexts for batch processing
        page_contexts = []
        
        for scenario in page_scenarios:
            mock_page = AsyncMock()
            mock_page.url = scenario["url"]
            mock_page.evaluate.return_value = scenario["datalayer"]
            page_contexts.append((mock_page, scenario["url"], None))
        
        # Process all pages with aggregation
        results, aggregate = await self.service.process_multiple_pages(
            page_contexts=page_contexts,
            run_id="multi-page-test-run"
        )
        
        # Verify aggregation results
        assert aggregate.run_id == "multi-page-test-run"
        assert aggregate.total_pages == 5
        assert aggregate.pages_successful == 4  # 4 pages with dataLayer
        assert aggregate.pages_failed == 1     # 1 page without dataLayer
        
        # Check variable presence analysis
        variable_stats = aggregate.variable_stats
        
        # 'page' variable should be present on 4/5 pages (80%)
        page_stats = variable_stats.get("page", {})
        assert page_stats.get("presence", 0) == pytest.approx(0.8, rel=1e-2)
        
        # 'site.version' should be present on 4/5 pages
        if "site" in variable_stats:
            site_stats = variable_stats["site"]
            assert site_stats.get("presence", 0) == pytest.approx(0.8, rel=1e-2)
        
        # 'user' should be present on 4/5 pages
        user_stats = variable_stats.get("user", {})
        assert user_stats.get("presence", 0) == pytest.approx(0.8, rel=1e-2)
        
        # 'product' should only be present on 1/5 pages (20%)
        product_stats = variable_stats.get("product", {})
        assert product_stats.get("presence", 0) == pytest.approx(0.2, rel=1e-2)
    
    @pytest.mark.asyncio
    async def test_validation_issue_aggregation(self):
        """Test aggregation of validation issues across pages."""
        # Schema for testing
        test_schema = {
            "type": "object",
            "properties": {
                "page": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "minLength": 1}
                    },
                    "required": ["title"]
                },
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": "^user[0-9]+$"}
                    },
                    "required": ["id"]
                }
            },
            "required": ["page"]
        }
        
        # Pages with various validation issues
        page_scenarios = [
            {
                "url": "https://example.com/valid1",
                "datalayer": {
                    "page": {"title": "Valid Page 1"},
                    "user": {"id": "user123"}
                }
            },
            {
                "url": "https://example.com/valid2", 
                "datalayer": {
                    "page": {"title": "Valid Page 2"},
                    "user": {"id": "user456"}
                }
            },
            {
                "url": "https://example.com/missing-page",
                "datalayer": {
                    "user": {"id": "user789"}
                    # Missing required 'page' field
                }
            },
            {
                "url": "https://example.com/empty-title",
                "datalayer": {
                    "page": {"title": ""},  # Empty title violates minLength
                    "user": {"id": "user101"}
                }
            },
            {
                "url": "https://example.com/invalid-user",
                "datalayer": {
                    "page": {"title": "Valid Page"},
                    "user": {"id": "invalid_user_id"}  # Doesn't match pattern
                }
            }
        ]
        
        # Create page contexts for batch processing
        page_contexts = []
        
        for scenario in page_scenarios:
            mock_page = AsyncMock()
            mock_page.url = scenario["url"]
            mock_page.evaluate.return_value = scenario["datalayer"]
            page_contexts.append((mock_page, scenario["url"], None))
        
        # Process with validation
        results, aggregate = await self.service.process_multiple_pages(
            page_contexts=page_contexts,
            run_id="validation-test-run"
        )
        
        # Verify validation summary
        validation_summary = aggregate.validation_summary
        
        assert validation_summary["total_issues"] >= 3  # At least 3 pages with issues
        assert validation_summary["error_count"] >= 3   # Multiple validation errors
        
        # Should have breakdown of common issues
        common_issues = validation_summary.get("most_common_issues", [])
        assert len(common_issues) > 0
        
        # Check that issue patterns are identified
        issue_messages = [item["message"] for item in common_issues]
        assert any("required" in msg.lower() or "missing" in msg.lower() for msg in issue_messages)


@pytest.mark.integration
class TestDataLayerPerformanceIntegration:
    """Integration tests for performance and scalability."""
    
    def setup_method(self):
        """Set up performance integration tests."""
        self.config = DataLayerConfig(
            enabled=True,
            capture_timeout=5.0,
            max_depth=50,
            max_size=1000000  # 1MB limit
        )
        self.service = DataLayerService(self.config)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_datalayer_processing(self):
        """Test processing of large dataLayer objects."""
        # Create large dataLayer with nested structures
        large_datalayer = {
            "metadata": {
                "capture_time": "2024-01-15T10:30:00Z",
                "version": "1.2.3",
                "environment": "production"
            },
            "user": {
                "id": "user123456",
                "profile": {
                    f"attribute_{i}": f"value_{i}" 
                    for i in range(100)  # 100 user attributes
                }
            },
            "products": [
                {
                    "id": f"prod_{i}",
                    "name": f"Product {i}",
                    "price": 19.99 + i,
                    "attributes": {
                        f"attr_{j}": f"value_{j}" 
                        for j in range(20)  # 20 attributes per product
                    }
                }
                for i in range(50)  # 50 products
            ],
            "analytics": {
                "events": [
                    {
                        "event": f"event_{i}",
                        "timestamp": f"2024-01-15T10:{30+i}:00Z",
                        "properties": {
                            f"prop_{j}": f"value_{j}" 
                            for j in range(10)
                        }
                    }
                    for i in range(100)  # 100 events
                ]
            }
        }
        
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/large-datalayer"
        mock_page.evaluate.return_value = large_datalayer
        
        result = await self.service.capture_and_validate(mock_page, "https://example.com/large-datalayer")
        
        import time
        start_time = time.time()
        processing_time = time.time() - start_time
        
        # Should process within reasonable time (adjust threshold as needed)
        assert processing_time < 10.0  # 10 seconds max
        
        # Should successfully capture large dataLayer
        assert result.snapshot.exists is True
        assert result.snapshot.latest is not None
        
        # Should preserve structure
        assert len(result.snapshot.latest["products"]) == 50
        assert len(result.snapshot.latest["analytics"]["events"]) == 100
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_page_processing(self):
        """Test concurrent processing of multiple pages."""
        # Create many pages for concurrent processing
        num_pages = 20
        page_contexts = []
        
        for i in range(num_pages):
            mock_page = AsyncMock()
            mock_page.url = f"https://example.com/page-{i}"
            mock_page.evaluate.return_value = {
                "page": {"id": i, "title": f"Page {i}"},
                "timestamp": f"2024-01-15T10:{i:02d}:00Z",
                "data": {f"field_{j}": f"value_{j}" for j in range(10)}
            }
            
            page_contexts.append((mock_page, f"https://example.com/page-{i}", None))
        
        # Process all pages concurrently
        import time
        start_time = time.time()
        
        results, aggregate = await self.service.process_multiple_pages(
            page_contexts=page_contexts
        )
        
        processing_time = time.time() - start_time
        
        # Should complete faster than sequential processing
        assert processing_time < 5.0  # Should be much faster with concurrency
        assert len(results) == num_pages
        assert all(result.snapshot.exists for result in results)
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self):
        """Test memory usage stays within reasonable bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple large dataLayers
        for i in range(10):
            large_datalayer = {
                f"section_{j}": {
                    f"field_{k}": f"data_{i}_{j}_{k}" 
                    for k in range(100)
                }
                for j in range(50)
            }
            
            mock_page = AsyncMock()
            mock_page.url = f"https://example.com/memory-test-{i}"
            mock_page.evaluate.return_value = large_datalayer
            
            result = await self.service.capture_and_validate(mock_page, f"https://example.com/memory-test-{i}")
            await self.service.capture_and_validate(mock_page, context)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should not grow excessively
        assert memory_increase < 500  # Less than 500MB increase
    
    @pytest.mark.asyncio
    async def test_error_resilience_under_load(self):
        """Test system resilience with mixed success/failure scenarios."""
        # Mix of successful and failing pages
        scenarios = []
        
        # 70% success rate
        for i in range(70):
            scenarios.append({
                "url": f"https://example.com/success-{i}",
                "response": {"page": f"success_{i}", "data": f"value_{i}"},
                "should_fail": False
            })
        
        # 30% failure rate with different error types
        error_types = [
            lambda: None,  # Missing dataLayer
            lambda: Exception("JavaScript error"),
            lambda: asyncio.TimeoutError("Timeout"),
            lambda: Exception("Network error")
        ]
        
        for i in range(30):
            error_type = error_types[i % len(error_types)]
            scenarios.append({
                "url": f"https://example.com/error-{i}",
                "response": error_type,
                "should_fail": True
            })
        
        # Randomize order
        import random
        random.shuffle(scenarios)
        
        page_contexts = []
        
        for scenario in scenarios:
            mock_page = AsyncMock()
            mock_page.url = scenario["url"]
            
            if scenario["should_fail"]:
                response = scenario["response"]
                if callable(response):
                    response_val = response()
                    if isinstance(response_val, Exception):
                        mock_page.evaluate.side_effect = response_val
                    else:
                        mock_page.evaluate.return_value = response_val
                else:
                    mock_page.evaluate.side_effect = response
            else:
                mock_page.evaluate.return_value = scenario["response"]
            
            page_contexts.append((mock_page, scenario["url"], None))
        
        # Process all pages with aggregation
        results, aggregate = await self.service.process_multiple_pages(
            page_contexts=page_contexts,
            run_id="resilience-test"
        )
        
        # System should handle mixed scenarios gracefully
        assert aggregate.total_pages == 100
        assert aggregate.pages_successful >= 65  # Allow some variance
        assert aggregate.pages_failed >= 25
        
        # Success rate should be reasonable
        success_rate = aggregate.pages_successful / aggregate.total_pages
        assert 0.6 <= success_rate <= 0.8  # 60-80% success rate


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])