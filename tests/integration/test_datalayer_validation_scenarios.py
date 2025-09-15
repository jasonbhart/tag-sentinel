"""Integration tests for real-world validation scenarios."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from typing import Dict, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.audit.datalayer.models import DLContext, ValidationSeverity


@pytest.mark.integration
class TestRealWorldValidationScenarios:
    """Integration tests with real-world dataLayer validation scenarios."""
    
    @pytest.mark.asyncio
    async def test_google_analytics_4_validation(self, integration_service):
        """Test validation against Google Analytics 4 dataLayer structure."""
        ga4_schema = {
            "type": "object",
            "properties": {
                "event": {"type": "string"},
                "page_title": {"type": "string"},
                "page_location": {"type": "string", "format": "uri"},
                "page_referrer": {"type": "string", "format": "uri"},
                "language": {"type": "string", "pattern": "^[a-z]{2}(-[A-Z]{2})?$"},
                "user_id": {"type": "string"},
                "session_id": {"type": "string"},
                "custom_parameters": {
                    "type": "object",
                    "additionalProperties": {"type": ["string", "number", "boolean"]}
                },
                "ecommerce": {
                    "type": "object",
                    "properties": {
                        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
                        "value": {"type": "number", "minimum": 0},
                        "transaction_id": {"type": "string"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item_id": {"type": "string"},
                                    "item_name": {"type": "string"},
                                    "category": {"type": "string"},
                                    "quantity": {"type": "integer", "minimum": 1},
                                    "price": {"type": "number", "minimum": 0}
                                },
                                "required": ["item_id", "item_name"]
                            }
                        }
                    }
                }
            }
        }
        
        # Valid GA4 dataLayer scenarios
        valid_scenarios = [
            {
                "name": "page_view",
                "data": {
                    "event": "page_view",
                    "page_title": "Homepage",
                    "page_location": "https://example.com/",
                    "language": "en-US",
                    "user_id": "12345",
                    "custom_parameters": {
                        "content_group1": "homepage",
                        "custom_parameter": "test_value"
                    }
                }
            },
            {
                "name": "purchase_event",
                "data": {
                    "event": "purchase",
                    "page_title": "Order Confirmation", 
                    "page_location": "https://example.com/order-confirmation",
                    "user_id": "12345",
                    "ecommerce": {
                        "currency": "USD",
                        "value": 129.99,
                        "transaction_id": "TXN123456",
                        "items": [
                            {
                                "item_id": "SKU001",
                                "item_name": "Wireless Headphones",
                                "category": "Electronics",
                                "quantity": 1,
                                "price": 129.99
                            }
                        ]
                    }
                }
            }
        ]
        
        # Test valid scenarios
        for scenario in valid_scenarios:
            mock_page = AsyncMock()
            mock_page.url = "https://example.com/ga4-test"
            mock_page.evaluate.return_value = scenario["data"]
            
            context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://example.com/ga4-test"})
            result = await integration_service.capture_and_validate(mock_page, context, ga4_schema)
            
            # Should pass validation
            assert result.snapshot.exists
            assert len(result.issues) == 0, f"Validation failed for {scenario['name']}: {result.issues}"
        
        # Invalid GA4 scenarios
        invalid_scenarios = [
            {
                "name": "invalid_currency",
                "data": {
                    "event": "purchase",
                    "ecommerce": {
                        "currency": "US",  # Invalid - should be 3 characters
                        "value": 100,
                        "items": [{"item_id": "123", "item_name": "Test"}]
                    }
                }
            },
            {
                "name": "missing_item_name",
                "data": {
                    "event": "purchase", 
                    "ecommerce": {
                        "currency": "USD",
                        "items": [{"item_id": "123"}]  # Missing required item_name
                    }
                }
            },
            {
                "name": "invalid_language",
                "data": {
                    "event": "page_view",
                    "language": "english"  # Invalid format - should be like "en" or "en-US"
                }
            }
        ]
        
        # Test invalid scenarios
        for scenario in invalid_scenarios:
            mock_page = AsyncMock()
            mock_page.url = "https://example.com/ga4-invalid"
            mock_page.evaluate.return_value = scenario["data"]
            
            context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://example.com/ga4-invalid"})
            result = await integration_service.capture_and_validate(mock_page, context, ga4_schema)
            
            # Should have validation errors
            assert result.snapshot.exists
            assert len(result.issues) > 0, f"Expected validation errors for {scenario['name']}"
            
            # Should have error severity issues
            error_issues = [issue for issue in result.issues if issue.severity == ValidationSeverity.CRITICAL]
            assert len(error_issues) > 0
    
    @pytest.mark.asyncio
    async def test_adobe_analytics_validation(self, integration_service):
        """Test validation against Adobe Analytics dataLayer structure."""
        adobe_schema = {
            "type": "object", 
            "properties": {
                "digitalData": {
                    "type": "object",
                    "properties": {
                        "page": {
                            "type": "object",
                            "properties": {
                                "pageInfo": {
                                    "type": "object",
                                    "properties": {
                                        "pageName": {"type": "string"},
                                        "server": {"type": "string"},
                                        "hierarchy": {"type": "string"},
                                        "siteSection": {"type": "string"}
                                    },
                                    "required": ["pageName"]
                                },
                                "category": {
                                    "type": "object",
                                    "properties": {
                                        "primaryCategory": {"type": "string"},
                                        "subCategory1": {"type": "string"},
                                        "pageType": {"type": "string"}
                                    }
                                }
                            },
                            "required": ["pageInfo"]
                        },
                        "user": {
                            "type": "array",
                            "items": {
                                "type": "object", 
                                "properties": {
                                    "profile": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "profileInfo": {
                                                    "type": "object",
                                                    "properties": {
                                                        "profileID": {"type": "string"},
                                                        "userName": {"type": "string"},
                                                        "email": {"type": "string", "format": "email"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "product": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "productInfo": {
                                        "type": "object", 
                                        "properties": {
                                            "productID": {"type": "string"},
                                            "productName": {"type": "string"},
                                            "description": {"type": "string"},
                                            "price": {"type": "number", "minimum": 0}
                                        },
                                        "required": ["productID", "productName"]
                                    },
                                    "category": {
                                        "type": "object",
                                        "properties": {
                                            "primaryCategory": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "required": ["page"]
                }
            },
            "required": ["digitalData"]
        }
        
        # Valid Adobe Analytics dataLayer
        valid_adobe_data = {
            "digitalData": {
                "page": {
                    "pageInfo": {
                        "pageName": "Homepage",
                        "server": "www.example.com", 
                        "hierarchy": "home",
                        "siteSection": "main"
                    },
                    "category": {
                        "primaryCategory": "home",
                        "pageType": "homepage"
                    }
                },
                "user": [
                    {
                        "profile": [
                            {
                                "profileInfo": {
                                    "profileID": "user123",
                                    "userName": "john_doe",
                                    "email": "john@example.com"
                                }
                            }
                        ]
                    }
                ],
                "product": [
                    {
                        "productInfo": {
                            "productID": "PROD001",
                            "productName": "Wireless Mouse",
                            "description": "Ergonomic wireless mouse", 
                            "price": 29.99
                        },
                        "category": {
                            "primaryCategory": "electronics"
                        }
                    }
                ]
            }
        }
        
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/adobe-test"
        mock_page.evaluate.return_value = valid_adobe_data
        
        context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://example.com/adobe-test"})
        result = await integration_service.capture_and_validate(mock_page, context, adobe_schema)
        
        # Should pass validation
        assert result.snapshot.exists
        assert len(result.issues) == 0
        
        # Verify complex nested structure preserved
        digital_data = result.snapshot.latest["digitalData"]
        assert digital_data["page"]["pageInfo"]["pageName"] == "Homepage"
        assert len(digital_data["user"]) == 1
        assert len(digital_data["product"]) == 1
    
    @pytest.mark.asyncio
    async def test_custom_business_schema_validation(self, integration_service):
        """Test validation against custom business-specific schema."""
        # Custom schema for a retail business
        retail_schema = {
            "type": "object",
            "properties": {
                "site": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
                        "environment": {"type": "string", "enum": ["dev", "staging", "production"]}
                    },
                    "required": ["name", "version", "environment"]
                },
                "page": {
                    "type": "object", 
                    "properties": {
                        "type": {"type": "string", "enum": ["home", "category", "product", "cart", "checkout", "account"]},
                        "template": {"type": "string"},
                        "cms_id": {"type": "string"},
                        "last_modified": {"type": "string", "format": "date-time"}
                    },
                    "required": ["type"]
                },
                "customer": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": "^CUST[0-9]{6}$"},
                        "tier": {"type": "string", "enum": ["bronze", "silver", "gold", "platinum"]},
                        "lifetime_value": {"type": "number", "minimum": 0},
                        "acquisition_channel": {"type": "string"}
                    }
                },
                "session": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": "^sess_[a-zA-Z0-9]{16}$"},
                        "start_time": {"type": "string", "format": "date-time"},
                        "page_views": {"type": "integer", "minimum": 1},
                        "device_type": {"type": "string", "enum": ["desktop", "mobile", "tablet"]}
                    },
                    "required": ["id", "page_views", "device_type"]
                }
            },
            "required": ["site", "page", "session"]
        }
        
        # Valid business data
        valid_business_data = {
            "site": {
                "name": "RetailCorp Online Store",
                "version": "2.1.3", 
                "environment": "production"
            },
            "page": {
                "type": "product",
                "template": "product_detail_v2",
                "cms_id": "page_12345",
                "last_modified": "2024-01-15T09:30:00Z"
            },
            "customer": {
                "id": "CUST123456",
                "tier": "gold",
                "lifetime_value": 1250.75,
                "acquisition_channel": "organic_search"
            },
            "session": {
                "id": "sess_abc123def4567890",
                "start_time": "2024-01-15T10:00:00Z", 
                "page_views": 5,
                "device_type": "desktop"
            }
        }
        
        mock_page = AsyncMock()
        mock_page.url = "https://retailcorp.com/products/widget"
        mock_page.evaluate.return_value = valid_business_data
        
        context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://retailcorp.com/products/widget"})
        result = await integration_service.capture_and_validate(mock_page, context, retail_schema)
        
        # Should pass validation
        assert result.snapshot.exists
        assert len(result.issues) == 0
        
        # Test invalid business data scenarios
        invalid_scenarios = [
            {
                "name": "invalid_version_format",
                "data": {**valid_business_data, "site": {**valid_business_data["site"], "version": "v2.1"}},
                "expected_errors": ["pattern"]
            },
            {
                "name": "invalid_customer_id",
                "data": {**valid_business_data, "customer": {**valid_business_data["customer"], "id": "customer123"}},
                "expected_errors": ["pattern"]  
            },
            {
                "name": "invalid_enum_value",
                "data": {**valid_business_data, "page": {**valid_business_data["page"], "type": "blog"}},
                "expected_errors": ["enum"]
            },
            {
                "name": "missing_required_field", 
                "data": {k: v for k, v in valid_business_data.items() if k != "session"},
                "expected_errors": ["required"]
            }
        ]
        
        for scenario in invalid_scenarios:
            mock_page = AsyncMock()
            mock_page.url = f"https://retailcorp.com/invalid-{scenario['name']}"
            mock_page.evaluate.return_value = scenario["data"]
            
            context = DLContext(url=f"https://retailcorp.com/invalid-{scenario['name']}")
            result = await integration_service.capture_and_validate(mock_page, context, retail_schema)
            
            # Should have validation errors
            assert result.snapshot.exists
            assert len(result.issues) > 0, f"Expected validation errors for {scenario['name']}"
            
            # Check for expected error types
            error_messages = [issue.message.lower() for issue in result.issues]
            assert any(expected in ' '.join(error_messages) for expected in scenario["expected_errors"])
    
    @pytest.mark.asyncio
    async def test_schema_evolution_compatibility(self, integration_service):
        """Test backward compatibility with evolving schemas."""
        # Base schema v1
        schema_v1 = {
            "type": "object",
            "properties": {
                "page": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "type": {"type": "string"}
                    },
                    "required": ["title"]
                },
                "user": {
                    "type": "object", 
                    "properties": {
                        "id": {"type": "string"}
                    }
                }
            },
            "required": ["page"]
        }
        
        # Evolved schema v2 (backward compatible)
        schema_v2 = {
            "type": "object",
            "properties": {
                "page": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "type": {"type": "string"},
                        "version": {"type": "string"},  # New optional field
                        "meta": {  # New optional nested object
                            "type": "object",
                            "properties": {
                                "author": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "required": ["title"]  # Same requirement
                },
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "preferences": {  # New optional field
                            "type": "object",
                            "properties": {
                                "theme": {"type": "string"},
                                "notifications": {"type": "boolean"}
                            }
                        }
                    }
                },
                "analytics": {  # New optional top-level object
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "tracking_id": {"type": "string"}
                    }
                }
            },
            "required": ["page"]  # Same requirement
        }
        
        # Data that should work with both schemas (v1 compatible)
        v1_compatible_data = {
            "page": {
                "title": "Test Page",
                "type": "homepage"
            },
            "user": {
                "id": "user123"
            }
        }
        
        # Data with v2 features
        v2_enhanced_data = {
            "page": {
                "title": "Enhanced Test Page",
                "type": "homepage", 
                "version": "2.1",
                "meta": {
                    "author": "Content Team",
                    "tags": ["homepage", "featured", "responsive"]
                }
            },
            "user": {
                "id": "user123",
                "preferences": {
                    "theme": "dark",
                    "notifications": True
                }
            },
            "analytics": {
                "session_id": "sess_abc123",
                "tracking_id": "GA_123456"
            }
        }
        
        # Test v1 data against both schemas
        for schema_version, schema in [("v1", schema_v1), ("v2", schema_v2)]:
            mock_page = AsyncMock()
            mock_page.url = f"https://example.com/compat-test-{schema_version}"
            mock_page.evaluate.return_value = v1_compatible_data
            
            context = DLContext(url=f"https://example.com/compat-test-{schema_version}")
            result = await integration_service.capture_and_validate(mock_page, context, schema)
            
            # v1 data should pass validation against both schemas
            assert result.snapshot.exists
            assert len(result.issues) == 0, f"v1 data failed validation against schema {schema_version}"
        
        # Test v2 data against v2 schema only
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/v2-enhanced"
        mock_page.evaluate.return_value = v2_enhanced_data
        
        context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://example.com/v2-enhanced"})
        result = await integration_service.capture_and_validate(mock_page, context, schema_v2)
        
        # v2 enhanced data should pass v2 schema validation
        assert result.snapshot.exists
        assert len(result.issues) == 0
        
        # Verify enhanced features are captured
        latest = result.snapshot.latest
        assert latest["page"]["version"] == "2.1"
        assert len(latest["page"]["meta"]["tags"]) == 3
        assert latest["user"]["preferences"]["theme"] == "dark"
        assert "analytics" in latest


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])