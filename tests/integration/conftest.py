"""Shared fixtures for DataLayer integration tests."""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.audit.datalayer.config import DataLayerConfig, RedactionConfig, SchemaConfig
from app.audit.datalayer.service import DataLayerService
from app.audit.datalayer.models import DLContext


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def integration_config():
    """DataLayer configuration for integration tests."""
    from app.audit.datalayer.config import CaptureConfig, PerformanceConfig
    
    return DataLayerConfig(
        environment="test",
        capture=CaptureConfig(
            enabled=True,
            max_depth=8,
            max_size_bytes=500000,  # 500KB for integration tests
            execution_timeout_ms=10000,
            event_detection_patterns=["event", "gtm", "custom_event"],
            batch_processing=True  # Enable batch processing to fix validation
        ),
        redaction=RedactionConfig(
            enabled=False,  # Disable redaction for integration tests
            default_method="hash"
        ),
        validation=SchemaConfig(
            enabled=True,
            strict_mode=False
        ),
        performance=PerformanceConfig(
            max_concurrent_captures=1,  # Single capture to avoid validation warning
            enable_caching=False
        )
    )


@pytest.fixture
def integration_service(integration_config):
    """DataLayer service configured for integration testing."""
    return DataLayerService(integration_config)


@pytest.fixture
def mock_gtm_page():
    """Mock page with GTM-style dataLayer."""
    gtm_data = [
        {"gtm.start": 1640995200000},
        {"event": "gtm.js", "gtm.uniqueEventId": 1},
        {"event": "page_view", "page_title": "Home Page", "page_type": "homepage"},
        {"user_id": "user123", "user_type": "registered"},
        {"event": "scroll", "scroll_depth": 25},
        {"products": ["item1", "item2"], "category": "electronics"},
        {"event": "click", "element": "header_logo", "position": "top"}
    ]
    
    mock_page = AsyncMock()
    mock_page.url = "https://example.com/gtm-test"
    mock_page.evaluate.return_value = {
        'exists': True,
        'objectName': 'dataLayer',
        'latest': {},  # GTM style is typically array-based, latest state would be empty
        'events': gtm_data,  # GTM data is the events array
        'truncated': False
    }

    return mock_page


@pytest.fixture
def mock_object_page():
    """Mock page with object-style dataLayer."""
    object_data = {
        "page": {
            "title": "Product Detail Page",
            "type": "product",
            "category": "electronics",
            "language": "en-US"
        },
        "user": {
            "id": "user456",
            "segment": "premium_customer",
            "logged_in": True,
            "preferences": {
                "newsletter": True,
                "notifications": False
            }
        },
        "product": {
            "id": "prod789",
            "name": "Wireless Headphones",
            "brand": "TechBrand",
            "price": 199.99,
            "currency": "USD",
            "in_stock": True,
            "variants": [
                {"color": "black", "size": "medium"},
                {"color": "white", "size": "medium"}
            ]
        },
        "ecommerce": {
            "currency": "USD",
            "value": 199.99,
            "items": [
                {
                    "item_id": "prod789",
                    "item_name": "Wireless Headphones",
                    "category": "electronics",
                    "brand": "TechBrand",
                    "price": 199.99,
                    "quantity": 1
                }
            ]
        }
    }
    
    mock_page = AsyncMock()
    mock_page.url = "https://example.com/object-test"
    mock_page.evaluate.return_value = {
        'exists': True,
        'objectName': 'dataLayer',
        'latest': object_data,  # Object style stores current state in latest
        'events': [],  # Object style typically doesn't have events array
        'truncated': False
    }

    return mock_page


@pytest.fixture
def mock_empty_page():
    """Mock page without dataLayer."""
    mock_page = AsyncMock()
    mock_page.url = "https://example.com/no-datalayer"
    mock_page.evaluate.return_value = {
        'exists': False,
        'objectName': 'dataLayer',
        'latest': None,
        'events': None,
        'truncated': False
    }

    return mock_page


@pytest.fixture
def mock_error_page():
    """Mock page that throws JavaScript error."""
    mock_page = AsyncMock()
    mock_page.url = "https://example.com/js-error"
    mock_page.evaluate.side_effect = Exception("ReferenceError: dataLayer is not defined")
    
    return mock_page


@pytest.fixture
def sensitive_data_page():
    """Mock page with sensitive data for redaction testing."""
    sensitive_data = {
        "user_info": {
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
            "ssn": "123-45-6789",
            "name": "John Doe",
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
                "state": "CA",
                "zip": "12345"
            }
        },
        "payment": {
            "credit_card": "4111-1111-1111-1111",
            "billing_email": "billing@example.com",
            "billing_phone": "555-987-6543"
        },
        "contact": {
            "support_email": "support@company.com",
            "sales_phone": "555-000-1234"
        },
        "safe_data": {
            "product_id": "prod123",
            "session_id": "sess_abc123",
            "page_views": 5,
            "referrer": "https://google.com"
        }
    }
    
    mock_page = AsyncMock()
    mock_page.url = "https://example.com/sensitive-data"
    mock_page.evaluate.return_value = {
        'exists': True,
        'objectName': 'dataLayer',
        'latest': sensitive_data,  # Sensitive data as latest state
        'events': [],
        'truncated': False
    }

    return mock_page


@pytest.fixture
def ecommerce_schema():
    """Comprehensive e-commerce JSON schema for validation testing."""
    return {
        "type": "object",
        "properties": {
            "page": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "type": {
                        "type": "string", 
                        "enum": ["homepage", "category", "product", "cart", "checkout", "account"]
                    },
                    "category": {"type": "string"},
                    "language": {"type": "string", "pattern": "^[a-z]{2}-[A-Z]{2}$"}
                },
                "required": ["title", "type"]
            },
            "user": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": "^user[0-9]+$"},
                    "segment": {"type": "string"},
                    "logged_in": {"type": "boolean"},
                    "preferences": {
                        "type": "object",
                        "properties": {
                            "newsletter": {"type": "boolean"},
                            "notifications": {"type": "boolean"}
                        }
                    }
                }
            },
            "product": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string", "minLength": 1},
                    "brand": {"type": "string"},
                    "price": {"type": "number", "minimum": 0},
                    "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
                    "in_stock": {"type": "boolean"},
                    "variants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "color": {"type": "string"},
                                "size": {"type": "string"}
                            }
                        }
                    }
                }
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
                                "category": {"type": "string"},
                                "brand": {"type": "string"},
                                "price": {"type": "number", "minimum": 0},
                                "quantity": {"type": "integer", "minimum": 1}
                            },
                            "required": ["item_id", "item_name", "price", "quantity"]
                        }
                    }
                },
                "required": ["currency", "items"]
            }
        },
        "required": ["page"]
    }


@pytest.fixture
def multi_page_scenarios():
    """Multiple page scenarios for aggregation testing."""
    return [
        {
            "url": "https://shop.example.com/",
            "datalayer": {
                "page": {"title": "Welcome to Our Store", "type": "homepage"},
                "user": {"segment": "new_visitor"},
                "site": {"version": "2.1.0", "environment": "production"}
            },
            "events": [{"event": "page_view", "page_type": "homepage"}]
        },
        {
            "url": "https://shop.example.com/electronics",
            "datalayer": {
                "page": {"title": "Electronics", "type": "category", "category": "electronics"},
                "user": {"segment": "returning_customer", "id": "user123"},
                "site": {"version": "2.1.0", "environment": "production"},
                "category": {"items_count": 45, "sort": "popularity"}
            },
            "events": [
                {"event": "page_view", "page_type": "category"},
                {"event": "view_item_list", "category": "electronics"}
            ]
        },
        {
            "url": "https://shop.example.com/electronics/smartphones/iphone",
            "datalayer": {
                "page": {"title": "iPhone 15 Pro", "type": "product"},
                "user": {"segment": "returning_customer", "id": "user123"},
                "product": {
                    "id": "iphone15pro",
                    "name": "iPhone 15 Pro", 
                    "price": 999.99,
                    "category": "smartphones",
                    "brand": "Apple"
                },
                "site": {"version": "2.1.0", "environment": "production"}
            },
            "events": [
                {"event": "page_view", "page_type": "product"},
                {"event": "view_item", "item_id": "iphone15pro", "value": 999.99}
            ]
        },
        {
            "url": "https://shop.example.com/cart",
            "datalayer": {
                "page": {"title": "Shopping Cart", "type": "cart"},
                "user": {"segment": "returning_customer", "id": "user123"},
                "cart": {
                    "items": [
                        {"id": "iphone15pro", "quantity": 1, "price": 999.99}
                    ],
                    "total_items": 1,
                    "total_value": 999.99
                },
                "site": {"version": "2.1.0", "environment": "production"}
            },
            "events": [{"event": "page_view", "page_type": "cart"}]
        },
        {
            "url": "https://shop.example.com/account/profile",
            "datalayer": {
                "page": {"title": "My Profile", "type": "account"},
                "user": {
                    "segment": "returning_customer", 
                    "id": "user123",
                    "logged_in": True,
                    "account_type": "premium"
                },
                "site": {"version": "2.1.0", "environment": "production"}
            },
            "events": [{"event": "page_view", "page_type": "account"}]
        }
    ]


@pytest.fixture
def temp_schema_files(ecommerce_schema):
    """Temporary schema files for testing."""
    files = {}
    
    # Main e-commerce schema
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ecommerce_schema, f)
        files['ecommerce'] = Path(f.name)
    
    # Simple page schema
    simple_schema = {
        "type": "object",
        "properties": {
            "page": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1}
                },
                "required": ["title"]
            }
        },
        "required": ["page"]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(simple_schema, f)
        files['simple'] = Path(f.name)
    
    # Invalid schema (for error testing)
    invalid_schema = {
        "type": "invalid_type",
        "properties": "not_an_object"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(invalid_schema, f)
        files['invalid'] = Path(f.name)
    
    yield files
    
    # Cleanup
    for file_path in files.values():
        if file_path.exists():
            file_path.unlink()


@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing."""
    return {
        "metadata": {
            "capture_timestamp": "2024-01-15T10:30:00Z",
            "version": "3.2.1",
            "environment": "production",
            "feature_flags": {f"flag_{i}": i % 2 == 0 for i in range(50)}
        },
        "user": {
            "id": "user987654321",
            "profile": {
                f"preference_{i}": f"value_{i}" 
                for i in range(200)  # 200 user preferences
            },
            "history": {
                "purchases": [
                    {
                        "id": f"order_{i}",
                        "date": f"2024-01-{i%30+1:02d}T10:00:00Z",
                        "total": 29.99 + i,
                        "items": [
                            {
                                "id": f"item_{i}_{j}",
                                "name": f"Product {i}-{j}",
                                "price": 9.99 + j
                            }
                            for j in range(3)  # 3 items per order
                        ]
                    }
                    for i in range(20)  # 20 purchase history items
                ]
            }
        },
        "catalog": {
            "categories": [
                {
                    "id": f"cat_{i}",
                    "name": f"Category {i}",
                    "products": [
                        {
                            "id": f"prod_{i}_{j}",
                            "name": f"Product {i}-{j}",
                            "price": 19.99 + j,
                            "attributes": {
                                f"attr_{k}": f"value_{k}" 
                                for k in range(15)  # 15 attributes per product
                            }
                        }
                        for j in range(25)  # 25 products per category
                    ]
                }
                for i in range(8)  # 8 categories
            ]
        },
        "analytics": {
            "session": {
                "id": "sess_abc123def456",
                "events": [
                    {
                        "event": f"event_type_{i%10}",
                        "timestamp": f"2024-01-15T10:{i%60:02d}:00Z",
                        "properties": {
                            f"prop_{j}": f"value_{i}_{j}" 
                            for j in range(12)  # 12 properties per event
                        }
                    }
                    for i in range(150)  # 150 events
                ]
            }
        }
    }


# Test configuration
def pytest_configure(config):
    """Configure pytest with integration-specific markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests requiring browser automation"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow integration tests (use -m 'not slow' to skip)"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance/load tests"
    )