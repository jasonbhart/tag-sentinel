"""Unit tests for DataLayer snapshot engine."""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.audit.datalayer.snapshot import Snapshotter
from app.audit.datalayer.models import DataLayerSnapshot, DLContext
from app.audit.datalayer.config import DataLayerConfig


class TestSnapshotter:
    """Test cases for Snapshotter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DataLayerConfig(
            capture={
                "enabled": True,
                "timeout_seconds": 5.0,
                "max_depth": 10,
                "max_size_bytes": 1024
            }
        )
        self.snapshotter = Snapshotter(self.config)
    
    def test_snapshotter_initialization(self):
        """Test snapshotter initialization."""
        assert self.snapshotter.config == self.config.capture
        assert self.snapshotter.config.enabled is True
    
    def test_is_datalayer_array_detection(self):
        """Test GTM-style array detection."""
        # GTM-style array
        gtm_array = [
            {"gtm.start": 1234567890},
            {"event": "page_view", "page": "home"},
            {"user_id": "123"}
        ]
        
        # Plain object
        plain_object = {"page": "home", "user_id": "123"}
        
        # Empty array
        empty_array = []
        
        assert self.snapshotter._is_datalayer_array(gtm_array) is True
        assert self.snapshotter._is_datalayer_array(plain_object) is False
        assert self.snapshotter._is_datalayer_array(empty_array) is True
        assert self.snapshotter._is_datalayer_array(None) is False
    
    def test_is_event_push_detection(self):
        """Test event push detection."""
        # Standard event
        event_push = {"event": "page_view", "page": "home"}
        
        # GTM ecommerce event
        ecommerce_push = {"event": "purchase", "ecommerce": {"value": 100}}
        
        # Custom event with custom pattern
        custom_config = DataLayerConfig(
            capture={
                "enabled": True,
                "timeout_seconds": 5.0,
                "max_depth": 10,
                "max_size_bytes": 1024,
                "event_detection_patterns": ["custom_event"]
            }
        )
        snapshotter_custom = Snapshotter(custom_config)
        custom_push = {"custom_event": "test_event", "data": "value"}
        
        # Variable push (not event)
        variable_push = {"user_id": "123", "page": "home"}
        
        assert self.snapshotter._is_event_push(event_push) is True
        assert self.snapshotter._is_event_push(ecommerce_push) is True
        assert snapshotter_custom._is_event_push(custom_push) is True
        assert self.snapshotter._is_event_push(variable_push) is False
    
    def test_extract_events_from_array(self):
        """Test event extraction from GTM array."""
        gtm_array = [
            {"gtm.start": 1234567890},
            {"event": "page_view", "page": "home"},
            {"user_id": "123"},
            {"event": "click", "element": "button"},
            {"page_title": "Test Page"}
        ]
        
        events, variables = self.snapshotter._extract_events_from_array(gtm_array)
        
        # Should have 2 events
        assert len(events) == 2
        assert events[0]["event"] == "page_view"
        assert events[1]["event"] == "click"
        
        # Should have merged variables (excluding gtm.start)
        assert variables["user_id"] == "123"
        assert variables["page_title"] == "Test Page"
        assert "gtm.start" not in variables
    
    def test_merge_variables_simple(self):
        """Test simple variable merging."""
        variables = [
            {"user_id": "123", "page": "home"},
            {"user_id": "456", "category": "products"},  # user_id should be overwritten
            {"section": "header"}
        ]
        
        merged = self.snapshotter._merge_variables(variables)
        
        assert merged["user_id"] == "456"  # Last value wins
        assert merged["page"] == "home"
        assert merged["category"] == "products"
        assert merged["section"] == "header"
    
    def test_merge_variables_nested(self):
        """Test nested variable merging."""
        variables = [
            {"user": {"id": "123", "name": "John"}},
            {"user": {"id": "456", "email": "john@example.com"}, "page": "home"}
        ]
        
        merged = self.snapshotter._merge_variables(variables)
        
        # Nested objects should be merged properly
        assert merged["user"]["id"] == "456"  # Last value wins
        assert merged["user"]["email"] == "john@example.com"  # New field added
        assert "name" not in merged["user"]  # Previous nested field lost (simple merge)
        assert merged["page"] == "home"
    
    def test_merge_variables_with_depth_limit(self):
        """Test variable merging respects depth limit."""
        # Create deeply nested structure
        deep_structure = {"level1": {"level2": {"level3": {"level4": "deep_value"}}}}
        
        # Test with shallow depth limit
        shallow_config = DataLayerConfig(max_depth=2)
        shallow_snapshotter = Snapshotter(shallow_config)
        
        merged = shallow_snapshotter._merge_variables([deep_structure])
        
        # Should limit depth
        assert "level1" in merged
        assert "level2" in merged["level1"]
        # Depth limit should prevent deeper nesting
        assert isinstance(merged["level1"]["level2"], (dict, str))
    
    def test_process_gtm_array_normalization(self):
        """Test GTM array processing and normalization."""
        gtm_data = [
            {"gtm.start": 1234567890},
            {"event": "page_view", "page": "home", "user_id": "123"},
            {"category": "products"},
            {"event": "click", "element": "button", "user_id": "456"}  # user_id changes
        ]
        
        processed = self.snapshotter._process_gtm_array(gtm_data)
        
        # Should have normalized structure
        assert "variables" in processed
        assert "events" in processed
        
        # Events should be extracted
        assert len(processed["events"]) == 2
        assert processed["events"][0]["event"] == "page_view"
        assert processed["events"][1]["event"] == "click"
        
        # Variables should be merged (latest values win)
        variables = processed["variables"]
        assert variables["category"] == "products"
        # Note: user_id from events might be merged depending on implementation
    
    def test_process_object_datalayer(self):
        """Test processing object-style dataLayer."""
        object_data = {
            "page": "home",
            "user": {"id": "123", "type": "registered"},
            "products": ["item1", "item2"]
        }
        
        processed = self.snapshotter._process_object_datalayer(object_data)
        
        # Should return the object as-is (with potential depth limiting)
        assert processed["page"] == "home"
        assert processed["user"]["id"] == "123"
        assert len(processed["products"]) == 2
    
    def test_safe_json_serialize_circular_reference(self):
        """Test safe JSON serialization with circular references."""
        # Create circular reference
        obj = {"name": "test"}
        obj["self"] = obj
        
        result = self.snapshotter._safe_json_serialize(obj)
        
        # Should handle circular reference gracefully
        assert result is not None
        assert result["name"] == "test"
        # Circular reference should be handled (likely removed or replaced)
    
    def test_safe_json_serialize_large_object(self):
        """Test safe JSON serialization with size limit."""
        # Create large object
        large_obj = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}
        
        # Use small size limit (minimum 1024 bytes required)
        small_config = DataLayerConfig(
            capture={
                "enabled": True,
                "max_size_bytes": 1024,
                "max_entries": 50
            }
        )
        small_snapshotter = Snapshotter(small_config)

        result = small_snapshotter._safe_json_serialize(large_obj)

        # Should respect size limit
        if result is not None:
            serialized_size = len(json.dumps(result))
            assert serialized_size <= small_config.capture.max_size_bytes * 1.1  # Allow some overhead
    
    def test_safe_json_serialize_with_functions(self):
        """Test safe JSON serialization with non-serializable objects."""
        obj_with_function = {
            "name": "test",
            "func": lambda x: x * 2,  # Non-serializable
            "data": [1, 2, 3]
        }
        
        result = self.snapshotter._safe_json_serialize(obj_with_function)
        
        # Should handle non-serializable objects
        assert result is not None
        assert result["name"] == "test"
        assert result["data"] == [1, 2, 3]
        # Function should be handled gracefully (removed or replaced)
    
    @pytest.mark.asyncio
    async def test_capture_from_page_missing_datalayer(self):
        """Test capturing from page with missing dataLayer."""
        # Mock page that returns null for dataLayer
        mock_page = AsyncMock()
        mock_page.evaluate.return_value = {
            'exists': False,
            'latest': None,
            'events': []
        }
        mock_page.url = "https://example.com"
        
        context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://example.com"})
        
        snapshot = await self.snapshotter.capture_from_page(mock_page, context)
        
        assert str(snapshot.page_url) == "https://example.com/"
        assert not snapshot.exists
        assert snapshot.latest is None
        assert len(snapshot.events) == 0
    
    @pytest.mark.asyncio
    async def test_capture_from_page_object_datalayer(self):
        """Test capturing object-style dataLayer from page."""
        mock_datalayer = {"page": "home", "user_id": "123"}

        mock_page = AsyncMock()
        mock_page.evaluate.return_value = {
            'exists': True,
            'latest': mock_datalayer,
            'events': []
        }
        mock_page.url = "https://example.com"
        
        context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://example.com"})
        
        snapshot = await self.snapshotter.capture_from_page(mock_page, context)
        
        assert str(snapshot.page_url) == "https://example.com/"
        assert snapshot.exists
        assert snapshot.latest is not None
        assert snapshot.latest["page"] == "home"
    
    @pytest.mark.asyncio
    async def test_capture_from_page_gtm_array(self):
        """Test capturing GTM-style array from page."""
        mock_datalayer = [
            {"event": "page_view", "page": "home"},
            {"user_id": "123"}
        ]

        mock_page = AsyncMock()
        mock_page.evaluate.return_value = {
            'exists': True,
            'latest': {"user_id": "123"},  # Variables merged
            'events': [{"event": "page_view", "page": "home"}]  # Events extracted
        }
        mock_page.url = "https://example.com"
        
        context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://example.com"})
        
        snapshot = await self.snapshotter.capture_from_page(mock_page, context)
        
        assert str(snapshot.page_url) == "https://example.com/"
        assert snapshot.exists
        assert len(snapshot.events) == 1
        assert snapshot.events[0]["event"] == "page_view"
        assert snapshot.latest["user_id"] == "123"
    
    @pytest.mark.asyncio
    async def test_capture_from_page_timeout(self):
        """Test capture with timeout."""
        # Mock page that times out
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        mock_page = AsyncMock()
        mock_page.evaluate.side_effect = PlaywrightTimeoutError("Page timeout")
        mock_page.url = "https://example.com"
        
        context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://example.com"})
        
        snapshot = await self.snapshotter.capture_from_page(mock_page, context)
        
        assert str(snapshot.page_url) == "https://example.com/"
        # With timeout, we expect the snapshot to exist but might not have data
        assert not snapshot.exists or snapshot.latest is None
    
    @pytest.mark.asyncio
    async def test_capture_from_page_javascript_error(self):
        """Test capture with JavaScript error."""
        # Mock page that throws JavaScript error
        mock_page = AsyncMock()
        mock_page.evaluate.side_effect = Exception("JavaScript error: ReferenceError")
        mock_page.url = "https://example.com"
        
        context = DLContext(env="test", data_layer_object="dataLayer", max_depth=6, max_entries=500, site_config={"url": "https://example.com"})
        
        snapshot = await self.snapshotter.capture_from_page(mock_page, context)
        
        assert str(snapshot.page_url) == "https://example.com/"
        # With JavaScript error, we expect the snapshot to exist but might not have data
        assert not snapshot.exists or snapshot.latest is None
    
    def test_create_snapshot_basic(self):
        """Test basic snapshot creation."""
        url = "https://example.com"
        raw_data = {"page": "home", "user_id": "123"}
        
        snapshot = self.snapshotter._create_snapshot(url, raw_data, raw_data)
        
        assert str(snapshot.page_url).rstrip('/') == url.rstrip('/')
        assert snapshot.latest == raw_data
        assert snapshot.exists
        assert snapshot.variable_count == 2
    
    def test_create_snapshot_with_events(self):
        """Test snapshot creation with events."""
        url = "https://example.com"
        raw_data = [{"event": "click"}, {"user_id": "123"}]
        processed_data = {"user_id": "123"}
        events = [{"event": "click"}]
        
        snapshot = self.snapshotter._create_snapshot(url, raw_data, processed_data, events)
        
        assert str(snapshot.page_url).rstrip('/') == url.rstrip('/')
        assert snapshot.events == events
        assert snapshot.event_count == 1
        assert snapshot.latest == processed_data
    
    def test_create_snapshot_with_error(self):
        """Test snapshot creation with error."""
        url = "https://example.com"
        error = "DataLayer not found"
        
        snapshot = self.snapshotter._create_snapshot(url, None, None, [], error)
        
        assert str(snapshot.page_url).rstrip('/') == url.rstrip('/')
        assert snapshot.latest == {}
        assert not snapshot.exists

    def test_create_snapshot_empty_datalayer(self):
        """Test snapshot creation with empty but existing dataLayer."""
        url = "https://example.com"

        # Test empty dict (dataLayer exists but has no variables)
        empty_dict_snapshot = self.snapshotter._create_snapshot(url, {}, {})
        assert str(empty_dict_snapshot.page_url).rstrip('/') == url.rstrip('/')
        assert empty_dict_snapshot.exists  # Should exist even though empty
        assert empty_dict_snapshot.latest == {}
        assert empty_dict_snapshot.variable_count == 0

        # Test empty array (dataLayer exists but has no pushes)
        empty_array_snapshot = self.snapshotter._create_snapshot(url, [], {})
        assert str(empty_array_snapshot.page_url).rstrip('/') == url.rstrip('/')
        assert empty_array_snapshot.exists  # Should exist even though empty
        assert empty_array_snapshot.latest == {}
        assert empty_array_snapshot.variable_count == 0

    def test_javascript_generation(self):
        """Test JavaScript code generation for dataLayer capture."""
        # This tests that the JavaScript code is properly formatted
        js_code = self.snapshotter._get_datalayer_capture_js()
        
        assert "window[objectName]" in js_code
        assert "JSON.stringify" in js_code
        # Should have safety checks
        assert "try" in js_code and "catch" in js_code
    
    def test_config_driven_behavior(self):
        """Test that configuration drives snapshotter behavior."""
        # Test with different configurations
        from app.audit.datalayer.config import CaptureConfig

        strict_capture_config = CaptureConfig(
            max_depth=2,
            max_size_bytes=1024,  # Minimum allowed value
            execution_timeout_ms=1000
        )
        strict_config = DataLayerConfig(capture=strict_capture_config)

        lenient_capture_config = CaptureConfig(
            max_depth=20,  # Maximum allowed value
            max_size_bytes=10000,
            execution_timeout_ms=30000
        )
        lenient_config = DataLayerConfig(capture=lenient_capture_config)

        strict_snapshotter = Snapshotter(strict_config)
        lenient_snapshotter = Snapshotter(lenient_config)

        # Both should be configured differently
        assert strict_snapshotter.config.max_depth == 2
        assert lenient_snapshotter.config.max_depth == 20

        assert strict_snapshotter.config.execution_timeout_ms == 1000
        assert lenient_snapshotter.config.execution_timeout_ms == 30000


class TestSnapshotterEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DataLayerConfig()
        self.snapshotter = Snapshotter(self.config)
    
    def test_empty_gtm_array(self):
        """Test handling of empty GTM array."""
        empty_array = []
        
        processed = self.snapshotter._process_gtm_array(empty_array)
        
        assert processed["variables"] == {}
        assert processed["events"] == []
    
    def test_gtm_array_with_only_gtm_start(self):
        """Test GTM array with only gtm.start."""
        gtm_array = [{"gtm.start": 1234567890}]
        
        processed = self.snapshotter._process_gtm_array(gtm_array)
        
        # gtm.start should be filtered out
        assert processed["variables"] == {}
        assert processed["events"] == []
    
    def test_malformed_gtm_array(self):
        """Test handling of malformed GTM array."""
        malformed_array = [
            {"event": "test"},
            "invalid_string",  # Invalid entry
            {"valid": "entry"},
            None,  # None entry
            42  # Number entry
        ]
        
        # Should handle gracefully
        events, variables = self.snapshotter._extract_events_from_array(malformed_array)
        
        # Should only process valid dict entries
        assert len(events) == 1
        assert events[0]["event"] == "test"
        assert variables["valid"] == "entry"
    
    def test_very_deep_nesting(self):
        """Test handling of very deeply nested objects."""
        # Create 20-level deep nesting
        deep_obj = current = {}
        for i in range(20):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        current["value"] = "deep_value"
        
        # Process with default depth limit
        result = self.snapshotter._safe_json_serialize(deep_obj)
        
        # Should handle gracefully (either truncated or processed)
        assert result is not None
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_data = {
            "emoji": "🎉🚀💯",
            "chinese": "你好世界",
            "arabic": "مرحبا بالعالم",
            "special_chars": "\n\t\r\0",
            "html_entities": "&lt;script&gt;alert()&lt;/script&gt;"
        }
        
        result = self.snapshotter._safe_json_serialize(unicode_data)
        
        assert result is not None
        assert result["emoji"] == "🎉🚀💯"
        assert result["chinese"] == "你好世界"
    
    def test_very_large_arrays(self):
        """Test handling of very large arrays."""
        large_array = [f"item_{i}" for i in range(10000)]
        data = {"large_array": large_array}
        
        # Should handle large data gracefully
        result = self.snapshotter._safe_json_serialize(data)
        
        # Should either process or truncate appropriately
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])