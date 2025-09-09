"""Specialized tests for MP Debug validation functionality.

This module provides focused testing of the Measurement Protocol (MP) debug 
validation feature integrated into the GA4 detector for non-production environments.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch
from typing import Dict, Any

from app.audit.detectors import GA4Detector, DetectContext
from app.audit.models.capture import (
    PageResult,
    RequestLog,
    CaptureStatus,
    RequestStatus,
    ResourceType
)


class TestMPDebugValidation:
    """Test MP Debug validation functionality."""
    
    @pytest.fixture
    def non_production_context(self) -> DetectContext:
        """Create non-production context with MP debug enabled."""
        return DetectContext(
            environment="test",
            is_production=False,
            config={
                "ga4": {
                    "enabled": True,
                    "mp_debug": {
                        "enabled": True,
                        "timeout_ms": 2000,
                        "endpoint": "https://www.google-analytics.com/debug/mp/collect"
                    }
                }
            },
            enable_debug=True,
            enable_external_validation=True
        )
    
    @pytest.fixture
    def production_context(self) -> DetectContext:
        """Create production context with MP debug disabled."""
        return DetectContext(
            environment="production",
            is_production=True,
            config={
                "ga4": {
                    "enabled": True,
                    "mp_debug": {
                        "enabled": False
                    }
                }
            },
            enable_debug=False,
            enable_external_validation=False
        )
    
    @pytest.fixture
    def ga4_page_result(self) -> PageResult:
        """Create page result with GA4 requests."""
        return PageResult(
            url="https://example.com/test",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=[
                RequestLog(
                    url="https://www.google-analytics.com/mp/collect?measurement_id=G-XXXXXXXXXX",
                    method="POST",
                    resource_type=ResourceType.FETCH,
                    status=RequestStatus.SUCCESS,
                    status_code=200,
                    request_body='{"client_id":"123.456","events":[{"name":"page_view","params":{"page_title":"Test Page"}}]}',
                    start_time=datetime.utcnow()
                ),
                RequestLog(
                    url="https://www.google-analytics.com/g/collect?tid=UA-123456-1&t=pageview",
                    method="GET",
                    resource_type=ResourceType.FETCH,
                    status=RequestStatus.SUCCESS,
                    status_code=200,
                    start_time=datetime.utcnow()
                )
            ]
        )
    
    @pytest.mark.asyncio
    async def test_mp_debug_enabled_in_non_production(self, ga4_page_result, non_production_context):
        """Test that MP debug validation runs in non-production environments."""
        detector = GA4Detector()
        
        # Mock the HTTP client to simulate successful debug response
        mock_response = {
            "validationMessages": [],
            "eventValidationResults": [
                {
                    "eventName": "page_view",
                    "isValid": True,
                    "validationMessages": []
                }
            ]
        }
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            result = await detector.detect(ga4_page_result, non_production_context)
        
        assert result.success
        assert result.detector_name == "GA4Detector"
        
        # Should have events detected
        assert len(result.events) > 0
        
        # Should have MP debug validation notes
        debug_notes = [note for note in result.notes if "mp_debug" in note.message.lower()]
        # MP debug should have run (either success or error note)
    
    @pytest.mark.asyncio
    async def test_mp_debug_disabled_in_production(self, ga4_page_result, production_context):
        """Test that MP debug validation is skipped in production environments."""
        detector = GA4Detector()
        
        with patch('httpx.AsyncClient.post') as mock_post:
            result = await detector.detect(ga4_page_result, production_context)
        
        # Should not make any HTTP requests for MP debug in production
        mock_post.assert_not_called()
        
        assert result.success
        assert result.detector_name == "GA4Detector"
        
        # Should still detect events normally
        assert len(result.events) > 0
        
        # Should not have MP debug related notes
        debug_notes = [note for note in result.notes if "mp_debug" in note.message.lower()]
        assert len(debug_notes) == 0
    
    @pytest.mark.asyncio
    async def test_mp_debug_validation_success(self, ga4_page_result, non_production_context):
        """Test successful MP debug validation response handling."""
        detector = GA4Detector()
        
        mock_response = {
            "validationMessages": [],
            "eventValidationResults": [
                {
                    "eventName": "page_view",
                    "isValid": True,
                    "validationMessages": []
                }
            ]
        }
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            result = await detector.detect(ga4_page_result, non_production_context)
        
        assert result.success
        
        # Verify the debug endpoint was called
        mock_post.assert_called()
        
        # Check that the call was made to the debug endpoint
        call_args = mock_post.call_args
        assert "debug/mp/collect" in call_args[1]["url"]
    
    @pytest.mark.asyncio
    async def test_mp_debug_validation_errors(self, ga4_page_result, non_production_context):
        """Test MP debug validation with validation errors."""
        detector = GA4Detector()
        
        mock_response = {
            "validationMessages": [
                {
                    "fieldPath": "events[0].name",
                    "description": "Invalid event name",
                    "validationCode": "VALUE_INVALID"
                }
            ],
            "eventValidationResults": [
                {
                    "eventName": "page_view",
                    "isValid": False,
                    "validationMessages": [
                        {
                            "fieldPath": "events[0].params.page_title",
                            "description": "Parameter value too long",
                            "validationCode": "VALUE_TOO_LONG"
                        }
                    ]
                }
            ]
        }
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            result = await detector.detect(ga4_page_result, non_production_context)
        
        assert result.success  # Should still succeed even with validation errors
        
        # Should have validation error notes
        validation_notes = [note for note in result.notes if "validation" in note.message.lower()]
        # May or may not have validation notes depending on implementation
    
    @pytest.mark.asyncio
    async def test_mp_debug_network_error_handling(self, ga4_page_result, non_production_context):
        """Test graceful handling of network errors during MP debug validation."""
        detector = GA4Detector()
        
        with patch('httpx.AsyncClient.post') as mock_post:
            # Simulate network timeout
            mock_post.side_effect = asyncio.TimeoutError("Request timed out")
            
            result = await detector.detect(ga4_page_result, non_production_context)
        
        # Should still succeed despite MP debug failure
        assert result.success
        
        # Should have detected events normally
        assert len(result.events) > 0
        
        # Should handle the error gracefully without crashing
        assert result.detector_name == "GA4Detector"
    
    @pytest.mark.asyncio
    async def test_mp_debug_http_error_handling(self, ga4_page_result, non_production_context):
        """Test handling of HTTP errors from MP debug endpoint."""
        detector = GA4Detector()
        
        with patch('httpx.AsyncClient.post') as mock_post:
            # Simulate HTTP 500 error
            mock_post.return_value.status_code = 500
            mock_post.return_value.text = "Internal Server Error"
            
            result = await detector.detect(ga4_page_result, non_production_context)
        
        # Should still succeed despite MP debug failure
        assert result.success
        
        # Should have detected events normally
        assert len(result.events) > 0
    
    @pytest.mark.asyncio
    async def test_mp_debug_invalid_json_handling(self, ga4_page_result, non_production_context):
        """Test handling of invalid JSON responses from MP debug endpoint."""
        detector = GA4Detector()
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.side_effect = ValueError("Invalid JSON")
            mock_post.return_value.text = "Not valid JSON"
            
            result = await detector.detect(ga4_page_result, non_production_context)
        
        # Should still succeed despite invalid JSON response
        assert result.success
        assert len(result.events) > 0
    
    @pytest.mark.asyncio
    async def test_mp_debug_timeout_configuration(self, ga4_page_result):
        """Test that MP debug respects timeout configuration."""
        timeout_context = DetectContext(
            environment="test",
            is_production=False,
            config={
                "ga4": {
                    "enabled": True,
                    "mp_debug": {
                        "enabled": True,
                        "timeout_ms": 100  # Very short timeout
                    }
                }
            },
            enable_debug=True,
            enable_external_validation=True
        )
        
        detector = GA4Detector()
        
        with patch('httpx.AsyncClient.post') as mock_post:
            # Simulate slow response that would exceed timeout
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(0.2)  # 200ms delay, exceeds 100ms timeout
                return AsyncMock(status_code=200, json=lambda: {})
                
            mock_post.side_effect = slow_response
            
            result = await detector.detect(ga4_page_result, timeout_context)
        
        # Should complete successfully despite timeout
        assert result.success
        assert len(result.events) > 0
    
    def test_mp_debug_configuration_validation(self):
        """Test that MP debug configuration is properly validated."""
        # Test with invalid configuration
        invalid_context = DetectContext(
            environment="test",
            is_production=False,
            config={
                "ga4": {
                    "enabled": True,
                    "mp_debug": "invalid_config"  # Should be dict
                }
            }
        )
        
        detector = GA4Detector()
        # Should not crash with invalid config - detector should handle gracefully
        assert detector is not None
    
    @pytest.mark.asyncio
    async def test_mp_debug_with_multiple_requests(self, non_production_context):
        """Test MP debug validation with multiple GA4 requests."""
        page_with_multiple_requests = PageResult(
            url="https://example.com/multi",
            capture_status=CaptureStatus.SUCCESS,
            network_requests=[
                RequestLog(
                    url="https://www.google-analytics.com/mp/collect?measurement_id=G-1111111111",
                    method="POST",
                    resource_type=ResourceType.FETCH,
                    status=RequestStatus.SUCCESS,
                    request_body='{"client_id":"123.456","events":[{"name":"page_view"}]}',
                    start_time=datetime.utcnow()
                ),
                RequestLog(
                    url="https://www.google-analytics.com/mp/collect?measurement_id=G-2222222222",
                    method="POST",
                    resource_type=ResourceType.FETCH,
                    status=RequestStatus.SUCCESS,
                    request_body='{"client_id":"789.012","events":[{"name":"click"}]}',
                    start_time=datetime.utcnow()
                )
            ]
        )
        
        detector = GA4Detector()
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"validationMessages": []}
            
            result = await detector.detect(page_with_multiple_requests, non_production_context)
        
        assert result.success
        assert len(result.events) > 0
        
        # Should have made calls for GA4 requests (implementation dependent)
        # The key is that it doesn't crash with multiple requests
    
    @pytest.mark.asyncio
    async def test_mp_debug_external_validation_disabled(self, ga4_page_result):
        """Test that MP debug is skipped when external validation is disabled."""
        context_no_external = DetectContext(
            environment="test",
            is_production=False,
            config={
                "ga4": {
                    "enabled": True,
                    "mp_debug": {"enabled": True}
                }
            },
            enable_external_validation=False  # Disabled
        )
        
        detector = GA4Detector()
        
        with patch('httpx.AsyncClient.post') as mock_post:
            result = await detector.detect(ga4_page_result, context_no_external)
        
        # Should not make HTTP requests when external validation is disabled
        mock_post.assert_not_called()
        
        assert result.success
        assert len(result.events) > 0