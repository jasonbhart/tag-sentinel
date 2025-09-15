"""Unit tests for network observer."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.audit.capture.network_observer import NetworkObserver
from app.audit.models.capture import RequestLog, RequestStatus, ResourceType


class TestNetworkObserver:
    """Tests for NetworkObserver class."""
    
    @pytest.fixture
    def mock_page(self):
        """Mock Playwright page."""
        page = AsyncMock()
        page.on = MagicMock()
        return page
    
    @pytest.fixture
    def observer(self, mock_page):
        """Create network observer for testing."""
        return NetworkObserver(mock_page)
    
    @pytest.fixture
    def mock_request(self):
        """Mock Playwright request."""
        request = MagicMock()
        request.url = "https://example.com/api/data"
        request.method = "GET"
        request.resource_type = "xhr"
        request.headers = {"User-Agent": "Test"}
        request.post_data = None
        request.timing = {
            'dns_start': 100.0,
            'dns_end': 120.0,
            'request_start': 150.0,
            'response_start': 200.0,
            'response_end': 250.0
        }
        request.failure = None
        return request
    
    @pytest.fixture
    def mock_response(self):
        """Mock Playwright response."""
        response = AsyncMock()
        response.status = 200
        response.status_text = "OK"
        response.headers = {"content-type": "application/json"}
        response.text.return_value = '{"data": "test"}'
        response.body.return_value = b'{"data": "test"}'
        return response
    
    def test_observer_initialization(self, observer, mock_page):
        """Test observer initialization."""
        assert observer.page == mock_page
        assert len(observer.requests) == 0
        assert len(observer.completed_requests) == 0
        assert len(observer._active_requests) == 0
        
        # Verify event listeners were set up
        assert mock_page.on.call_count == 4
        expected_events = ["request", "response", "requestfinished", "requestfailed"]
        actual_events = [call.args[0] for call in mock_page.on.call_args_list]
        for event in expected_events:
            assert event in actual_events
    
    def test_create_request_log(self, observer, mock_request):
        """Test creating request log from Playwright request."""
        request_log = observer._create_request_log(mock_request)
        
        assert isinstance(request_log, RequestLog)
        assert request_log.url == "https://example.com/api/data"
        assert request_log.method == "GET"
        assert request_log.resource_type == ResourceType.XHR
        assert request_log.status == RequestStatus.PENDING
        assert request_log.request_headers == {"User-Agent": "Test"}
    
    def test_resource_type_mapping(self, observer):
        """Test resource type mapping from Playwright types."""
        test_cases = [
            ("document", ResourceType.DOCUMENT),
            ("stylesheet", ResourceType.STYLESHEET),
            ("image", ResourceType.IMAGE),
            ("script", ResourceType.SCRIPT),
            ("xhr", ResourceType.XHR),
            ("fetch", ResourceType.FETCH),
            ("websocket", ResourceType.WEBSOCKET),
            ("unknown_type", ResourceType.OTHER)
        ]
        
        for pw_type, expected_type in test_cases:
            request = MagicMock()
            request.url = "https://example.com/test"
            request.method = "GET"
            request.resource_type = pw_type
            request.headers = {}
            request.post_data = None
            
            request_log = observer._create_request_log(request)
            assert request_log.resource_type == expected_type
    
    def test_extract_timing_data(self, observer, mock_request):
        """Test extracting timing data."""
        timing = observer._extract_timing_data(mock_request)
        
        assert timing is not None
        assert timing.dns_start == 100.0
        assert timing.dns_end == 120.0
        assert timing.request_start == 150.0
        assert timing.response_start == 200.0
        assert timing.response_end == 250.0
    
    def test_extract_timing_data_missing(self, observer):
        """Test extracting timing data when not available."""
        request = MagicMock()
        request.timing = None
        
        timing = observer._extract_timing_data(request)
        assert timing is None
    
    def test_on_request(self, observer, mock_request):
        """Test request start event handling."""
        observer._on_request(mock_request)
        
        assert len(observer.requests) == 1
        assert mock_request.url in observer.requests
        assert len(observer._active_requests) == 1
        
        request_log = observer.requests[mock_request.url]
        assert request_log.status == RequestStatus.PENDING
    
    def test_on_response(self, observer, mock_request, mock_response):
        """Test response received event handling."""
        # First add the request
        observer._on_request(mock_request)
        
        # Mock the response.request property
        mock_response.request = mock_request
        
        # Handle response
        observer._on_response(mock_response)
        
        request_log = observer.requests[mock_request.url]
        assert request_log.status_code == 200
        assert request_log.status_text == "OK"
        assert request_log.response_headers == {"content-type": "application/json"}
    
    def test_on_response_body_capture(self, observer, mock_request, mock_response):
        """Test response body capture logic."""
        observer._on_request(mock_request)
        mock_response.request = mock_request
        
        # Mock headers for text content
        mock_response.headers = {
            "content-type": "application/json",
            "content-length": "100"
        }
        
        # Mock async methods
        mock_response.text = AsyncMock(return_value='{"test": "data"}')
        mock_response.body = AsyncMock(return_value=b'{"test": "data"}')
        
        observer._on_response(mock_response)
        
        request_log = observer.requests[mock_request.url]
        # Response should have been processed synchronously
        assert request_log.status_code == 200
    
    def test_on_request_finished(self, observer, mock_request):
        """Test request finished event handling."""
        # Add request first
        observer._on_request(mock_request)
        
        # Mark as finished
        observer._on_request_finished(mock_request)
        
        request_log = observer.requests[mock_request.url]
        assert request_log.status == RequestStatus.SUCCESS
        assert request_log.end_time is not None
        assert len(observer.completed_requests) == 1
        assert mock_request.url not in observer._active_requests
    
    def test_on_request_failed(self, observer, mock_request):
        """Test request failed event handling."""
        mock_request.failure = "Connection timeout"
        
        # Add request first
        observer._on_request(mock_request)
        
        # Mark as failed
        observer._on_request_failed(mock_request)
        
        request_log = observer.requests[mock_request.url]
        assert request_log.status == RequestStatus.TIMEOUT  # "timeout" in failure message
        assert request_log.error_text == "Connection timeout"
        assert request_log.end_time is not None
        assert len(observer.completed_requests) == 1
    
    def test_on_request_failed_unknown_request(self, observer, mock_request):
        """Test handling failed event for unknown request."""
        mock_request.failure = "Network error"
        
        # Don't add request first, simulate missing request
        observer._on_request_failed(mock_request)
        
        # Should create failed request log
        assert len(observer.completed_requests) == 1
        request_log = observer.completed_requests[0]
        assert request_log.status == RequestStatus.FAILED
        assert request_log.error_text == "Network error"
    
    def test_failure_type_detection(self, observer, mock_request):
        """Test different failure type detection."""
        test_cases = [
            ("Connection timeout", RequestStatus.TIMEOUT),
            ("Request aborted", RequestStatus.ABORTED),
            ("Network error", RequestStatus.FAILED),
            ("Unknown error", RequestStatus.FAILED)
        ]
        
        for failure_text, expected_status in test_cases:
            mock_request.failure = failure_text
            observer._on_request(mock_request)
            observer._on_request_failed(mock_request)
            
            request_log = observer.completed_requests[-1]
            assert request_log.status == expected_status
            
            # Clear for next test
            observer.clear()
    
    def test_callback_system(self, observer, mock_request):
        """Test callback system for completed requests."""
        callback_calls = []
        
        def test_callback(request_log):
            callback_calls.append(request_log)
        
        observer.add_callback(test_callback)
        
        # Process request
        observer._on_request(mock_request)
        observer._on_request_finished(mock_request)
        
        assert len(callback_calls) == 1
        assert callback_calls[0].url == mock_request.url
    
    def test_callback_error_handling(self, observer, mock_request):
        """Test callback error handling."""
        def failing_callback(request_log):
            raise Exception("Callback error")
        
        observer.add_callback(failing_callback)
        
        # Should not raise error
        observer._on_request(mock_request)
        observer._on_request_finished(mock_request)
        
        # Request should still be processed
        assert len(observer.completed_requests) == 1
    
    def test_get_requests_methods(self, observer, mock_request):
        """Test methods for getting requests."""
        # Add completed request
        observer._on_request(mock_request)
        observer._on_request_finished(mock_request)
        
        # Add active request
        mock_request2 = MagicMock()
        mock_request2.url = "https://example.com/active"
        mock_request2.method = "POST"
        mock_request2.resource_type = "xhr"
        mock_request2.headers = {}
        mock_request2.post_data = None
        
        observer._on_request(mock_request2)
        
        # Test getter methods
        completed = observer.get_completed_requests()
        active = observer.get_active_requests()
        all_requests = observer.get_all_requests()
        
        assert len(completed) == 1
        assert len(active) == 1
        assert len(all_requests) == 2
    
    def test_get_requests_by_host(self, observer):
        """Test filtering requests by host."""
        # Add requests for different hosts
        for i, host in enumerate(["example.com", "api.example.com", "other.com"]):
            request = MagicMock()
            request.url = f"https://{host}/path{i}"
            request.method = "GET"
            request.resource_type = "xhr"
            request.headers = {}
            request.post_data = None
            
            observer._on_request(request)
            observer._on_request_finished(request)
        
        example_requests = observer.get_requests_by_host("example.com")
        api_requests = observer.get_requests_by_host("api.example.com")
        
        assert len(example_requests) == 1
        assert len(api_requests) == 1
        assert example_requests[0].host == "example.com"
    
    def test_get_requests_by_type(self, observer):
        """Test filtering requests by resource type."""
        resource_types = ["xhr", "script", "image"]
        
        for i, res_type in enumerate(resource_types):
            request = MagicMock()
            request.url = f"https://example.com/resource{i}"
            request.method = "GET"
            request.resource_type = res_type
            request.headers = {}
            request.post_data = None
            
            observer._on_request(request)
            observer._on_request_finished(request)
        
        xhr_requests = observer.get_requests_by_type(ResourceType.XHR)
        script_requests = observer.get_requests_by_type(ResourceType.SCRIPT)
        
        assert len(xhr_requests) == 1
        assert len(script_requests) == 1
    
    def test_get_failed_requests(self, observer):
        """Test getting failed requests."""
        # Add successful request
        success_request = MagicMock()
        success_request.url = "https://example.com/success"
        success_request.method = "GET"
        success_request.resource_type = "xhr"
        success_request.headers = {}
        success_request.post_data = None
        
        observer._on_request(success_request)
        observer._on_request_finished(success_request)
        
        # Add failed request
        fail_request = MagicMock()
        fail_request.url = "https://example.com/fail"
        fail_request.method = "GET"
        fail_request.resource_type = "xhr"
        fail_request.headers = {}
        fail_request.post_data = None
        fail_request.failure = "Network error"
        
        observer._on_request(fail_request)
        observer._on_request_failed(fail_request)
        
        failed_requests = observer.get_failed_requests()
        assert len(failed_requests) == 1
        assert failed_requests[0].url == "https://example.com/fail"
    
    def test_finalize_pending_requests(self, observer, mock_request):
        """Test finalizing pending requests."""
        # Add request but don't finish it
        observer._on_request(mock_request)
        
        assert len(observer._active_requests) == 1
        assert len(observer.completed_requests) == 0
        
        # Finalize pending requests
        observer.finalize_pending_requests()
        
        assert len(observer._active_requests) == 0
        assert len(observer.completed_requests) == 1
        
        request_log = observer.completed_requests[0]
        assert request_log.status == RequestStatus.TIMEOUT
        assert "did not complete" in request_log.error_text
    
    def test_clear(self, observer, mock_request):
        """Test clearing observer data."""
        # Add some requests
        observer._on_request(mock_request)
        observer._on_request_finished(mock_request)
        
        assert len(observer.requests) > 0
        assert len(observer.completed_requests) > 0
        
        # Clear
        observer.clear()
        
        assert len(observer.requests) == 0
        assert len(observer.completed_requests) == 0
        assert len(observer._active_requests) == 0
        assert len(observer._callbacks) == 0
    
    def test_get_stats(self, observer):
        """Test statistics generation."""
        # Add various request types
        request_configs = [
            ("https://example.com/doc", "document", True),  # Successful
            ("https://example.com/script.js", "script", True),  # Successful  
            ("https://example.com/api", "xhr", False),  # Failed
            ("https://example.com/img.png", "image", True),  # Successful
        ]
        
        for i, (url, res_type, success) in enumerate(request_configs):
            request = MagicMock()
            request.url = url
            request.method = "GET"
            request.resource_type = res_type
            request.headers = {}
            request.post_data = None
            
            observer._on_request(request)
            
            if success:
                observer._on_request_finished(request)
            else:
                request.failure = "Failed"
                observer._on_request_failed(request)
        
        stats = observer.get_stats()
        
        assert stats['total_requests'] == 4
        assert stats['successful_requests'] == 3
        assert stats['failed_requests'] == 1
        assert stats['document_requests'] == 1
        assert stats['script_requests'] == 1
        assert stats['xhr_requests'] == 1
        assert stats['image_requests'] == 1
    
    def test_repr(self, observer, mock_request):
        """Test string representation."""
        observer._on_request(mock_request)
        observer._on_request_finished(mock_request)
        
        repr_str = repr(observer)
        assert "NetworkObserver" in repr_str
        assert "total=1" in repr_str
        assert "completed=1" in repr_str
        assert "pending=0" in repr_str
        assert "failed=0" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])