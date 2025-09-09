"""Unit tests for capture data models."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from app.audit.models.capture import (
    RequestLog, CookieRecord, ConsoleLog, PageResult, TimingData, ArtifactPaths,
    RequestStatus, ResourceType, ConsoleLevel, CaptureStatus
)


class TestTimingData:
    """Tests for TimingData model."""
    
    def test_timing_data_creation(self):
        """Test basic timing data creation."""
        timing = TimingData(
            dns_start=100.0,
            dns_end=120.0,
            connect_start=120.0,
            connect_end=150.0,
            request_start=150.0,
            response_start=200.0,
            response_end=250.0
        )
        
        assert timing.dns_start == 100.0
        assert timing.dns_end == 120.0
        assert timing.total_time == 100.0  # 250 - 150
    
    def test_timing_data_partial(self):
        """Test timing data with partial information."""
        timing = TimingData(request_start=100.0)
        assert timing.total_time is None
        
        timing = TimingData(request_start=100.0, response_end=200.0)
        assert timing.total_time == 100.0
    
    def test_timing_data_empty(self):
        """Test timing data with no information."""
        timing = TimingData()
        assert timing.total_time is None


class TestRequestLog:
    """Tests for RequestLog model."""
    
    def test_request_log_creation(self):
        """Test basic request log creation."""
        request = RequestLog(
            url="https://example.com/api/data",
            method="GET",
            resource_type=ResourceType.XHR
        )
        
        assert request.url == "https://example.com/api/data"
        assert request.method == "GET"
        assert request.resource_type == ResourceType.XHR
        assert request.status == RequestStatus.PENDING
        assert request.host == "example.com"
    
    def test_request_log_validation(self):
        """Test URL validation."""
        # Valid URL
        request = RequestLog(
            url="https://example.com",
            method="GET",
            resource_type=ResourceType.DOCUMENT
        )
        assert request.url == "https://example.com"
        
        # Invalid URL should raise error
        with pytest.raises(ValueError):
            RequestLog(
                url="not-a-url",
                method="GET", 
                resource_type=ResourceType.DOCUMENT
            )
    
    def test_request_success_property(self):
        """Test is_successful property."""
        request = RequestLog(
            url="https://example.com",
            method="GET",
            resource_type=ResourceType.DOCUMENT,
            status=RequestStatus.SUCCESS,
            status_code=200
        )
        assert request.is_successful is True
        
        request.status_code = 404
        assert request.is_successful is False
        
        request.status = RequestStatus.FAILED
        assert request.is_successful is False
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        start = datetime.utcnow()
        end = start + timedelta(milliseconds=500)
        
        request = RequestLog(
            url="https://example.com",
            method="GET",
            resource_type=ResourceType.DOCUMENT,
            start_time=start,
            end_time=end
        )
        
        # Should be approximately 500ms
        assert abs(request.duration_ms - 500) < 10


class TestCookieRecord:
    """Tests for CookieRecord model."""
    
    def test_cookie_record_creation(self):
        """Test basic cookie record creation."""
        cookie = CookieRecord(
            name="session_id",
            value="abc123",
            domain="example.com",
            size=50,
            is_first_party=True
        )
        
        assert cookie.name == "session_id"
        assert cookie.value == "abc123"
        assert cookie.domain == "example.com"
        assert cookie.is_first_party is True
    
    def test_cookie_from_playwright(self):
        """Test creating cookie from Playwright cookie object."""
        pw_cookie = {
            'name': 'test_cookie',
            'value': 'test_value',
            'domain': 'example.com',
            'path': '/',
            'expires': -1,  # Session cookie
            'secure': True,
            'httpOnly': False,
            'sameSite': 'Strict'
        }
        
        cookie = CookieRecord.from_playwright_cookie(pw_cookie, "example.com", redact_value=False)
        
        assert cookie.name == "test_cookie"
        assert cookie.value == "test_value"
        assert cookie.domain == "example.com"
        assert cookie.secure is True
        assert cookie.http_only is False
        assert cookie.same_site == "Strict"
        assert cookie.is_session is True
        assert cookie.is_first_party is True
    
    def test_cookie_value_redaction(self):
        """Test cookie value redaction."""
        pw_cookie = {
            'name': 'secret_cookie',
            'value': 'secret_value',
            'domain': 'example.com',
            'expires': -1
        }
        
        cookie = CookieRecord.from_playwright_cookie(pw_cookie, "example.com", redact_value=True)
        
        assert cookie.value is None
        assert cookie.value_redacted is True
    
    def test_first_party_detection(self):
        """Test first-party cookie detection."""
        # Same domain
        pw_cookie = {'name': 'test', 'value': 'val', 'domain': 'example.com', 'expires': -1}
        cookie = CookieRecord.from_playwright_cookie(pw_cookie, "example.com")
        assert cookie.is_first_party is True
        
        # Subdomain
        pw_cookie['domain'] = '.example.com'
        cookie = CookieRecord.from_playwright_cookie(pw_cookie, "sub.example.com")
        assert cookie.is_first_party is True
        
        # Third party
        pw_cookie['domain'] = 'tracker.com'
        cookie = CookieRecord.from_playwright_cookie(pw_cookie, "example.com")
        assert cookie.is_first_party is False


class TestConsoleLog:
    """Tests for ConsoleLog model."""
    
    def test_console_log_creation(self):
        """Test basic console log creation."""
        log = ConsoleLog(
            level=ConsoleLevel.ERROR,
            text="JavaScript error occurred",
            url="https://example.com/script.js",
            line_number=42
        )
        
        assert log.level == ConsoleLevel.ERROR
        assert log.text == "JavaScript error occurred"
        assert log.url == "https://example.com/script.js"
        assert log.line_number == 42
    
    def test_console_log_from_playwright(self):
        """Test creating console log from Playwright message."""
        # Mock Playwright message
        class MockMessage:
            def __init__(self):
                self.type = 'error'
                self.text = 'Uncaught TypeError'
                self.location = {
                    'url': 'https://example.com/app.js',
                    'lineNumber': 10,
                    'columnNumber': 5
                }
        
        message = MockMessage()
        log = ConsoleLog.from_playwright_message(message)
        
        assert log.level == ConsoleLevel.ERROR
        assert log.text == "Uncaught TypeError"
        assert log.url == "https://example.com/app.js"
        assert log.line_number == 10
        assert log.column_number == 5


class TestArtifactPaths:
    """Tests for ArtifactPaths model."""
    
    def test_artifact_paths_creation(self):
        """Test artifact paths creation."""
        artifacts = ArtifactPaths(
            har_file=Path("/tmp/test.har"),
            screenshot_file=Path("/tmp/test.png")
        )
        
        assert artifacts.har_file == Path("/tmp/test.har")
        assert artifacts.screenshot_file == Path("/tmp/test.png")
        assert artifacts.has_artifacts is True
    
    def test_empty_artifacts(self):
        """Test empty artifact paths."""
        artifacts = ArtifactPaths()
        assert artifacts.has_artifacts is False


class TestPageResult:
    """Tests for PageResult model."""
    
    def test_page_result_creation(self):
        """Test basic page result creation."""
        result = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS
        )
        
        assert result.url == "https://example.com"
        assert result.capture_status == CaptureStatus.SUCCESS
        assert result.is_successful is True
        assert len(result.network_requests) == 0
        assert len(result.cookies) == 0
        assert len(result.console_logs) == 0
    
    def test_page_result_url_validation(self):
        """Test URL validation in page result."""
        # Valid URL
        result = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS
        )
        assert result.url == "https://example.com"
        
        # Invalid URL should raise error
        with pytest.raises(ValueError):
            PageResult(
                url="not-a-url",
                capture_status=CaptureStatus.SUCCESS
            )
    
    def test_add_data_methods(self):
        """Test methods for adding captured data."""
        result = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS
        )
        
        # Add request
        request = RequestLog(
            url="https://example.com/api",
            method="GET",
            resource_type=ResourceType.XHR
        )
        result.add_request(request)
        assert len(result.network_requests) == 1
        
        # Add cookie
        cookie = CookieRecord(
            name="test", 
            domain="example.com", 
            size=10,
            is_first_party=True
        )
        result.add_cookie(cookie)
        assert len(result.cookies) == 1
        
        # Add console log
        log = ConsoleLog(level=ConsoleLevel.INFO, text="Info message")
        result.add_console_log(log)
        assert len(result.console_logs) == 1
        
        # Add page error
        result.add_page_error("JavaScript error")
        assert len(result.page_errors) == 1
    
    def test_request_filtering_properties(self):
        """Test properties that filter requests."""
        result = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS
        )
        
        # Add successful request
        success_req = RequestLog(
            url="https://example.com/success",
            method="GET",
            resource_type=ResourceType.DOCUMENT,
            status=RequestStatus.SUCCESS,
            status_code=200
        )
        result.add_request(success_req)
        
        # Add failed request
        fail_req = RequestLog(
            url="https://example.com/fail",
            method="GET",
            resource_type=ResourceType.DOCUMENT,
            status=RequestStatus.FAILED,
            status_code=404
        )
        result.add_request(fail_req)
        
        assert len(result.successful_requests) == 1
        assert len(result.failed_requests) == 1
    
    def test_cookie_filtering_properties(self):
        """Test properties that filter cookies."""
        result = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS
        )
        
        # Add first-party cookie
        first_party = CookieRecord(
            name="session", 
            domain="example.com", 
            size=10,
            is_first_party=True
        )
        result.add_cookie(first_party)
        
        # Add third-party cookie
        third_party = CookieRecord(
            name="tracking", 
            domain="tracker.com", 
            size=10,
            is_first_party=False
        )
        result.add_cookie(third_party)
        
        assert len(result.first_party_cookies) == 1
        assert len(result.third_party_cookies) == 1
    
    def test_error_count_calculation(self):
        """Test error count calculation."""
        result = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS
        )
        
        # Add console error
        error_log = ConsoleLog(level=ConsoleLevel.ERROR, text="Error")
        result.add_console_log(error_log)
        
        # Add failed request
        fail_req = RequestLog(
            url="https://example.com/fail",
            method="GET",
            resource_type=ResourceType.DOCUMENT,
            status=RequestStatus.FAILED
        )
        result.add_request(fail_req)
        
        # Add page error
        result.add_page_error("JS Error")
        
        assert result.error_count == 3  # 1 console + 1 network + 1 page
    
    def test_export_summary(self):
        """Test export summary functionality."""
        result = PageResult(
            url="https://example.com",
            final_url="https://www.example.com",
            title="Example Site",
            capture_status=CaptureStatus.SUCCESS,
            load_time_ms=1500.0
        )
        
        # Add some data
        request = RequestLog(
            url="https://example.com/api",
            method="GET",
            resource_type=ResourceType.XHR,
            status=RequestStatus.SUCCESS,
            status_code=200
        )
        result.add_request(request)
        
        cookie = CookieRecord(name="test", domain="example.com", size=10, is_first_party=True)
        result.add_cookie(cookie)
        
        summary = result.export_summary()
        
        assert summary['url'] == "https://example.com"
        assert summary['final_url'] == "https://www.example.com"
        assert summary['title'] == "Example Site"
        assert summary['status'] == CaptureStatus.SUCCESS
        assert summary['load_time_ms'] == 1500.0
        assert summary['total_requests'] == 1
        assert summary['successful_requests'] == 1
        assert summary['total_cookies'] == 1
        assert summary['first_party_cookies'] == 1


if __name__ == "__main__":
    pytest.main([__file__])