"""Pydantic models for browser capture session data and artifacts.

This module defines the data models used by the Browser Capture Engine,
including network requests, cookies, console logs, and comprehensive page results.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class RequestStatus(str, Enum):
    """Status of network request lifecycle."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ABORTED = "aborted"


class ResourceType(str, Enum):
    """Types of network resources."""
    DOCUMENT = "document"
    STYLESHEET = "stylesheet"
    IMAGE = "image"
    MEDIA = "media"
    FONT = "font"
    SCRIPT = "script"
    TEXTTRACK = "texttrack"
    XHR = "xhr"
    FETCH = "fetch"
    EVENTSOURCE = "eventsource"
    WEBSOCKET = "websocket"
    MANIFEST = "manifest"
    OTHER = "other"


class ConsoleLevel(str, Enum):
    """Console log levels."""
    LOG = "log"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    DEBUG = "debug"
    TRACE = "trace"


class CaptureStatus(str, Enum):
    """Overall status of page capture."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TimingData(BaseModel):
    """Network request timing information."""
    
    dns_start: Optional[float] = Field(
        default=None, 
        description="DNS lookup start time in milliseconds"
    )
    dns_end: Optional[float] = Field(
        default=None, 
        description="DNS lookup end time in milliseconds"
    )
    connect_start: Optional[float] = Field(
        default=None, 
        description="Connection start time in milliseconds"
    )
    connect_end: Optional[float] = Field(
        default=None, 
        description="Connection end time in milliseconds"
    )
    request_start: Optional[float] = Field(
        default=None, 
        description="Request start time in milliseconds"
    )
    response_start: Optional[float] = Field(
        default=None, 
        description="Response start time in milliseconds"
    )
    response_end: Optional[float] = Field(
        default=None, 
        description="Response end time in milliseconds"
    )
    
    @property
    def total_time(self) -> Optional[float]:
        """Calculate total request time in milliseconds."""
        if self.request_start is not None and self.response_end is not None:
            return self.response_end - self.request_start
        return None


class RequestLog(BaseModel):
    """Complete network request lifecycle data."""
    
    # Request identification
    url: str = Field(description="Request URL")
    method: str = Field(description="HTTP method")
    resource_type: ResourceType = Field(description="Type of requested resource")
    
    # Request data
    request_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Request headers"
    )
    request_body: Optional[str] = Field(
        default=None,
        description="Request body (if applicable)"
    )
    
    # Response data
    status_code: Optional[int] = Field(
        default=None,
        description="HTTP status code"
    )
    status_text: Optional[str] = Field(
        default=None,
        description="HTTP status text"
    )
    response_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Response headers"
    )
    response_body: Optional[str] = Field(
        default=None,
        description="Response body (if captured)"
    )
    response_size: Optional[int] = Field(
        default=None,
        description="Response size in bytes"
    )
    
    # Timing and performance
    timing: Optional[TimingData] = Field(
        default=None,
        description="Detailed timing information"
    )
    start_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="Request start timestamp"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Request end timestamp"
    )
    
    # Request lifecycle
    status: RequestStatus = Field(
        default=RequestStatus.PENDING,
        description="Request lifecycle status"
    )
    error_text: Optional[str] = Field(
        default=None,
        description="Error message if request failed"
    )
    
    # Protocol information
    protocol: Optional[str] = Field(
        default=None,
        description="Protocol used (HTTP/1.1, HTTP/2, etc.)"
    )
    remote_address: Optional[str] = Field(
        default=None,
        description="Remote server IP address"
    )
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate URL format."""
        try:
            result = urlparse(v)
            if not result.scheme or not result.netloc:
                raise ValueError("Invalid URL format")
        except Exception as e:
            raise ValueError(f"Invalid URL: {e}")
        return v
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate request duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    @property
    def host(self) -> str:
        """Extract host from URL."""
        return urlparse(self.url).netloc
    
    @property
    def is_successful(self) -> bool:
        """Check if request was successful."""
        # If we have status code information, use it to determine success
        if self.status_code is not None:
            return (
                self.status == RequestStatus.SUCCESS and
                200 <= self.status_code < 400
            )
        # Otherwise, just check if the request finished successfully
        return self.status == RequestStatus.SUCCESS


class CookieRecord(BaseModel):
    """Cookie information with privacy-conscious handling."""
    
    name: str = Field(description="Cookie name")
    value: Optional[str] = Field(
        default=None,
        description="Cookie value (may be redacted for privacy)"
    )
    domain: str = Field(description="Cookie domain")
    path: str = Field(default="/", description="Cookie path")
    
    # Cookie attributes
    expires: Optional[datetime] = Field(
        default=None,
        description="Cookie expiration time"
    )
    max_age: Optional[int] = Field(
        default=None,
        description="Cookie max age in seconds"
    )
    secure: bool = Field(default=False, description="Secure flag")
    http_only: bool = Field(default=False, description="HttpOnly flag")
    same_site: Optional[str] = Field(
        default=None,
        description="SameSite attribute (Strict, Lax, None)"
    )
    
    # Analysis metadata
    size: int = Field(description="Total cookie size in bytes")
    is_first_party: bool = Field(
        default=True,
        description="Whether cookie is first-party"
    )
    is_session: bool = Field(
        default=True,
        description="Whether cookie is session-only"
    )
    value_redacted: bool = Field(
        default=False,
        description="Whether cookie value was redacted"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the cookie"
    )
    
    @classmethod
    def from_playwright_cookie(cls, cookie: dict, page_host: str, redact_value: bool = True):
        """Create CookieRecord from Playwright cookie object."""
        # Determine if cookie is first-party
        cookie_domain = cookie.get('domain', '').lstrip('.')
        is_first_party = (
            cookie_domain == page_host or
            page_host.endswith(f'.{cookie_domain}') or
            cookie_domain.endswith(f'.{page_host}')
        )
        
        # Calculate cookie size
        name_size = len(cookie.get('name', ''))
        value_size = len(cookie.get('value', ''))
        domain_size = len(cookie.get('domain', ''))
        path_size = len(cookie.get('path', '/'))
        total_size = name_size + value_size + domain_size + path_size
        
        # Determine if session cookie
        is_session = cookie.get('expires', -1) == -1
        
        return cls(
            name=cookie.get('name', ''),
            value=None if redact_value else cookie.get('value'),
            domain=cookie.get('domain', ''),
            path=cookie.get('path', '/'),
            expires=datetime.fromtimestamp(cookie['expires']) if cookie.get('expires', -1) != -1 else None,
            secure=cookie.get('secure', False),
            http_only=cookie.get('httpOnly', False),
            same_site=cookie.get('sameSite'),
            size=total_size,
            is_first_party=is_first_party,
            is_session=is_session,
            value_redacted=redact_value
        )


class ConsoleLog(BaseModel):
    """Console event capture."""
    
    level: ConsoleLevel = Field(description="Console log level")
    text: str = Field(description="Console message text")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When console event occurred"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL where console event originated"
    )
    line_number: Optional[int] = Field(
        default=None,
        description="Line number where event occurred"
    )
    column_number: Optional[int] = Field(
        default=None,
        description="Column number where event occurred"
    )
    stack_trace: Optional[str] = Field(
        default=None,
        description="Stack trace for errors"
    )
    
    @classmethod
    def from_playwright_message(cls, message):
        """Create ConsoleLog from Playwright console message."""
        # Map Playwright types to our enum
        level_map = {
            'log': ConsoleLevel.LOG,
            'info': ConsoleLevel.INFO,
            'warn': ConsoleLevel.WARN,
            'error': ConsoleLevel.ERROR,
            'debug': ConsoleLevel.DEBUG,
            'trace': ConsoleLevel.TRACE,
        }
        
        level = level_map.get(message.type, ConsoleLevel.LOG)
        text = message.text
        location = message.location
        
        return cls(
            level=level,
            text=text,
            url=location.get('url') if location else None,
            line_number=location.get('lineNumber') if location else None,
            column_number=location.get('columnNumber') if location else None
        )


class ArtifactPaths(BaseModel):
    """Paths to debug artifacts generated during capture."""
    
    har_file: Optional[Path] = Field(
        default=None,
        description="Path to HAR file"
    )
    screenshot_file: Optional[Path] = Field(
        default=None,
        description="Path to screenshot file"
    )
    trace_file: Optional[Path] = Field(
        default=None,
        description="Path to Playwright trace file"
    )
    page_source: Optional[Path] = Field(
        default=None,
        description="Path to page source HTML"
    )
    
    @property
    def has_artifacts(self) -> bool:
        """Check if any artifacts were generated."""
        return any([
            self.har_file,
            self.screenshot_file,
            self.trace_file,
            self.page_source
        ])


class PageResult(BaseModel):
    """Comprehensive output from page capture session."""
    
    # Page identification
    url: str = Field(description="Page URL that was captured")
    final_url: Optional[str] = Field(
        default=None,
        description="Final URL after redirects"
    )
    title: Optional[str] = Field(
        default=None,
        description="Page title"
    )
    
    # Capture metadata
    capture_status: CaptureStatus = Field(description="Overall capture status")
    capture_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When capture was performed"
    )
    load_time_ms: Optional[float] = Field(
        default=None,
        description="Total page load time in milliseconds"
    )
    
    # Captured data
    network_requests: List[RequestLog] = Field(
        default_factory=list,
        description="All network requests made during page load"
    )
    cookies: List[CookieRecord] = Field(
        default_factory=list,
        description="Cookies set during page session"
    )
    console_logs: List[ConsoleLog] = Field(
        default_factory=list,
        description="Console messages and errors"
    )
    
    # Error information
    page_errors: List[str] = Field(
        default_factory=list,
        description="JavaScript errors that occurred"
    )
    capture_error: Optional[str] = Field(
        default=None,
        description="Error message if capture failed"
    )
    
    # Artifacts
    artifacts: Optional[ArtifactPaths] = Field(
        default=None,
        description="Paths to generated debug artifacts"
    )
    
    # Performance metrics
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional performance and timing metrics"
    )
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate URL format."""
        try:
            result = urlparse(v)
            if not result.scheme or not result.netloc:
                raise ValueError("Invalid URL format")
        except Exception as e:
            raise ValueError(f"Invalid URL: {e}")
        return v
    
    @property
    def successful_requests(self) -> List[RequestLog]:
        """Get only successful network requests."""
        return [req for req in self.network_requests if req.is_successful]
    
    @property
    def failed_requests(self) -> List[RequestLog]:
        """Get only failed network requests."""
        return [req for req in self.network_requests if not req.is_successful]
    
    @property
    def error_count(self) -> int:
        """Total number of errors (console + page + network)."""
        console_errors = len([log for log in self.console_logs if log.level == ConsoleLevel.ERROR])
        network_errors = len(self.failed_requests)
        return console_errors + network_errors + len(self.page_errors)
    
    @property
    def first_party_cookies(self) -> List[CookieRecord]:
        """Get only first-party cookies."""
        return [cookie for cookie in self.cookies if cookie.is_first_party]
    
    @property
    def third_party_cookies(self) -> List[CookieRecord]:
        """Get only third-party cookies."""
        return [cookie for cookie in self.cookies if not cookie.is_first_party]
    
    @property
    def is_successful(self) -> bool:
        """Check if overall capture was successful."""
        return self.capture_status == CaptureStatus.SUCCESS
    
    def add_request(self, request: RequestLog) -> None:
        """Add a network request to the result."""
        self.network_requests.append(request)
    
    def add_cookie(self, cookie: CookieRecord) -> None:
        """Add a cookie to the result."""
        self.cookies.append(cookie)
    
    def add_console_log(self, log: ConsoleLog) -> None:
        """Add a console log to the result."""
        self.console_logs.append(log)
    
    def add_page_error(self, error: str) -> None:
        """Add a page error to the result."""
        self.page_errors.append(error)
    
    def set_artifacts(self, artifacts: ArtifactPaths) -> None:
        """Set artifact paths."""
        self.artifacts = artifacts
    
    def export_summary(self) -> Dict[str, Any]:
        """Export a summary of the page result for reporting."""
        return {
            "url": self.url,
            "final_url": self.final_url,
            "title": self.title,
            "status": self.capture_status,
            "capture_time": self.capture_time.isoformat(),
            "load_time_ms": self.load_time_ms,
            "total_requests": len(self.network_requests),
            "successful_requests": len(self.successful_requests),
            "failed_requests": len(self.failed_requests),
            "total_cookies": len(self.cookies),
            "first_party_cookies": len(self.first_party_cookies),
            "third_party_cookies": len(self.third_party_cookies),
            "console_logs": len(self.console_logs),
            "page_errors": len(self.page_errors),
            "error_count": self.error_count,
            "has_artifacts": self.artifacts.has_artifacts if self.artifacts else False
        }