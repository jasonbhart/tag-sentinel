"""Efficient indexing system for audit data to enable fast rule evaluation.

This module provides comprehensive indexing of audit data (requests, cookies, events, 
timelines, and page metadata) to support linear-time rule evaluation with complex 
filtering and aggregation queries. Performance optimized for large-scale processing.
"""

import re
import sys
import bisect
import hashlib
import logging
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache, wraps
from typing import Dict, List, Any, Optional, Set, Union, Tuple, Callable, Iterator
from urllib.parse import urlparse

from pydantic import BaseModel, Field

# Import audit data models
from ..models.capture import PageResult, RequestLog, CookieRecord, ConsoleLog
from ..detectors.base import TagEvent, DetectorNote


# Performance optimization utilities

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Performance monitoring for indexing operations."""
    
    def __init__(self):
        self.metrics = {
            'index_build_time': 0.0,
            'query_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_usage': 0,
            'indexed_items': 0
        }
    
    def record_index_time(self, time_ms: float):
        """Record index building time."""
        self.metrics['index_build_time'] = time_ms
    
    def record_query_time(self, time_ms: float):
        """Record query execution time."""
        self.metrics['query_times'].append(time_ms)
    
    def record_cache_hit(self):
        """Record cache hit."""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        self.metrics['cache_misses'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        query_times = self.metrics['query_times']
        return {
            'index_build_time_ms': self.metrics['index_build_time'],
            'total_queries': len(query_times),
            'avg_query_time_ms': sum(query_times) / len(query_times) if query_times else 0,
            'max_query_time_ms': max(query_times) if query_times else 0,
            'cache_hit_ratio': (
                self.metrics['cache_hits'] / 
                (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
            ),
            'indexed_items': self.metrics['indexed_items'],
            'memory_usage_mb': self.metrics['memory_usage'] / (1024 * 1024)
        }


def cache_query(max_size: int = 256):
    """Decorator to cache query results."""
    def decorator(func):
        @lru_cache(maxsize=max_size)
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert unhashable types to strings for caching
            cache_args = []
            for arg in args:
                if hasattr(arg, '__dict__'):
                    cache_args.append(str(hash(str(arg.__dict__))))
                else:
                    cache_args.append(arg)
            
            cache_kwargs = {}
            for key, value in kwargs.items():
                if hasattr(value, '__dict__'):
                    cache_kwargs[key] = str(hash(str(value.__dict__)))
                else:
                    cache_kwargs[key] = value
            
            return func(*cache_args, **cache_kwargs)
        return wrapper
    return decorator


class IndexStats(BaseModel):
    """Statistics about index performance and usage."""
    
    build_time_ms: float = Field(description="Time to build indexes")
    memory_usage_bytes: int = Field(description="Estimated memory usage")
    
    # Item counts
    total_requests: int = Field(default=0)
    total_cookies: int = Field(default=0)
    total_events: int = Field(default=0)
    total_pages: int = Field(default=0)
    
    # Index sizes
    request_index_size: int = Field(default=0)
    cookie_index_size: int = Field(default=0)
    event_index_size: int = Field(default=0)
    
    # Query performance
    avg_query_time_ms: float = Field(default=0.0)
    cache_hit_ratio: float = Field(default=0.0)


class OptimizedFilter:
    """High-performance filter with pre-compiled patterns."""
    
    def __init__(self, field: str, operator: str, value: Any):
        self.field = field
        self.operator = operator
        self.value = value
        self._compiled_regex = None
        self._value_set = None
        
        # Pre-compile regex patterns
        if operator in ('regex', 'matches'):
            self._compiled_regex = re.compile(str(value))
        
        # Convert lists to sets for O(1) lookup
        elif operator in ('in', 'not_in') and isinstance(value, (list, tuple)):
            self._value_set = set(value)
    
    def apply(self, item: Any) -> bool:
        """Apply filter to item with optimized operations."""
        try:
            field_value = getattr(item, self.field) if hasattr(item, self.field) else None
            
            if field_value is None:
                return self.operator in ('is_null', 'ne')
            
            # Optimized operations
            if self.operator == 'eq':
                return field_value == self.value
            elif self.operator == 'ne':
                return field_value != self.value
            elif self.operator == 'gt':
                return field_value > self.value
            elif self.operator == 'gte':
                return field_value >= self.value
            elif self.operator == 'lt':
                return field_value < self.value
            elif self.operator == 'lte':
                return field_value <= self.value
            elif self.operator == 'in':
                return field_value in (self._value_set or self.value)
            elif self.operator == 'not_in':
                return field_value not in (self._value_set or self.value)
            elif self.operator == 'contains':
                return self.value in str(field_value)
            elif self.operator == 'startswith':
                return str(field_value).startswith(str(self.value))
            elif self.operator == 'endswith':
                return str(field_value).endswith(str(self.value))
            elif self.operator in ('regex', 'matches'):
                return bool(self._compiled_regex.search(str(field_value)))
            elif self.operator == 'is_null':
                return field_value is None
            elif self.operator == 'not_null':
                return field_value is not None
            else:
                return False
                
        except (AttributeError, TypeError, ValueError):
            return False


class BatchProcessor:
    """Batch processor for efficient bulk operations."""
    
    def __init__(self, batch_size: int = 1000, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_batches(
        self, 
        items: List[Any], 
        processor_func: Callable[[List[Any]], Any],
        parallel: bool = True
    ) -> List[Any]:
        """Process items in batches."""
        if not items:
            return []
        
        # Split into batches
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        if not parallel or len(batches) == 1:
            # Sequential processing
            results = []
            for batch in batches:
                result = processor_func(batch)
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            return results
        
        # Parallel processing
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(processor_func, batch): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    result = future.result()
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}")
        
        return results


class TimelineEntry(BaseModel):
    """Single entry in a page timeline."""
    
    timestamp: datetime = Field(description="When the event occurred")
    event_type: str = Field(description="Type of event (request, cookie, tag_event, etc.)")
    source_id: str = Field(description="ID or reference to the source object")
    summary: str = Field(description="Brief description of the event")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event metadata"
    )
    
    @property
    def timestamp_ms(self) -> int:
        """Get timestamp as milliseconds since epoch."""
        return int(self.timestamp.timestamp() * 1000)


class RequestIndex(BaseModel):
    """Index structure for efficient request lookups."""
    
    # Core indexes
    by_page: Dict[str, List[RequestLog]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Requests grouped by page URL"
    )
    by_host: Dict[str, List[RequestLog]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Requests grouped by host"
    )
    by_resource_type: Dict[str, List[RequestLog]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Requests grouped by resource type"
    )
    by_status: Dict[str, List[RequestLog]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Requests grouped by status"
    )
    
    # Time-based indexes
    chronological: List[RequestLog] = Field(
        default_factory=list,
        description="All requests in chronological order"
    )
    
    @property
    def all_requests(self) -> List[RequestLog]:
        """Get all requests (alias for chronological for API compatibility)."""
        return self.chronological
    
    # Analytics-specific indexes  
    by_vendor: Dict[str, List[RequestLog]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Analytics requests grouped by vendor (ga4, gtm, etc.)"
    )
    by_pattern: Dict[str, List[RequestLog]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Requests grouped by URL patterns"
    )
    
    # Statistics
    total_count: int = Field(default=0, description="Total number of requests")
    successful_count: int = Field(default=0, description="Number of successful requests")
    failed_count: int = Field(default=0, description="Number of failed requests")


class CookieIndex(BaseModel):
    """Index structure for efficient cookie lookups."""
    
    # Core indexes
    by_page: Dict[str, List[CookieRecord]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Cookies grouped by page URL"
    )
    by_domain: Dict[str, List[CookieRecord]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Cookies grouped by domain"
    )
    by_name: Dict[str, List[CookieRecord]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Cookies grouped by name"
    )
    
    # Classification indexes
    first_party: List[CookieRecord] = Field(
        default_factory=list,
        description="First-party cookies only"
    )
    third_party: List[CookieRecord] = Field(
        default_factory=list,
        description="Third-party cookies only"
    )
    session_cookies: List[CookieRecord] = Field(
        default_factory=list,
        description="Session cookies (no expiration)"
    )
    persistent_cookies: List[CookieRecord] = Field(
        default_factory=list,
        description="Persistent cookies (with expiration)"
    )
    
    # Security attribute indexes
    secure_cookies: List[CookieRecord] = Field(
        default_factory=list,
        description="Cookies with Secure flag"
    )
    http_only_cookies: List[CookieRecord] = Field(
        default_factory=list,
        description="Cookies with HttpOnly flag"
    )
    same_site_cookies: Dict[str, List[CookieRecord]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Cookies grouped by SameSite attribute"
    )
    
    # Time-based indexes
    chronological: List[CookieRecord] = Field(
        default_factory=list,
        description="All cookies in chronological order"
    )
    
    @property
    def all_cookies(self) -> List[CookieRecord]:
        """Get all cookies (alias for chronological for API compatibility)."""
        return self.chronological
    
    # Statistics
    total_count: int = Field(default=0, description="Total number of cookies")
    first_party_count: int = Field(default=0, description="Number of first-party cookies")
    third_party_count: int = Field(default=0, description="Number of third-party cookies")


class EventIndex(BaseModel):
    """Index structure for efficient tag event lookups."""
    
    # Core indexes
    by_page: Dict[str, List[TagEvent]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Events grouped by page URL"
    )
    by_vendor: Dict[str, List[TagEvent]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Events grouped by vendor"
    )
    by_name: Dict[str, List[TagEvent]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Events grouped by event name"
    )
    by_category: Dict[str, List[TagEvent]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Events grouped by category"
    )
    
    # ID-based indexes
    by_measurement_id: Dict[str, List[TagEvent]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Events grouped by measurement/tracking ID"
    )
    
    # Time-based indexes
    chronological: List[TagEvent] = Field(
        default_factory=list,
        description="All events in chronological order"
    )
    
    @property
    def all_events(self) -> List[TagEvent]:
        """Get all events (alias for chronological for API compatibility)."""
        return self.chronological
    
    # Status indexes
    by_status: Dict[str, List[TagEvent]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Events grouped by status"
    )
    by_confidence: Dict[str, List[TagEvent]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Events grouped by confidence level"
    )
    
    # Statistics
    total_count: int = Field(default=0, description="Total number of events")
    by_vendor_count: Dict[str, int] = Field(
        default_factory=dict,
        description="Event counts by vendor"
    )


class PageIndex(BaseModel):
    """Index structure for page-level metadata."""
    
    # Page metadata
    pages: List[PageResult] = Field(
        default_factory=list,
        description="All page results"
    )
    by_url: Dict[str, PageResult] = Field(
        default_factory=dict,
        description="Pages indexed by URL"
    )
    by_host: Dict[str, List[PageResult]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Pages grouped by host"
    )
    by_status: Dict[str, List[PageResult]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Pages grouped by capture status"
    )
    
    # Timeline data
    timelines: Dict[str, List[TimelineEntry]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Timeline entries for each page"
    )
    
    # Statistics
    total_pages: int = Field(default=0, description="Total number of pages")
    successful_pages: int = Field(default=0, description="Successfully captured pages")
    failed_pages: int = Field(default=0, description="Failed page captures")


class RunSummary(BaseModel):
    """Summary statistics for an entire audit run."""
    
    run_id: Optional[str] = Field(default=None, description="Unique run identifier")
    start_time: Optional[datetime] = Field(default=None, description="Run start time")
    end_time: Optional[datetime] = Field(default=None, description="Run end time")
    
    # Page statistics
    total_pages: int = Field(default=0, description="Total pages processed")
    successful_pages: int = Field(default=0, description="Successfully processed pages")
    failed_pages: int = Field(default=0, description="Failed page processes")
    
    # Request statistics
    total_requests: int = Field(default=0, description="Total network requests")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")
    
    # Cookie statistics
    total_cookies: int = Field(default=0, description="Total cookies captured")
    first_party_cookies: int = Field(default=0, description="First-party cookies")
    third_party_cookies: int = Field(default=0, description="Third-party cookies")
    
    # Event statistics
    total_events: int = Field(default=0, description="Total tag events detected")
    events_by_vendor: Dict[str, int] = Field(
        default_factory=dict,
        description="Event counts by vendor"
    )
    
    # Performance metrics
    average_page_load_time: Optional[float] = Field(
        default=None,
        description="Average page load time in milliseconds"
    )
    total_data_captured: int = Field(
        default=0,
        description="Total data captured in bytes"
    )
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate run duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate page success rate as percentage."""
        if self.total_pages == 0:
            return 0.0
        return (self.successful_pages / self.total_pages) * 100.0


class AuditIndexes(BaseModel):
    """Complete set of indexes for an audit run with performance optimizations."""
    
    requests: RequestIndex = Field(
        default_factory=RequestIndex,
        description="Request index"
    )
    cookies: CookieIndex = Field(
        default_factory=CookieIndex,
        description="Cookie index"
    )
    events: EventIndex = Field(
        default_factory=EventIndex,
        description="Tag event index"
    )
    pages: PageIndex = Field(
        default_factory=PageIndex,
        description="Page index"
    )
    summary: RunSummary = Field(
        default_factory=RunSummary,
        description="Run-level summary"
    )
    
    # Performance monitoring
    stats: Optional[IndexStats] = Field(
        default=None,
        description="Index performance statistics"
    )
    _performance_monitor: Optional[PerformanceMonitor] = None
    
    # Index metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When indexes were created"
    )
    total_index_size: int = Field(
        default=0,
        description="Total memory usage estimate in bytes"
    )
    
    def initialize_performance_monitor(self):
        """Initialize performance monitoring."""
        if not self._performance_monitor:
            self._performance_monitor = PerformanceMonitor()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self._performance_monitor:
            return self._performance_monitor.get_stats()
        return {}
    
    @cache_query(max_size=128)
    def get_requests_by_domain_cached(self, domain: str) -> List[RequestLog]:
        """Get requests by domain with caching."""
        domain_requests = []
        for host, requests in self.requests.by_host.items():
            if domain in host:
                domain_requests.extend(requests)
        return domain_requests
    
    @cache_query(max_size=64)
    def get_cookies_by_classification_cached(self, classification: str) -> List[CookieRecord]:
        """Get cookies by classification with caching."""
        if hasattr(self.cookies, 'by_classification'):
            return self.cookies.by_classification.get(classification, [])
        return []
    
    def optimize_memory_usage(self):
        """Optimize memory usage by cleaning up redundant data."""
        # Remove duplicate references
        seen_requests = set()
        for page_url, requests in list(self.requests.by_page.items()):
            filtered_requests = []
            for req in requests:
                req_key = (req.url, req.method, req.timestamp)
                if req_key not in seen_requests:
                    seen_requests.add(req_key)
                    filtered_requests.append(req)
            self.requests.by_page[page_url] = filtered_requests
        
        # Compact string storage by interning common strings
        common_domains = Counter()
        for requests in self.requests.by_page.values():
            for req in requests:
                if req.url:
                    domain = urlparse(req.url).netloc
                    common_domains[domain] += 1
        
        # Intern frequently used domains
        for domain, count in common_domains.most_common(50):
            if count > 5:  # Only intern domains used more than 5 times
                sys.intern(domain)
    
    def get_memory_usage_estimate(self) -> int:
        """Estimate memory usage in bytes."""
        import sys
        
        total_size = 0
        
        # Estimate request index size
        for requests in self.requests.by_page.values():
            for req in requests:
                total_size += sys.getsizeof(req.url or "")
                total_size += sys.getsizeof(req.method or "")
                total_size += 100  # Approximate overhead per request
        
        # Estimate cookie index size  
        for cookies in self.cookies.by_page.values():
            for cookie in cookies:
                total_size += sys.getsizeof(cookie.name or "")
                total_size += sys.getsizeof(cookie.value or "")
                total_size += 80  # Approximate overhead per cookie
        
        # Estimate event index size
        for events in self.events.by_page.values():
            total_size += len(events) * 150  # Approximate size per event
        
        self.total_index_size = total_size
        return total_size


class IndexBuilder:
    """Builds efficient indexes from audit data for rule evaluation."""
    
    def __init__(self):
        self._vendor_patterns = {
            'ga4': [
                re.compile(r'google-analytics\.com/g/collect'),
                re.compile(r'googletagmanager\.com.*GA_MEASUREMENT_ID'),
            ],
            'gtm': [
                re.compile(r'googletagmanager\.com/gtm\.js'),
                re.compile(r'googletagmanager\.com.*GTM-'),
            ],
            'facebook': [
                re.compile(r'facebook\.net.*tr'),
                re.compile(r'facebook\.com.*tr'),
            ]
        }
    
    def build(self, run_data: List[PageResult], 
              tag_events: Optional[List[TagEvent]] = None,
              run_id: Optional[str] = None) -> AuditIndexes:
        """Build comprehensive indexes from audit run data.
        
        Args:
            run_data: List of PageResult objects from audit run
            tag_events: Optional list of detected tag events
            run_id: Optional run identifier
            
        Returns:
            Complete AuditIndexes with all data indexed
        """
        indexes = AuditIndexes()
        
        # Set run metadata
        if run_id:
            indexes.summary.run_id = run_id
        
        if run_data:
            indexes.summary.start_time = min(page.capture_time for page in run_data)
            indexes.summary.end_time = max(page.capture_time for page in run_data)
        
        # Build indexes
        self._build_page_indexes(run_data, indexes)
        self._build_request_indexes(run_data, indexes)
        self._build_cookie_indexes(run_data, indexes)
        
        if tag_events:
            self._build_event_indexes(tag_events, indexes)
        
        # Build timelines
        self._build_timelines(run_data, tag_events or [], indexes)
        
        # Calculate summary statistics
        self._calculate_summary_stats(indexes)
        
        # Estimate memory usage
        self._estimate_index_size(indexes)
        
        return indexes
    
    def _build_page_indexes(self, run_data: List[PageResult], indexes: AuditIndexes) -> None:
        """Build page-level indexes."""
        for page in run_data:
            # Add to main collections
            indexes.pages.pages.append(page)
            indexes.pages.by_url[page.url] = page
            
            # Group by host
            host = urlparse(page.url).netloc
            indexes.pages.by_host[host].append(page)
            
            # Group by status
            indexes.pages.by_status[page.capture_status.value].append(page)
        
        # Update statistics
        indexes.pages.total_pages = len(run_data)
        indexes.pages.successful_pages = len(indexes.pages.by_status.get('success', []))
        indexes.pages.failed_pages = len(run_data) - indexes.pages.successful_pages
    
    def _build_request_indexes(self, run_data: List[PageResult], indexes: AuditIndexes) -> None:
        """Build request indexes across all pages."""
        all_requests = []
        
        for page in run_data:
            for request in page.network_requests:
                all_requests.append(request)
                
                # Index by page
                indexes.requests.by_page[page.url].append(request)
                
                # Index by host
                host = request.host
                indexes.requests.by_host[host].append(request)
                
                # Index by resource type
                indexes.requests.by_resource_type[request.resource_type.value].append(request)
                
                # Index by status
                indexes.requests.by_status[request.status.value].append(request)
                
                # Index by vendor (analytics detection)
                vendor = self._detect_request_vendor(request)
                if vendor:
                    indexes.requests.by_vendor[vendor].append(request)
        
        # Sort chronologically
        all_requests.sort(key=lambda r: r.start_time)
        indexes.requests.chronological = all_requests
        
        # Update statistics
        indexes.requests.total_count = len(all_requests)
        indexes.requests.successful_count = len([r for r in all_requests if r.is_successful])
        indexes.requests.failed_count = len(all_requests) - indexes.requests.successful_count
    
    def _build_cookie_indexes(self, run_data: List[PageResult], indexes: AuditIndexes) -> None:
        """Build cookie indexes across all pages."""
        all_cookies = []
        
        for page in run_data:
            for cookie in page.cookies:
                all_cookies.append(cookie)
                
                # Index by page
                indexes.cookies.by_page[page.url].append(cookie)
                
                # Index by domain and name
                indexes.cookies.by_domain[cookie.domain].append(cookie)
                indexes.cookies.by_name[cookie.name].append(cookie)
                
                # Classification indexes
                if cookie.is_first_party:
                    indexes.cookies.first_party.append(cookie)
                else:
                    indexes.cookies.third_party.append(cookie)
                
                if cookie.is_session:
                    indexes.cookies.session_cookies.append(cookie)
                else:
                    indexes.cookies.persistent_cookies.append(cookie)
                
                # Security attribute indexes
                if cookie.secure:
                    indexes.cookies.secure_cookies.append(cookie)
                
                if cookie.http_only:
                    indexes.cookies.http_only_cookies.append(cookie)
                
                if cookie.same_site:
                    indexes.cookies.same_site_cookies[cookie.same_site].append(cookie)
        
        # Update statistics
        indexes.cookies.total_count = len(all_cookies)
        indexes.cookies.first_party_count = len(indexes.cookies.first_party)
        indexes.cookies.third_party_count = len(indexes.cookies.third_party)
    
    def _build_event_indexes(self, tag_events: List[TagEvent], indexes: AuditIndexes) -> None:
        """Build tag event indexes."""
        for event in tag_events:
            # Index by page
            indexes.events.by_page[event.page_url].append(event)
            
            # Index by vendor
            indexes.events.by_vendor[event.vendor.value].append(event)
            
            # Index by name and category
            indexes.events.by_name[event.name].append(event)
            if event.category:
                indexes.events.by_category[event.category].append(event)
            
            # Index by measurement ID
            if event.id:
                indexes.events.by_measurement_id[event.id].append(event)
            
            # Index by status and confidence
            indexes.events.by_status[event.status.value].append(event)
            indexes.events.by_confidence[event.confidence.value].append(event)
        
        # Sort chronologically
        tag_events_sorted = sorted(tag_events, key=lambda e: e.detected_at)
        indexes.events.chronological = tag_events_sorted
        
        # Update statistics
        indexes.events.total_count = len(tag_events)
        for event in tag_events:
            vendor = event.vendor.value
            indexes.events.by_vendor_count[vendor] = indexes.events.by_vendor_count.get(vendor, 0) + 1
    
    def _build_timelines(self, run_data: List[PageResult], 
                        tag_events: List[TagEvent], indexes: AuditIndexes) -> None:
        """Build chronological timelines for each page."""
        for page in run_data:
            timeline = []
            
            # Add request events to timeline
            for request in page.network_requests:
                timeline.append(TimelineEntry(
                    timestamp=request.start_time,
                    event_type="request",
                    source_id=request.url,
                    summary=f"{request.method} {request.url} ({request.resource_type.value})",
                    metadata={
                        "status": request.status.value,
                        "resource_type": request.resource_type.value,
                        "host": request.host
                    }
                ))
            
            # Add cookie events to timeline
            for cookie in page.cookies:
                timeline.append(TimelineEntry(
                    timestamp=page.capture_time,  # Use page capture time as approximation
                    event_type="cookie",
                    source_id=f"{cookie.domain}:{cookie.name}",
                    summary=f"Cookie set: {cookie.name} (domain: {cookie.domain})",
                    metadata={
                        "domain": cookie.domain,
                        "is_first_party": cookie.is_first_party,
                        "secure": cookie.secure,
                        "http_only": cookie.http_only
                    }
                ))
            
            # Add tag events to timeline
            page_events = [e for e in tag_events if e.page_url == page.url]
            for event in page_events:
                timeline.append(TimelineEntry(
                    timestamp=event.detected_at,
                    event_type="tag_event",
                    source_id=f"{event.vendor.value}:{event.name}",
                    summary=f"{event.vendor.value} event: {event.name}",
                    metadata={
                        "vendor": event.vendor.value,
                        "status": event.status.value,
                        "confidence": event.confidence.value,
                        "measurement_id": event.id
                    }
                ))
            
            # Sort timeline chronologically
            timeline.sort(key=lambda t: t.timestamp)
            indexes.pages.timelines[page.url] = timeline
    
    def _detect_request_vendor(self, request: RequestLog) -> Optional[str]:
        """Detect analytics vendor from request URL."""
        url = request.url.lower()
        
        for vendor, patterns in self._vendor_patterns.items():
            for pattern in patterns:
                if pattern.search(url):
                    return vendor
        
        return None
    
    def _calculate_summary_stats(self, indexes: AuditIndexes) -> None:
        """Calculate run-level summary statistics."""
        # Page statistics
        indexes.summary.total_pages = indexes.pages.total_pages
        indexes.summary.successful_pages = indexes.pages.successful_pages
        indexes.summary.failed_pages = indexes.pages.failed_pages
        
        # Request statistics
        indexes.summary.total_requests = indexes.requests.total_count
        indexes.summary.successful_requests = indexes.requests.successful_count
        indexes.summary.failed_requests = indexes.requests.failed_count
        
        # Cookie statistics
        indexes.summary.total_cookies = indexes.cookies.total_count
        indexes.summary.first_party_cookies = indexes.cookies.first_party_count
        indexes.summary.third_party_cookies = indexes.cookies.third_party_count
        
        # Event statistics
        indexes.summary.total_events = indexes.events.total_count
        indexes.summary.events_by_vendor = indexes.events.by_vendor_count.copy()
        
        # Performance metrics
        successful_pages = [p for p in indexes.pages.pages if p.is_successful and p.load_time_ms]
        if successful_pages:
            avg_load_time = sum(p.load_time_ms for p in successful_pages) / len(successful_pages)
            indexes.summary.average_page_load_time = avg_load_time
    
    def _estimate_index_size(self, indexes: AuditIndexes) -> None:
        """Estimate total memory usage of indexes."""
        # This is a rough estimation - in practice you'd want more precise measurements
        size = 0
        
        # Estimate based on counts and typical object sizes
        size += indexes.requests.total_count * 500  # ~500 bytes per request
        size += indexes.cookies.total_count * 200   # ~200 bytes per cookie
        size += indexes.events.total_count * 300    # ~300 bytes per event
        size += indexes.pages.total_pages * 100     # ~100 bytes per page entry
        
        # Add index overhead (hash tables, etc.)
        size += len(indexes.requests.by_page) * 50
        size += len(indexes.cookies.by_domain) * 50
        size += len(indexes.events.by_vendor) * 50
        
        indexes.total_index_size = size


def build_audit_indexes(page_results: List[PageResult], tag_events: Optional[List[TagEvent]] = None) -> AuditIndexes:
    """Build comprehensive indexes for a collection of page results.
    
    Args:
        page_results: List of page capture results to index
        tag_events: Optional list of tag events from detectors
        
    Returns:
        Complete audit indexes for efficient querying
    """
    builder = IndexBuilder()
    return builder.build(page_results, tag_events)


def build_page_index(page_result: PageResult, tag_events: Optional[List[TagEvent]] = None) -> PageIndex:
    """Build index for a single page result.
    
    Args:
        page_result: Page capture result to index
        tag_events: Optional tag events for this page
        
    Returns:
        Page index with timeline and metadata
    """
    # Create a simple page index with just this page
    page_index = PageIndex(pages=[page_result])
    return page_index


def build_run_summary(indexes: AuditIndexes) -> RunSummary:
    """Build summary statistics for an audit run.
    
    Args:
        indexes: Complete audit indexes
        
    Returns:
        Summary statistics for the audit run
    """
    builder = IndexBuilder()
    builder._build_run_summary(indexes)
    return indexes.summary


# Query Interface & Filtering

class QueryFilter(BaseModel):
    """Base query filter for indexed data."""
    
    field: str = Field(description="Field to filter on")
    operator: str = Field(description="Filter operator (eq, ne, gt, lt, in, regex, etc.)")
    value: Any = Field(description="Filter value")
    
    def apply(self, item: Any) -> bool:
        """Apply filter to an item."""
        field_value = self._get_field_value(item, self.field)
        
        if self.operator == "eq":
            return field_value == self.value
        elif self.operator == "ne":
            return field_value != self.value
        elif self.operator == "gt":
            return field_value > self.value
        elif self.operator == "lt":
            return field_value < self.value
        elif self.operator == "gte":
            return field_value >= self.value
        elif self.operator == "lte":
            return field_value <= self.value
        elif self.operator == "in":
            return field_value in self.value
        elif self.operator == "not_in":
            return field_value not in self.value
        elif self.operator == "regex":
            pattern = re.compile(self.value) if isinstance(self.value, str) else self.value
            return bool(pattern.search(str(field_value)))
        elif self.operator == "contains":
            return self.value in str(field_value)
        elif self.operator == "startswith":
            return str(field_value).startswith(self.value)
        elif self.operator == "endswith":
            return str(field_value).endswith(self.value)
        elif self.operator == "exists":
            return field_value is not None
        elif self.operator == "not_exists":
            return field_value is None
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")
    
    @staticmethod
    def _get_field_value(item: Any, field: str) -> Any:
        """Extract field value from item using dot notation."""
        current = item
        for part in field.split('.'):
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current


class QueryBuilder:
    """Fluent query builder for indexed audit data."""
    
    def __init__(self, indexes: AuditIndexes):
        self.indexes = indexes
        self.filters: List[QueryFilter] = []
        self.limit_value: Optional[int] = None
        self.offset_value: int = 0
        self.sort_field: Optional[str] = None
        self.sort_desc: bool = False
    
    def filter(self, field: str, operator: str, value: Any) -> 'QueryBuilder':
        """Add a filter condition."""
        self.filters.append(QueryFilter(field=field, operator=operator, value=value))
        return self
    
    def where(self, field: str, value: Any) -> 'QueryBuilder':
        """Add equality filter (shorthand)."""
        return self.filter(field, "eq", value)
    
    def where_regex(self, field: str, pattern: str) -> 'QueryBuilder':
        """Add regex filter (shorthand)."""
        return self.filter(field, "regex", pattern)
    
    def where_in(self, field: str, values: List[Any]) -> 'QueryBuilder':
        """Add 'in' filter (shorthand)."""
        return self.filter(field, "in", values)
    
    def limit(self, count: int) -> 'QueryBuilder':
        """Limit result count."""
        self.limit_value = count
        return self
    
    def offset(self, count: int) -> 'QueryBuilder':
        """Set result offset."""
        self.offset_value = count
        return self
    
    def sort_by(self, field: str, descending: bool = False) -> 'QueryBuilder':
        """Sort results by field."""
        self.sort_field = field
        self.sort_desc = descending
        return self
    
    def requests(self) -> List[RequestLog]:
        """Query network requests."""
        return self._execute_query(self.indexes.requests.chronological)
    
    def cookies(self) -> List[CookieRecord]:
        """Query cookies."""
        return self._execute_query(self.indexes.cookies.chronological)
    
    def events(self) -> List[TagEvent]:
        """Query tag events."""
        return self._execute_query(self.indexes.events.chronological)
    
    def pages(self) -> List[PageIndex]:
        """Query page indexes."""
        return self._execute_query(self.indexes.pages.pages)
    
    def _execute_query(self, items: List[Any]) -> List[Any]:
        """Execute query on a collection of items."""
        # Apply filters
        filtered_items = items
        for filter_obj in self.filters:
            filtered_items = [item for item in filtered_items if filter_obj.apply(item)]
        
        # Sort if specified
        if self.sort_field:
            def sort_key(item):
                return QueryFilter._get_field_value(item, self.sort_field)
            filtered_items.sort(key=sort_key, reverse=self.sort_desc)
        
        # Apply pagination
        start = self.offset_value
        end = start + self.limit_value if self.limit_value else None
        return filtered_items[start:end]


class QueryAggregator:
    """Aggregation operations for indexed audit data."""
    
    def __init__(self, indexes: AuditIndexes):
        self.indexes = indexes
    
    def count_requests_by_domain(self, filters: Optional[List[QueryFilter]] = None) -> Dict[str, int]:
        """Count requests grouped by domain."""
        requests = self._apply_filters(self.indexes.requests.chronological, filters or [])
        counts = defaultdict(int)
        for req in requests:
            domain = urlparse(req.url).netloc
            counts[domain] += 1
        return dict(counts)
    
    def count_requests_by_status(self, filters: Optional[List[QueryFilter]] = None) -> Dict[str, int]:
        """Count requests grouped by status code ranges."""
        requests = self._apply_filters(self.indexes.requests.chronological, filters or [])
        counts = defaultdict(int)
        for req in requests:
            if req.status_code:
                if 200 <= req.status_code < 300:
                    counts['2xx'] += 1
                elif 300 <= req.status_code < 400:
                    counts['3xx'] += 1
                elif 400 <= req.status_code < 500:
                    counts['4xx'] += 1
                elif 500 <= req.status_code < 600:
                    counts['5xx'] += 1
                else:
                    counts['other'] += 1
            else:
                counts['unknown'] += 1
        return dict(counts)
    
    def count_cookies_by_attributes(self, filters: Optional[List[QueryFilter]] = None) -> Dict[str, int]:
        """Count cookies grouped by security attributes."""
        cookies = self._apply_filters(self.indexes.cookies.chronological, filters or [])
        counts = {
            'secure': 0,
            'http_only': 0,
            'same_site': 0,
            'first_party': 0,
            'third_party': 0,
            'session': 0,
            'persistent': 0
        }
        
        for cookie in cookies:
            if cookie.secure:
                counts['secure'] += 1
            if cookie.http_only:
                counts['http_only'] += 1
            if cookie.same_site:
                counts['same_site'] += 1
            if cookie.is_first_party:
                counts['first_party'] += 1
            else:
                counts['third_party'] += 1
            if cookie.is_session:
                counts['session'] += 1
            else:
                counts['persistent'] += 1
        
        return counts
    
    def count_events_by_vendor(self, filters: Optional[List[QueryFilter]] = None) -> Dict[str, int]:
        """Count tag events grouped by vendor."""
        events = self._apply_filters(self.indexes.events.chronological, filters or [])
        counts = defaultdict(int)
        for event in events:
            counts[event.vendor.value] += 1
        return dict(counts)
    
    def get_timeline_stats(self, filters: Optional[List[QueryFilter]] = None) -> Dict[str, Any]:
        """Get timeline statistics across all pages."""
        pages = self._apply_filters(self.indexes.pages.pages, filters or [])
        
        total_events = 0
        event_types = defaultdict(int)
        avg_timeline_length = 0
        
        for page in pages:
            page_timeline = self.indexes.pages.timelines.get(page.url, [])
            timeline_length = len(page_timeline)
            total_events += timeline_length
            avg_timeline_length += timeline_length
            
            for entry in page_timeline:
                event_types[entry.event_type] += 1
        
        if pages:
            avg_timeline_length /= len(pages)
        
        return {
            'total_events': total_events,
            'average_timeline_length': avg_timeline_length,
            'event_types': dict(event_types),
            'pages_analyzed': len(pages)
        }
    
    def get_performance_stats(self, filters: Optional[List[QueryFilter]] = None) -> Dict[str, Any]:
        """Get performance statistics across pages."""
        pages = self._apply_filters(self.indexes.pages.pages, filters or [])
        
        load_times = [p.load_time_ms for p in pages if p.load_time_ms is not None]
        request_counts = [len(p.network_requests) for p in pages]
        
        if not load_times:
            return {
                'pages_analyzed': len(pages),
                'load_time_stats': None,
                'request_count_stats': {
                    'min': min(request_counts) if request_counts else 0,
                    'max': max(request_counts) if request_counts else 0,
                    'avg': sum(request_counts) / len(request_counts) if request_counts else 0
                }
            }
        
        return {
            'pages_analyzed': len(pages),
            'load_time_stats': {
                'min': min(load_times),
                'max': max(load_times),
                'avg': sum(load_times) / len(load_times),
                'median': sorted(load_times)[len(load_times) // 2]
            },
            'request_count_stats': {
                'min': min(request_counts),
                'max': max(request_counts),
                'avg': sum(request_counts) / len(request_counts)
            }
        }
    
    def _apply_filters(self, items: List[Any], filters: List[QueryFilter]) -> List[Any]:
        """Apply filters to a collection of items."""
        filtered_items = items
        for filter_obj in filters:
            filtered_items = [item for item in filtered_items if filter_obj.apply(item)]
        return filtered_items


class AuditQuery:
    """High-level query interface for audit data."""
    
    def __init__(self, indexes: AuditIndexes):
        self.indexes = indexes
        self.builder = QueryBuilder(indexes)
        self.aggregator = QueryAggregator(indexes)
    
    def query(self) -> QueryBuilder:
        """Start a new query builder."""
        return QueryBuilder(self.indexes)
    
    def aggregate(self) -> QueryAggregator:
        """Get aggregation interface."""
        return self.aggregator
    
    # Convenience methods for common queries
    def get_requests_by_domain(self, domain: str) -> List[RequestLog]:
        """Get all requests for a specific domain."""
        return self.query().where_regex('url', f'https?://{re.escape(domain)}').requests()
    
    def get_failed_requests(self) -> List[RequestLog]:
        """Get all failed requests."""
        from app.audit.models.capture import RequestStatus
        return self.query().filter('status', 'ne', RequestStatus.SUCCESS).requests()
    
    def get_third_party_cookies(self) -> List[CookieRecord]:
        """Get all third-party cookies."""
        return self.query().where('is_first_party', False).cookies()
    
    def get_insecure_cookies(self) -> List[CookieRecord]:
        """Get cookies without secure attributes."""
        return (self.query()
                .where('secure', False)
                .filter('http_only', 'eq', False)
                .cookies())
    
    def get_events_by_vendor(self, vendor: str) -> List[TagEvent]:
        """Get all events from a specific vendor."""
        return self.query().where('vendor', vendor).events()
    
    def get_slow_pages(self, threshold_ms: float = 3000) -> List[PageIndex]:
        """Get pages that loaded slower than threshold."""
        return self.query().filter('load_time_ms', 'gt', threshold_ms).pages()
    
    def get_error_pages(self) -> List[PageIndex]:
        """Get pages with errors."""
        return self.query().filter('has_errors', 'eq', True).pages()
    
    # Direct data access proxy methods for convenience
    def requests(self) -> List[RequestLog]:
        """Get all requests (proxy to query().requests())."""
        return self.query().requests()
    
    def cookies(self) -> List[CookieRecord]:
        """Get all cookies (proxy to query().cookies())."""
        return self.query().cookies()
    
    def events(self) -> List[TagEvent]:
        """Get all events (proxy to query().events())."""
        return self.query().events()
    
    def pages(self) -> List[PageIndex]:
        """Get all pages (proxy to query().pages())."""
        return self.query().pages()