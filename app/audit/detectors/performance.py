"""Performance optimization utilities for analytics detectors.

This module provides caching, request filtering, and performance monitoring
to optimize detector performance for high-volume processing.
"""

import hashlib
import re
import time
from collections import defaultdict
from functools import wraps, lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ..models.capture import RequestLog


F = TypeVar('F', bound=Callable[..., Any])


class PerformanceMetrics(BaseModel):
    """Performance metrics for detector operations."""
    
    operation_name: str = Field(description="Name of the operation")
    total_calls: int = Field(default=0, description="Total number of calls")
    total_time_ms: float = Field(default=0.0, description="Total processing time")
    cache_hits: int = Field(default=0, description="Cache hits")
    cache_misses: int = Field(default=0, description="Cache misses") 
    average_time_ms: float = Field(default=0.0, description="Average processing time")
    min_time_ms: float = Field(default=float('inf'), description="Minimum processing time")
    max_time_ms: float = Field(default=0.0, description="Maximum processing time")
    
    def record_operation(self, processing_time_ms: float, cache_hit: bool = False) -> None:
        """Record an operation execution."""
        self.total_calls += 1
        self.total_time_ms += processing_time_ms
        self.min_time_ms = min(self.min_time_ms, processing_time_ms)
        self.max_time_ms = max(self.max_time_ms, processing_time_ms)
        self.average_time_ms = self.total_time_ms / self.total_calls
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total_cacheable = self.cache_hits + self.cache_misses
        return (self.cache_hits / total_cacheable * 100) if total_cacheable > 0 else 0.0


class PerformanceMonitor:
    """Global performance monitor for all detector operations."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.start_time = time.time()
    
    def get_metrics(self, operation_name: str) -> PerformanceMetrics:
        """Get or create metrics for an operation."""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = PerformanceMetrics(operation_name=operation_name)
        return self.metrics[operation_name]
    
    def record_operation(self, operation_name: str, processing_time_ms: float, cache_hit: bool = False) -> None:
        """Record an operation execution."""
        metrics = self.get_metrics(operation_name)
        metrics.record_operation(processing_time_ms, cache_hit)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary for all operations."""
        return {
            "uptime_seconds": time.time() - self.start_time,
            "operations": {name: {
                "total_calls": metrics.total_calls,
                "average_time_ms": round(metrics.average_time_ms, 3),
                "total_time_ms": round(metrics.total_time_ms, 3),
                "cache_hit_rate": round(metrics.cache_hit_rate, 2),
                "min_time_ms": round(metrics.min_time_ms, 3) if metrics.min_time_ms != float('inf') else 0,
                "max_time_ms": round(metrics.max_time_ms, 3)
            } for name, metrics in self.metrics.items()},
            "total_calls": sum(m.total_calls for m in self.metrics.values()),
            "total_time_ms": round(sum(m.total_time_ms for m in self.metrics.values()), 3)
        }


# Global performance monitor
performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str) -> Callable[[F], F]:
    """Decorator to monitor performance of detector operations.
    
    Args:
        operation_name: Name of the operation for metrics
        
    Returns:
        Decorated function with performance monitoring
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            processing_time_ms = (end_time - start_time) * 1000
            performance_monitor.record_operation(operation_name, processing_time_ms)
            
            return result
        return wrapper
    return decorator


class RequestFilter:
    """High-performance request filtering to reduce processing overhead."""
    
    def __init__(self):
        # Pre-compiled patterns for common filtering
        self.static_patterns = {
            'images': re.compile(r'\.(jpg|jpeg|png|gif|svg|ico|webp)(\?|$)', re.IGNORECASE),
            'css': re.compile(r'\.css(\?|$)', re.IGNORECASE),
            'fonts': re.compile(r'\.(woff|woff2|ttf|otf|eot)(\?|$)', re.IGNORECASE),
            'media': re.compile(r'\.(mp4|mp3|avi|mov|wmv|flv)(\?|$)', re.IGNORECASE)
        }
        
        # Domain-based filters
        self.analytics_domains = {
            'google-analytics.com',
            'googletagmanager.com',
            'facebook.com',
            'doubleclick.net',
            'googlesyndication.com',
            'google.com'
        }
        
        # Cache for domain checks
        self.domain_cache: Dict[str, bool] = {}
    
    @lru_cache(maxsize=1000)
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL with caching."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ""
    
    def is_analytics_request(self, request: RequestLog) -> bool:
        """Fast check if request is potentially analytics-related."""
        domain = self.extract_domain(request.url)
        
        # Check cache first
        if domain in self.domain_cache:
            return self.domain_cache[domain]
        
        # Check against known analytics domains
        is_analytics = any(analytics_domain in domain for analytics_domain in self.analytics_domains)
        
        # Cache result
        if len(self.domain_cache) < 10000:  # Prevent unlimited growth
            self.domain_cache[domain] = is_analytics
        
        return is_analytics
    
    def should_process_request(self, request: RequestLog) -> bool:
        """Determine if request should be processed by detectors."""
        url = request.url.lower()
        
        # Skip static resources
        for pattern_name, pattern in self.static_patterns.items():
            if pattern.search(url):
                return False
        
        # Only process analytics-related requests
        return self.is_analytics_request(request)
    
    def filter_requests(self, requests: List[RequestLog]) -> List[RequestLog]:
        """Filter requests to only those relevant for analytics detection."""
        return [req for req in requests if self.should_process_request(req)]


class PatternCache:
    """High-performance pattern matching with intelligent caching."""
    
    def __init__(self, max_cache_size: int = 5000):
        self.max_cache_size = max_cache_size
        self.pattern_cache: Dict[str, bool] = {}
        self.compiled_patterns: Dict[str, re.Pattern] = {}
        
        # Pattern hit counters for cache eviction
        self.pattern_hits: Dict[str, int] = defaultdict(int)
    
    def compile_pattern(self, pattern: str) -> re.Pattern:
        """Compile regex pattern with caching."""
        if pattern not in self.compiled_patterns:
            try:
                self.compiled_patterns[pattern] = re.compile(pattern)
            except re.error:
                # Fall back to literal matching for invalid regex
                self.compiled_patterns[pattern] = re.compile(re.escape(pattern))
        
        return self.compiled_patterns[pattern]
    
    @lru_cache(maxsize=2000)
    def normalize_url_for_matching(self, url: str) -> str:
        """Normalize URL for consistent pattern matching."""
        # Remove query parameters for caching purposes
        if '?' in url:
            url = url.split('?')[0]
        
        # Convert to lowercase for case-insensitive matching
        return url.lower()
    
    def match_pattern(self, url: str, pattern: str) -> bool:
        """Match URL against pattern with caching."""
        # Create cache key
        normalized_url = self.normalize_url_for_matching(url)
        cache_key = f"{normalized_url}::{pattern}"
        
        # Check cache
        if cache_key in self.pattern_cache:
            self.pattern_hits[cache_key] += 1
            performance_monitor.record_operation("pattern_match", 0.1, cache_hit=True)
            return self.pattern_cache[cache_key]
        
        # Perform pattern matching
        start_time = time.perf_counter()
        compiled_pattern = self.compile_pattern(pattern)
        
        try:
            result = bool(compiled_pattern.search(url))
        except:
            result = False
        
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        
        # Cache result with LRU eviction
        if len(self.pattern_cache) >= self.max_cache_size:
            self._evict_least_used()
        
        self.pattern_cache[cache_key] = result
        self.pattern_hits[cache_key] = 1
        
        performance_monitor.record_operation("pattern_match", processing_time_ms, cache_hit=False)
        return result
    
    def _evict_least_used(self) -> None:
        """Evict least-used patterns from cache."""
        if not self.pattern_hits:
            return
        
        # Find least-used patterns
        sorted_patterns = sorted(self.pattern_hits.items(), key=lambda x: x[1])
        patterns_to_remove = sorted_patterns[:len(sorted_patterns) // 4]  # Remove bottom 25%
        
        for pattern_key, _ in patterns_to_remove:
            self.pattern_cache.pop(pattern_key, None)
            self.pattern_hits.pop(pattern_key, None)


class ParameterCache:
    """Cache for parameter extraction results."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
    
    def _get_cache_key(self, request_body: str, content_type: str) -> str:
        """Generate cache key for request data."""
        # Use hash of body + content type for caching
        combined = f"{content_type}::{request_body[:500]}"  # Limit for performance
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_parameters(self, request_body: str, content_type: str) -> Optional[Dict[str, Any]]:
        """Get cached parameter extraction result."""
        cache_key = self._get_cache_key(request_body, content_type)
        
        if cache_key in self.cache:
            self.access_times[cache_key] = time.time()
            performance_monitor.record_operation("parameter_extraction", 0.1, cache_hit=True)
            return self.cache[cache_key]
        
        return None
    
    def cache_parameters(self, request_body: str, content_type: str, parameters: Dict[str, Any]) -> None:
        """Cache parameter extraction result."""
        cache_key = self._get_cache_key(request_body, content_type)
        
        # Evict old entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[cache_key] = parameters
        self.access_times[cache_key] = time.time()
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entries."""
        if not self.access_times:
            return
        
        # Remove oldest 25% of entries
        sorted_entries = sorted(self.access_times.items(), key=lambda x: x[1])
        entries_to_remove = sorted_entries[:len(sorted_entries) // 4]
        
        for cache_key, _ in entries_to_remove:
            self.cache.pop(cache_key, None)
            self.access_times.pop(cache_key, None)


# Global performance utilities
request_filter = RequestFilter()
pattern_cache = PatternCache()
parameter_cache = ParameterCache()


def optimized_pattern_match(url: str, pattern: str) -> bool:
    """Optimized pattern matching with caching."""
    return pattern_cache.match_pattern(url, pattern)


def filter_requests_for_processing(requests: List[RequestLog]) -> List[RequestLog]:
    """Filter requests to only those relevant for analytics processing."""
    return request_filter.filter_requests(requests)


class BatchProcessor:
    """Batch processing utilities for high-throughput detection."""
    
    @staticmethod
    def batch_process(items: List[Any], 
                     processor: Callable[[Any], Any],
                     batch_size: int = 50,
                     max_workers: Optional[int] = None) -> List[Any]:
        """Process items in batches for better performance.
        
        Args:
            items: Items to process
            processor: Function to process each item
            batch_size: Size of each batch
            max_workers: Maximum worker threads (None = CPU count)
            
        Returns:
            List of processed results
        """
        if not items:
            return []
        
        # For CPU-bound tasks, use threading is not beneficial
        # Use simple batch processing for memory efficiency
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = [processor(item) for item in batch]
            results.extend(batch_results)
        
        return results


def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary."""
    return performance_monitor.get_summary()


def reset_performance_metrics() -> None:
    """Reset all performance metrics."""
    global performance_monitor
    performance_monitor = PerformanceMonitor()
    
    # Clear caches
    pattern_cache.pattern_cache.clear()
    pattern_cache.pattern_hits.clear()
    parameter_cache.cache.clear()
    parameter_cache.access_times.clear()
    request_filter.domain_cache.clear()