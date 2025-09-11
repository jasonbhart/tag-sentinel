"""Pydantic models for crawl configuration and output data structures.

This module defines the data models used throughout the crawling system,
including configuration validation, page plans, and metrics tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator, AnyHttpUrl
import re


class DiscoveryMode(str, Enum):
    """URL discovery modes supported by the crawler."""
    SEEDS = "seeds"           # Only crawl explicitly provided seed URLs
    SITEMAP = "sitemap"       # Discover URLs from sitemap.xml
    DOM = "dom"               # Follow links discovered in page DOM
    HYBRID = "hybrid"         # Combine sitemap + DOM discovery


class LoadWaitStrategy(str, Enum):
    """Strategies for waiting for page load completion."""
    NETWORKIDLE = "networkidle"    # Wait for network inactivity
    SELECTOR = "selector"          # Wait for specific element
    TIMEOUT = "timeout"            # Fixed timeout
    CUSTOM = "custom"              # Custom JavaScript condition


class CrawlConfig(BaseModel):
    """Configuration for a crawling session.
    
    This model defines all parameters needed to configure a crawl,
    including discovery mode, limits, filtering, and politeness settings.
    """
    
    # Discovery configuration
    discovery_mode: DiscoveryMode = Field(
        default=DiscoveryMode.SEEDS,
        description="Method for discovering URLs to crawl"
    )
    
    seeds: List[AnyHttpUrl] = Field(
        default_factory=list,
        description="Initial seed URLs to start crawling from"
    )
    
    sitemap_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="URL of sitemap.xml file for URL discovery"
    )
    
    # Scope filtering
    include_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns for URLs to include in crawl"
    )
    
    exclude_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns for URLs to exclude from crawl"
    )
    
    same_site_only: bool = Field(
        default=True,
        description="Restrict crawling to same site as seed URLs"
    )
    
    # Limits and performance
    max_pages: int = Field(
        default=500,
        ge=1,
        le=100000,
        description="Maximum number of pages to crawl"
    )
    
    max_depth: Optional[int] = Field(
        default=None,
        ge=0,
        le=50,
        description="Maximum link depth from seed URLs (None = unlimited)"
    )
    
    max_concurrency: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of concurrent requests"
    )
    
    requests_per_second: float = Field(
        default=2.0,
        gt=0.0,
        le=100.0,
        description="Maximum requests per second per host"
    )
    
    max_concurrent_per_host: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum concurrent requests per host"
    )
    
    # Timeouts and retries
    page_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Page load timeout in seconds"
    )
    
    navigation_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Navigation timeout in seconds"
    )
    
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts per URL"
    )
    
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Base delay between retries in seconds"
    )
    
    # Load wait configuration
    load_wait_strategy: LoadWaitStrategy = Field(
        default=LoadWaitStrategy.NETWORKIDLE,
        description="Strategy for determining when page load is complete"
    )
    
    load_wait_timeout: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Maximum time to wait for load condition in seconds"
    )
    
    load_wait_selector: Optional[str] = Field(
        default=None,
        description="CSS selector to wait for (when using selector strategy)"
    )
    
    load_wait_js: Optional[str] = Field(
        default=None,
        description="JavaScript condition to wait for (when using custom strategy)"
    )
    
    # User agent and headers
    user_agent: Optional[str] = Field(
        default=None,
        description="Custom User-Agent string (None = use default)"
    )
    
    extra_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers to send with requests"
    )
    
    @field_validator('include_patterns', 'exclude_patterns')
    @classmethod
    def validate_regex_patterns(cls, v):
        """Validate that regex patterns compile correctly."""
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        return v
    
    @model_validator(mode='after')
    def validate_sitemap_url(self):
        """Validate sitemap URL when discovery mode requires it."""
        if self.discovery_mode in (DiscoveryMode.SITEMAP, DiscoveryMode.HYBRID):
            if not self.sitemap_url:
                raise ValueError("sitemap_url required for sitemap-based discovery")
        return self
    
    @model_validator(mode='after')
    def validate_seeds(self):
        """Validate that seeds are provided when required."""
        if self.discovery_mode in (DiscoveryMode.SEEDS, DiscoveryMode.DOM, DiscoveryMode.HYBRID):
            if not self.seeds:
                raise ValueError("seeds required for seed-based or DOM discovery")
        return self
    
    @model_validator(mode='after')
    def validate_load_wait_selector(self):
        """Validate selector when using selector wait strategy."""
        if self.load_wait_strategy == LoadWaitStrategy.SELECTOR:
            if not self.load_wait_selector:
                raise ValueError("load_wait_selector required for selector wait strategy")
        return self
    
    @model_validator(mode='after')
    def validate_load_wait_js(self):
        """Validate JavaScript when using custom wait strategy."""
        if self.load_wait_strategy == LoadWaitStrategy.CUSTOM:
            if not self.load_wait_js:
                raise ValueError("load_wait_js required for custom wait strategy")
        return self
    
    @model_validator(mode='after')
    def validate_discovery_requirements(cls, model):
        """Validate that required fields are present for the selected discovery mode."""
        # Check sitemap URL requirement
        if model.discovery_mode in (DiscoveryMode.SITEMAP, DiscoveryMode.HYBRID):
            if not model.sitemap_url:
                raise ValueError("sitemap_url required for sitemap-based discovery")
        
        # Check seeds requirement  
        if model.discovery_mode in (DiscoveryMode.SEEDS, DiscoveryMode.DOM, DiscoveryMode.HYBRID):
            if not model.seeds:
                raise ValueError("seeds required for seed-based or DOM discovery")
        
        return model


class PagePlan(BaseModel):
    """Plan for processing a discovered page.
    
    This model represents a URL that has been discovered and is ready
    for processing by the browser capture engine.
    """
    
    url: AnyHttpUrl = Field(
        description="Normalized URL to be processed"
    )
    
    source_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="URL where this page was discovered (None for seeds)"
    )
    
    depth: int = Field(
        default=0,
        ge=0,
        description="Link depth from original seed URLs"
    )
    
    discovery_method: str = Field(
        description="How this URL was discovered (seeds, sitemap, dom)"
    )
    
    discovered_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this URL was discovered"
    )
    
    priority: int = Field(
        default=0,
        description="Processing priority (higher = more important)"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about this page"
    )
    
    # Load wait configuration (can override global config per URL)
    load_wait_strategy: Optional[LoadWaitStrategy] = Field(
        default=None,
        description="Override global load wait strategy for this page"
    )
    
    load_wait_timeout: Optional[int] = Field(
        default=None,
        ge=1,
        le=60,
        description="Override global load wait timeout for this page"
    )


class CrawlStats(BaseModel):
    """Statistics tracking for crawl progress and performance."""
    
    # URL statistics
    urls_discovered: int = Field(default=0, description="Total URLs discovered")
    urls_queued: int = Field(default=0, description="URLs currently in queue")
    urls_processed: int = Field(default=0, description="URLs successfully processed")
    urls_failed: int = Field(default=0, description="URLs that failed processing")
    urls_skipped: int = Field(default=0, description="URLs skipped due to scope/limits")
    urls_deduplicated: int = Field(default=0, description="Duplicate URLs filtered out")
    
    # Host statistics
    unique_hosts: Set[str] = Field(default_factory=set, description="Unique hosts encountered")
    rate_limit_hits: int = Field(default=0, description="Number of rate limit encounters")
    
    # Error statistics
    timeout_errors: int = Field(default=0, description="Pages that timed out")
    network_errors: int = Field(default=0, description="Network connectivity errors")
    http_4xx_errors: int = Field(default=0, description="HTTP 4xx client errors")
    http_5xx_errors: int = Field(default=0, description="HTTP 5xx server errors")
    
    # Performance statistics
    start_time: Optional[datetime] = Field(default=None, description="Crawl start time")
    end_time: Optional[datetime] = Field(default=None, description="Crawl end time")
    total_bytes_downloaded: int = Field(default=0, description="Total bytes downloaded")
    average_response_time: float = Field(default=0.0, description="Average response time in seconds")
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate crawl duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.urls_processed + self.urls_failed
        if total == 0:
            return 0.0
        return (self.urls_processed / total) * 100.0
    
    @property
    def pages_per_second(self) -> float:
        """Calculate processing rate in pages per second."""
        duration = self.duration
        if duration and duration > 0:
            return self.urls_processed / duration
        return 0.0


class CrawlMetrics(BaseModel):
    """Comprehensive metrics for crawl monitoring and reporting."""
    
    config: CrawlConfig = Field(description="Configuration used for this crawl")
    stats: CrawlStats = Field(default_factory=CrawlStats, description="Current statistics")
    
    # Real-time status
    is_running: bool = Field(default=False, description="Whether crawl is currently active")
    current_url: Optional[str] = Field(default=None, description="Currently processing URL")
    queue_size: int = Field(default=0, description="Current queue size")
    
    # Error details
    recent_errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent error details for debugging"
    )
    
    def add_error(self, url: str, error_type: str, error_message: str, host: str = None):
        """Add an error to recent errors list."""
        error_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "url": url,
            "error_type": error_type,
            "error_message": error_message,
            "host": host or urlparse(url).netloc
        }
        
        self.recent_errors.append(error_info)
        
        # Keep only last 100 errors to prevent memory bloat
        if len(self.recent_errors) > 100:
            self.recent_errors = self.recent_errors[-100:]
    
    def export_summary(self) -> Dict[str, Any]:
        """Export a summary of metrics for reporting."""
        return {
            "discovery_mode": self.config.discovery_mode,
            "max_pages": self.config.max_pages,
            "max_concurrency": self.config.max_concurrency,
            "urls_discovered": self.stats.urls_discovered,
            "urls_processed": self.stats.urls_processed,
            "urls_failed": self.stats.urls_failed,
            "success_rate": self.stats.success_rate,
            "duration_seconds": self.stats.duration,
            "pages_per_second": self.stats.pages_per_second,
            "unique_hosts": len(self.stats.unique_hosts),
            "total_errors": len(self.recent_errors),
            "is_running": self.is_running,
            "queue_size": self.queue_size
        }