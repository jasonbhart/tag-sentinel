"""Audit engine package for Tag Sentinel.

This package provides the core crawling and URL discovery functionality
for the Tag Sentinel web analytics auditing platform.
"""

from .crawler import Crawler
from .models.crawl import (
    CrawlConfig,
    DiscoveryMode,
    LoadWaitStrategy,
    PagePlan,
    CrawlMetrics,
    CrawlStats
)
from .utils.url_normalizer import normalize, is_valid_http_url, are_same_site
from .utils.scope_matcher import ScopeMatcher

__all__ = [
    # Main crawler
    'Crawler',
    
    # Models
    'CrawlConfig',
    'DiscoveryMode', 
    'LoadWaitStrategy',
    'PagePlan',
    'CrawlMetrics',
    'CrawlStats',
    
    # Utilities
    'normalize',
    'is_valid_http_url',
    'are_same_site',
    'ScopeMatcher'
]

__version__ = "0.1.0"