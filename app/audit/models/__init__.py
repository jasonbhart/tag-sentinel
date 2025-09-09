"""Audit data models package."""

from .crawl import (
    DiscoveryMode,
    LoadWaitStrategy,
    CrawlConfig,
    PagePlan,
    CrawlStats,
    CrawlMetrics
)

from .capture import (
    RequestLog,
    CookieRecord,
    ConsoleLog,
    PageResult,
    TimingData,
    ArtifactPaths,
    RequestStatus,
    ResourceType,
    ConsoleLevel,
    CaptureStatus,
)

__all__ = [
    # Crawl models
    'DiscoveryMode',
    'LoadWaitStrategy', 
    'CrawlConfig',
    'PagePlan',
    'CrawlStats',
    'CrawlMetrics',
    
    # Capture models
    'RequestLog',
    'CookieRecord',
    'ConsoleLog',
    'PageResult',
    'TimingData',
    'ArtifactPaths',
    'RequestStatus',
    'ResourceType',
    'ConsoleLevel',
    'CaptureStatus',
]