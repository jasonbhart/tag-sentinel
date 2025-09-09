"""Frontier queue with deduplication and backpressure handling.

This module implements an async queue system for URL management during crawls,
providing deduplication, backpressure handling, and priority support.
"""

import asyncio
from asyncio import Queue
from typing import Dict, Optional, Set, Any, List, Tuple
import time
from dataclasses import dataclass
from enum import Enum
import logging

from ..models.crawl import PagePlan
from ..utils.url_normalizer import normalize, URLNormalizationError


logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Priority levels for queue items."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class QueueItem:
    """Item stored in the frontier queue with priority and metadata."""
    page_plan: PagePlan
    priority: QueuePriority = QueuePriority.NORMAL
    enqueued_at: float = None
    
    def __post_init__(self):
        if self.enqueued_at is None:
            self.enqueued_at = time.time()
    
    def __lt__(self, other):
        """Support priority queue ordering (higher priority = lower value for heapq)."""
        return self.priority.value > other.priority.value


class FrontierQueueStats:
    """Statistics tracking for frontier queue operations."""
    
    def __init__(self):
        self.enqueued_total = 0
        self.dequeued_total = 0
        self.deduplicated_total = 0
        self.backpressure_events = 0
        self.queue_size_max = 0
        self.start_time = time.time()
    
    def export(self) -> Dict[str, Any]:
        """Export statistics as dictionary."""
        return {
            "enqueued_total": self.enqueued_total,
            "dequeued_total": self.dequeued_total,
            "deduplicated_total": self.deduplicated_total,
            "backpressure_events": self.backpressure_events,
            "queue_size_max": self.queue_size_max,
            "uptime_seconds": time.time() - self.start_time
        }


class FrontierQueue:
    """Async queue for URL frontier with deduplication and backpressure.
    
    Features:
    - URL deduplication using normalized URLs
    - Backpressure handling when queue reaches capacity
    - Priority support for different types of URLs
    - Memory-efficient seen-set tracking
    - Comprehensive statistics and monitoring
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        backpressure_threshold: float = 0.8,
        enable_bloom_filter: bool = False,
        bloom_capacity: int = 100000,
        bloom_error_rate: float = 0.1
    ):
        """Initialize the frontier queue.
        
        Args:
            max_size: Maximum number of items in queue
            backpressure_threshold: Queue fullness ratio to trigger backpressure
            enable_bloom_filter: Use Bloom filter for memory-efficient deduplication
            bloom_capacity: Bloom filter expected capacity
            bloom_error_rate: Bloom filter false positive rate
        """
        self._queue = asyncio.PriorityQueue(maxsize=max_size)
        self._max_size = max_size
        self._backpressure_threshold = backpressure_threshold
        
        # Deduplication tracking
        self._seen_urls: Set[str] = set()
        self._bloom_filter = None
        
        if enable_bloom_filter:
            try:
                from pybloom_live import BloomFilter
                self._bloom_filter = BloomFilter(capacity=bloom_capacity, error_rate=bloom_error_rate)
            except ImportError:
                logger.warning("pybloom_live not available, falling back to set-based deduplication")
        
        # Statistics and monitoring
        self._stats = FrontierQueueStats()
        self._lock = asyncio.Lock()
        self._backpressure_condition = asyncio.Condition()
        self._closed = False
        
        # Priority counter for stable sorting
        self._counter = 0
    
    async def put(
        self, 
        page_plan: PagePlan, 
        priority: QueuePriority = QueuePriority.NORMAL,
        wait_on_backpressure: bool = True
    ) -> bool:
        """Add a page plan to the queue.
        
        Args:
            page_plan: PagePlan to add to queue
            priority: Priority level for processing
            wait_on_backpressure: Whether to wait if queue is under backpressure
            
        Returns:
            True if item was added, False if deduplicated or queue full
            
        Raises:
            QueueClosedError: If queue has been closed
        """
        if self._closed:
            raise QueueClosedError("Queue has been closed")
        
        # Normalize URL for deduplication
        try:
            normalized_url = normalize(str(page_plan.url))
        except URLNormalizationError:
            logger.debug(f"Skipping invalid URL: {page_plan.url}")
            return False
        
        # Check for duplicates
        if await self._is_duplicate(normalized_url):
            self._stats.deduplicated_total += 1
            logger.debug(f"Deduplicated URL: {normalized_url}")
            return False
        
        # Handle backpressure
        if wait_on_backpressure:
            await self._handle_backpressure()
        elif self._is_backpressure_active():
            logger.debug(f"Queue under backpressure, dropping URL: {normalized_url}")
            self._stats.backpressure_events += 1
            return False
        
        # Create queue item with priority and counter for stable sorting
        queue_item = QueueItem(
            page_plan=page_plan,
            priority=priority
        )
        
        try:
            # Use counter to ensure stable priority ordering
            self._counter += 1
            await self._queue.put((-priority.value, self._counter, queue_item))
            
            # Track URL as seen
            await self._mark_as_seen(normalized_url)
            
            # Update statistics
            self._stats.enqueued_total += 1
            self._stats.queue_size_max = max(self._stats.queue_size_max, self._queue.qsize())
            
            logger.debug(f"Enqueued URL with priority {priority.name}: {normalized_url}")
            return True
            
        except asyncio.QueueFull:
            logger.warning(f"Queue full, dropping URL: {normalized_url}")
            self._stats.backpressure_events += 1
            return False
    
    async def get(self, timeout: Optional[float] = None) -> Optional[PagePlan]:
        """Get the next page plan from the queue.
        
        Args:
            timeout: Maximum time to wait for an item
            
        Returns:
            Next PagePlan or None if timeout/queue closed
            
        Raises:
            QueueClosedError: If queue has been closed
        """
        if self._closed and self._queue.empty():
            raise QueueClosedError("Queue has been closed and is empty")
        
        try:
            if timeout:
                priority, counter, queue_item = await asyncio.wait_for(
                    self._queue.get(), timeout=timeout
                )
            else:
                priority, counter, queue_item = await self._queue.get()
            
            self._stats.dequeued_total += 1
            
            # Notify backpressure condition that space is available
            async with self._backpressure_condition:
                self._backpressure_condition.notify_all()
            
            logger.debug(f"Dequeued URL: {queue_item.page_plan.url}")
            return queue_item.page_plan
            
        except asyncio.TimeoutError:
            logger.debug("Queue get timeout")
            return None
    
    async def close(self):
        """Close the queue and prevent new items from being added."""
        self._closed = True
        
        # Wake up any waiting put() operations
        async with self._backpressure_condition:
            self._backpressure_condition.notify_all()
        
        logger.info("Frontier queue closed")
    
    def qsize(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def is_closed(self) -> bool:
        """Check if queue is closed."""
        return self._closed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        stats = self._stats.export()
        stats.update({
            "current_size": self.qsize(),
            "max_capacity": self._max_size,
            "backpressure_threshold": self._backpressure_threshold,
            "seen_urls_count": len(self._seen_urls),
            "is_closed": self._closed,
            "backpressure_active": self._is_backpressure_active()
        })
        return stats
    
    async def _is_duplicate(self, normalized_url: str) -> bool:
        """Check if URL has been seen before."""
        async with self._lock:
            if self._bloom_filter:
                # First check Bloom filter
                if normalized_url in self._bloom_filter:
                    # Possible duplicate, check exact set
                    return normalized_url in self._seen_urls
                return False
            else:
                return normalized_url in self._seen_urls
    
    async def _mark_as_seen(self, normalized_url: str):
        """Mark URL as seen for deduplication."""
        async with self._lock:
            self._seen_urls.add(normalized_url)
            if self._bloom_filter:
                self._bloom_filter.add(normalized_url)
    
    def _is_backpressure_active(self) -> bool:
        """Check if queue is under backpressure."""
        if self._max_size <= 0:
            return False
        return self.qsize() >= (self._max_size * self._backpressure_threshold)
    
    async def _handle_backpressure(self):
        """Handle backpressure by waiting for queue space."""
        if not self._is_backpressure_active():
            return
        
        self._stats.backpressure_events += 1
        logger.debug("Queue under backpressure, waiting for space...")
        
        async with self._backpressure_condition:
            while self._is_backpressure_active() and not self._closed:
                await self._backpressure_condition.wait()
    
    async def bulk_put(
        self, 
        page_plans: List[Tuple[PagePlan, QueuePriority]], 
        max_failures: int = 10
    ) -> Tuple[int, int]:
        """Add multiple page plans to the queue efficiently.
        
        Args:
            page_plans: List of (PagePlan, QueuePriority) tuples
            max_failures: Maximum failures before stopping
            
        Returns:
            Tuple of (successful_adds, failed_adds)
        """
        successful = 0
        failed = 0
        
        for page_plan, priority in page_plans:
            if failed >= max_failures:
                logger.warning(f"Stopping bulk put after {max_failures} failures")
                break
            
            try:
                success = await self.put(page_plan, priority, wait_on_backpressure=False)
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error adding URL to queue: {e}")
                failed += 1
        
        logger.info(f"Bulk put completed: {successful} successful, {failed} failed")
        return successful, failed
    
    def get_memory_usage_estimate(self) -> Dict[str, int]:
        """Estimate memory usage of queue components."""
        # Rough estimates in bytes
        seen_urls_bytes = len(self._seen_urls) * 100  # Approximate average URL length
        queue_items_bytes = self.qsize() * 500  # Approximate PagePlan size
        
        return {
            "seen_urls_bytes": seen_urls_bytes,
            "queue_items_bytes": queue_items_bytes,
            "total_estimate_bytes": seen_urls_bytes + queue_items_bytes,
            "seen_urls_count": len(self._seen_urls),
            "queue_items_count": self.qsize()
        }


class QueueClosedError(Exception):
    """Raised when attempting to operate on a closed queue."""
    pass