"""Unit tests for frontier queue functionality."""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.audit.queue.frontier_queue import FrontierQueue, QueuePriority, QueueClosedError
from app.audit.models.crawl import PagePlan


class TestFrontierQueue:
    """Test cases for frontier queue."""
    
    @pytest.mark.asyncio
    async def test_basic_enqueue_dequeue(self):
        """Test basic enqueue and dequeue operations."""
        queue = FrontierQueue(max_size=100)
        
        page_plan = PagePlan(
            url="https://example.com/page1",
            depth=0,
            discovery_method="test"
        )
        
        # Test enqueue
        success = await queue.put(page_plan, QueuePriority.NORMAL)
        assert success is True
        assert queue.qsize() == 1
        
        # Test dequeue
        retrieved = await queue.get()
        assert retrieved is not None
        assert str(retrieved.url) == "https://example.com/page1"
        assert queue.qsize() == 0
        
        await queue.close()
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test priority-based dequeue ordering."""
        queue = FrontierQueue(max_size=100)
        
        # Add pages with different priorities
        page_normal = PagePlan(url="https://example.com/normal", depth=0, discovery_method="test")
        page_high = PagePlan(url="https://example.com/high", depth=0, discovery_method="test")
        
        await queue.put(page_normal, QueuePriority.NORMAL)
        await queue.put(page_high, QueuePriority.HIGH)
        
        # High priority should come first
        first = await queue.get()
        assert str(first.url) == "https://example.com/high"
        
        second = await queue.get()
        assert str(second.url) == "https://example.com/normal"
        
        await queue.close()
    
    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Test URL deduplication."""
        queue = FrontierQueue(max_size=100)
        
        page_plan1 = PagePlan(url="https://example.com/page", depth=0, discovery_method="test")
        page_plan2 = PagePlan(url="https://example.com/page", depth=1, discovery_method="test")  # Duplicate URL
        
        success1 = await queue.put(page_plan1, QueuePriority.NORMAL)
        success2 = await queue.put(page_plan2, QueuePriority.NORMAL)  # Should be deduplicated
        
        assert success1 is True
        assert success2 is False  # Duplicate rejected
        assert queue.qsize() == 1
        
        await queue.close()
    
    @pytest.mark.asyncio
    async def test_backpressure(self):
        """Test backpressure handling."""
        queue = FrontierQueue(max_size=2, backpressure_threshold=0.5)  # Small queue for testing
        
        page1 = PagePlan(url="https://example.com/1", depth=0, discovery_method="test")
        page2 = PagePlan(url="https://example.com/2", depth=0, discovery_method="test")
        page3 = PagePlan(url="https://example.com/3", depth=0, discovery_method="test")
        
        # Fill queue - use wait_on_backpressure=False to avoid hanging
        success1 = await queue.put(page1, QueuePriority.NORMAL, wait_on_backpressure=False)
        success2 = await queue.put(page2, QueuePriority.NORMAL, wait_on_backpressure=False)
        
        # First should succeed, second should fail due to backpressure
        assert success1 is True
        assert success2 is False  # Should be rejected due to backpressure
        
        # This should also trigger backpressure
        success3 = await queue.put(page3, QueuePriority.NORMAL, wait_on_backpressure=False)
        assert success3 is False  # Should be rejected due to backpressure
        
        await queue.close()
    
    @pytest.mark.asyncio
    async def test_bulk_put(self):
        """Test bulk put operations."""
        queue = FrontierQueue(max_size=100)
        
        page_plans = [
            (PagePlan(url=f"https://example.com/page{i}", depth=0, discovery_method="test"), QueuePriority.NORMAL)
            for i in range(5)
        ]
        
        successful, failed = await queue.bulk_put(page_plans)
        
        assert successful == 5
        assert failed == 0
        assert queue.qsize() == 5
        
        await queue.close()
    
    @pytest.mark.asyncio
    async def test_queue_stats(self):
        """Test queue statistics."""
        queue = FrontierQueue(max_size=100)
        
        page_plan = PagePlan(url="https://example.com/page", depth=0, discovery_method="test")
        await queue.put(page_plan, QueuePriority.NORMAL)
        
        stats = queue.get_stats()
        
        assert stats["enqueued_total"] == 1
        assert stats["current_size"] == 1
        assert stats["max_capacity"] == 100
        
        await queue.close()
    
    @pytest.mark.asyncio
    async def test_closed_queue_operations(self):
        """Test operations on closed queue."""
        queue = FrontierQueue(max_size=100)
        await queue.close()
        
        page_plan = PagePlan(url="https://example.com/page", depth=0, discovery_method="test")
        
        # Should raise error when putting to closed queue
        with pytest.raises(QueueClosedError):
            await queue.put(page_plan, QueuePriority.NORMAL)


if __name__ == "__main__":
    pytest.main([__file__])