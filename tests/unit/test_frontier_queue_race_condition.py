"""Unit tests for frontier queue race condition fixes."""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import patch

from app.audit.queue.frontier_queue import FrontierQueue, QueuePriority
from app.audit.models.crawl import PagePlan


class TestFrontierQueueRaceCondition:
    """Test cases for frontier queue race condition fixes."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Create a frontier queue for testing."""
        queue = FrontierQueue(max_size=10)
        yield queue
        await queue.close()

    @pytest.mark.asyncio
    async def test_no_duplicates_under_concurrent_access(self, queue):
        """Test that duplicates are prevented under concurrent access."""
        url = "https://example.com/test-page"
        page_plan = PagePlan(
            url=url,
            source_url="https://example.com",
            depth=1,
            discovery_method="test"
        )

        # Number of concurrent tasks trying to enqueue the same URL
        num_tasks = 20

        async def enqueue_task():
            return await queue.put(page_plan, QueuePriority.NORMAL)

        # Run multiple tasks concurrently
        results = await asyncio.gather(*[enqueue_task() for _ in range(num_tasks)])

        # Only one should succeed
        successful_enqueues = sum(1 for result in results if result)
        assert successful_enqueues == 1

        # Queue should contain exactly one item
        assert queue.qsize() == 1

        # Deduplication stats should reflect the duplicates
        stats = queue.get_stats()
        assert stats.deduplicated_total == num_tasks - 1

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_different_urls(self, queue):
        """Test concurrent enqueuing of different URLs works correctly."""
        urls = [f"https://example.com/page-{i}" for i in range(10)]

        async def enqueue_task(url):
            page_plan = PagePlan(
                url=url,
                source_url="https://example.com",
                depth=1,
                discovery_method="test"
            )
            return await queue.put(page_plan, QueuePriority.NORMAL)

        # Run tasks concurrently
        results = await asyncio.gather(*[enqueue_task(url) for url in urls])

        # All should succeed
        assert all(results)
        assert queue.qsize() == 10

        # No deduplication should occur
        stats = queue.get_stats()
        assert stats.deduplicated_total == 0

    @pytest.mark.asyncio
    async def test_backpressure_with_duplicates(self, queue):
        """Test that backpressure handling doesn't cause duplicate race conditions."""
        # Fill up the queue first
        for i in range(10):  # max_size is 10
            page_plan = PagePlan(
                url=f"https://example.com/fill-{i}",
                source_url="https://example.com",
                depth=1,
                discovery_method="test"
            )
            await queue.put(page_plan, QueuePriority.NORMAL, wait_on_backpressure=False)

        # Queue should be full
        assert queue.qsize() == 10

        # Now try to add the same URL multiple times when queue is under backpressure
        duplicate_url = "https://example.com/duplicate"
        duplicate_plan = PagePlan(
            url=duplicate_url,
            source_url="https://example.com",
            depth=1,
            discovery_method="test"
        )

        # Multiple tasks trying to add the same URL when queue is full
        async def enqueue_duplicate():
            return await queue.put(duplicate_plan, QueuePriority.NORMAL, wait_on_backpressure=False)

        results = await asyncio.gather(*[enqueue_duplicate() for _ in range(5)])

        # All should fail due to backpressure (queue full)
        assert all(not result for result in results)

        # No URL should be added to seen_urls if enqueue failed
        # This tests that the race condition fix properly handles backpressure failures
        stats = queue.get_stats()
        assert stats.backpressure_events > 0

    @pytest.mark.asyncio
    async def test_queue_full_exception_handling(self, queue):
        """Test that queue full exceptions properly clean up reserved URLs."""
        # Use a smaller queue for this test
        small_queue = FrontierQueue(max_size=2)

        try:
            # Fill the queue
            for i in range(2):
                page_plan = PagePlan(
                    url=f"https://example.com/fill-{i}",
                    source_url="https://example.com",
                    depth=1,
                    discovery_method="test"
                )
                await small_queue.put(page_plan, QueuePriority.NORMAL, wait_on_backpressure=False)

            # Now try to add one more (should fail due to queue full)
            overflow_plan = PagePlan(
                url="https://example.com/overflow",
                source_url="https://example.com",
                depth=1,
                discovery_method="test"
            )

            result = await small_queue.put(overflow_plan, QueuePriority.NORMAL, wait_on_backpressure=False)
            assert result is False

            # The overflow URL should NOT be in seen_urls since enqueue failed
            # This tests that the race condition fix properly cleans up on failure
            assert small_queue.qsize() == 2

            # Try to add the same URL again - it should not be deduplicated
            # (proving it wasn't left in seen_urls from the failed attempt)
            result2 = await small_queue.put(overflow_plan, QueuePriority.NORMAL, wait_on_backpressure=False)
            assert result2 is False

            # Should still show backpressure events, not deduplication
            stats = small_queue.get_stats()
            assert stats.backpressure_events >= 2
            # The overflow URL wasn't successfully added, so no deduplication should occur

        finally:
            await small_queue.close()

    @pytest.mark.asyncio
    async def test_reservation_and_release_timing(self, queue):
        """Test the timing of URL reservation and release."""
        url = "https://example.com/timing-test"
        page_plan = PagePlan(
            url=url,
            source_url="https://example.com",
            depth=1,
            discovery_method="test"
        )

        # Mock the queue.put to simulate a delay, testing our reservation logic
        original_put = queue._queue.put

        async def slow_put(*args, **kwargs):
            # Simulate slow queue operation
            await asyncio.sleep(0.1)
            return await original_put(*args, **kwargs)

        with patch.object(queue._queue, 'put', side_effect=slow_put):
            # Start two concurrent enqueue operations
            task1 = asyncio.create_task(queue.put(page_plan, QueuePriority.NORMAL))
            task2 = asyncio.create_task(queue.put(page_plan, QueuePriority.NORMAL))

            results = await asyncio.gather(task1, task2)

        # Exactly one should succeed due to URL reservation
        successful_enqueues = sum(1 for result in results if result)
        assert successful_enqueues == 1

        # One should be deduplicated
        stats = queue.get_stats()
        assert stats.deduplicated_total == 1

    @pytest.mark.asyncio
    async def test_bloom_filter_interaction_with_race_fix(self, queue):
        """Test that bloom filter works correctly with race condition fixes."""
        # Enable bloom filter
        await queue.close()  # Close first

        bloom_queue = FrontierQueue(max_size=100, enable_bloom_filter=True)

        try:
            url = "https://example.com/bloom-test"
            page_plan = PagePlan(
                url=url,
                source_url="https://example.com",
                depth=1,
                discovery_method="test"
            )

            # Add URL once
            result1 = await bloom_queue.put(page_plan, QueuePriority.NORMAL)
            assert result1 is True

            # Try to add same URL multiple times concurrently
            async def enqueue_task():
                return await bloom_queue.put(page_plan, QueuePriority.NORMAL)

            results = await asyncio.gather(*[enqueue_task() for _ in range(10)])

            # All subsequent attempts should fail due to deduplication
            assert all(not result for result in results)

            # Should show deduplication stats
            stats = bloom_queue.get_stats()
            assert stats.deduplicated_total == 10

        finally:
            await bloom_queue.close()


if __name__ == "__main__":
    pytest.main([__file__])