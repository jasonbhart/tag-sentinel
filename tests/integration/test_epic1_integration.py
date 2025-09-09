#!/usr/bin/env python3
"""
Basic functionality test for Tag Sentinel crawler implementation.

This script tests the core components without requiring external dependencies.
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.asyncio
async def test_url_normalizer():
    """Test URL normalization functionality."""
    print("Testing URL normalizer...")
    
    from app.audit.utils.url_normalizer import normalize, are_same_site, is_valid_http_url
    
    # Test URL normalization
    test_cases = [
        ("HTTP://Example.COM:80/Path/?param=value#fragment", "http://example.com/Path/?param=value"),
        ("https://site.com:443/", "https://site.com/"),
        ("http://site.com/path#fragment", "http://site.com/path"),
    ]
    
    for input_url, expected in test_cases:
        result = normalize(input_url)
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"✓ {input_url} -> {result}")
    
    # Test same site comparison
    assert are_same_site("https://www.example.com/page1", "https://blog.example.com/page2") == True
    assert are_same_site("https://example.com/page1", "https://other.com/page2") == False
    print("✓ Same site comparison works")
    
    # Test URL validation
    assert is_valid_http_url("https://example.com") == True
    assert is_valid_http_url("ftp://example.com") == False
    assert is_valid_http_url("not-a-url") == False
    print("✓ URL validation works")
    
    print("URL normalizer tests passed!\n")


@pytest.mark.asyncio
async def test_scope_matcher():
    """Test scope matching functionality."""
    print("Testing scope matcher...")
    
    from app.audit.utils.scope_matcher import ScopeMatcher
    
    # Test include/exclude patterns
    matcher = ScopeMatcher(
        include_patterns=[".*example\\.com.*"],  # Fixed pattern to match example.com
        exclude_patterns=[".*/admin/.*"],
        same_site_only=False
    )
    
    # Test URLs
    test_cases = [
        ("https://example.com/page", True),
        ("https://sub.example.com/page", True),
        ("https://example.com/admin/panel", False),  # Excluded
        ("https://other.com/page", False),  # Not included
    ]
    
    for url, expected in test_cases:
        result = matcher.is_in_scope(url)
        assert result == expected, f"Expected {expected} for {url}, got {result}"
        print(f"✓ {url} -> {'in scope' if result else 'out of scope'}")
    
    print("Scope matcher tests passed!\n")


@pytest.mark.asyncio
async def test_crawl_models():
    """Test crawl configuration models."""
    print("Testing crawl models...")
    
    from app.audit.models.crawl import CrawlConfig, DiscoveryMode, PagePlan
    
    # Test configuration model
    config = CrawlConfig(
        discovery_mode=DiscoveryMode.SEEDS,
        seeds=["https://example.com"],
        max_pages=100,
        include_patterns=[".*example.*"]
    )
    
    assert config.discovery_mode == DiscoveryMode.SEEDS
    assert len(config.seeds) == 1
    print("✓ CrawlConfig creation works")
    
    # Test PagePlan model
    page_plan = PagePlan(
        url="https://example.com",
        depth=0,
        discovery_method="seeds"
    )
    
    assert str(page_plan.url) == "https://example.com/"  # Pydantic adds trailing slash
    assert page_plan.depth == 0
    print("✓ PagePlan creation works")
    
    print("Crawl models tests passed!\n")


@pytest.mark.asyncio
async def test_seed_provider():
    """Test seed list provider."""
    print("Testing seed provider...")
    
    from app.audit.input.seed_provider import SeedListProvider
    
    # Test with direct seeds
    provider = SeedListProvider(seeds=[
        "https://example.com/page1",
        "https://example.com/page2",
        "invalid-url",  # Should be skipped
        "https://example.com/page1"  # Duplicate, should be skipped
    ])
    
    page_plans = []
    async for page_plan in provider.discover_urls():
        page_plans.append(page_plan)
    
    # Should have 2 valid unique URLs
    assert len(page_plans) == 2
    urls = [str(pp.url) for pp in page_plans]
    assert "https://example.com/page1" in urls
    assert "https://example.com/page2" in urls
    print(f"✓ Discovered {len(page_plans)} URLs from seeds")
    
    stats = provider.get_stats()
    print(f"✓ Provider stats: {stats['valid_urls']} valid, {stats['invalid_urls']} invalid")
    
    print("Seed provider tests passed!\n")


@pytest.mark.asyncio
async def test_frontier_queue():
    """Test frontier queue functionality."""
    print("Testing frontier queue...")
    
    from app.audit.queue.frontier_queue import FrontierQueue, QueuePriority
    from app.audit.models.crawl import PagePlan
    
    queue = FrontierQueue(max_size=100)
    
    # Test enqueuing
    page_plan1 = PagePlan(url="https://example.com/page1", depth=0, discovery_method="test")
    page_plan2 = PagePlan(url="https://example.com/page2", depth=1, discovery_method="test")
    
    success1 = await queue.put(page_plan1, QueuePriority.HIGH)
    success2 = await queue.put(page_plan2, QueuePriority.NORMAL)
    
    assert success1 == True
    assert success2 == True
    assert queue.qsize() == 2
    print("✓ Enqueuing works")
    
    # Test dequeuing (high priority should come first)
    retrieved1 = await queue.get()
    assert str(retrieved1.url) == "https://example.com/page1"  # Should work as-is
    print("✓ Priority dequeuing works")
    
    retrieved2 = await queue.get(timeout=0.1)
    assert str(retrieved2.url) == "https://example.com/page2"  # Should work as-is
    print("✓ Normal dequeuing works")
    
    await queue.close()
    print("Frontier queue tests passed!\n")


@pytest.mark.asyncio
async def test_basic_crawler():
    """Test basic crawler initialization and configuration."""
    print("Testing basic crawler...")
    
    from app.audit import Crawler, CrawlConfig, DiscoveryMode
    
    # Create a simple configuration
    config = CrawlConfig(
        discovery_mode=DiscoveryMode.SEEDS,
        seeds=["https://example.com", "https://example.com/about"],
        max_pages=2,
        max_concurrency=1
    )
    
    # Test crawler creation and initialization
    crawler = Crawler(config)
    assert crawler.config.discovery_mode == DiscoveryMode.SEEDS
    assert len(crawler.config.seeds) == 2
    print("✓ Crawler initialization works")
    
    # Test metrics initialization
    metrics = crawler.get_metrics()
    assert metrics.config == config
    assert metrics.is_running == False
    assert metrics.stats.urls_processed == 0
    print("✓ Metrics initialization works")
    
    # Test scope matcher initialization
    assert crawler.scope_matcher is not None
    print("✓ Scope matcher initialization works")
    
    # Test frontier queue initialization
    assert crawler.frontier_queue is not None
    print("✓ Frontier queue initialization works")
    
    # Test cleanup
    await crawler.stop()
    print("✓ Crawler cleanup works")
    
    print("Basic crawler tests passed!\n")


async def main():
    """Run all tests."""
    print("Tag Sentinel Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_url_normalizer,
        test_scope_matcher, 
        test_crawl_models,
        test_seed_provider,
        test_frontier_queue,
        test_basic_crawler,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("🎉 All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())