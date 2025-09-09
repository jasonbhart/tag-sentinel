#!/usr/bin/env python3
"""
Basic crawling example for Tag Sentinel.

This example demonstrates how to use the crawler to discover URLs
from seed lists and sitemaps without browser automation.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.audit import Crawler, CrawlConfig, DiscoveryMode


async def basic_seed_crawl():
    """Example of crawling from seed URLs only."""
    print("=== Basic Seed Crawl Example ===")
    
    # Configure crawl
    config = CrawlConfig(
        discovery_mode=DiscoveryMode.SEEDS,
        seeds=[
            "https://example.com",
            "https://example.com/about",
            "https://example.com/contact"
        ],
        max_pages=10,
        max_concurrency=2,
        include_patterns=[".*example\\.com.*"],
        same_site_only=True
    )
    
    # Create crawler
    crawler = Crawler(config)
    
    try:
        # Execute crawl
        page_count = 0
        async for page_plan in crawler.crawl():
            page_count += 1
            print(f"Discovered: {page_plan.url} (depth: {page_plan.depth}, method: {page_plan.discovery_method})")
            
            # Stop after a few pages for demo
            if page_count >= 5:
                break
        
        # Print final metrics
        metrics = crawler.get_metrics()
        print(f"\nCrawl completed:")
        print(f"- Pages discovered: {metrics.stats.urls_processed}")
        print(f"- URLs skipped: {metrics.stats.urls_skipped}")
        print(f"- Success rate: {metrics.stats.success_rate:.1f}%")
        
    finally:
        await crawler.stop()


async def sitemap_crawl_example():
    """Example of crawling from sitemap."""
    print("\n=== Sitemap Crawl Example ===")
    
    # Configure sitemap crawl
    config = CrawlConfig(
        discovery_mode=DiscoveryMode.SITEMAP,
        sitemap_url="https://example.com/sitemap.xml",
        max_pages=20,
        max_concurrency=3,
        same_site_only=True,
        requests_per_second=1.0  # Be polite
    )
    
    crawler = Crawler(config)
    
    try:
        page_count = 0
        async for page_plan in crawler.crawl():
            page_count += 1
            print(f"From sitemap: {page_plan.url}")
            
            # Stop after a few for demo
            if page_count >= 3:
                break
        
        # Show component stats
        stats = crawler.get_component_stats()
        if 'provider_sitemap' in stats:
            sitemap_stats = stats['provider_sitemap']
            print(f"\nSitemap stats:")
            print(f"- Sitemaps processed: {sitemap_stats.get('sitemaps_processed', 0)}")
            print(f"- URLs discovered: {sitemap_stats.get('urls_discovered', 0)}")
            print(f"- Invalid URLs: {sitemap_stats.get('invalid_urls', 0)}")
        
    except Exception as e:
        print(f"Sitemap crawl failed (expected for example.com): {e}")
    finally:
        await crawler.stop()


async def file_based_seed_example():
    """Example of loading seeds from a file."""
    print("\n=== File-based Seed Example ===")
    
    # Create a temporary seed file
    seed_file = Path("/tmp/test_seeds.txt")
    seed_content = """
# Example seed URLs
https://example.com
https://example.com/page1
https://example.com/page2

# This is a comment and will be ignored
https://example.com/page3
"""
    
    seed_file.write_text(seed_content.strip())
    
    try:
        # Import seed provider directly for file example
        from app.audit.input import SeedListProvider
        
        # Load seeds from file
        seed_provider = SeedListProvider(seed_files=[seed_file])
        page_plans = []
        
        async for page_plan in seed_provider.discover_urls():
            page_plans.append(page_plan)
            print(f"Loaded seed: {page_plan.url} (source: {page_plan.metadata.get('source', 'unknown')})")
        
        print(f"\nLoaded {len(page_plans)} URLs from seed file")
        
        # Show provider stats
        stats = seed_provider.get_stats()
        print(f"Stats: {stats['valid_urls']} valid, {stats['invalid_urls']} invalid")
        
    finally:
        # Cleanup
        if seed_file.exists():
            seed_file.unlink()


async def main():
    """Run all examples."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce log noise from HTTP client
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    
    try:
        await basic_seed_crawl()
        await sitemap_crawl_example()
        await file_based_seed_example()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Tag Sentinel Crawler Examples")
    print("=" * 40)
    asyncio.run(main())