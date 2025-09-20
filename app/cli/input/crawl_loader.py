"""Crawl URL loader for CLI input mode.

This module provides a simple interface for discovering URLs by crawling
from a base URL, wrapping the existing Crawler components.
"""

import asyncio
import logging
from typing import List
from urllib.parse import urlparse

from ...audit.models.crawl import CrawlConfig, DiscoveryMode
from ...audit.crawler import Crawler, CrawlerError
from ...audit.input.seed_provider import SeedListProvider


logger = logging.getLogger(__name__)


async def load_urls_from_crawl(
    base_url: str,
    max_urls: int = 500,
    max_depth: int = 2,
    max_concurrency: int = 3,
    timeout: int = 30
) -> List[str]:
    """Load URLs by crawling from a base URL.

    Args:
        base_url: Base URL to start crawling from
        max_urls: Maximum number of URLs to discover
        max_depth: Maximum crawl depth
        max_concurrency: Maximum concurrent requests
        timeout: Page load timeout in seconds

    Returns:
        List of URLs discovered through crawling

    Raises:
        CrawlerError: If crawling fails
    """
    urls = []

    logger.info(f"Starting crawl from base URL: {base_url}")

    try:
        # Parse the base URL to determine scope
        parsed_url = urlparse(base_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Create crawler configuration
        config = CrawlConfig(
            discovery_mode=DiscoveryMode.DOM,
            max_pages=max_urls,
            max_depth=max_depth,
            max_concurrency=max_concurrency,
            requests_per_second=5.0,  # Conservative rate limiting
            max_concurrent_per_host=max_concurrency,
            respect_robots_txt=True,
            follow_redirects=True,
            timeout_ms=timeout * 1000,
            user_agent="TagSentinel/1.0 CLI (+https://github.com/tag-sentinel)",

            # Scope configuration - stay within the same domain
            allowed_domains=[parsed_url.netloc],
            blocked_domains=[],
            url_patterns=[],
            blocked_url_patterns=[
                r".*\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot)$",
                r".*[&?]format=(json|xml|rss|atom).*",
                r".*/api/.*",
                r".*/admin/.*"
            ],

            # Only discover via DOM links, not additional sitemaps
            discover_sitemaps=False,
            follow_sitemap_links=False,
            extract_dom_links=True,
        )

        # Create crawler instance
        crawler = Crawler(config)

        # Add the base URL as a seed
        async with crawler:
            # Add seed URL
            await crawler.add_seed_url(base_url, metadata={
                "source": "cli_crawl",
                "depth": 0
            })

            # Collect URLs from the crawler
            async for page_plan in crawler.discover_pages():
                urls.append(page_plan.url)

                # Early exit if we've reached the limit
                if len(urls) >= max_urls:
                    break

            # Get final metrics
            metrics = crawler.get_metrics()
            logger.info(
                f"Crawl completed: {metrics.stats.pages_discovered} discovered, "
                f"{len(urls)} collected, depth {metrics.stats.max_depth_reached}"
            )

    except Exception as e:
        error_msg = f"Failed to crawl from {base_url}: {e}"
        logger.error(error_msg)
        raise CrawlerError(error_msg)

    if not urls:
        raise CrawlerError(f"No URLs discovered from crawl base: {base_url}")

    return urls