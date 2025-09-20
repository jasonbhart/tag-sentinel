"""Sitemap URL loader for CLI input mode.

This module provides a simple interface for loading URLs from sitemap files
for use in the CLI runner, wrapping the existing SitemapProvider.
"""

import asyncio
import logging
from typing import List
from pathlib import Path

from ...audit.input.sitemap_provider import SitemapProvider, SitemapProviderError


logger = logging.getLogger(__name__)


async def load_urls_from_sitemap(
    sitemap_url: str,
    max_urls: int = 500,
    timeout: int = 30
) -> List[str]:
    """Load URLs from a sitemap.xml file.

    Args:
        sitemap_url: URL of the sitemap.xml file
        max_urls: Maximum number of URLs to extract
        timeout: HTTP request timeout in seconds

    Returns:
        List of URLs extracted from the sitemap

    Raises:
        SitemapProviderError: If sitemap cannot be loaded or parsed
    """
    urls = []

    logger.info(f"Loading URLs from sitemap: {sitemap_url}")

    try:
        async with SitemapProvider(
            sitemap_url=sitemap_url,
            max_urls=max_urls,
            timeout=timeout,
            validate_urls=True,
            skip_invalid=True
        ) as provider:

            async for page_plan in provider.discover_urls():
                urls.append(page_plan.url)

                # Early exit if we've reached the limit
                if len(urls) >= max_urls:
                    break

            # Log provider statistics
            stats = provider.get_stats()
            logger.info(
                f"Sitemap loaded: {stats['urls_discovered']} discovered, "
                f"{len(urls)} valid, {stats['invalid_urls']} invalid, "
                f"{stats['duplicate_urls']} duplicates"
            )

    except Exception as e:
        error_msg = f"Failed to load sitemap {sitemap_url}: {e}"
        logger.error(error_msg)
        raise SitemapProviderError(error_msg)

    if not urls:
        raise SitemapProviderError(f"No valid URLs found in sitemap: {sitemap_url}")

    return urls