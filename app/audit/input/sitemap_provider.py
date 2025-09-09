"""Sitemap XML provider for URL discovery from sitemap files.

This module handles URL discovery from sitemap.xml files, supporting
standard sitemaps, sitemap index files, and gzip-compressed sitemaps.
"""

import asyncio
import gzip
import xml.etree.ElementTree as ET
from typing import List, AsyncIterator, Optional, Set
from urllib.parse import urljoin, urlparse
import logging
import aiohttp
from datetime import datetime

from ..models.crawl import PagePlan
from ..utils.url_normalizer import is_valid_http_url, normalize


logger = logging.getLogger(__name__)


class SitemapProviderError(Exception):
    """Raised when sitemap provider encounters an error."""
    pass


class SitemapProvider:
    """Provider for URLs from sitemap.xml files.
    
    Features:
    - Standard sitemap.xml format parsing
    - Sitemap index file support (nested sitemaps)
    - Gzip-compressed sitemap support
    - Recursive sitemap discovery
    - URL validation and filtering
    """
    
    # XML namespaces commonly used in sitemaps
    SITEMAP_NS = {
        'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
        'image': 'http://www.google.com/schemas/sitemap-image/1.1',
        'news': 'http://www.google.com/schemas/sitemap-news/0.9',
        'video': 'http://www.google.com/schemas/sitemap-video/1.1'
    }
    
    def __init__(
        self,
        sitemap_url: str,
        max_urls: int = 50000,
        max_depth: int = 5,
        timeout: int = 30,
        user_agent: str = "TagSentinel/1.0 (+https://github.com/tag-sentinel)",
        validate_urls: bool = True,
        skip_invalid: bool = True
    ):
        """Initialize sitemap provider.
        
        Args:
            sitemap_url: URL of the sitemap.xml file
            max_urls: Maximum URLs to discover from sitemaps
            max_depth: Maximum depth for nested sitemap discovery
            timeout: HTTP request timeout in seconds
            user_agent: User-Agent string for HTTP requests
            validate_urls: Whether to validate discovered URLs
            skip_invalid: Whether to skip invalid URLs or raise error
        """
        self.sitemap_url = sitemap_url
        self.max_urls = max_urls
        self.max_depth = max_depth
        self.timeout = timeout
        self.user_agent = user_agent
        self.validate_urls = validate_urls
        self.skip_invalid = skip_invalid
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._stats = {
            "sitemaps_processed": 0,
            "sitemaps_failed": 0,
            "urls_discovered": 0,
            "invalid_urls": 0,
            "duplicate_urls": 0,
            "compressed_sitemaps": 0,
            "index_sitemaps": 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session is available."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {'User-Agent': self.user_agent}
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
            )
    
    async def close(self):
        """Close HTTP session and cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def discover_urls(self, depth: int = 0) -> AsyncIterator[PagePlan]:
        """Discover URLs from sitemap sources.
        
        Args:
            depth: Initial depth level for discovered URLs
            
        Yields:
            PagePlan objects for each URL found in sitemaps
        """
        await self._ensure_session()
        
        seen_urls: Set[str] = set()
        processed_sitemaps: Set[str] = set()
        
        try:
            async for page_plan in self._process_sitemap(
                self.sitemap_url, depth, seen_urls, processed_sitemaps, 0
            ):
                if len(seen_urls) >= self.max_urls:
                    logger.warning(f"Reached maximum URL limit ({self.max_urls})")
                    break
                yield page_plan
                
        except Exception as e:
            error_msg = f"Failed to process sitemap {self.sitemap_url}: {e}"
            logger.error(error_msg)
            if not self.skip_invalid:
                raise SitemapProviderError(error_msg)
        
        logger.info(f"Sitemap discovery completed: {self._stats}")
    
    async def _process_sitemap(
        self,
        sitemap_url: str,
        depth: int,
        seen_urls: Set[str],
        processed_sitemaps: Set[str],
        sitemap_depth: int
    ) -> AsyncIterator[PagePlan]:
        """Process a single sitemap file recursively.
        
        Args:
            sitemap_url: URL of sitemap to process
            depth: Depth level for discovered URLs
            seen_urls: Set of seen URLs for deduplication
            processed_sitemaps: Set of processed sitemap URLs
            sitemap_depth: Current sitemap nesting depth
            
        Yields:
            PagePlan objects for URLs in the sitemap
        """
        # Prevent infinite recursion and re-processing
        if sitemap_depth >= self.max_depth:
            logger.warning(f"Maximum sitemap depth reached: {sitemap_url}")
            return
        
        normalized_sitemap_url = normalize(sitemap_url)
        if normalized_sitemap_url in processed_sitemaps:
            logger.debug(f"Sitemap already processed: {sitemap_url}")
            return
        
        processed_sitemaps.add(normalized_sitemap_url)
        
        try:
            # Fetch sitemap content
            sitemap_content = await self._fetch_sitemap(sitemap_url)
            
            # Parse XML content
            try:
                root = ET.fromstring(sitemap_content)
            except ET.ParseError as e:
                self._stats["sitemaps_failed"] += 1
                error_msg = f"Invalid XML in sitemap {sitemap_url}: {e}"
                logger.error(error_msg)
                if not self.skip_invalid:
                    raise SitemapProviderError(error_msg)
                return
            
            # Determine sitemap type and process accordingly
            if self._is_sitemap_index(root):
                self._stats["index_sitemaps"] += 1
                logger.debug(f"Processing sitemap index: {sitemap_url}")
                
                # Process nested sitemaps
                async for page_plan in self._process_sitemap_index(
                    root, sitemap_url, depth, seen_urls, processed_sitemaps, sitemap_depth
                ):
                    yield page_plan
            else:
                # Process regular sitemap
                logger.debug(f"Processing sitemap: {sitemap_url}")
                async for page_plan in self._process_regular_sitemap(
                    root, sitemap_url, depth, seen_urls
                ):
                    yield page_plan
            
            self._stats["sitemaps_processed"] += 1
            
        except Exception as e:
            self._stats["sitemaps_failed"] += 1
            error_msg = f"Error processing sitemap {sitemap_url}: {e}"
            logger.error(error_msg)
            if not self.skip_invalid:
                raise SitemapProviderError(error_msg)
    
    async def _fetch_sitemap(self, sitemap_url: str) -> bytes:
        """Fetch sitemap content from URL.
        
        Args:
            sitemap_url: URL of sitemap to fetch
            
        Returns:
            Sitemap content as bytes
        """
        logger.debug(f"Fetching sitemap: {sitemap_url}")
        
        async with self._session.get(sitemap_url) as response:
            if response.status != 200:
                raise SitemapProviderError(
                    f"HTTP {response.status} fetching sitemap: {sitemap_url}"
                )
            
            content = await response.read()
            
            # Check if content is gzipped
            if self._is_gzipped_content(content, response.headers):
                try:
                    content = gzip.decompress(content)
                    self._stats["compressed_sitemaps"] += 1
                    logger.debug(f"Decompressed gzipped sitemap: {sitemap_url}")
                except Exception as e:
                    raise SitemapProviderError(f"Failed to decompress gzipped sitemap: {e}")
            
            return content
    
    def _is_gzipped_content(self, content: bytes, headers) -> bool:
        """Check if content is gzipped.
        
        Args:
            content: Content bytes to check
            headers: HTTP response headers
            
        Returns:
            True if content appears to be gzipped
        """
        # Check Content-Encoding header
        if headers.get('content-encoding') == 'gzip':
            return True
        
        # Check gzip magic number
        return content.startswith(b'\x1f\x8b')
    
    def _is_sitemap_index(self, root: ET.Element) -> bool:
        """Check if XML represents a sitemap index file.
        
        Args:
            root: XML root element
            
        Returns:
            True if this is a sitemap index file
        """
        # Look for sitemapindex root element
        if root.tag.endswith('}sitemapindex') or root.tag == 'sitemapindex':
            return True
        
        # Look for sitemap children (index files contain sitemap elements)
        for child in root:
            if child.tag.endswith('}sitemap') or child.tag == 'sitemap':
                return True
        
        return False
    
    async def _process_sitemap_index(
        self,
        root: ET.Element,
        base_url: str,
        depth: int,
        seen_urls: Set[str],
        processed_sitemaps: Set[str],
        sitemap_depth: int
    ) -> AsyncIterator[PagePlan]:
        """Process sitemap index file.
        
        Args:
            root: XML root element of sitemap index
            base_url: Base URL for resolving relative URLs
            depth: Depth level for URLs
            seen_urls: Set of seen URLs
            processed_sitemaps: Set of processed sitemaps
            sitemap_depth: Current sitemap nesting depth
            
        Yields:
            PagePlan objects from nested sitemaps
        """
        # Find all sitemap elements
        sitemap_elements = root.findall('.//sitemap:sitemap', self.SITEMAP_NS)
        if not sitemap_elements:
            # Fallback without namespace
            sitemap_elements = root.findall('.//sitemap')
        
        for sitemap_elem in sitemap_elements:
            # Extract sitemap URL
            loc_elem = sitemap_elem.find('sitemap:loc', self.SITEMAP_NS)
            if loc_elem is None:
                loc_elem = sitemap_elem.find('loc')
            
            if loc_elem is None or not loc_elem.text:
                logger.warning("Sitemap element missing loc")
                continue
            
            nested_sitemap_url = urljoin(base_url, loc_elem.text.strip())
            
            # Process nested sitemap
            async for page_plan in self._process_sitemap(
                nested_sitemap_url, depth, seen_urls, processed_sitemaps, sitemap_depth + 1
            ):
                yield page_plan
    
    async def _process_regular_sitemap(
        self,
        root: ET.Element,
        base_url: str,
        depth: int,
        seen_urls: Set[str]
    ) -> AsyncIterator[PagePlan]:
        """Process regular sitemap file.
        
        Args:
            root: XML root element of sitemap
            base_url: Base URL for resolving relative URLs
            depth: Depth level for URLs
            seen_urls: Set of seen URLs
            
        Yields:
            PagePlan objects for URLs in sitemap
        """
        # Find all URL elements
        url_elements = root.findall('.//sitemap:url', self.SITEMAP_NS)
        if not url_elements:
            # Fallback without namespace
            url_elements = root.findall('.//url')
        
        for url_elem in url_elements:
            # Extract URL location
            loc_elem = url_elem.find('sitemap:loc', self.SITEMAP_NS)
            if loc_elem is None:
                loc_elem = url_elem.find('loc')
            
            if loc_elem is None or not loc_elem.text:
                logger.debug("URL element missing loc")
                continue
            
            url = urljoin(base_url, loc_elem.text.strip())
            
            # Extract metadata
            metadata = self._extract_url_metadata(url_elem, base_url)
            
            # Process URL
            async for page_plan in self._process_url(url, depth, seen_urls, metadata):
                yield page_plan
    
    def _extract_url_metadata(self, url_elem: ET.Element, base_url: str) -> dict:
        """Extract metadata from sitemap URL element.
        
        Args:
            url_elem: URL XML element
            base_url: Base URL for context
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "source": "sitemap",
            "sitemap_url": base_url
        }
        
        # Extract lastmod
        lastmod_elem = url_elem.find('sitemap:lastmod', self.SITEMAP_NS)
        if lastmod_elem is None:
            lastmod_elem = url_elem.find('lastmod')
        if lastmod_elem is not None and lastmod_elem.text:
            try:
                # Parse ISO 8601 date
                lastmod = datetime.fromisoformat(lastmod_elem.text.strip().replace('Z', '+00:00'))
                metadata["lastmod"] = lastmod.isoformat()
            except (ValueError, TypeError):
                logger.debug(f"Invalid lastmod format: {lastmod_elem.text}")
        
        # Extract priority
        priority_elem = url_elem.find('sitemap:priority', self.SITEMAP_NS)
        if priority_elem is None:
            priority_elem = url_elem.find('priority')
        if priority_elem is not None and priority_elem.text:
            try:
                priority = float(priority_elem.text.strip())
                if 0.0 <= priority <= 1.0:
                    metadata["priority"] = priority
            except (ValueError, TypeError):
                logger.debug(f"Invalid priority format: {priority_elem.text}")
        
        # Extract changefreq
        changefreq_elem = url_elem.find('sitemap:changefreq', self.SITEMAP_NS)
        if changefreq_elem is None:
            changefreq_elem = url_elem.find('changefreq')
        if changefreq_elem is not None and changefreq_elem.text:
            metadata["changefreq"] = changefreq_elem.text.strip()
        
        return metadata
    
    async def _process_url(
        self,
        url: str,
        depth: int,
        seen_urls: Set[str],
        metadata: dict
    ) -> AsyncIterator[PagePlan]:
        """Process a single URL from sitemap.
        
        Args:
            url: URL to process
            depth: Depth level for the URL
            seen_urls: Set of seen URLs
            metadata: URL metadata from sitemap
            
        Yields:
            PagePlan if URL is valid and not duplicate
        """
        self._stats["urls_discovered"] += 1
        
        try:
            # Validate URL if requested
            if self.validate_urls:
                if not is_valid_http_url(url):
                    self._stats["invalid_urls"] += 1
                    logger.debug(f"Invalid URL from sitemap: {url}")
                    if not self.skip_invalid:
                        raise SitemapProviderError(f"Invalid URL: {url}")
                    return
            
            # Normalize for deduplication
            normalized_url = normalize(url)
            
            # Check for duplicates
            if normalized_url in seen_urls:
                self._stats["duplicate_urls"] += 1
                logger.debug(f"Duplicate URL from sitemap: {normalized_url}")
                return
            
            seen_urls.add(normalized_url)
            
            # Create PagePlan
            page_plan = PagePlan(
                url=normalized_url,
                source_url=None,
                depth=depth,
                discovery_method="sitemap",
                metadata=metadata
            )
            
            logger.debug(f"Valid URL from sitemap: {normalized_url}")
            yield page_plan
            
        except Exception as e:
            self._stats["invalid_urls"] += 1
            error_msg = f"Error processing URL '{url}' from sitemap: {e}"
            logger.warning(error_msg)
            if not self.skip_invalid:
                raise SitemapProviderError(error_msg)
    
    def get_stats(self) -> dict:
        """Get provider statistics.
        
        Returns:
            Dictionary with discovery statistics
        """
        return {
            "provider": "sitemap",
            **self._stats,
            "sitemap_url": self.sitemap_url,
            "max_urls": self.max_urls,
            "max_depth": self.max_depth
        }