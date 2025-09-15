"""DOM link discovery provider for extracting links from web pages.

This module handles URL discovery from web page DOM using Playwright,
supporting JavaScript-rendered content and configurable wait conditions.

NOTE: This module requires EPIC 2 (Browser Capture Engine) to be implemented
for full Playwright integration.
"""

import asyncio
from typing import List, AsyncIterator, Optional, Set, Dict, Any
from urllib.parse import urljoin, urlparse
import logging

from ..models.crawl import PagePlan, LoadWaitStrategy
from ..utils.url_normalizer import is_valid_http_url, normalize

# Placeholder imports - will be replaced when EPIC 2 is implemented
try:
    from playwright.async_api import Page, BrowserContext
except ImportError:
    # Fallback for when Playwright is not installed
    Page = None
    BrowserContext = None


logger = logging.getLogger(__name__)


class DomProviderError(Exception):
    """Raised when DOM provider encounters an error."""
    pass


class DomLinkProvider:
    """Provider for URLs discovered from web page DOM.
    
    This provider extracts links from web pages after they have fully loaded,
    including JavaScript-rendered content. It supports configurable wait
    conditions and custom selectors.
    
    Features:
    - JavaScript-rendered link extraction
    - Configurable wait conditions (networkidle, selector, timeout)
    - Custom link selectors and filtering
    - Handles page load failures gracefully
    - Extracts canonical links and meta redirects
    """
    
    def __init__(
        self,
        browser_context: Optional['BrowserContext'] = None,
        load_wait_strategy: LoadWaitStrategy = LoadWaitStrategy.NETWORKIDLE,
        load_wait_timeout: int = 30,
        load_wait_selector: Optional[str] = None,
        load_wait_js: Optional[str] = None,
        custom_selectors: Optional[List[str]] = None,
        extract_canonical: bool = True,
        extract_meta_refresh: bool = True,
        validate_urls: bool = True,
        skip_invalid: bool = True
    ):
        """Initialize DOM link provider.
        
        Args:
            browser_context: Playwright browser context (from EPIC 2)
            load_wait_strategy: Strategy for waiting for page load
            load_wait_timeout: Maximum time to wait for load condition
            load_wait_selector: CSS selector to wait for
            load_wait_js: JavaScript condition to wait for
            custom_selectors: Additional CSS selectors for link extraction
            extract_canonical: Whether to extract canonical URLs
            extract_meta_refresh: Whether to extract meta refresh URLs
            validate_urls: Whether to validate discovered URLs
            skip_invalid: Whether to skip invalid URLs or raise error
        """
        self.browser_context = browser_context
        self.load_wait_strategy = load_wait_strategy
        self.load_wait_timeout = load_wait_timeout
        self.load_wait_selector = load_wait_selector
        self.load_wait_js = load_wait_js
        self.custom_selectors = custom_selectors or []
        self.extract_canonical = extract_canonical
        self.extract_meta_refresh = extract_meta_refresh
        self.validate_urls = validate_urls
        self.skip_invalid = skip_invalid
        
        # Default selectors for link extraction
        self.default_selectors = [
            'a[href]',           # Standard links
            'area[href]',        # Image map areas
            'link[rel="alternate"][href]',  # Alternate links
        ]
        
        self._stats = {
            "pages_processed": 0,
            "pages_failed": 0,
            "links_discovered": 0,
            "canonical_links": 0,
            "meta_refresh_links": 0,
            "invalid_urls": 0,
            "duplicate_urls": 0,
            "javascript_links": 0,
            "load_timeouts": 0
        }
    
    async def discover_urls_from_page(
        self, 
        source_url: str, 
        depth: int = 1,
        seen_urls: Optional[Set[str]] = None
    ) -> AsyncIterator[PagePlan]:
        """Discover URLs from a single web page.
        
        Args:
            source_url: URL of the page to extract links from
            depth: Depth level for discovered URLs
            seen_urls: Set of already seen URLs for deduplication
            
        Yields:
            PagePlan objects for links found on the page
        """
        if seen_urls is None:
            seen_urls = set()
        
        if not self.browser_context:
            raise DomProviderError("Browser context required for DOM link discovery")
        
        page: Optional['Page'] = None
        
        try:
            # Create new page for this URL
            page = await self.browser_context.new_page()
            
            # Configure page for link extraction
            await self._configure_page(page)
            
            # Navigate to the page
            logger.debug(f"Navigating to page: {source_url}")
            await page.goto(source_url, wait_until="domcontentloaded")
            
            # Wait for load condition and capture timing
            load_state_info = await self._wait_for_load_condition(page, source_url)

            # Extract links from the page
            links = await self._extract_links_from_page(page, source_url)
            
            self._stats["pages_processed"] += 1
            logger.debug(f"Extracted {len(links)} links from {source_url}")
            
            # Process each discovered link
            for link_info in links:
                async for page_plan in self._process_link(
                    link_info, source_url, depth, seen_urls, load_state_info
                ):
                    yield page_plan
                    
        except Exception as e:
            self._stats["pages_failed"] += 1
            error_msg = f"Failed to process page {source_url}: {e}"
            logger.error(error_msg)
            if not self.skip_invalid:
                raise DomProviderError(error_msg)
        finally:
            if page:
                await page.close()
    
    async def _configure_page(self, page: 'Page'):
        """Configure page for optimal link extraction.
        
        Args:
            page: Playwright page object
        """
        # Set reasonable viewport
        await page.set_viewport_size({"width": 1280, "height": 720})
        
        # Disable images and media for faster loading (optional)
        # This can be made configurable
        await page.route("**/*.{png,jpg,jpeg,gif,svg,ico,webp}", lambda route: route.abort())
        await page.route("**/*.{mp4,avi,mov,wmv,flv,webm}", lambda route: route.abort())
        
        # Set user agent if needed
        # await page.set_extra_http_headers({"User-Agent": "..."})
    
    async def _wait_for_load_condition(self, page: 'Page', url: str):
        """Wait for page load condition based on configured strategy.

        Args:
            page: Playwright page object
            url: URL being loaded (for logging)

        Returns:
            Dict with load state information including timing
        """
        import time
        start_time = time.time()

        try:
            if self.load_wait_strategy == LoadWaitStrategy.NETWORKIDLE:
                await page.wait_for_load_state("networkidle", timeout=self.load_wait_timeout * 1000)

            elif self.load_wait_strategy == LoadWaitStrategy.SELECTOR:
                if not self.load_wait_selector:
                    raise DomProviderError("load_wait_selector required for selector wait strategy")
                await page.wait_for_selector(
                    self.load_wait_selector,
                    timeout=self.load_wait_timeout * 1000
                )

            elif self.load_wait_strategy == LoadWaitStrategy.TIMEOUT:
                await asyncio.sleep(min(self.load_wait_timeout, 10))  # Cap at 10 seconds

            elif self.load_wait_strategy == LoadWaitStrategy.CUSTOM:
                if not self.load_wait_js:
                    raise DomProviderError("load_wait_js required for custom wait strategy")
                await page.wait_for_function(
                    self.load_wait_js,
                    timeout=self.load_wait_timeout * 1000
                )

            end_time = time.time()
            wait_duration = round((end_time - start_time) * 1000)  # Convert to ms

            return {
                'load_strategy': self.load_wait_strategy,
                'wait_ms': wait_duration,
                'selector': self.load_wait_selector if self.load_wait_strategy == LoadWaitStrategy.SELECTOR else None,
                'timeout_ms': self.load_wait_timeout * 1000,
                'success': True
            }

        except Exception as e:
            end_time = time.time()
            wait_duration = round((end_time - start_time) * 1000)
            self._stats["load_timeouts"] += 1
            logger.warning(f"Load wait timeout for {url}: {e}")

            # Return load state info even on timeout
            return {
                'load_strategy': self.load_wait_strategy,
                'wait_ms': wait_duration,
                'selector': self.load_wait_selector if self.load_wait_strategy == LoadWaitStrategy.SELECTOR else None,
                'timeout_ms': self.load_wait_timeout * 1000,
                'success': False,
                'error': str(e)
            }
    
    async def _extract_links_from_page(
        self, 
        page: 'Page', 
        base_url: str
    ) -> List[Dict[str, Any]]:
        """Extract all links from the page.
        
        Args:
            page: Playwright page object
            base_url: Base URL for resolving relative links
            
        Returns:
            List of link information dictionaries
        """
        links = []
        
        # Combine default and custom selectors
        all_selectors = self.default_selectors + self.custom_selectors
        
        # Extract links using CSS selectors
        for selector in all_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    href = await element.get_attribute('href')
                    if href:
                        link_info = {
                            'url': urljoin(base_url, href.strip()),
                            'text': (await element.text_content() or '').strip(),
                            'selector': selector,
                            'type': 'standard'
                        }
                        links.append(link_info)
                        
            except Exception as e:
                logger.debug(f"Error extracting links with selector '{selector}': {e}")
        
        # Extract canonical link
        if self.extract_canonical:
            try:
                canonical_elem = await page.query_selector('link[rel="canonical"][href]')
                if canonical_elem:
                    canonical_href = await canonical_elem.get_attribute('href')
                    if canonical_href:
                        link_info = {
                            'url': urljoin(base_url, canonical_href.strip()),
                            'text': '',
                            'selector': 'link[rel="canonical"]',
                            'type': 'canonical'
                        }
                        links.append(link_info)
                        self._stats["canonical_links"] += 1
            except Exception as e:
                logger.debug(f"Error extracting canonical link: {e}")
        
        # Extract meta refresh redirect
        if self.extract_meta_refresh:
            try:
                meta_refresh = await page.query_selector('meta[http-equiv="refresh"][content]')
                if meta_refresh:
                    content = await meta_refresh.get_attribute('content')
                    if content:
                        # Parse meta refresh content (format: "delay;url=...")
                        parts = content.split(';', 1)
                        if len(parts) > 1:
                            url_part = parts[1].strip()
                            if url_part.lower().startswith('url='):
                                refresh_url = url_part[4:].strip('\'"')
                                link_info = {
                                    'url': urljoin(base_url, refresh_url),
                                    'text': '',
                                    'selector': 'meta[http-equiv="refresh"]',
                                    'type': 'meta_refresh'
                                }
                                links.append(link_info)
                                self._stats["meta_refresh_links"] += 1
            except Exception as e:
                logger.debug(f"Error extracting meta refresh: {e}")
        
        # Check for JavaScript-generated links (simplified approach)
        # This could be expanded to execute custom JavaScript for link discovery
        try:
            js_links = await page.evaluate('''
                () => {
                    const links = [];
                    // Look for onclick handlers that might navigate
                    const clickElements = document.querySelectorAll('[onclick]');
                    for (const elem of clickElements) {
                        const onclick = elem.getAttribute('onclick');
                        const match = onclick.match(/(?:location\\.href|window\\.open)\\s*=\\s*['"](.*?)['"]/) ||
                                     onclick.match(/(?:location\\.href|window\\.open)\\(['"](.*?)['"]\\)/) ||
                                     onclick.match(/(?:location)\\s*=\\s*['"](.*?)['"]/) ||
                                     onclick.match(/(?:location)\\(['"](.*?)['"]\\)/);
                        if (match && match[1]) {
                            links.push({
                                url: match[1],
                                text: (elem.textContent || '').trim(),
                                type: 'javascript_onclick'
                            });
                        }
                    }
                    return links;
                }
            ''')
            
            for js_link in js_links:
                link_info = {
                    'url': urljoin(base_url, js_link['url']),
                    'text': js_link['text'],
                    'selector': '[onclick]',
                    'type': 'javascript'
                }
                links.append(link_info)
                self._stats["javascript_links"] += 1
                
        except Exception as e:
            logger.debug(f"Error extracting JavaScript links: {e}")
        
        self._stats["links_discovered"] += len(links)
        return links
    
    async def _process_link(
        self,
        link_info: Dict[str, Any],
        source_url: str,
        depth: int,
        seen_urls: Set[str],
        load_state_info: Dict[str, Any] = None
    ) -> AsyncIterator[PagePlan]:
        """Process a single discovered link.

        Args:
            link_info: Link information dictionary
            source_url: URL where link was discovered
            depth: Depth level for the link
            seen_urls: Set of seen URLs
            load_state_info: Load state information from the source page

        Yields:
            PagePlan if link is valid and not duplicate
        """
        url = link_info['url']
        
        try:
            # Filter out non-HTTP(S) URLs
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                logger.debug(f"Skipping non-HTTP(S) URL: {url}")
                return
            
            # Validate URL if requested
            if self.validate_urls:
                if not is_valid_http_url(url):
                    self._stats["invalid_urls"] += 1
                    logger.debug(f"Invalid URL from DOM: {url}")
                    if not self.skip_invalid:
                        raise DomProviderError(f"Invalid URL: {url}")
                    return
            
            # Normalize for deduplication
            normalized_url = normalize(url)
            
            # Check for duplicates
            if normalized_url in seen_urls:
                self._stats["duplicate_urls"] += 1
                logger.debug(f"Duplicate URL from DOM: {normalized_url}")
                return
            
            seen_urls.add(normalized_url)
            
            # Create PagePlan
            metadata = {
                "source": "dom",
                "link_text": link_info.get('text', ''),
                "link_selector": link_info.get('selector', ''),
                "link_type": link_info.get('type', 'standard'),
                "source_page": source_url
            }

            # Include load state information if available
            if load_state_info:
                metadata["load_state"] = load_state_info
            
            page_plan = PagePlan(
                url=normalized_url,
                source_url=source_url,
                depth=depth,
                discovery_method="dom",
                metadata=metadata
            )
            
            logger.debug(f"Valid link from DOM: {normalized_url} (from {source_url})")
            yield page_plan
            
        except Exception as e:
            self._stats["invalid_urls"] += 1
            error_msg = f"Error processing link '{url}' from {source_url}: {e}"
            logger.warning(error_msg)
            if not self.skip_invalid:
                raise DomProviderError(error_msg)
    
    def get_stats(self) -> dict:
        """Get provider statistics.
        
        Returns:
            Dictionary with discovery statistics
        """
        return {
            "provider": "dom",
            **self._stats,
            "load_wait_strategy": self.load_wait_strategy.value,
            "load_wait_timeout": self.load_wait_timeout,
            "custom_selectors": len(self.custom_selectors),
            "extract_canonical": self.extract_canonical,
            "extract_meta_refresh": self.extract_meta_refresh
        }


# Placeholder implementation for when Playwright is not available
class MockDomLinkProvider(DomLinkProvider):
    """Mock DOM provider for testing when Playwright is not available."""
    
    def __init__(self, **kwargs):
        super().__init__(browser_context=None, **kwargs)
        logger.warning("Using mock DOM provider - Playwright not available")
    
    async def discover_urls_from_page(self, source_url: str, depth: int = 1, seen_urls: Optional[Set[str]] = None) -> AsyncIterator[PagePlan]:
        """Mock implementation that yields no URLs."""
        logger.warning(f"Mock DOM provider cannot extract links from {source_url}")
        return
        yield  # Make this a generator