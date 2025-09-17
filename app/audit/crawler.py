"""Main crawler engine for orchestrating URL discovery and page processing.

This module coordinates all crawler components including input providers,
scope matching, queue management, and rate limiting to produce PagePlans
for downstream processing by the browser capture engine.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, AsyncIterator, Optional, Dict, Any, Set
from contextlib import asynccontextmanager

from .models.crawl import CrawlConfig, CrawlMetrics, CrawlStats, PagePlan, DiscoveryMode
from .utils.scope_matcher import ScopeMatcher, create_scope_matcher_from_config
from .queue.frontier_queue import FrontierQueue, QueuePriority, QueueClosedError
from .queue.rate_limiter import PerHostRateLimiter
from .input.seed_provider import SeedListProvider
from .input.sitemap_provider import SitemapProvider
from .input.dom_provider import DomLinkProvider, MockDomLinkProvider

# Import placeholder for browser context - will be replaced when EPIC 2 is implemented
try:
    from playwright.async_api import BrowserContext
except ImportError:
    BrowserContext = None


logger = logging.getLogger(__name__)


class CrawlerError(Exception):
    """Raised when crawler encounters a fatal error."""
    pass


class Crawler:
    """Main crawler engine for URL discovery and orchestration.
    
    The crawler coordinates multiple components to discover and filter URLs
    according to configured rules, producing PagePlans for downstream processing.
    
    Components:
    - Input providers (seeds, sitemap, DOM)
    - Scope matcher for URL filtering
    - Frontier queue with deduplication
    - Per-host rate limiting
    - Worker management and coordination
    """
    
    def __init__(
        self,
        config: CrawlConfig,
        browser_context: Optional['BrowserContext'] = None
    ):
        """Initialize crawler with configuration.
        
        Args:
            config: Crawl configuration
            browser_context: Playwright browser context (from EPIC 2)
        """
        self.config = config
        self.browser_context = browser_context
        
        # Initialize components
        self.scope_matcher = create_scope_matcher_from_config(config)
        self.frontier_queue = FrontierQueue(
            max_size=max(config.max_pages * 2, 1000),  # Buffer for discovery
            backpressure_threshold=0.8
        )
        self.rate_limiter = PerHostRateLimiter(
            default_requests_per_second=config.requests_per_second,
            default_max_concurrent=config.max_concurrent_per_host
        )
        
        # Initialize input providers
        self._input_providers = self._create_input_providers()
        
        # State tracking
        self._metrics = CrawlMetrics(config=config)
        self._running = False
        self._discovery_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # URL tracking for emission and DOM discovery
        self._processed_urls: Set[str] = set()
        self._processing_urls: Set[str] = set()
        self._emitted_urls: Set[str] = set()
        self._urls_emitted = 0
        self._lock = asyncio.Lock()

        # DOM discovery task pool
        self._dom_semaphore = asyncio.Semaphore(config.max_concurrency)
        self._dom_tasks: Set[asyncio.Task] = set()

        # Robots.txt cache
        self._robots_cache: Dict[str, Dict[str, bool]] = {}

        # Failed URLs (for potential retry)
        self._failed_urls: Set[str] = set()
    
    def _create_input_providers(self) -> Dict[str, Any]:
        """Create input providers based on discovery mode."""
        providers = {}
        
        # Always create seed provider if seeds are available
        if self.config.seeds:
            providers['seeds'] = SeedListProvider(
                seeds=[str(url) for url in self.config.seeds]
            )
        
        # Create sitemap provider if needed
        if (self.config.discovery_mode in (DiscoveryMode.SITEMAP, DiscoveryMode.HYBRID) 
            and self.config.sitemap_url):
            providers['sitemap'] = SitemapProvider(
                sitemap_url=str(self.config.sitemap_url),
                max_urls=self.config.max_pages,
                timeout=self.config.page_timeout
            )
        
        # Create DOM provider if needed
        if (self.config.discovery_mode in (DiscoveryMode.DOM, DiscoveryMode.HYBRID)):
            if self.browser_context:
                providers['dom'] = DomLinkProvider(
                    browser_context=self.browser_context,
                    load_wait_strategy=self.config.load_wait_strategy,
                    load_wait_timeout=self.config.load_wait_timeout,
                    load_wait_selector=self.config.load_wait_selector,
                    load_wait_js=self.config.load_wait_js
                )
            else:
                logger.warning("Browser context not available, using mock DOM provider")
                providers['dom'] = MockDomLinkProvider()
        
        return providers

    async def _is_allowed_by_robots(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        if not self.config.respect_robots:
            return True

        try:
            from urllib.parse import urlparse, urljoin
            parsed = urlparse(url)
            host = parsed.netloc.lower()

            # Check cache first
            if host in self._robots_cache:
                path = parsed.path or '/'
                # Simple robots.txt check - look for exact path or wildcard rules
                robots_rules = self._robots_cache[host]
                if path in robots_rules:
                    return robots_rules[path]
                # Check for wildcard match (simple implementation)
                for rule_path, allowed in robots_rules.items():
                    if rule_path.endswith('*') and path.startswith(rule_path[:-1]):
                        return allowed
                return True  # Default allow if no specific rule

            # Fetch robots.txt (simple implementation)
            robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
            self._robots_cache[host] = {}  # Initialize cache for this host

            try:
                import aiohttp
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(robots_url) as response:
                        if response.status == 200:
                            robots_text = await response.text()
                            # Parse robots.txt for User-agent: * rules
                            lines = robots_text.split('\n')
                            in_user_agent_section = False
                            for line in lines:
                                line = line.strip()
                                if line.lower().startswith('user-agent:'):
                                    in_user_agent_section = '*' in line or 'user-agent: *' in line.lower()
                                elif in_user_agent_section and line.lower().startswith('disallow:'):
                                    disallowed_path = line.split(':', 1)[1].strip()
                                    if disallowed_path:
                                        self._robots_cache[host][disallowed_path] = False
            except Exception:
                # If robots.txt fetch fails, default to allowing
                pass

            # Check again with populated cache
            path = parsed.path or '/'
            robots_rules = self._robots_cache[host]
            if path in robots_rules:
                return robots_rules[path]
            # Check for wildcard match
            for rule_path, allowed in robots_rules.items():
                if rule_path.endswith('*') and path.startswith(rule_path[:-1]):
                    return allowed
            return True  # Default allow

        except Exception as e:
            logger.debug(f"Robots.txt check failed for {url}: {e}")
            return True  # Default to allow on error
    
    async def crawl(self) -> AsyncIterator[PagePlan]:
        """Execute the crawl and yield discovered PagePlans.
        
        Yields:
            PagePlan objects ready for processing by browser capture engine
        """
        if self._running:
            raise CrawlerError("Crawler is already running")
        
        self._running = True
        self._metrics.stats.start_time = datetime.utcnow()
        self._metrics.is_running = True
        
        logger.info(f"Starting crawl with config: {self.config.discovery_mode}")
        
        try:
            async with self._crawler_context():
                # Start URL discovery
                self._discovery_task = asyncio.create_task(self._discover_urls())

                # Yield PagePlans as they become available and trigger DOM discovery
                async for page_plan in self._process_queue():
                    if self._should_stop():
                        break
                    yield page_plan
                    
        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            raise CrawlerError(f"Crawl execution failed: {e}")
        finally:
            await self._cleanup()
            self._running = False
            self._metrics.is_running = False
            self._metrics.stats.end_time = datetime.utcnow()
            logger.info(f"Crawl completed: {self._metrics.export_summary()}")
    
    @asynccontextmanager
    async def _crawler_context(self):
        """Async context manager for crawler resources."""
        try:
            # Initialize sitemap provider context if needed
            if 'sitemap' in self._input_providers:
                sitemap_provider = self._input_providers['sitemap']
                await sitemap_provider.__aenter__()
            
            yield
            
        finally:
            # Cleanup sitemap provider
            if 'sitemap' in self._input_providers:
                sitemap_provider = self._input_providers['sitemap']
                await sitemap_provider.__aexit__(None, None, None)
    
    async def _discover_urls(self):
        """Background task for URL discovery from all input providers."""
        try:
            # Discover from seeds first (highest priority)
            if 'seeds' in self._input_providers:
                seed_provider = self._input_providers['seeds']
                async for page_plan in seed_provider.discover_urls(depth=0):
                    await self._enqueue_page_plan(page_plan, QueuePriority.HIGH)
            
            # Discover from sitemap
            if 'sitemap' in self._input_providers:
                sitemap_provider = self._input_providers['sitemap']
                async for page_plan in sitemap_provider.discover_urls(depth=0):
                    await self._enqueue_page_plan(page_plan, QueuePriority.NORMAL)
            
            # DOM discovery happens during processing (in workers)
            
        except Exception as e:
            logger.error(f"URL discovery failed: {e}")
        finally:
            logger.debug("URL discovery completed")
    
    async def _enqueue_page_plan(self, page_plan: PagePlan, priority: QueuePriority):
        """Enqueue a PagePlan if it passes scope checks.

        Args:
            page_plan: PagePlan to enqueue
            priority: Queue priority for the page
        """
        # Track discovered URLs (before scope/limit checks)
        self._metrics.stats.urls_discovered += 1

        # Track unique hosts
        try:
            from urllib.parse import urlparse
            parsed = urlparse(str(page_plan.url))
            if parsed.netloc:
                self._metrics.stats.unique_hosts.add(parsed.netloc.lower())
        except Exception:
            # If URL parsing fails, still continue with processing
            pass

        # Check scope
        if not self.scope_matcher.is_in_scope(str(page_plan.url)):
            self._metrics.stats.urls_skipped += 1
            logger.debug(f"URL out of scope: {page_plan.url}")
            return

        # Check robots.txt
        if not await self._is_allowed_by_robots(str(page_plan.url)):
            self._metrics.stats.urls_skipped += 1
            logger.debug(f"URL disallowed by robots.txt: {page_plan.url}")
            return

        # Check limits
        if self._urls_emitted >= self.config.max_pages:
            logger.info(f"Reached max pages limit: {self.config.max_pages}")
            return

        # Enqueue
        success = await self.frontier_queue.put(page_plan, priority)
        if success:
            self._metrics.stats.urls_queued += 1
            self._metrics.queue_size = self.frontier_queue.qsize()
            logger.debug(f"Enqueued: {page_plan.url}")
        else:
            self._metrics.stats.urls_skipped += 1
    
    async def _schedule_dom_discovery(self, page_plan: PagePlan):
        """Schedule DOM discovery for a page without blocking emission.

        Args:
            page_plan: PagePlan to discover links from
        """
        if (self.config.discovery_mode not in (DiscoveryMode.DOM, DiscoveryMode.HYBRID)
            or 'dom' not in self._input_providers):
            return

        # Create DOM discovery task
        task = asyncio.create_task(self._discover_dom_links(page_plan))
        self._dom_tasks.add(task)

        # Clean up completed tasks
        task.add_done_callback(lambda t: self._dom_tasks.discard(t))
    
    async def _discover_dom_links(self, page_plan: PagePlan):
        """Discover DOM links from a page using rate limiting and concurrency control.

        Args:
            page_plan: PagePlan to discover links from
        """
        url = str(page_plan.url)

        # Check if already processing, processed, or permanently failed
        async with self._lock:
            if url in self._processing_urls or url in self._processed_urls or url in self._failed_urls:
                return
            self._processing_urls.add(url)

        # Use semaphore to limit concurrent DOM discovery
        async with self._dom_semaphore:
            try:
                # Rate limiting
                limiter = await self.rate_limiter.acquire(url, timeout=30.0)
                if not limiter:
                    logger.warning(f"Rate limit timeout for DOM discovery: {url}")
                    self._metrics.add_error(url, "rate_limit", "Timeout waiting for rate limit")
                    # Don't mark as processed - allow retry later
                    async with self._lock:
                        self._processing_urls.discard(url)
                    return

                try:
                    # Apply download delay if configured
                    if self.config.download_delay_ms:
                        await asyncio.sleep(self.config.download_delay_ms / 1000.0)

                    dom_provider = self._input_providers['dom']
                    new_depth = page_plan.depth + 1

                    # Check depth limit
                    if self.config.max_depth is None or new_depth <= self.config.max_depth:
                        async for discovered_plan in dom_provider.discover_urls_from_page(
                            url, new_depth, self._processed_urls
                        ):
                            await self._enqueue_page_plan(discovered_plan, QueuePriority.LOW)

                    # Record success and mark as processed
                    await self.rate_limiter.record_response(url, 200)
                    self._metrics.stats.urls_dom_processed += 1

                    async with self._lock:
                        self._processing_urls.discard(url)
                        self._processed_urls.add(url)

                finally:
                    await limiter.release()

            except Exception as e:
                logger.error(f"Error in DOM discovery for {url}: {e}")
                self._metrics.add_error(url, "dom_discovery_error", str(e))
                self._metrics.stats.urls_failed += 1

                # Record error with rate limiter for backoff calculations
                error_type = self._classify_error(e)
                await self.rate_limiter.record_error(url, error_type)

                # Classify error type for retry decision
                async with self._lock:
                    self._processing_urls.discard(url)
                    if self._is_permanent_error(e):
                        # Permanent error - don't retry
                        self._failed_urls.add(url)
                    # Transient error - allow retry by not marking as processed or failed
    

    def _is_permanent_error(self, error: Exception) -> bool:
        """Determine if an error is permanent and should not be retried."""
        error_str = str(error).lower()

        # Permanent HTTP errors
        if any(code in error_str for code in ['404', '403', '401', '400']):
            return True

        # DNS resolution errors
        if any(term in error_str for term in ['dns', 'resolve', 'name resolution']):
            return True

        # Invalid URL errors
        if 'invalid url' in error_str or 'malformed url' in error_str:
            return True

        # Otherwise consider it transient
        return False

    def _classify_error(self, error: Exception) -> str:
        """Classify an error for rate limiter backoff calculations.

        Args:
            error: The exception that occurred

        Returns:
            Error type string for rate limiter
        """
        error_name = type(error).__name__.lower()
        error_str = str(error).lower()

        # Network-related errors
        if any(term in error_str for term in ['timeout', 'timed out']):
            return 'timeout'
        elif any(term in error_str for term in ['connection', 'network', 'dns', 'resolve']):
            return 'network'
        # Check for specific HTTP status codes in error messages
        elif any(code in error_str for code in ['500', '502', '503', '504']) or \
             ('http' in error_str and any(code in error_str for code in ['500', '502', '503', '504'])):
            return 'server_error'
        elif '429' in error_str or ('http' in error_str and '429' in error_str):
            return 'rate_limited'
        elif any(code in error_str for code in ['404', '403', '401']) or \
             ('http' in error_str and any(code in error_str for code in ['404', '403', '401'])):
            return 'client_error'
        else:
            return 'unknown'
    
    async def _process_queue(self) -> AsyncIterator[PagePlan]:
        """Process the frontier queue and yield PagePlans for browser capture.

        Yields:
            PagePlan objects ready for browser processing
        """
        while not self._should_stop():
            try:
                page_plan = await self.frontier_queue.get(timeout=1.0)
                if page_plan is None:
                    # Check if discovery is complete and queue is empty
                    if (self._discovery_task and self._discovery_task.done()
                        and self.frontier_queue.empty()):
                        break
                    continue

                url = str(page_plan.url)

                # Check if already emitted (prevent duplicates)
                async with self._lock:
                    if url in self._emitted_urls:
                        continue
                    self._emitted_urls.add(url)
                    self._urls_emitted += 1
                    self._metrics.stats.urls_emitted += 1

                # Update metrics
                self._metrics.current_url = url
                self._metrics.queue_size = self.frontier_queue.qsize()

                # Schedule DOM discovery for this page (non-blocking)
                await self._schedule_dom_discovery(page_plan)

                yield page_plan

            except QueueClosedError:
                break
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
    
    def _should_stop(self) -> bool:
        """Check if crawler should stop based on limits and conditions."""
        # Check page limit (based on emitted pages, not processed)
        if self._urls_emitted >= self.config.max_pages:
            return True

        # Check shutdown signal
        if self._shutdown_event.is_set():
            return True

        # Check if discovery is done and queue is empty
        if (self._discovery_task and self._discovery_task.done()
            and self.frontier_queue.empty()
            and not self._processing_urls):
            return True

        return False
    
    async def stop(self):
        """Gracefully stop the crawler."""
        logger.info("Stopping crawler...")
        self._shutdown_event.set()

        # Cancel discovery task
        if self._discovery_task and not self._discovery_task.done():
            self._discovery_task.cancel()

        # Close queue
        await self.frontier_queue.close()

        # Wait for DOM discovery tasks to finish
        if self._dom_tasks:
            await asyncio.gather(*self._dom_tasks, return_exceptions=True)

        await self._cleanup()
    
    async def _cleanup(self):
        """Cleanup crawler resources."""
        # Stop rate limiter
        await self.rate_limiter.close()

        # Cancel any remaining tasks
        if self._discovery_task and not self._discovery_task.done():
            self._discovery_task.cancel()

        # Cancel remaining DOM discovery tasks
        for task in self._dom_tasks:
            if not task.done():
                task.cancel()

        if self._dom_tasks:
            await asyncio.gather(*self._dom_tasks, return_exceptions=True)

        self._dom_tasks.clear()
        logger.info("Crawler cleanup completed")
    
    def get_metrics(self) -> CrawlMetrics:
        """Get current crawler metrics.
        
        Returns:
            Current CrawlMetrics object
        """
        # Update queue stats
        self._metrics.queue_size = self.frontier_queue.qsize()
        
        # Update stats from components
        queue_stats = self.frontier_queue.get_stats()
        self._metrics.stats.urls_deduplicated = queue_stats.get("deduplicated_total", 0)
        
        rate_stats = self.rate_limiter.get_summary_stats()
        self._metrics.stats.rate_limit_hits = rate_stats.get("total_rate_limited", 0)
        
        return self._metrics
    
    def get_component_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed statistics from all components.
        
        Returns:
            Dictionary with stats from all crawler components
        """
        stats = {
            "crawler": self._metrics.export_summary(),
            "frontier_queue": self.frontier_queue.get_stats(),
            "rate_limiter": self.rate_limiter.get_summary_stats(),
            "scope_matcher": self.scope_matcher.get_scope_info()
        }
        
        # Add input provider stats
        for name, provider in self._input_providers.items():
            if hasattr(provider, 'get_stats'):
                stats[f"provider_{name}"] = provider.get_stats()
        
        return stats