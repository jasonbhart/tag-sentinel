"""Browser factory for creating and managing Playwright browser contexts.

This module provides the BrowserFactory class that handles browser lifecycle management,
context creation with configuration, and cleanup. It supports different browser engines,
custom headers, viewport configuration, and debug modes.
"""

import asyncio
import logging
import hashlib
import json
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any, List, AsyncGenerator
from pathlib import Path
from collections import OrderedDict

from playwright.async_api import (
    Browser, 
    BrowserContext, 
    Playwright, 
    async_playwright,
    Page
)

logger = logging.getLogger(__name__)


class BrowserEngineType:
    """Supported browser engine types."""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class BrowserConfig:
    """Configuration for browser creation and context setup."""
    
    def __init__(
        self,
        engine: str = BrowserEngineType.CHROMIUM,
        headless: bool = True,
        devtools: bool = False,
        slow_mo: int = 0,
        viewport: Optional[Dict[str, int]] = None,
        user_agent: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        proxy: Optional[Dict[str, str]] = None,
        ignore_https_errors: bool = False,
        java_script_enabled: bool = True,
        downloads_path: Optional[Path] = None,
        trace: bool = False,
        video: bool = False,
        screenshots: bool = False,
        har_path: Optional[Path] = None,
        permissions: Optional[List[str]] = None,
        geolocation: Optional[Dict[str, float]] = None,
        timezone: Optional[str] = None,
        locale: Optional[str] = None,
        color_scheme: Optional[str] = None,
        **kwargs
    ):
        """Initialize browser configuration.
        
        Args:
            engine: Browser engine to use (chromium, firefox, webkit)
            headless: Run browser in headless mode
            devtools: Open DevTools automatically
            slow_mo: Slow down operations by specified milliseconds
            viewport: Viewport size dict with 'width' and 'height'
            user_agent: Custom User-Agent string
            extra_headers: Additional HTTP headers for all requests
            proxy: Proxy configuration dict
            ignore_https_errors: Ignore SSL/TLS certificate errors
            java_script_enabled: Enable JavaScript execution
            downloads_path: Directory for downloads
            trace: Enable Playwright tracing
            video: Enable video recording
            screenshots: Enable screenshot capture
            har_path: Path for HAR file recording
            permissions: List of permissions to grant
            geolocation: Geolocation dict with 'latitude' and 'longitude'
            timezone: Timezone ID (e.g., 'America/New_York')
            locale: Locale for the browser context
            color_scheme: Color scheme preference ('dark' or 'light')
        """
        self.engine = engine
        self.headless = headless
        self.devtools = devtools
        self.slow_mo = slow_mo
        self.viewport = viewport or {'width': 1920, 'height': 1080}
        self.user_agent = user_agent
        self.extra_headers = extra_headers or {}
        self.proxy = proxy
        self.ignore_https_errors = ignore_https_errors
        self.java_script_enabled = java_script_enabled
        self.downloads_path = downloads_path
        self.trace = trace
        self.video = video
        self.screenshots = screenshots
        self.har_path = har_path
        self.permissions = permissions or []
        self.geolocation = geolocation
        self.timezone = timezone
        self.locale = locale
        self.color_scheme = color_scheme
        self.extra_options = kwargs
    
    def to_browser_options(self) -> Dict[str, Any]:
        """Convert to Playwright browser launch options."""
        options = {
            'headless': self.headless,
            'slow_mo': self.slow_mo,
        }
        
        if self.devtools and not self.headless:
            options['devtools'] = True
            
        # Add extra options
        options.update(self.extra_options)
        
        return options
    
    def to_context_options(self) -> Dict[str, Any]:
        """Convert to Playwright browser context options."""
        options = {}
        
        if self.viewport:
            options['viewport'] = self.viewport
        
        if self.user_agent:
            options['user_agent'] = self.user_agent
            
        if self.extra_headers:
            options['extra_http_headers'] = self.extra_headers
            
        if self.proxy:
            options['proxy'] = self.proxy
            
        if self.ignore_https_errors:
            options['ignore_https_errors'] = self.ignore_https_errors
            
        if not self.java_script_enabled:
            options['java_script_enabled'] = False
            
        if self.downloads_path:
            options['accept_downloads'] = True
            
        if self.permissions:
            options['permissions'] = self.permissions
            
        if self.geolocation:
            options['geolocation'] = self.geolocation
            
        if self.timezone:
            options['timezone_id'] = self.timezone
            
        if self.locale:
            options['locale'] = self.locale
            
        if self.color_scheme:
            options['color_scheme'] = self.color_scheme
        
        # Recording options
        if self.video:
            options['record_video_dir'] = str(self.downloads_path or Path.cwd() / "videos")
            
        if self.har_path:
            options['record_har_path'] = str(self.har_path)
            
        return options


class BrowserFactory:
    """Factory for creating and managing Playwright browser instances."""
    
    def __init__(self, config: BrowserConfig, enable_context_reuse: bool = False):
        """Initialize browser factory with configuration.

        Args:
            config: Browser configuration object
            enable_context_reuse: Enable context reuse pool for efficiency
        """
        self.config = config
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self._context_count = 0
        self._max_contexts = 50  # Prevent memory leaks

        # Context reuse pool
        self._enable_context_reuse = enable_context_reuse
        self._context_pool: OrderedDict[str, BrowserContext] = OrderedDict()
        self._max_pooled_contexts = 5
        
    async def start(self) -> None:
        """Start Playwright and launch browser."""
        if self.playwright is not None:
            logger.warning("Browser factory already started")
            return
            
        logger.info(f"Starting browser factory with engine: {self.config.engine}")
        
        try:
            self.playwright = await async_playwright().start()
            
            # Select browser engine
            if self.config.engine == BrowserEngineType.FIREFOX:
                browser_type = self.playwright.firefox
            elif self.config.engine == BrowserEngineType.WEBKIT:
                browser_type = self.playwright.webkit
            else:
                browser_type = self.playwright.chromium
            
            # Launch browser with options
            browser_options = self.config.to_browser_options()
            self.browser = await browser_type.launch(**browser_options)
            
            logger.info(f"Browser launched successfully (headless={self.config.headless})")
            
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop browser and cleanup resources."""
        logger.info("Stopping browser factory")

        try:
            # Clear context pool
            for context in self._context_pool.values():
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing pooled context: {e}")
            self._context_pool.clear()

            if self.browser:
                await self.browser.close()
                self.browser = None

            if self.playwright:
                await self.playwright.stop()
                self.playwright = None

            self._context_count = 0
            logger.info("Browser factory stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping browser factory: {e}")
    
    async def create_context(self, **context_overrides) -> BrowserContext:
        """Create a new browser context.
        
        Args:
            **context_overrides: Override default context options
            
        Returns:
            New browser context
            
        Raises:
            RuntimeError: If browser factory not started
        """
        if not self.browser:
            raise RuntimeError("Browser factory not started. Call start() first.")
        
        # Prevent memory leaks from too many contexts
        if self._context_count >= self._max_contexts:
            logger.warning(f"Context count limit ({self._max_contexts}) reached")
            
        try:
            # Merge configuration options with overrides
            context_options = self.config.to_context_options()
            context_options.update(context_overrides)
            
            context = await self.browser.new_context(**context_options)
            self._context_count += 1
            
            logger.debug(f"Created browser context #{self._context_count}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to create browser context: {e}")
            raise
    
    @asynccontextmanager
    async def context(self, **context_overrides) -> AsyncGenerator[BrowserContext, None]:
        """Context manager for browser context lifecycle.

        Args:
            **context_overrides: Override default context options

        Yields:
            Browser context that will be automatically closed or returned to pool
        """
        context = None
        trace_path = context_overrides.pop('record_trace_path', None)
        original_options = context_overrides.copy()

        try:
            # Try to get from pool first
            context = await self._get_pooled_context(context_overrides)

            if context is None:
                # Create new context if none available in pool
                context = await self.create_context(**context_overrides)

            # Start tracing if requested
            if trace_path:
                await context.tracing.start(screenshots=True, snapshots=True, sources=True)

            yield context

        finally:
            if context:
                # Stop tracing before returning to pool or closing
                if trace_path:
                    try:
                        await context.tracing.stop(path=trace_path)
                        logger.debug(f"Trace saved: {trace_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save trace: {e}")

                # Return to pool or close based on configuration and context state
                await self._return_context_to_pool(context, original_options)
                self._context_count -= 1
    
    @asynccontextmanager 
    async def page(self, **context_overrides) -> AsyncGenerator[Page, None]:
        """Context manager for a single page.
        
        Args:
            **context_overrides: Override default context options
            
        Yields:
            Page instance that will be automatically closed
        """
        async with self.context(**context_overrides) as context:
            page = await context.new_page()
            try:
                yield page
            finally:
                await page.close()
    
    async def get_browser_version(self) -> Optional[str]:
        """Get browser version information.
        
        Returns:
            Browser version string or None if not available
        """
        if not self.browser:
            return None
            
        try:
            return self.browser.version
        except Exception as e:
            logger.error(f"Failed to get browser version: {e}")
            return None

    def _get_context_key(self, context_options: Dict[str, Any]) -> str:
        """Generate a hash key for context options to enable reuse.

        Args:
            context_options: Context creation options

        Returns:
            Hash string that can be used as cache key
        """
        # Exclude certain options that shouldn't affect reuse
        reusable_options = context_options.copy()
        reusable_options.pop('record_har_path', None)  # HAR recording paths are unique
        reusable_options.pop('record_video_dir', None)  # Video paths are unique

        # Create stable hash from options
        options_str = json.dumps(reusable_options, sort_keys=True)
        return hashlib.md5(options_str.encode()).hexdigest()

    async def _get_pooled_context(self, context_options: Dict[str, Any]) -> Optional[BrowserContext]:
        """Get a reusable context from the pool if available.

        Args:
            context_options: Context creation options

        Returns:
            Existing context or None if not available
        """
        if not self._enable_context_reuse:
            return None

        context_key = self._get_context_key(context_options)

        if context_key in self._context_pool:
            # Move to end (most recently used)
            context = self._context_pool[context_key]
            del self._context_pool[context_key]
            self._context_pool[context_key] = context

            # Clear cookies and local storage for isolation
            try:
                await context.clear_cookies()
                await context.clear_permissions()
                logger.debug(f"Reusing browser context (key: {context_key[:8]})")
                return context
            except Exception as e:
                logger.warning(f"Failed to clear context state, creating new one: {e}")
                # Remove from pool if it can't be cleared
                del self._context_pool[context_key]

        return None

    async def _return_context_to_pool(self, context: BrowserContext, context_options: Dict[str, Any]) -> None:
        """Return a context to the pool for reuse.

        Args:
            context: Browser context to pool
            context_options: Original context creation options
        """
        if not self._enable_context_reuse:
            await context.close()
            return

        context_key = self._get_context_key(context_options)

        # Don't pool contexts with unique recording requirements
        if 'record_har_path' in context_options or 'record_video_dir' in context_options:
            await context.close()
            return

        # Manage pool size with LRU eviction
        if len(self._context_pool) >= self._max_pooled_contexts:
            # Remove oldest context
            oldest_key, oldest_context = self._context_pool.popitem(last=False)
            await oldest_context.close()
            logger.debug(f"Evicted oldest context from pool (key: {oldest_key[:8]})")

        # Add to pool
        self._context_pool[context_key] = context
        logger.debug(f"Added context to pool (key: {context_key[:8]}, total: {len(self._context_pool)})")

    async def health_check(self) -> bool:
        """Check if browser factory is healthy.
        
        Returns:
            True if browser is running and responsive
        """
        if not self.browser:
            return False
            
        try:
            # Try to create a temporary page to test browser responsiveness
            async with self.page() as page:
                await page.goto("about:blank", timeout=5000)
                return True
        except Exception as e:
            logger.error(f"Browser health check failed: {e}")
            return False
    
    async def restart_browser(self) -> None:
        """Restart browser (useful for recovering from crashes).
        
        Raises:
            Exception: If restart fails
        """
        logger.info("Restarting browser")
        config = self.config  # Save config
        await self.stop()
        self.config = config  # Restore config
        await self.start()
        logger.info("Browser restarted successfully")
    
    @property
    def is_running(self) -> bool:
        """Check if browser factory is running."""
        if self.browser is None:
            return False
        try:
            # Use _is_closed() for backward compatibility with tests
            return not self.browser._is_closed()
        except Exception:
            # Fallback for cases where _is_closed() is not available
            return True
    
    @property
    def context_count(self) -> int:
        """Get current number of active contexts."""
        return self._context_count
    
    def __repr__(self) -> str:
        """String representation of browser factory."""
        return (
            f"BrowserFactory(engine={self.config.engine}, "
            f"headless={self.config.headless}, "
            f"running={self.is_running}, "
            f"contexts={self.context_count})"
        )


# Convenience function for creating browser factory
def create_browser_factory(
    engine: str = BrowserEngineType.CHROMIUM,
    headless: bool = True,
    **kwargs
) -> BrowserFactory:
    """Create a browser factory with simple configuration.
    
    Args:
        engine: Browser engine to use
        headless: Run in headless mode
        **kwargs: Additional configuration options
        
    Returns:
        Configured BrowserFactory instance
    """
    config = BrowserConfig(engine=engine, headless=headless, **kwargs)
    return BrowserFactory(config)


# Default browser factory for common use cases
def create_default_factory() -> BrowserFactory:
    """Create browser factory with default configuration."""
    return create_browser_factory(
        engine=BrowserEngineType.CHROMIUM,
        headless=True,
        viewport={'width': 1920, 'height': 1080},
        ignore_https_errors=True,
    )


def create_debug_factory() -> BrowserFactory:
    """Create browser factory optimized for debugging."""
    return create_browser_factory(
        engine=BrowserEngineType.CHROMIUM,
        headless=False,
        devtools=True,
        slow_mo=500,
        viewport={'width': 1920, 'height': 1080},
        ignore_https_errors=True,
    )