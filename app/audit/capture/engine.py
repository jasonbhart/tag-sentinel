"""Main capture engine that coordinates all browser capture components.

This module provides the CaptureEngine class that orchestrates browser factory,
page sessions, and all observers to provide a unified interface for capturing
web pages with comprehensive analytics tracking data.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from urllib.parse import urlparse

from .browser_factory import BrowserFactory, BrowserConfig, create_default_factory
from .page_session import PageSession, PageSessionConfig, WaitStrategy
from ..models.capture import PageResult, CaptureStatus
from ..models.crawl import PagePlan

logger = logging.getLogger(__name__)


class CaptureEngineConfig:
    """Configuration for the capture engine."""
    
    def __init__(
        self,
        # Browser configuration
        browser_config: Optional[BrowserConfig] = None,
        max_concurrent_pages: int = 5,
        page_timeout_ms: int = 30000,
        
        # Default page session configuration
        default_wait_strategy: str = WaitStrategy.NETWORKIDLE,
        default_wait_timeout_ms: int = 30000,
        enable_network_capture: bool = True,
        enable_console_capture: bool = True,
        enable_cookie_capture: bool = True,
        redact_cookie_values: bool = True,
        filter_console_noise: bool = True,
        
        # Artifacts configuration
        artifacts_enabled: bool = False,
        artifacts_dir: Optional[Path] = None,
        take_screenshots: bool = False,
        screenshot_on_error: bool = True,
        enable_har: bool = False,
        enable_trace: bool = False,
        
        # Error handling
        retry_attempts: int = 3,
        retry_delay_ms: int = 1000,
        continue_on_error: bool = True,
        
        # Performance
        memory_limit_mb: Optional[int] = None,
        cleanup_interval_pages: int = 50,
        
        **kwargs
    ):
        """Initialize capture engine configuration.
        
        Args:
            browser_config: Browser factory configuration
            max_concurrent_pages: Maximum concurrent page captures
            page_timeout_ms: Timeout for individual page captures
            default_wait_strategy: Default page load wait strategy
            default_wait_timeout_ms: Default page load wait timeout
            enable_network_capture: Enable network request capture
            enable_console_capture: Enable console message capture
            enable_cookie_capture: Enable cookie collection
            redact_cookie_values: Redact cookie values for privacy
            filter_console_noise: Filter noisy console messages
            artifacts_enabled: Enable artifact generation
            artifacts_dir: Directory for debug artifacts
            take_screenshots: Take screenshots of pages
            screenshot_on_error: Take screenshots on errors
            enable_har: Generate HAR files
            enable_trace: Generate Playwright traces
            retry_attempts: Number of retry attempts for failed captures
            retry_delay_ms: Delay between retry attempts
            continue_on_error: Continue processing after errors
            memory_limit_mb: Memory usage limit in MB
            cleanup_interval_pages: Pages processed between cleanups
        """
        self.browser_config = browser_config or BrowserConfig()
        self.max_concurrent_pages = max_concurrent_pages
        self.page_timeout_ms = page_timeout_ms
        
        self.default_wait_strategy = default_wait_strategy
        self.default_wait_timeout_ms = default_wait_timeout_ms
        self.enable_network_capture = enable_network_capture
        self.enable_console_capture = enable_console_capture
        self.enable_cookie_capture = enable_cookie_capture
        self.redact_cookie_values = redact_cookie_values
        self.filter_console_noise = filter_console_noise
        
        self.artifacts_enabled = artifacts_enabled
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.take_screenshots = take_screenshots
        self.screenshot_on_error = screenshot_on_error
        self.enable_har = enable_har
        self.enable_trace = enable_trace
        
        self.retry_attempts = retry_attempts
        self.retry_delay_ms = retry_delay_ms
        self.continue_on_error = continue_on_error
        
        self.memory_limit_mb = memory_limit_mb
        self.cleanup_interval_pages = cleanup_interval_pages
        
        # Store extra config
        self.extra_config = kwargs
    
    def create_page_session_config(self, **overrides) -> PageSessionConfig:
        """Create page session config with defaults and overrides.
        
        Args:
            **overrides: Configuration overrides
            
        Returns:
            PageSessionConfig instance
        """
        config_params = {
            'wait_strategy': self.default_wait_strategy,
            'wait_timeout_ms': self.default_wait_timeout_ms,
            'enable_network_capture': self.enable_network_capture,
            'enable_console_capture': self.enable_console_capture,
            'enable_cookie_capture': self.enable_cookie_capture,
            'redact_cookie_values': self.redact_cookie_values,
            'filter_console_noise': self.filter_console_noise,
            'artifacts_dir': self.artifacts_dir if self.artifacts_enabled else None,
            'take_screenshot': self.take_screenshots,
            'screenshot_on_error': self.screenshot_on_error,
            'enable_har': self.enable_har,
            'enable_trace': self.enable_trace,
        }
        
        # Apply overrides
        config_params.update(overrides)
        
        return PageSessionConfig(**config_params)


class CaptureEngine:
    """Main capture engine that coordinates all components."""
    
    def __init__(self, config: Optional[CaptureEngineConfig] = None):
        """Initialize capture engine.
        
        Args:
            config: Engine configuration (uses defaults if None)
        """
        self.config = config or CaptureEngineConfig()
        self.browser_factory: Optional[BrowserFactory] = None
        self._is_running = False
        self._pages_processed = 0
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._callbacks: List[Callable[[PageResult], None]] = []
        
        # Statistics
        self.stats = {
            'pages_attempted': 0,
            'pages_successful': 0,
            'pages_failed': 0,
            'pages_timeout': 0,
            'total_duration_ms': 0,
            'start_time': None,
            'errors': [],
        }
    
    async def start(self) -> None:
        """Start the capture engine and initialize browser."""
        if self._is_running:
            logger.warning("Capture engine already running")
            return
        
        logger.info("Starting capture engine")
        
        try:
            # Create browser factory
            self.browser_factory = BrowserFactory(self.config.browser_config)
            await self.browser_factory.start()
            
            # Initialize concurrency control
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_pages)
            
            # Reset statistics
            self.stats['start_time'] = datetime.utcnow()
            self._is_running = True
            
            logger.info("Capture engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start capture engine: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the capture engine and cleanup resources."""
        logger.info("Stopping capture engine")
        
        try:
            if self.browser_factory:
                await self.browser_factory.stop()
                self.browser_factory = None
            
            self._is_running = False
            self._semaphore = None
            
            logger.info("Capture engine stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping capture engine: {e}")
    
    def add_callback(self, callback: Callable[[PageResult], None]) -> None:
        """Add callback to be called for each completed page capture.
        
        Args:
            callback: Function to call with PageResult
        """
        self._callbacks.append(callback)
    
    async def capture_page(
        self, 
        url: str, 
        session_config_overrides: Optional[Dict[str, Any]] = None
    ) -> PageResult:
        """Capture a single page.
        
        Args:
            url: URL to capture
            session_config_overrides: Configuration overrides for this page
            
        Returns:
            PageResult with captured data
        """
        if not self._is_running:
            raise RuntimeError("Capture engine not started. Call start() first.")
        
        session_config_overrides = session_config_overrides or {}
        
        # Create page session configuration
        page_config = self.config.create_page_session_config(**session_config_overrides)
        
        # Perform capture with retry logic
        return await self._capture_page_with_retry(url, page_config)
    
    async def _capture_page_with_retry(self, url: str, config: PageSessionConfig) -> PageResult:
        """Capture page with retry logic.
        
        Args:
            url: URL to capture
            config: Page session configuration
            
        Returns:
            PageResult (may indicate failure)
        """
        last_error = None
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                async with self._semaphore:
                    return await self._capture_page_single_attempt(url, config)
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Capture attempt {attempt + 1} failed for {url}: {e}")
                
                if attempt < self.config.retry_attempts:
                    delay = self.config.retry_delay_ms / 1000.0 * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All capture attempts failed for {url}: {e}")
        
        # Create failed result
        result = PageResult(
            url=url,
            capture_status=CaptureStatus.FAILED,
            capture_error=str(last_error),
            capture_time=datetime.utcnow()
        )
        
        self._update_stats(result)
        self._call_callbacks(result)
        return result
    
    async def _capture_page_single_attempt(self, url: str, config: PageSessionConfig) -> PageResult:
        """Perform single page capture attempt.
        
        Args:
            url: URL to capture
            config: Page session configuration
            
        Returns:
            PageResult with captured data
        """
        start_time = datetime.utcnow()
        
        try:
            # Create new page context
            async with self.browser_factory.page() as page:
                # Create and execute page session
                session = PageSession(page, config)
                result = await session.capture_page(url)
                
                self._pages_processed += 1
                
                # Periodic cleanup
                if self._pages_processed % self.config.cleanup_interval_pages == 0:
                    await self._perform_cleanup()
                
                self._update_stats(result)
                self._call_callbacks(result)
                
                return result
                
        except Exception as e:
            # Create error result
            result = PageResult(
                url=url,
                capture_status=CaptureStatus.FAILED,
                capture_error=str(e),
                capture_time=start_time,
                load_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            self._update_stats(result)
            self._call_callbacks(result)
            raise e
    
    async def capture_pages(
        self, 
        urls: List[str],
        session_config_overrides: Optional[Dict[str, Any]] = None
    ) -> List[PageResult]:
        """Capture multiple pages concurrently.
        
        Args:
            urls: List of URLs to capture
            session_config_overrides: Configuration overrides
            
        Returns:
            List of PageResult objects
        """
        if not self._is_running:
            raise RuntimeError("Capture engine not started. Call start() first.")
        
        logger.info(f"Capturing {len(urls)} pages")
        
        # Create capture tasks
        tasks = []
        for url in urls:
            task = asyncio.create_task(
                self.capture_page(url, session_config_overrides)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                if not self.config.continue_on_error:
                    # Cancel remaining tasks
                    for remaining_task in tasks:
                        if not remaining_task.done():
                            remaining_task.cancel()
                    raise
                
                logger.error(f"Page capture failed: {e}")
        
        logger.info(f"Batch capture completed: {len(results)} results")
        return results
    
    async def capture_page_plans(
        self, 
        page_plans: List[PagePlan]
    ) -> List[PageResult]:
        """Capture pages from PagePlan objects.
        
        Args:
            page_plans: List of PagePlan objects with configuration
            
        Returns:
            List of PageResult objects
        """
        results = []
        
        for plan in page_plans:
            # Create session config overrides from plan
            overrides = {}
            
            if plan.load_wait_strategy:
                overrides['wait_strategy'] = plan.load_wait_strategy.value
            
            if plan.load_wait_timeout:
                overrides['wait_timeout_ms'] = plan.load_wait_timeout * 1000
            
            # Add plan metadata to overrides
            overrides['pre_steps'] = []  # Could be populated from plan.metadata
            
            try:
                result = await self.capture_page(str(plan.url), overrides)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to capture page plan {plan.url}: {e}")
                if not self.config.continue_on_error:
                    break
        
        return results
    
    def _update_stats(self, result: PageResult) -> None:
        """Update engine statistics.
        
        Args:
            result: Completed page result
        """
        self.stats['pages_attempted'] += 1
        
        if result.is_successful:
            self.stats['pages_successful'] += 1
        elif result.capture_status == CaptureStatus.TIMEOUT:
            self.stats['pages_timeout'] += 1
        else:
            self.stats['pages_failed'] += 1
        
        if result.load_time_ms:
            self.stats['total_duration_ms'] += result.load_time_ms
        
        # Track errors
        if result.capture_error:
            self.stats['errors'].append({
                'url': result.url,
                'error': result.capture_error,
                'timestamp': result.capture_time.isoformat(),
            })
            
            # Limit error history
            if len(self.stats['errors']) > 100:
                self.stats['errors'] = self.stats['errors'][-100:]
    
    def _call_callbacks(self, result: PageResult) -> None:
        """Call all registered callbacks.
        
        Args:
            result: Completed page result
        """
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in capture engine callback: {e}")
    
    async def _perform_cleanup(self) -> None:
        """Perform periodic cleanup to manage memory usage."""
        try:
            # Check browser health
            if self.browser_factory and not await self.browser_factory.health_check():
                logger.warning("Browser health check failed, restarting browser")
                await self.browser_factory.restart_browser()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.debug(f"Cleanup performed after {self._pages_processed} pages")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator['CaptureEngine', None]:
        """Context manager for engine lifecycle.
        
        Yields:
            Started capture engine that will be automatically stopped
        """
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats['pages_attempted'] > 0:
            stats['success_rate'] = (stats['pages_successful'] / stats['pages_attempted']) * 100
            stats['average_load_time_ms'] = stats['total_duration_ms'] / stats['pages_attempted']
        else:
            stats['success_rate'] = 0
            stats['average_load_time_ms'] = 0
        
        # Calculate runtime
        if stats['start_time']:
            runtime = (datetime.utcnow() - stats['start_time']).total_seconds()
            stats['runtime_seconds'] = runtime
            
            if runtime > 0:
                stats['pages_per_second'] = stats['pages_attempted'] / runtime
            else:
                stats['pages_per_second'] = 0
        
        # Browser stats
        if self.browser_factory:
            stats['browser_running'] = self.browser_factory.is_running
            stats['browser_contexts'] = self.browser_factory.context_count
        
        stats['is_running'] = self._is_running
        stats['pages_processed'] = self._pages_processed
        
        return stats
    
    def export_summary(self) -> Dict[str, Any]:
        """Export summary for reporting.
        
        Returns:
            Dictionary with summary information
        """
        stats = self.get_stats()
        
        return {
            'engine_status': 'running' if self._is_running else 'stopped',
            'pages_processed': stats['pages_attempted'],
            'success_rate': f"{stats['success_rate']:.1f}%",
            'average_load_time': f"{stats['average_load_time_ms']:.0f}ms",
            'runtime': f"{stats.get('runtime_seconds', 0):.1f}s",
            'processing_rate': f"{stats.get('pages_per_second', 0):.2f} pages/sec",
            'error_count': len(stats['errors']),
            'configuration': {
                'max_concurrent': self.config.max_concurrent_pages,
                'wait_strategy': self.config.default_wait_strategy,
                'artifacts_enabled': self.config.artifacts_enabled,
            }
        }
    
    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._is_running
    
    def __repr__(self) -> str:
        """String representation of capture engine."""
        stats = self.get_stats()
        return (
            f"CaptureEngine(running={self._is_running}, "
            f"processed={stats['pages_attempted']}, "
            f"success_rate={stats['success_rate']:.1f}%)"
        )


# Convenience functions for common use cases

def create_capture_engine(
    headless: bool = True,
    max_concurrent_pages: int = 5,
    enable_artifacts: bool = False,
    artifacts_dir: Optional[Path] = None,
    **kwargs
) -> CaptureEngine:
    """Create capture engine with common configuration.
    
    Args:
        headless: Run browser in headless mode
        max_concurrent_pages: Maximum concurrent page captures
        enable_artifacts: Enable artifact generation
        artifacts_dir: Directory for artifacts
        **kwargs: Additional configuration options
        
    Returns:
        Configured CaptureEngine instance
    """
    browser_config = BrowserConfig(headless=headless)
    
    engine_config = CaptureEngineConfig(
        browser_config=browser_config,
        max_concurrent_pages=max_concurrent_pages,
        artifacts_enabled=enable_artifacts,
        artifacts_dir=artifacts_dir,
        **kwargs
    )
    
    return CaptureEngine(engine_config)


def create_debug_capture_engine(artifacts_dir: Optional[Path] = None) -> CaptureEngine:
    """Create capture engine optimized for debugging.
    
    Args:
        artifacts_dir: Directory for debug artifacts
        
    Returns:
        Debug-optimized CaptureEngine instance
    """
    browser_config = BrowserConfig(
        headless=False,
        devtools=True,
        slow_mo=500
    )
    
    engine_config = CaptureEngineConfig(
        browser_config=browser_config,
        max_concurrent_pages=1,  # Sequential processing for debugging
        artifacts_enabled=True,
        artifacts_dir=artifacts_dir or Path.cwd() / "debug_artifacts",
        take_screenshots=True,
        enable_har=True,
        enable_trace=True,
    )
    
    return CaptureEngine(engine_config)